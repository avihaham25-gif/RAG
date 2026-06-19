"""Cross-encoder reranker for improving retrieval precision."""

import signal
import time
from models import Chunk, ScoredChunk
from config import RAGConfig


class RerankerTimeoutError(Exception):
    """Raised when reranker scoring exceeds the configured timeout."""
    pass


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.
    
    The cross-encoder jointly encodes (query, document) pairs to produce
    a relevance score, providing more accurate relevance estimates than
    bi-encoder similarity.
    
    Falls back to original ranking if:
    - The model fails to initialize
    - Scoring exceeds the configured timeout (default: 10 seconds)
    - Any other error occurs during scoring
    """
    
    def __init__(self, config: RAGConfig = None):
        """Initialize the reranker.
        
        Args:
            config: RAGConfig instance. Uses defaults if None.
        """
        if config is None:
            config = RAGConfig()
        self.model_name = config.reranker_model
        self.top_k = config.reranker_top_k
        self.timeout = config.reranker_timeout
        self._model = None
        self._initialized = False
        self._init_failed = False
    
    def _ensure_loaded(self) -> bool:
        """Lazily load the cross-encoder model.
        
        Returns:
            True if model is loaded successfully, False otherwise.
        """
        if self._initialized:
            return self._model is not None
        
        if self._init_failed:
            return False
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            return True
        except ImportError:
            print(f"Warning: sentence-transformers not installed, reranking disabled")
            self._init_failed = True
            self._initialized = True
            return False
        except Exception as e:
            print(f"Warning: failed to load reranker model '{self.model_name}': {e}")
            self._init_failed = True
            self._initialized = True
            return False
    
    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int = None
    ) -> list[ScoredChunk]:
        """Rerank chunks by relevance to the query using cross-encoder.
        
        Args:
            query: The original user query.
            chunks: List of ScoredChunk from retrieval (sorted by retrieval score).
            top_k: Number of top results to return. Defaults to config.reranker_top_k.
            
        Returns:
            List of ScoredChunk sorted by reranker relevance score (descending),
            limited to top_k results. Each chunk has source_method="reranker"
            and the relevance_score attached to metadata.
            
            Falls back to original top-k if reranking fails.
        """
        if top_k is None:
            top_k = self.top_k
        
        # If no chunks, return empty
        if not chunks:
            return []
        
        # If fewer chunks than top_k, we'll return all of them (Req 5.3)
        effective_k = min(top_k, len(chunks))
        
        # Try to load model; if fails, return original ranking (Req 5.5)
        if not self._ensure_loaded():
            return self._fallback(chunks, effective_k)
        
        # Score each chunk against the query with timeout
        try:
            scores = self._score_with_timeout(query, chunks)
        except (RerankerTimeoutError, Exception) as e:
            print(f"Warning: reranking failed ({e}), using original ranking")
            return self._fallback(chunks, effective_k)
        
        # Normalize scores to [0.0, 1.0] range
        scores = self._normalize_scores(scores)
        
        # Create reranked results
        scored_pairs = list(zip(chunks, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with updated scores and metadata
        results = []
        for original_chunk, score in scored_pairs[:effective_k]:
            # Attach relevance score to metadata (Req 5.4)
            updated_metadata = dict(original_chunk.chunk.metadata)
            updated_metadata["relevance_score"] = score
            
            reranked_chunk = Chunk(
                content=original_chunk.chunk.content,
                metadata=updated_metadata,
            )
            
            results.append(ScoredChunk(
                chunk=reranked_chunk,
                score=score,
                source_method="reranker",
            ))
        
        return results
    
    def _score_with_timeout(self, query: str, chunks: list[ScoredChunk]) -> list[float]:
        """Score chunks with a timeout constraint.
        
        Args:
            query: The search query.
            chunks: Chunks to score.
            
        Returns:
            List of raw scores from the cross-encoder.
            
        Raises:
            RerankerTimeoutError: If scoring exceeds timeout.
        """
        # Prepare query-document pairs
        pairs = [(query, chunk.chunk.content) for chunk in chunks]
        
        start_time = time.time()
        
        # Score using the cross-encoder
        scores = self._model.predict(pairs)
        
        elapsed = time.time() - start_time
        if elapsed > self.timeout:
            raise RerankerTimeoutError(
                f"Reranker scoring took {elapsed:.1f}s, exceeding timeout of {self.timeout}s"
            )
        
        return list(scores)
    
    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to [0.0, 1.0] range using sigmoid.
        
        Cross-encoder raw scores can be any real number.
        We apply sigmoid to map them to [0, 1].
        """
        import math
        
        normalized = []
        for score in scores:
            try:
                sigmoid = 1.0 / (1.0 + math.exp(-float(score)))
            except (OverflowError, ValueError):
                sigmoid = 0.0 if score < 0 else 1.0
            normalized.append(sigmoid)
        
        return normalized
    
    def _fallback(self, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return original top-k chunks without reranking.
        
        Preserves original scores and order but updates source_method.
        
        Args:
            chunks: Original ranked chunks.
            top_k: Number of results to return.
            
        Returns:
            Top-k chunks from original ranking with source_method="reranker".
        """
        results = []
        for chunk in chunks[:top_k]:
            # Keep original score but mark as coming through reranker (fallback)
            updated_metadata = dict(chunk.chunk.metadata)
            updated_metadata["relevance_score"] = chunk.score
            
            fallback_chunk = Chunk(
                content=chunk.chunk.content,
                metadata=updated_metadata,
            )
            
            results.append(ScoredChunk(
                chunk=fallback_chunk,
                score=chunk.score,
                source_method="reranker",
            ))
        
        return results
