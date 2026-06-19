"""Reciprocal Rank Fusion (RRF) for merging ranked result lists."""

from models import Chunk, ScoredChunk
from config import RAGConfig


class RRFFusion:
    """Merges ranked lists using Reciprocal Rank Fusion.
    
    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranker i
    
    Where:
    - k is a smoothing constant (default 60, configurable via rrf_k)
    - rank_i(d) is the rank of document d in list i (1-based)
    """
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.k = config.rrf_k
        self.top_n = config.fusion_candidates
    
    def fuse(
        self,
        dense_results: list[ScoredChunk],
        sparse_results: list[ScoredChunk],
        top_n: int = None,
    ) -> list[ScoredChunk]:
        """Merge ranked lists using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense vector search (sorted by score desc).
            sparse_results: Results from sparse BM25 search (sorted by score desc).
            top_n: Number of results to return. Defaults to config.fusion_candidates.
            
        Returns:
            List of ScoredChunk sorted by fused RRF score descending,
            with source_method="rrf".
            
        Handles edge cases:
        - If one list is empty, returns results from the other list
        - If both are empty, returns empty list
        """
        if top_n is None:
            top_n = self.top_n
        
        # Edge case: both empty
        if not dense_results and not sparse_results:
            return []
        
        # Edge case: one empty - return the other (Req 4.6)
        if not dense_results:
            return [
                ScoredChunk(chunk=sc.chunk, score=sc.score, source_method="rrf")
                for sc in sparse_results[:top_n]
            ]
        if not sparse_results:
            return [
                ScoredChunk(chunk=sc.chunk, score=sc.score, source_method="rrf")
                for sc in dense_results[:top_n]
            ]
        
        # Build RRF scores
        # Use chunk content + source as identity key
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        
        # Process dense results
        for rank, scored_chunk in enumerate(dense_results, start=1):
            key = self._chunk_key(scored_chunk.chunk)
            rrf_score = 1.0 / (self.k + rank)
            chunk_scores[key] = chunk_scores.get(key, 0.0) + rrf_score
            chunk_map[key] = scored_chunk.chunk
        
        # Process sparse results
        for rank, scored_chunk in enumerate(sparse_results, start=1):
            key = self._chunk_key(scored_chunk.chunk)
            rrf_score = 1.0 / (self.k + rank)
            chunk_scores[key] = chunk_scores.get(key, 0.0) + rrf_score
            chunk_map[key] = scored_chunk.chunk
        
        # Sort by fused score descending
        sorted_keys = sorted(chunk_scores.keys(), key=lambda k: chunk_scores[k], reverse=True)
        
        # Return top_n results
        results = []
        for key in sorted_keys[:top_n]:
            results.append(ScoredChunk(
                chunk=chunk_map[key],
                score=chunk_scores[key],
                source_method="rrf",
            ))
        
        return results
    
    def _chunk_key(self, chunk: Chunk) -> str:
        """Generate a unique key for a chunk based on content and source."""
        source = chunk.metadata.get("source", "")
        chunk_idx = chunk.metadata.get("chunk_index", -1)
        # Use content hash + source + index for unique identification
        return f"{source}:{chunk_idx}:{hash(chunk.content)}"
