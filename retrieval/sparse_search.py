"""Sparse BM25 keyword search."""

from models import Chunk, ScoredChunk
from config import RAGConfig
from indexing.bm25_index import BM25Index


class SparseSearch:
    """Sparse keyword search using BM25 index.
    
    Wraps BM25Index to perform keyword-based search.
    Handles missing index gracefully.
    """
    
    def __init__(self, config: RAGConfig = None, bm25_index: BM25Index = None):
        if config is None:
            config = RAGConfig()
        self.config = config
        self.k = config.bm25_search_k
        
        if bm25_index is not None:
            self._index = bm25_index
        else:
            self._index = BM25Index(config)
            self._index.load()
    
    def search(self, query: str, k: int = None) -> list[ScoredChunk]:
        """Search BM25 index for matching chunks.
        
        Args:
            query: The search query string.
            k: Number of results. Defaults to config.bm25_search_k.
            
        Returns:
            List of ScoredChunk with source_method="bm25", sorted by score descending.
            Returns empty list if index is not available.
        """
        if k is None:
            k = self.k
        
        if not self._index.is_built:
            return []
        
        results = self._index.search(query, k=k)
        
        scored_chunks = []
        for chunk, score in results:
            scored_chunks.append(ScoredChunk(
                chunk=chunk,
                score=float(score),
                source_method="bm25",
            ))
        
        return scored_chunks
