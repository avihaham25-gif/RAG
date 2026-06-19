"""Retrieval pipeline combining dense search, sparse search, and RRF fusion."""

from models import ScoredChunk
from config import RAGConfig
from indexing.bm25_index import BM25Index
from retrieval.dense_search import DenseSearch
from retrieval.sparse_search import SparseSearch
from retrieval.rrf_fusion import RRFFusion


class RetrievalPipeline:
    """Combines dense and sparse search with RRF fusion.
    
    Performs hybrid retrieval:
    1. Dense vector search (ChromaDB)
    2. Sparse keyword search (BM25)
    3. Merge results via Reciprocal Rank Fusion
    
    Returns fused results as ScoredChunk objects.
    """
    
    def __init__(self, config: RAGConfig = None, bm25_index: BM25Index = None):
        if config is None:
            config = RAGConfig()
        self.config = config
        self.dense_search = DenseSearch(config)
        self.sparse_search = SparseSearch(config, bm25_index=bm25_index)
        self.rrf_fusion = RRFFusion(config)
    
    def retrieve(self, query: str, top_n: int = None) -> list[ScoredChunk]:
        """Perform hybrid retrieval combining dense and sparse search.
        
        Args:
            query: The user's search query.
            top_n: Number of fused results to return. Defaults to config.fusion_candidates.
            
        Returns:
            List of ScoredChunk sorted by RRF fused score descending.
            Returns empty list if both searches return nothing (Req 4.7).
        """
        # Perform both searches
        dense_results = self.dense_search.search(query)
        sparse_results = self.sparse_search.search(query)
        
        # Fuse results
        fused = self.rrf_fusion.fuse(dense_results, sparse_results, top_n=top_n)
        
        return fused
