"""Shared configuration for the RAG system.

This module provides a single source of truth for all configuration parameters
used across the RAG pipeline components (indexing, retrieval, generation).
"""

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Central configuration dataclass for the RAG system.

    All components should import and use this config rather than
    hardcoding values independently.
    """

    # Embedding
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Chunking
    max_chunk_size: int = 1500       # characters (200-10000)
    chunk_overlap: int = 200         # characters (< 50% of max_chunk_size)
    min_chunk_size: int = 100        # minimum content characters

    # Retrieval
    dense_search_k: int = 50         # candidates from vector search
    bm25_search_k: int = 50          # candidates from BM25
    rrf_k: int = 60                  # RRF smoothing parameter (1-1000)
    fusion_candidates: int = 20      # chunks passed to reranker (1-100)

    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_top_k: int = 5          # final chunks to LLM (1-20)
    reranker_timeout: float = 10.0   # seconds

    # Confidence
    confidence_threshold: float = 0.3  # minimum relevance score
    faithfulness_threshold: float = 0.7  # low-confidence warning trigger

    # Paths
    db_directory: str = "./db"
    bm25_index_path: str = "./db/bm25_index.pkl"
    manifest_path: str = "./db/indexing_manifest.json"

    # LLM
    llm_model: str = "mistral"
    llm_temperature: float = 0.0
    llm_timeout: int = 60

    def __post_init__(self) -> None:
        """Validate configuration parameter ranges."""
        # max_chunk_size must be between 200 and 10000
        if not (200 <= self.max_chunk_size <= 10000):
            raise ValueError(
                f"max_chunk_size must be between 200 and 10000, got {self.max_chunk_size}"
            )

        # chunk_overlap must be less than 50% of max_chunk_size
        if self.chunk_overlap >= self.max_chunk_size * 0.5:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than 50% of "
                f"max_chunk_size ({self.max_chunk_size}), i.e. < {self.max_chunk_size * 0.5}"
            )

        # min_chunk_size must be less than max_chunk_size
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) must be less than "
                f"max_chunk_size ({self.max_chunk_size})"
            )

        # rrf_k must be between 1 and 1000
        if not (1 <= self.rrf_k <= 1000):
            raise ValueError(
                f"rrf_k must be between 1 and 1000, got {self.rrf_k}"
            )

        # fusion_candidates must be between 1 and 100
        if not (1 <= self.fusion_candidates <= 100):
            raise ValueError(
                f"fusion_candidates must be between 1 and 100, got {self.fusion_candidates}"
            )

        # reranker_top_k must be between 1 and 20
        if not (1 <= self.reranker_top_k <= 20):
            raise ValueError(
                f"reranker_top_k must be between 1 and 20, got {self.reranker_top_k}"
            )

        # confidence_threshold must be between 0.0 and 1.0
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

        # faithfulness_threshold must be between 0.0 and 1.0
        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            raise ValueError(
                f"faithfulness_threshold must be between 0.0 and 1.0, got {self.faithfulness_threshold}"
            )
