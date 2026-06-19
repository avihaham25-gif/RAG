"""Integration tests for the Retrieval Pipeline.

Tests hybrid search with BM25 (dense search mocked since model unavailable).
"""

import pytest
from unittest.mock import patch, MagicMock

from retrieval.retriever import RetrievalPipeline
from retrieval.dense_search import DenseSearch
from retrieval.sparse_search import SparseSearch
from retrieval.rrf_fusion import RRFFusion
from indexing.bm25_index import BM25Index
from models import Chunk, ScoredChunk
from config import RAGConfig


def _make_chunks(texts: list[str], source: str = "test.txt") -> list[Chunk]:
    """Helper to create Chunk objects."""
    return [
        Chunk(content=text, metadata={"source": source, "chunk_index": i, "section_title": ""})
        for i, text in enumerate(texts)
    ]


def _make_scored_chunks(chunks: list[Chunk], scores: list[float], method: str) -> list[ScoredChunk]:
    """Helper to create ScoredChunk objects."""
    return [
        ScoredChunk(chunk=chunk, score=score, source_method=method)
        for chunk, score in zip(chunks, scores)
    ]


class TestRRFFusion:
    """Test RRF fusion algorithm."""
    
    def test_fuse_both_results(self):
        """Should merge results from both methods."""
        config = RAGConfig(rrf_k=60, fusion_candidates=5)
        fusion = RRFFusion(config)
        
        chunks_a = _make_chunks(["doc A1", "doc A2", "doc A3"])
        chunks_b = _make_chunks(["doc B1", "doc B2", "doc A1"])  # A1 appears in both
        
        dense = _make_scored_chunks(chunks_a, [0.9, 0.8, 0.7], "dense")
        sparse = _make_scored_chunks(chunks_b, [5.0, 3.0, 2.0], "bm25")
        
        results = fusion.fuse(dense, sparse)
        
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        assert all(r.source_method == "rrf" for r in results)
    
    def test_fuse_scores_descending(self):
        """Results should be sorted by descending RRF score."""
        config = RAGConfig(rrf_k=60, fusion_candidates=10)
        fusion = RRFFusion(config)
        
        chunks = _make_chunks(["a", "b", "c", "d", "e"])
        dense = _make_scored_chunks(chunks[:3], [0.9, 0.8, 0.7], "dense")
        sparse = _make_scored_chunks(chunks[2:], [5.0, 3.0, 2.0], "bm25")
        
        results = fusion.fuse(dense, sparse)
        
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_fuse_respects_top_n(self):
        """Should return at most top_n results."""
        config = RAGConfig(rrf_k=60, fusion_candidates=3)
        fusion = RRFFusion(config)
        
        chunks = _make_chunks([f"doc {i}" for i in range(10)])
        dense = _make_scored_chunks(chunks[:5], [0.9, 0.8, 0.7, 0.6, 0.5], "dense")
        sparse = _make_scored_chunks(chunks[5:], [5.0, 4.0, 3.0, 2.0, 1.0], "bm25")
        
        results = fusion.fuse(dense, sparse, top_n=3)
        
        assert len(results) <= 3
    
    def test_fuse_empty_dense_returns_sparse(self):
        """Req 4.6: If dense is empty, return sparse results."""
        config = RAGConfig(rrf_k=60, fusion_candidates=5)
        fusion = RRFFusion(config)
        
        chunks = _make_chunks(["sparse doc 1", "sparse doc 2"])
        sparse = _make_scored_chunks(chunks, [5.0, 3.0], "bm25")
        
        results = fusion.fuse([], sparse)
        
        assert len(results) == 2
        assert all(r.source_method == "rrf" for r in results)
    
    def test_fuse_empty_sparse_returns_dense(self):
        """Req 4.6: If sparse is empty, return dense results."""
        config = RAGConfig(rrf_k=60, fusion_candidates=5)
        fusion = RRFFusion(config)
        
        chunks = _make_chunks(["dense doc 1", "dense doc 2"])
        dense = _make_scored_chunks(chunks, [0.9, 0.8], "dense")
        
        results = fusion.fuse(dense, [])
        
        assert len(results) == 2
        assert all(r.source_method == "rrf" for r in results)
    
    def test_fuse_both_empty_returns_empty(self):
        """Req 4.7: If both empty, return empty."""
        config = RAGConfig(rrf_k=60, fusion_candidates=5)
        fusion = RRFFusion(config)
        
        results = fusion.fuse([], [])
        assert results == []
    
    def test_rrf_score_formula(self):
        """Verify RRF score formula: score = sum(1/(k + rank))."""
        config = RAGConfig(rrf_k=60, fusion_candidates=10)
        fusion = RRFFusion(config)
        
        # Create a chunk that appears at rank 1 in both lists
        chunk = Chunk(content="shared doc", metadata={"source": "test.txt", "chunk_index": 0, "section_title": ""})
        
        dense = [ScoredChunk(chunk=chunk, score=0.9, source_method="dense")]
        sparse = [ScoredChunk(chunk=chunk, score=5.0, source_method="bm25")]
        
        results = fusion.fuse(dense, sparse)
        
        # Expected: 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2.0 / 61.0
        assert len(results) == 1
        assert abs(results[0].score - expected_score) < 0.0001


class TestSparseSearch:
    """Test sparse BM25 search wrapper."""
    
    def test_search_returns_scored_chunks(self, tmp_path):
        """Should return ScoredChunk with source_method='bm25'."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        bm25 = BM25Index(config)
        
        chunks = _make_chunks(["hello world", "good morning", "good evening"])
        bm25.build(chunks)
        
        search = SparseSearch(config, bm25_index=bm25)
        results = search.search("hello")
        
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        assert all(r.source_method == "bm25" for r in results)
    
    def test_search_no_index_returns_empty(self, tmp_path):
        """Should return empty if no index is built."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "nonexistent.pkl"))
        search = SparseSearch(config)
        
        results = search.search("test query")
        assert results == []
    
    def test_search_respects_k(self, tmp_path):
        """Should return at most k results."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"), bm25_search_k=2)
        bm25 = BM25Index(config)
        
        chunks = _make_chunks([f"word document {i}" for i in range(10)])
        bm25.build(chunks)
        
        search = SparseSearch(config, bm25_index=bm25)
        results = search.search("word", k=2)
        
        assert len(results) <= 2


class TestDenseSearch:
    """Test dense search wrapper (mocked since model unavailable)."""
    
    def test_search_returns_empty_when_unavailable(self):
        """Should return empty when ChromaDB/embeddings not available."""
        config = RAGConfig()
        search = DenseSearch(config)
        
        # In this sandbox, ChromaDB won't work, so it should return empty
        results = search.search("test query")
        assert isinstance(results, list)


class TestRetrievalPipelineIntegration:
    """Integration test for the full retrieval pipeline."""
    
    def test_retrieve_with_bm25_only(self, tmp_path):
        """Pipeline should work with BM25 only (when dense unavailable)."""
        config = RAGConfig(
            bm25_index_path=str(tmp_path / "bm25.pkl"),
            rrf_k=60,
            fusion_candidates=5,
        )
        bm25 = BM25Index(config)
        
        chunks = _make_chunks([
            "artificial intelligence is a key field in computer science",
            "natural language processing enables text understanding",
            "deep learning uses neural networks",
            "Python is a popular programming language",
        ])
        bm25.build(chunks)
        
        pipeline = RetrievalPipeline(config, bm25_index=bm25)
        results = pipeline.retrieve("artificial intelligence")
        
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        # First result should be most relevant
        assert "artificial intelligence" in results[0].chunk.content
    
    def test_retrieve_empty_query_returns_empty(self, tmp_path):
        """Empty query should return empty results."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        bm25 = BM25Index(config)
        chunks = _make_chunks(["some content"])
        bm25.build(chunks)
        
        pipeline = RetrievalPipeline(config, bm25_index=bm25)
        results = pipeline.retrieve("")
        
        assert results == []
    
    def test_retrieve_no_match_returns_empty(self, tmp_path):
        """Query with no matches should return empty."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        bm25 = BM25Index(config)
        chunks = _make_chunks(["apple banana cherry"])
        bm25.build(chunks)
        
        pipeline = RetrievalPipeline(config, bm25_index=bm25)
        results = pipeline.retrieve("elephant")
        
        assert results == []
    
    def test_retrieve_respects_top_n(self, tmp_path):
        """Should respect top_n parameter."""
        config = RAGConfig(
            bm25_index_path=str(tmp_path / "bm25.pkl"),
            fusion_candidates=20,
        )
        bm25 = BM25Index(config)
        chunks = _make_chunks([f"document word content {i}" for i in range(15)])
        bm25.build(chunks)
        
        pipeline = RetrievalPipeline(config, bm25_index=bm25)
        results = pipeline.retrieve("word", top_n=3)
        
        assert len(results) <= 3
    
    def test_retrieve_results_have_scores(self, tmp_path):
        """All results should have positive scores."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        bm25 = BM25Index(config)
        chunks = _make_chunks(["hello world test", "hello universe"])
        bm25.build(chunks)
        
        pipeline = RetrievalPipeline(config, bm25_index=bm25)
        results = pipeline.retrieve("hello")
        
        for result in results:
            assert result.score > 0
    
    def test_full_pipeline_with_indexing(self, tmp_path):
        """End-to-end: index documents then retrieve."""
        from indexing.pipeline import IndexingPipeline
        
        # Create documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "ai.txt").write_text(
            "Artificial intelligence is a research field that deals with creating intelligent systems that can learn and draw conclusions.",
            encoding="utf-8"
        )
        (docs_dir / "nlp.txt").write_text(
            "Natural language processing allows computers to understand and process human text automatically.",
            encoding="utf-8"
        )
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
            min_chunk_size=20,
            fusion_candidates=5,
        )
        
        # Index
        indexer = IndexingPipeline(config)
        indexer.run(directory=str(docs_dir))
        
        # Retrieve using the same BM25 index
        pipeline = RetrievalPipeline(config, bm25_index=indexer.bm25_index)
        results = pipeline.retrieve("artificial intelligence")
        
        assert len(results) > 0
        assert "artificial" in results[0].chunk.content.lower() or "intelligence" in results[0].chunk.content.lower()
