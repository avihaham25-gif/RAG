"""Unit tests for the Cross-Encoder Reranker."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import math

from retrieval.reranker import CrossEncoderReranker, RerankerTimeoutError
from models import Chunk, ScoredChunk
from config import RAGConfig


def _make_scored_chunks(texts: list[str], scores: list[float]) -> list[ScoredChunk]:
    """Helper to create ScoredChunk objects."""
    return [
        ScoredChunk(
            chunk=Chunk(content=text, metadata={"source": "test.txt", "chunk_index": i, "section_title": ""}),
            score=score,
            source_method="rrf",
        )
        for i, (text, score) in enumerate(zip(texts, scores))
    ]


class TestCrossEncoderRerankerInit:
    """Test initialization."""
    
    def test_default_config(self):
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert reranker.top_k == 5
        assert reranker.timeout == 10.0
    
    def test_custom_config(self):
        config = RAGConfig(
            reranker_model="custom/model",
            reranker_top_k=10,
            reranker_timeout=5.0,
        )
        reranker = CrossEncoderReranker(config)
        assert reranker.model_name == "custom/model"
        assert reranker.top_k == 10
        assert reranker.timeout == 5.0
    
    def test_lazy_loading(self):
        """Model should not be loaded at init time."""
        reranker = CrossEncoderReranker()
        assert reranker._model is None
        assert reranker._initialized is False


class TestCrossEncoderRerankerFallback:
    """Test fallback behavior when model is unavailable."""
    
    def test_fallback_when_model_unavailable(self):
        """Should return original ranking when model can't load (Req 5.5)."""
        reranker = CrossEncoderReranker()
        # Force init failure
        reranker._init_failed = True
        reranker._initialized = True
        
        chunks = _make_scored_chunks(
            ["doc A", "doc B", "doc C"],
            [0.9, 0.7, 0.5]
        )
        
        results = reranker.rerank("test query", chunks, top_k=3)
        
        assert len(results) == 3
        assert all(r.source_method == "reranker" for r in results)
        # Scores should be preserved from original
        assert results[0].score == 0.9
        assert results[1].score == 0.7
        assert results[2].score == 0.5
    
    def test_fallback_respects_top_k(self):
        """Fallback should still limit to top_k."""
        reranker = CrossEncoderReranker()
        reranker._init_failed = True
        reranker._initialized = True
        
        chunks = _make_scored_chunks(
            ["doc A", "doc B", "doc C", "doc D", "doc E"],
            [0.9, 0.8, 0.7, 0.6, 0.5]
        )
        
        results = reranker.rerank("query", chunks, top_k=2)
        assert len(results) == 2
    
    def test_fallback_attaches_relevance_score_to_metadata(self):
        """Even in fallback, relevance_score should be in metadata (Req 5.4)."""
        reranker = CrossEncoderReranker()
        reranker._init_failed = True
        reranker._initialized = True
        
        chunks = _make_scored_chunks(["doc A"], [0.85])
        results = reranker.rerank("query", chunks)
        
        assert "relevance_score" in results[0].chunk.metadata
        assert results[0].chunk.metadata["relevance_score"] == 0.85


class TestCrossEncoderRerankerWithMockedModel:
    """Test reranking with mocked cross-encoder."""
    
    @pytest.fixture
    def mocked_reranker(self):
        """Create a reranker with mocked model."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        reranker._model = mock_model
        reranker._initialized = True
        return reranker, mock_model
    
    def test_rerank_changes_order(self, mocked_reranker):
        """Reranker should reorder chunks based on cross-encoder scores."""
        reranker, mock_model = mocked_reranker
        
        # Original order: A(0.9), B(0.7), C(0.5)
        chunks = _make_scored_chunks(
            ["less relevant doc", "most relevant doc", "medium relevant doc"],
            [0.9, 0.7, 0.5]
        )
        
        # Cross-encoder says B is most relevant, then C, then A
        mock_model.predict.return_value = [-1.0, 3.0, 1.0]
        
        results = reranker.rerank("test query", chunks, top_k=3)
        
        assert len(results) == 3
        # B should now be first (highest cross-encoder score)
        assert "most relevant" in results[0].chunk.content
        # C should be second
        assert "medium relevant" in results[1].chunk.content
        # A should be last
        assert "less relevant" in results[2].chunk.content
    
    def test_rerank_scores_normalized_to_0_1(self, mocked_reranker):
        """Scores should be normalized to [0.0, 1.0] via sigmoid (Req 5.1)."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(["doc A", "doc B"], [0.9, 0.7])
        mock_model.predict.return_value = [5.0, -5.0]
        
        results = reranker.rerank("query", chunks, top_k=2)
        
        for result in results:
            assert 0.0 <= result.score <= 1.0
    
    def test_rerank_top_k_limits_results(self, mocked_reranker):
        """Should return at most top_k results (Req 5.2)."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(
            [f"doc {i}" for i in range(10)],
            [0.9 - i * 0.05 for i in range(10)]
        )
        mock_model.predict.return_value = list(range(10, 0, -1))
        
        results = reranker.rerank("query", chunks, top_k=3)
        assert len(results) == 3
    
    def test_fewer_chunks_than_top_k(self, mocked_reranker):
        """Req 5.3: If fewer chunks than top_k, return all without error."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(["doc A", "doc B"], [0.9, 0.7])
        mock_model.predict.return_value = [2.0, 1.0]
        
        # top_k=5 but only 2 chunks
        results = reranker.rerank("query", chunks, top_k=5)
        assert len(results) == 2
    
    def test_rerank_attaches_relevance_score_metadata(self, mocked_reranker):
        """Req 5.4: Score should be attached to chunk metadata."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(["doc A"], [0.5])
        mock_model.predict.return_value = [2.0]  # sigmoid(2.0) ~ 0.88
        
        results = reranker.rerank("query", chunks)
        
        assert "relevance_score" in results[0].chunk.metadata
        assert 0.0 < results[0].chunk.metadata["relevance_score"] < 1.0
    
    def test_rerank_source_method_is_reranker(self, mocked_reranker):
        """All results should have source_method='reranker'."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(["doc A", "doc B"], [0.9, 0.7])
        mock_model.predict.return_value = [1.0, 2.0]
        
        results = reranker.rerank("query", chunks, top_k=2)
        
        assert all(r.source_method == "reranker" for r in results)
    
    def test_rerank_preserves_chunk_content(self, mocked_reranker):
        """Chunk content should not be modified."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(["original content here"], [0.9])
        mock_model.predict.return_value = [1.5]
        
        results = reranker.rerank("query", chunks)
        assert results[0].chunk.content == "original content here"
    
    def test_rerank_preserves_original_metadata(self, mocked_reranker):
        """Original metadata fields should be preserved."""
        reranker, mock_model = mocked_reranker
        
        chunks = _make_scored_chunks(["doc"], [0.9])
        mock_model.predict.return_value = [1.0]
        
        results = reranker.rerank("query", chunks)
        
        assert results[0].chunk.metadata["source"] == "test.txt"
        assert results[0].chunk.metadata["chunk_index"] == 0


class TestCrossEncoderRerankerTimeout:
    """Test timeout behavior."""
    
    def test_timeout_triggers_fallback(self):
        """Req 5.5: If scoring exceeds timeout, fall back to original."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        reranker._model = mock_model
        reranker._initialized = True
        reranker.timeout = 0.001  # Very short timeout
        
        import time
        def slow_predict(pairs):
            time.sleep(0.1)  # Exceed timeout
            return [1.0] * len(pairs)
        
        mock_model.predict.side_effect = slow_predict
        
        chunks = _make_scored_chunks(["doc A", "doc B"], [0.9, 0.7])
        results = reranker.rerank("query", chunks, top_k=2)
        
        # Should fall back to original ranking
        assert len(results) == 2
        assert results[0].score == 0.9  # Original score preserved
    
    def test_model_exception_triggers_fallback(self):
        """Any exception during scoring should trigger fallback."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Out of memory")
        reranker._model = mock_model
        reranker._initialized = True
        
        chunks = _make_scored_chunks(["doc A"], [0.9])
        results = reranker.rerank("query", chunks)
        
        # Should fall back without crashing
        assert len(results) == 1
        assert results[0].score == 0.9


class TestCrossEncoderRerankerEdgeCases:
    """Test edge cases."""
    
    def test_empty_chunks_returns_empty(self):
        """Empty input should return empty output."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", [])
        assert results == []
    
    def test_single_chunk(self):
        """Single chunk should be returned as-is (with fallback)."""
        reranker = CrossEncoderReranker()
        reranker._init_failed = True
        reranker._initialized = True
        
        chunks = _make_scored_chunks(["only doc"], [0.95])
        results = reranker.rerank("query", chunks)
        
        assert len(results) == 1
        assert results[0].chunk.content == "only doc"
    
    def test_normalize_extreme_scores(self):
        """Sigmoid normalization should handle extreme values."""
        reranker = CrossEncoderReranker()
        
        # Very large positive -> ~1.0
        # Very large negative -> ~0.0
        scores = reranker._normalize_scores([100.0, -100.0, 0.0])
        
        assert scores[0] > 0.99  # Large positive -> near 1
        assert scores[1] < 0.01  # Large negative -> near 0
        assert abs(scores[2] - 0.5) < 0.01  # Zero -> 0.5
