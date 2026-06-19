"""Unit tests for RAGConfig dataclass and its validation logic."""

import pytest
from config import RAGConfig


class TestRAGConfigDefaults:
    """Test that default configuration creates successfully."""

    def test_default_config_creates_without_error(self):
        config = RAGConfig()
        assert config.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert config.max_chunk_size == 1500
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.dense_search_k == 50
        assert config.bm25_search_k == 50
        assert config.rrf_k == 60
        assert config.fusion_candidates == 20
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert config.reranker_top_k == 5
        assert config.reranker_timeout == 10.0
        assert config.confidence_threshold == 0.3
        assert config.faithfulness_threshold == 0.7
        assert config.db_directory == "./db"
        assert config.bm25_index_path == "./db/bm25_index.pkl"
        assert config.manifest_path == "./db/indexing_manifest.json"
        assert config.llm_model == "mistral"
        assert config.llm_temperature == 0.0
        assert config.llm_timeout == 60


class TestRAGConfigValidation:
    """Test validation logic in __post_init__."""

    def test_max_chunk_size_too_low(self):
        with pytest.raises(ValueError, match="max_chunk_size must be between 200 and 10000"):
            RAGConfig(max_chunk_size=199)

    def test_max_chunk_size_too_high(self):
        with pytest.raises(ValueError, match="max_chunk_size must be between 200 and 10000"):
            RAGConfig(max_chunk_size=10001)

    def test_max_chunk_size_boundary_low(self):
        config = RAGConfig(max_chunk_size=200, chunk_overlap=50, min_chunk_size=50)
        assert config.max_chunk_size == 200

    def test_max_chunk_size_boundary_high(self):
        config = RAGConfig(max_chunk_size=10000)
        assert config.max_chunk_size == 10000

    def test_chunk_overlap_too_large(self):
        # overlap must be < 50% of max_chunk_size (1500 * 0.5 = 750)
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than 50%"):
            RAGConfig(chunk_overlap=750)

    def test_chunk_overlap_at_boundary(self):
        # Exactly 50% should fail (must be strictly less than)
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than 50%"):
            RAGConfig(max_chunk_size=1000, chunk_overlap=500)

    def test_chunk_overlap_just_under_boundary(self):
        config = RAGConfig(max_chunk_size=1000, chunk_overlap=499, min_chunk_size=100)
        assert config.chunk_overlap == 499

    def test_min_chunk_size_too_large(self):
        with pytest.raises(ValueError, match="min_chunk_size.*must be less than"):
            RAGConfig(min_chunk_size=1500)

    def test_min_chunk_size_equal_to_max(self):
        with pytest.raises(ValueError, match="min_chunk_size.*must be less than"):
            RAGConfig(max_chunk_size=500, min_chunk_size=500, chunk_overlap=100)

    def test_rrf_k_too_low(self):
        with pytest.raises(ValueError, match="rrf_k must be between 1 and 1000"):
            RAGConfig(rrf_k=0)

    def test_rrf_k_too_high(self):
        with pytest.raises(ValueError, match="rrf_k must be between 1 and 1000"):
            RAGConfig(rrf_k=1001)

    def test_rrf_k_boundary_low(self):
        config = RAGConfig(rrf_k=1)
        assert config.rrf_k == 1

    def test_rrf_k_boundary_high(self):
        config = RAGConfig(rrf_k=1000)
        assert config.rrf_k == 1000

    def test_fusion_candidates_too_low(self):
        with pytest.raises(ValueError, match="fusion_candidates must be between 1 and 100"):
            RAGConfig(fusion_candidates=0)

    def test_fusion_candidates_too_high(self):
        with pytest.raises(ValueError, match="fusion_candidates must be between 1 and 100"):
            RAGConfig(fusion_candidates=101)

    def test_reranker_top_k_too_low(self):
        with pytest.raises(ValueError, match="reranker_top_k must be between 1 and 20"):
            RAGConfig(reranker_top_k=0)

    def test_reranker_top_k_too_high(self):
        with pytest.raises(ValueError, match="reranker_top_k must be between 1 and 20"):
            RAGConfig(reranker_top_k=21)

    def test_confidence_threshold_too_low(self):
        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            RAGConfig(confidence_threshold=-0.1)

    def test_confidence_threshold_too_high(self):
        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            RAGConfig(confidence_threshold=1.1)

    def test_confidence_threshold_boundaries(self):
        config_low = RAGConfig(confidence_threshold=0.0)
        assert config_low.confidence_threshold == 0.0
        config_high = RAGConfig(confidence_threshold=1.0)
        assert config_high.confidence_threshold == 1.0

    def test_faithfulness_threshold_too_low(self):
        with pytest.raises(ValueError, match="faithfulness_threshold must be between 0.0 and 1.0"):
            RAGConfig(faithfulness_threshold=-0.1)

    def test_faithfulness_threshold_too_high(self):
        with pytest.raises(ValueError, match="faithfulness_threshold must be between 0.0 and 1.0"):
            RAGConfig(faithfulness_threshold=1.1)

    def test_faithfulness_threshold_boundaries(self):
        config_low = RAGConfig(faithfulness_threshold=0.0)
        assert config_low.faithfulness_threshold == 0.0
        config_high = RAGConfig(faithfulness_threshold=1.0)
        assert config_high.faithfulness_threshold == 1.0


class TestRAGConfigCustomValues:
    """Test that custom valid values are accepted."""

    def test_custom_valid_config(self):
        config = RAGConfig(
            max_chunk_size=3000,
            chunk_overlap=500,
            min_chunk_size=200,
            rrf_k=100,
            fusion_candidates=50,
            reranker_top_k=10,
            confidence_threshold=0.5,
            faithfulness_threshold=0.8,
        )
        assert config.max_chunk_size == 3000
        assert config.chunk_overlap == 500
        assert config.min_chunk_size == 200
        assert config.rrf_k == 100
        assert config.fusion_candidates == 50
        assert config.reranker_top_k == 10
        assert config.confidence_threshold == 0.5
        assert config.faithfulness_threshold == 0.8
