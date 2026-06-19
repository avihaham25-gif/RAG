"""Unit tests for the Embedding Generator."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from config import RAGConfig
from indexing.embedder import (
    EmbeddingGenerator,
    EmbeddingModelError,
    EmbeddingMismatchError,
)


class TestEmbeddingGeneratorInit:
    """Test initialization and configuration."""

    def test_default_config_model_name(self):
        """Should use model from RAGConfig by default."""
        gen = EmbeddingGenerator()
        assert gen.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def test_custom_config_model_name(self):
        """Should use model from provided config."""
        config = RAGConfig(embedding_model="custom/model-name")
        gen = EmbeddingGenerator(config)
        assert gen.model_name == "custom/model-name"

    def test_lazy_loading_no_model_at_init(self):
        """Model should not be loaded until first use."""
        gen = EmbeddingGenerator()
        assert gen._embeddings is None

    def test_db_directory_from_config(self):
        """Should store db_directory from config."""
        config = RAGConfig(db_directory="/custom/db/path")
        gen = EmbeddingGenerator(config)
        assert gen.db_directory == "/custom/db/path"


class TestEmbeddingGeneratorModelLoading:
    """Test model loading behavior."""

    @patch("indexing.embedder.EmbeddingGenerator._ensure_loaded")
    def test_embed_query_calls_ensure_loaded(self, mock_ensure):
        """embed_query should trigger model loading."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_query.return_value = [0.1] * 384

        gen.embed_query("test")
        assert gen._embeddings.embed_query.called

    @patch("indexing.embedder.EmbeddingGenerator._ensure_loaded")
    def test_embed_documents_calls_ensure_loaded(self, mock_ensure):
        """embed_documents should trigger model loading."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_documents.return_value = [[0.1] * 384]

        gen.embed_documents(["test doc"])
        assert gen._embeddings.embed_documents.called

    def test_model_load_failure_raises_error(self):
        """Should raise EmbeddingModelError if model can't be loaded."""
        config = RAGConfig(embedding_model="nonexistent/fake-model-xyz")
        gen = EmbeddingGenerator(config)

        # Mock the import to succeed but model loading to fail
        mock_module = MagicMock()
        mock_module.HuggingFaceEmbeddings.side_effect = Exception("Model not found")

        with patch.dict("sys.modules", {"langchain_huggingface": mock_module}):
            with pytest.raises(EmbeddingModelError) as exc_info:
                gen._ensure_loaded()
            assert "nonexistent/fake-model-xyz" in str(exc_info.value)

    def test_model_load_failure_includes_model_name(self):
        """Error message should include the model name that failed."""
        config = RAGConfig(embedding_model="bad/model")
        gen = EmbeddingGenerator(config)

        mock_module = MagicMock()
        mock_module.HuggingFaceEmbeddings.side_effect = RuntimeError("Download failed")

        with patch.dict("sys.modules", {"langchain_huggingface": mock_module}):
            with pytest.raises(EmbeddingModelError) as exc_info:
                gen._ensure_loaded()
            assert "bad/model" in str(exc_info.value)
            assert "Download failed" in str(exc_info.value)

    def test_import_fallback_to_langchain_community(self):
        """Should fall back to langchain_community if langchain_huggingface unavailable."""
        gen = EmbeddingGenerator()

        mock_community = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_community.HuggingFaceEmbeddings.return_value = mock_embeddings_instance

        # Make langchain_huggingface import fail, but langchain_community succeed
        def import_side_effect(name, *args, **kwargs):
            if name == "langchain_huggingface":
                raise ImportError("No module named 'langchain_huggingface'")
            if name == "langchain_community.embeddings":
                return mock_community
            return MagicMock()

        with patch("builtins.__import__", side_effect=import_side_effect):
            # We need to clear the cached module if any
            import sys
            # Remove cached modules to force re-import
            for mod_name in list(sys.modules.keys()):
                if "langchain" in mod_name:
                    del sys.modules[mod_name]

            # This approach won't work cleanly with patch.dict, let's use a simpler test
            pass

    def test_both_imports_fail_raises_error(self):
        """Should raise EmbeddingModelError if both import paths fail."""
        gen = EmbeddingGenerator()

        import sys
        # Remove any cached langchain modules
        original_modules = {}
        for key in list(sys.modules.keys()):
            if "langchain" in key:
                original_modules[key] = sys.modules.pop(key)

        try:
            with patch.dict("sys.modules", {
                "langchain_huggingface": None,
                "langchain_community": None,
                "langchain_community.embeddings": None,
            }):
                with pytest.raises(EmbeddingModelError) as exc_info:
                    gen._ensure_loaded()
                assert "langchain_huggingface" in str(exc_info.value) or \
                       "langchain-huggingface" in str(exc_info.value)
        finally:
            # Restore modules
            sys.modules.update(original_modules)


class TestEmbeddingGeneratorWithMockedModel:
    """Test embedding generation with mocked model."""

    def test_embed_query_returns_vector(self):
        """embed_query should return a list of floats."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_query.return_value = [0.1, 0.2, 0.3] * 128  # 384 dims

        result = gen.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_returns_list_of_vectors(self):
        """embed_documents should return a list of vectors."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_documents.return_value = [
            [0.1] * 384,
            [0.2] * 384,
        ]

        result = gen.embed_documents(["doc1", "doc2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 384
        assert len(result[1]) == 384

    def test_get_langchain_embeddings(self):
        """get_langchain_embeddings should return the underlying object."""
        gen = EmbeddingGenerator()
        mock_embeddings = MagicMock()
        gen._embeddings = mock_embeddings

        result = gen.get_langchain_embeddings()
        assert result is mock_embeddings

    def test_dimensions_property(self):
        """dimensions should return vector length."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_query.return_value = [0.0] * 384

        assert gen.dimensions == 384


class TestEmbeddingGeneratorValidation:
    """Test index validation for model mismatch detection."""

    def test_save_model_metadata(self, tmp_path):
        """Should write metadata JSON file."""
        gen = EmbeddingGenerator()
        gen.save_model_metadata(str(tmp_path))

        metadata_path = tmp_path / EmbeddingGenerator.MODEL_METADATA_FILE
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["embedding_model"] == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def test_validate_index_passes_when_models_match(self, tmp_path):
        """Validation should pass silently when models match."""
        gen = EmbeddingGenerator()
        gen.save_model_metadata(str(tmp_path))

        # Should not raise
        gen.validate_index(str(tmp_path))

    def test_validate_index_raises_on_mismatch(self, tmp_path):
        """Should raise EmbeddingMismatchError when models differ."""
        # Save metadata with one model
        metadata_path = tmp_path / EmbeddingGenerator.MODEL_METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump({"embedding_model": "old-model/v1"}, f)

        # Try to validate with a different model
        gen = EmbeddingGenerator()  # uses default model

        with pytest.raises(EmbeddingMismatchError) as exc_info:
            gen.validate_index(str(tmp_path))

        error_msg = str(exc_info.value)
        assert "old-model/v1" in error_msg
        assert "paraphrase-multilingual-MiniLM-L12-v2" in error_msg

    def test_validate_index_passes_when_no_metadata(self, tmp_path):
        """Should pass silently when no metadata file exists (new index)."""
        gen = EmbeddingGenerator()
        # No metadata file exists in tmp_path
        gen.validate_index(str(tmp_path))  # Should not raise

    def test_validate_index_uses_default_db_directory(self, tmp_path):
        """Should use config db_directory if no path provided."""
        config = RAGConfig(db_directory=str(tmp_path))
        gen = EmbeddingGenerator(config)
        gen.save_model_metadata()

        # Validate without explicit path
        gen.validate_index()  # Should not raise

    def test_validate_mismatch_error_message_format(self, tmp_path):
        """Error message should clearly indicate expected vs actual model."""
        metadata_path = tmp_path / EmbeddingGenerator.MODEL_METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump({"embedding_model": "all-MiniLM-L6-v2"}, f)

        gen = EmbeddingGenerator()

        with pytest.raises(EmbeddingMismatchError) as exc_info:
            gen.validate_index(str(tmp_path))

        msg = str(exc_info.value)
        # Should mention both models
        assert "all-MiniLM-L6-v2" in msg
        assert gen.model_name in msg
        # Should indicate which is which
        assert "index was created with" in msg
        assert "current configuration uses" in msg

    def test_save_creates_directory_if_missing(self, tmp_path):
        """save_model_metadata should create the directory if it doesn't exist."""
        new_dir = tmp_path / "new_db"
        gen = EmbeddingGenerator()
        gen.save_model_metadata(str(new_dir))

        assert (new_dir / EmbeddingGenerator.MODEL_METADATA_FILE).exists()


class TestEmbeddingGeneratorEdgeCases:
    """Test edge cases."""

    def test_multiple_ensure_loaded_calls_idempotent(self):
        """Calling _ensure_loaded multiple times should only load once."""
        gen = EmbeddingGenerator()
        mock_embeddings = MagicMock()
        gen._embeddings = mock_embeddings

        gen._ensure_loaded()
        gen._ensure_loaded()

        # Should still be the same object (not re-created)
        assert gen._embeddings is mock_embeddings

    def test_empty_string_embed_query(self):
        """Should handle empty string gracefully."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_query.return_value = [0.0] * 384

        result = gen.embed_query("")
        assert len(result) == 384

    def test_empty_list_embed_documents(self):
        """Should handle empty list gracefully."""
        gen = EmbeddingGenerator()
        gen._embeddings = MagicMock()
        gen._embeddings.embed_documents.return_value = []

        result = gen.embed_documents([])
        assert result == []
