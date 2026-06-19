"""Unified embedding generator for the RAG system.

Provides a single source of truth for embedding generation,
ensuring consistency between indexing and query time.
"""

import json
import os
from pathlib import Path

from config import RAGConfig


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load."""
    pass


class EmbeddingMismatchError(Exception):
    """Raised when the query embedding model doesn't match the index model."""
    pass


class EmbeddingGenerator:
    """Unified embedding generator using HuggingFace sentence-transformers.
    
    Wraps the embedding model to provide:
    - Single source of truth for model selection (from RAGConfig)
    - Model validation against stored indices
    - Consistent embeddings for both indexing and querying
    - Extensibility for future caching or batching
    
    Attributes:
        model_name: The embedding model identifier from config
        dimensions: The output vector dimensionality
    """
    
    # Model metadata file stored alongside the ChromaDB
    MODEL_METADATA_FILE = "embedding_model_metadata.json"
    
    def __init__(self, config: RAGConfig = None):
        """Initialize the embedding generator.
        
        Args:
            config: RAGConfig instance. Uses defaults if None.
            
        Raises:
            EmbeddingModelError: If the model fails to load.
        """
        if config is None:
            config = RAGConfig()
        
        self.model_name = config.embedding_model
        self.db_directory = config.db_directory
        self._embeddings = None
        self._dimensions = None
    
    @property
    def dimensions(self) -> int:
        """Get the output vector dimensionality."""
        if self._dimensions is None:
            # Generate a test embedding to determine dimensions
            self._ensure_loaded()
            test_vector = self._embeddings.embed_query("test")
            self._dimensions = len(test_vector)
        return self._dimensions
    
    def _ensure_loaded(self) -> None:
        """Lazily load the embedding model on first use.
        
        Raises:
            EmbeddingModelError: If the model fails to load.
        """
        if self._embeddings is not None:
            return
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError:
                raise EmbeddingModelError(
                    f"Failed to load embedding model '{self.model_name}': "
                    f"Neither langchain_huggingface nor langchain_community is installed. "
                    f"Install with: pip install langchain-huggingface"
                )
        
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name
            )
        except Exception as e:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self.model_name}': {e}"
            )
    
    def get_langchain_embeddings(self):
        """Get the underlying LangChain embeddings object.
        
        This is used for passing to ChromaDB or other LangChain integrations
        that expect an Embeddings instance.
        
        Returns:
            The HuggingFaceEmbeddings instance.
            
        Raises:
            EmbeddingModelError: If the model fails to load.
        """
        self._ensure_loaded()
        return self._embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding vector for a query string.
        
        Args:
            text: The query text to embed.
            
        Returns:
            List of floats representing the embedding vector.
            
        Raises:
            EmbeddingModelError: If the model fails to load.
        """
        self._ensure_loaded()
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of documents.
        
        Args:
            texts: List of document texts to embed.
            
        Returns:
            List of embedding vectors (list of floats each).
            
        Raises:
            EmbeddingModelError: If the model fails to load.
        """
        self._ensure_loaded()
        return self._embeddings.embed_documents(texts)
    
    def save_model_metadata(self, index_path: str = None) -> None:
        """Save model metadata alongside the index for future validation.
        
        Writes a JSON file containing the model name used to create the index.
        This is used later by validate_index() to detect mismatches.
        
        Args:
            index_path: Path to the index directory. Defaults to config db_directory.
        """
        if index_path is None:
            index_path = self.db_directory
        
        metadata_path = os.path.join(index_path, self.MODEL_METADATA_FILE)
        metadata = {
            "embedding_model": self.model_name,
            "model_name": self.model_name,
        }
        
        os.makedirs(index_path, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
    def validate_index(self, index_path: str = None) -> None:
        """Validate that the current model matches the one used to create the index.
        
        Reads the model metadata file from the index directory and compares
        the stored model name with the current configuration.
        
        Args:
            index_path: Path to the index directory. Defaults to config db_directory.
            
        Raises:
            EmbeddingMismatchError: If the models don't match.
        """
        if index_path is None:
            index_path = self.db_directory
        
        metadata_path = os.path.join(index_path, self.MODEL_METADATA_FILE)
        
        if not os.path.exists(metadata_path):
            # No metadata file - index may not exist yet, which is fine
            return
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        stored_model = metadata.get("embedding_model", "")
        
        if stored_model != self.model_name:
            raise EmbeddingMismatchError(
                f"Embedding model mismatch: index was created with '{stored_model}' "
                f"but current configuration uses '{self.model_name}'. "
                f"Please re-index documents with the current model or update the configuration."
            )
