"""Dense vector search using ChromaDB."""

from models import Chunk, ScoredChunk
from config import RAGConfig
from indexing.embedder import EmbeddingGenerator


class DenseSearch:
    """Dense vector similarity search using ChromaDB.
    
    Wraps ChromaDB to perform embedding-based similarity search.
    Falls back gracefully when ChromaDB or embeddings are unavailable.
    """
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.config = config
        self.k = config.dense_search_k
    
    def search(self, query: str, k: int = None) -> list[ScoredChunk]:
        """Search ChromaDB for similar chunks.
        
        Args:
            query: The search query string.
            k: Number of results to return. Defaults to config.dense_search_k.
            
        Returns:
            List of ScoredChunk with source_method="dense", sorted by score descending.
            Returns empty list if ChromaDB is unavailable.
        """
        if k is None:
            k = self.k
        
        try:
            # Try to import and use ChromaDB
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            
            embedder = EmbeddingGenerator(self.config)
            embeddings = embedder.get_langchain_embeddings()
            
            # Load existing vector store
            vectorstore = Chroma(
                persist_directory=self.config.db_directory,
                embedding_function=embeddings,
            )
            
            # Perform similarity search with scores
            results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
            
            scored_chunks = []
            for doc, score in results:
                chunk = Chunk(
                    content=doc.page_content,
                    metadata=doc.metadata if doc.metadata else {},
                )
                scored_chunks.append(ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    source_method="dense",
                ))
            
            return scored_chunks
            
        except Exception:
            # ChromaDB or embeddings not available
            return []
