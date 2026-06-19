"""Indexing pipeline orchestrator.

Coordinates the full document indexing flow:
Parse → Chunk → Embed → Store in ChromaDB + Build BM25 Index

This replaces the monolithic rag_script.py with a modular pipeline.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import RAGConfig
from models import Chunk, IndexingManifest
from parsers.registry import ParserRegistry
from chunking.semantic_chunker import SemanticChunker
from indexing.embedder import EmbeddingGenerator
from indexing.bm25_index import BM25Index


class IndexingPipeline:
    """Orchestrates the full document indexing pipeline.
    
    Flow:
    1. Scan input paths for supported documents
    2. Parse each document using ParserRegistry
    3. Chunk parsed text using SemanticChunker
    4. Generate embeddings and store in ChromaDB (when available)
    5. Build BM25 sparse index
    6. Write indexing manifest
    
    Error handling:
    - Per-document failures are caught and logged
    - Previously indexed documents are not lost on failure
    - Manifest is always written (even if all docs fail)
    """
    
    def __init__(self, config: RAGConfig = None):
        """Initialize the indexing pipeline.
        
        Args:
            config: RAGConfig instance. Uses defaults if None.
        """
        if config is None:
            config = RAGConfig()
        self.config = config
        self.parser_registry = ParserRegistry()
        self.chunker = SemanticChunker(config)
        self.bm25_index = BM25Index(config)
    
    def run(self, file_paths: list[str] = None, directory: str = None) -> IndexingManifest:
        """Run the full indexing pipeline.
        
        Accepts either a list of file paths or a directory to scan.
        If both are provided, file_paths takes precedence.
        
        Args:
            file_paths: Explicit list of file paths to index.
            directory: Directory to scan for supported files.
            
        Returns:
            IndexingManifest with results of the indexing run.
        """
        # Resolve input files
        if file_paths is None and directory is not None:
            file_paths = self._scan_directory(directory)
        elif file_paths is None:
            file_paths = []
        
        total_files = len(file_paths)
        
        # Track results
        all_chunks: list[Chunk] = []
        successful_files: list[dict] = []
        failed_files: list[dict] = []
        cumulative_chunks = 0
        
        # Process each file
        for file_idx, file_path in enumerate(file_paths, start=1):
            try:
                # Check if file type is supported
                parser = self.parser_registry.get_parser(file_path)
                if parser is None:
                    # Unsupported extension - skip silently (Req 1.6)
                    continue
                
                # Parse the document
                segments = self.parser_registry.parse_file(file_path)
                
                if not segments:
                    # Supported file type but no content extracted - count as failure
                    # (Req 10.2: output error and continue)
                    failed_files.append({
                        "name": os.path.basename(file_path),
                        "error": "No content could be extracted from the file",
                    })
                    print(f"[{file_idx}/{total_files}] Error: {file_path}: no content could be extracted")
                    continue
                
                # Chunk the parsed text
                file_chunks: list[Chunk] = []
                for segment in segments:
                    chunks = self.chunker.chunk_document(segment.content, segment.metadata.get("source", file_path))
                    file_chunks.extend(chunks)
                
                if not file_chunks:
                    # Parsed but no chunks generated - count as failure
                    failed_files.append({
                        "name": os.path.basename(file_path),
                        "error": "No chunks could be generated from the file content",
                    })
                    print(f"[{file_idx}/{total_files}] Error: {file_path}: no chunks generated")
                    continue
                
                # Accumulate chunks
                all_chunks.extend(file_chunks)
                cumulative_chunks += len(file_chunks)
                
                # Record success
                successful_files.append({
                    "name": os.path.basename(file_path),
                    "chunk_count": len(file_chunks),
                })
                
                # Progress output (Req 10.1)
                print(f"[{file_idx}/{total_files}] Indexed: {file_path} ({len(file_chunks)} chunks, {cumulative_chunks} total)")
                
            except Exception as e:
                # Per-document failure (Req 10.2)
                failed_files.append({
                    "name": os.path.basename(file_path),
                    "error": str(e),
                })
                print(f"[{file_idx}/{total_files}] Error: {file_path}: {e}")
                continue
        
        # Build BM25 index from all accumulated chunks
        if all_chunks:
            self.bm25_index.build(all_chunks)
        
        # Store embeddings (ChromaDB) - only if embedding generator is available
        # Note: In environments without the model, we skip embedding storage
        # but still build the BM25 index
        self._store_embeddings(all_chunks)
        
        # Write manifest (Req 10.4) - ALWAYS written, even if all fail (Req 10.3)
        manifest = IndexingManifest(
            successful_files=successful_files,
            failed_files=failed_files,
            total_chunks=cumulative_chunks,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._write_manifest(manifest)
        
        # Print summary
        print(f"\nIndexing complete: {len(successful_files)} files indexed, "
              f"{len(failed_files)} failed, {cumulative_chunks} total chunks")
        
        return manifest
    
    def _scan_directory(self, directory: str) -> list[str]:
        """Scan a directory for supported files.
        
        Args:
            directory: Path to directory to scan.
            
        Returns:
            List of file paths with supported extensions.
        """
        supported = self.parser_registry.supported_extensions
        file_paths = []
        
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: directory not found: {directory}")
            return []
        
        for ext in supported:
            file_paths.extend(str(p) for p in dir_path.rglob(f"*{ext}"))
        
        # Sort for deterministic ordering
        file_paths.sort()
        return file_paths
    
    def _store_embeddings(self, chunks: list[Chunk]) -> None:
        """Store chunk embeddings in ChromaDB (if available).
        
        This method gracefully handles the case where the embedding model
        or ChromaDB is not available.
        
        Args:
            chunks: List of chunks to embed and store.
        """
        if not chunks:
            return
        
        try:
            embedder = EmbeddingGenerator(self.config)
            embedder.save_model_metadata()
            
            # Try to store in ChromaDB
            try:
                from langchain_chroma import Chroma
            except ImportError:
                try:
                    from langchain_community.vectorstores import Chroma
                except ImportError:
                    # ChromaDB not available - skip embedding storage
                    return
            
            # Create documents for ChromaDB
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            embeddings = embedder.get_langchain_embeddings()
            
            # Create or update the vector store
            Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=self.config.db_directory,
            )
        except Exception as e:
            # If embedding fails, we still have the BM25 index
            print(f"Warning: embedding storage failed: {e}")
    
    def _write_manifest(self, manifest: IndexingManifest) -> None:
        """Write the indexing manifest to disk.
        
        Args:
            manifest: The manifest to write.
        """
        os.makedirs(self.config.db_directory, exist_ok=True)
        
        manifest_data = {
            "successful_files": manifest.successful_files,
            "failed_files": manifest.failed_files,
            "total_chunks": manifest.total_chunks,
            "timestamp": manifest.timestamp,
        }
        
        with open(self.config.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
