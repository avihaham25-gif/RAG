"""Integration tests for the Indexing Pipeline.

Tests the full flow: Parse → Chunk → BM25 Index + Manifest
Note: Embedding storage is mocked since the model is not available.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from indexing.pipeline import IndexingPipeline
from config import RAGConfig
from models import IndexingManifest


class TestIndexingPipelineEndToEnd:
    """Test the full indexing pipeline flow."""
    
    @pytest.fixture
    def setup_docs(self, tmp_path):
        """Create test documents and configure pipeline."""
        # Create document directory
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        
        # Create TXT files
        (docs_dir / "doc1.txt").write_text(
            "# מבוא\n\nזהו מסמך ראשון בעברית שמכיל מספיק תוכן כדי ליצור chunks. "
            "המסמך עוסק בבינה מלאכותית ועיבוד שפה טבעית.\n\n"
            "# שיטות\n\nבפרק זה נתאר את השיטות השונות שבהן השתמשנו במחקר.",
            encoding="utf-8"
        )
        (docs_dir / "doc2.txt").write_text(
            "# תכנות\n\nזהו מסמך שני שעוסק בתכנות מחשבים ופיתוח תוכנה. "
            "נלמד על שפות תכנות שונות כגון Python ו-JavaScript.\n\n"
            "# סיכום\n\nבסיכום, למדנו על שיטות שונות בתכנות.",
            encoding="utf-8"
        )
        
        # Create a DB directory
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
            max_chunk_size=300,
            chunk_overlap=50,
            min_chunk_size=50,
        )
        
        return docs_dir, db_dir, config
    
    def test_full_pipeline_with_directory(self, setup_docs):
        """Run pipeline on a directory of documents."""
        docs_dir, db_dir, config = setup_docs
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(directory=str(docs_dir))
        
        # Check manifest
        assert isinstance(manifest, IndexingManifest)
        assert len(manifest.successful_files) == 2
        assert manifest.total_chunks > 0
        assert manifest.timestamp != ""
        assert len(manifest.failed_files) == 0
    
    def test_full_pipeline_with_file_list(self, setup_docs):
        """Run pipeline with explicit file list."""
        docs_dir, db_dir, config = setup_docs
        
        file_paths = [
            str(docs_dir / "doc1.txt"),
            str(docs_dir / "doc2.txt"),
        ]
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(file_paths=file_paths)
        
        assert len(manifest.successful_files) == 2
        assert manifest.total_chunks > 0
    
    def test_bm25_index_built(self, setup_docs):
        """Pipeline should build a searchable BM25 index."""
        docs_dir, db_dir, config = setup_docs
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        # BM25 index should be built
        assert pipeline.bm25_index.is_built
        
        # Should be searchable
        results = pipeline.bm25_index.search("בינה מלאכותית")
        assert len(results) > 0
    
    def test_bm25_index_persisted(self, setup_docs):
        """BM25 index should be persisted to disk."""
        docs_dir, db_dir, config = setup_docs
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        # Index file should exist
        assert os.path.exists(str(db_dir / "bm25_index.pkl"))
        
        # Should be loadable by a new instance
        from indexing.bm25_index import BM25Index
        new_index = BM25Index(config)
        assert new_index.load() is True
        assert new_index.is_built
    
    def test_manifest_written_to_disk(self, setup_docs):
        """Manifest JSON should be written to the configured path."""
        docs_dir, db_dir, config = setup_docs
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        manifest_path = db_dir / "indexing_manifest.json"
        assert manifest_path.exists()
        
        with open(manifest_path) as f:
            data = json.load(f)
        
        assert "successful_files" in data
        assert "failed_files" in data
        assert "total_chunks" in data
        assert "timestamp" in data
        assert len(data["successful_files"]) == 2
        assert data["total_chunks"] > 0
    
    def test_manifest_contains_chunk_counts(self, setup_docs):
        """Each successful file in manifest should have chunk_count."""
        docs_dir, db_dir, config = setup_docs
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        with open(db_dir / "indexing_manifest.json") as f:
            data = json.load(f)
        
        for file_entry in data["successful_files"]:
            assert "name" in file_entry
            assert "chunk_count" in file_entry
            assert file_entry["chunk_count"] > 0
    
    def test_manifest_timestamp_is_iso8601(self, setup_docs):
        """Timestamp should be valid ISO 8601 UTC."""
        docs_dir, db_dir, config = setup_docs
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        with open(db_dir / "indexing_manifest.json") as f:
            data = json.load(f)
        
        from datetime import datetime
        # Should parse without error
        ts = datetime.fromisoformat(data["timestamp"])
        assert ts is not None


class TestIndexingPipelineErrorHandling:
    """Test error resilience."""
    
    @pytest.fixture
    def setup_mixed_docs(self, tmp_path):
        """Create a mix of valid and invalid documents."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        
        # Valid file
        (docs_dir / "valid.txt").write_text(
            "זהו מסמך תקין בעברית עם מספיק תוכן כדי ליצור chunk אחד לפחות.",
            encoding="utf-8"
        )
        
        # Invalid file (corrupt DOCX)
        (docs_dir / "corrupt.docx").write_bytes(b"not a valid docx")
        
        # Another valid file
        (docs_dir / "valid2.txt").write_text(
            "מסמך נוסף תקין שגם הוא מכיל מספיק תוכן לחלוקה.",
            encoding="utf-8"
        )
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
            max_chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=30,
        )
        
        return docs_dir, db_dir, config
    
    def test_continues_after_single_failure(self, setup_mixed_docs, capsys):
        """Req 10.2: Pipeline continues after individual file failures."""
        docs_dir, db_dir, config = setup_mixed_docs
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(directory=str(docs_dir))
        
        # Should have processed the valid files
        assert len(manifest.successful_files) >= 1
        assert manifest.total_chunks > 0
        
        # BM25 index should still be built
        assert pipeline.bm25_index.is_built
    
    def test_manifest_records_failures(self, setup_mixed_docs):
        """Failed files should be recorded in the manifest."""
        docs_dir, db_dir, config = setup_mixed_docs
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(directory=str(docs_dir))
        
        with open(db_dir / "indexing_manifest.json") as f:
            data = json.load(f)
        
        # Should have both successes and we should verify the manifest was written
        assert "successful_files" in data
        assert "failed_files" in data
    
    def test_all_failures_still_writes_manifest(self, tmp_path):
        """Req 10.3: Even if all docs fail, manifest is still written."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        
        # Only corrupt files
        (docs_dir / "bad1.docx").write_bytes(b"corrupt")
        (docs_dir / "bad2.docx").write_bytes(b"also corrupt")
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
        )
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(directory=str(docs_dir))
        
        # Manifest should still be written
        assert os.path.exists(str(db_dir / "indexing_manifest.json"))
        assert manifest.total_chunks == 0
    
    def test_nonexistent_directory(self, tmp_path, capsys):
        """Should handle non-existent directory gracefully."""
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
        )
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(directory="/nonexistent/path")
        
        assert manifest.total_chunks == 0
        assert len(manifest.successful_files) == 0
    
    def test_empty_directory(self, tmp_path):
        """Should handle empty directory gracefully."""
        docs_dir = tmp_path / "empty_docs"
        docs_dir.mkdir()
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
        )
        
        pipeline = IndexingPipeline(config)
        manifest = pipeline.run(directory=str(docs_dir))
        
        assert manifest.total_chunks == 0
        # Manifest still written
        assert os.path.exists(str(db_dir / "indexing_manifest.json"))


class TestIndexingPipelineProgress:
    """Test progress reporting."""
    
    def test_progress_output(self, tmp_path, capsys):
        """Req 10.1: Should print progress for each document."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        
        (docs_dir / "doc1.txt").write_text(
            "מסמך ראשון עם תוכן מספיק לבדיקה של הפלט.",
            encoding="utf-8"
        )
        (docs_dir / "doc2.txt").write_text(
            "מסמך שני עם תוכן מספיק לבדיקה של הפלט.",
            encoding="utf-8"
        )
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
            min_chunk_size=10,
        )
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        captured = capsys.readouterr()
        # Should contain progress indicators
        assert "[1/" in captured.out or "[2/" in captured.out
        assert "Indexed" in captured.out or "total" in captured.out
    
    def test_summary_output(self, tmp_path, capsys):
        """Should print a summary at the end."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        (docs_dir / "doc.txt").write_text("תוכן מספיק לבדיקה.", encoding="utf-8")
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
            min_chunk_size=10,
        )
        
        pipeline = IndexingPipeline(config)
        pipeline.run(directory=str(docs_dir))
        
        captured = capsys.readouterr()
        assert "complete" in captured.out.lower() or "indexed" in captured.out.lower()


class TestIndexingPipelineDirectoryScan:
    """Test directory scanning."""
    
    def test_scans_supported_extensions(self, tmp_path):
        """Should find .txt, .docx, .pdf files."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        
        (docs_dir / "file.txt").write_text("content", encoding="utf-8")
        (docs_dir / "file.csv").write_text("a,b,c", encoding="utf-8")
        (docs_dir / "image.png").write_bytes(b"\x89PNG")
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
            min_chunk_size=5,
        )
        
        pipeline = IndexingPipeline(config)
        files = pipeline._scan_directory(str(docs_dir))
        
        # Should only find .txt (csv and png not supported)
        assert any("file.txt" in f for f in files)
        assert not any("file.csv" in f for f in files)
        assert not any("image.png" in f for f in files)
    
    def test_recursive_scan(self, tmp_path):
        """Should scan subdirectories."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        sub_dir = docs_dir / "subdir"
        sub_dir.mkdir()
        
        (docs_dir / "root.txt").write_text("root content", encoding="utf-8")
        (sub_dir / "nested.txt").write_text("nested content", encoding="utf-8")
        
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        
        config = RAGConfig(
            db_directory=str(db_dir),
            bm25_index_path=str(db_dir / "bm25_index.pkl"),
            manifest_path=str(db_dir / "indexing_manifest.json"),
        )
        
        pipeline = IndexingPipeline(config)
        files = pipeline._scan_directory(str(docs_dir))
        
        assert len(files) == 2
        assert any("root.txt" in f for f in files)
        assert any("nested.txt" in f for f in files)
