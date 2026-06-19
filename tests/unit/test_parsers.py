"""Unit tests for parser base interface and shared data models."""

import pytest

from parsers.base import BaseParser, TextSegment
from models import Chunk, ScoredChunk, GenerationResult, IndexingManifest


class ConcreteParser(BaseParser):
    """Test implementation of BaseParser."""

    def parse(self, file_path: str) -> list[TextSegment]:
        return [TextSegment(content="test content", metadata={"source": file_path})]

    def supports(self, file_path: str) -> bool:
        return file_path.endswith(".test")


class TestBaseParser:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseParser()

    def test_concrete_implementation_works(self):
        parser = ConcreteParser()
        result = parser.parse("/path/to/file.test")
        assert len(result) == 1
        assert result[0].content == "test content"
        assert result[0].metadata["source"] == "/path/to/file.test"

    def test_supports_method(self):
        parser = ConcreteParser()
        assert parser.supports("file.test") is True
        assert parser.supports("file.txt") is False


class TestTextSegment:
    def test_creation_with_content_and_metadata(self):
        segment = TextSegment(content="hello", metadata={"source": "/tmp/test.txt"})
        assert segment.content == "hello"
        assert segment.metadata["source"] == "/tmp/test.txt"

    def test_default_metadata_is_empty_dict(self):
        segment = TextSegment(content="hello")
        assert segment.metadata == {}

    def test_metadata_default_is_independent_per_instance(self):
        segment1 = TextSegment(content="a")
        segment2 = TextSegment(content="b")
        segment1.metadata["source"] = "/tmp/a.txt"
        assert "source" not in segment2.metadata


class TestChunk:
    def test_creation_with_full_metadata(self):
        chunk = Chunk(
            content="chunk text",
            metadata={
                "source": "/tmp/doc.pdf",
                "chunk_index": 0,
                "section_title": "Intro",
            },
        )
        assert chunk.content == "chunk text"
        assert chunk.metadata["source"] == "/tmp/doc.pdf"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["section_title"] == "Intro"

    def test_default_metadata_is_empty_dict(self):
        chunk = Chunk(content="text")
        assert chunk.metadata == {}


class TestScoredChunk:
    def test_wraps_chunk_with_score(self):
        chunk = Chunk(
            content="text",
            metadata={"source": "f.txt", "chunk_index": 0, "section_title": ""},
        )
        scored = ScoredChunk(chunk=chunk, score=0.85, source_method="reranker")
        assert scored.chunk.content == "text"
        assert scored.score == 0.85
        assert scored.source_method == "reranker"

    def test_default_source_method(self):
        chunk = Chunk(content="x", metadata={})
        scored = ScoredChunk(chunk=chunk, score=0.5)
        assert scored.source_method == ""


class TestGenerationResult:
    def test_default_values(self):
        result = GenerationResult(title="כותרת", answer="תשובה")
        assert result.title == "כותרת"
        assert result.answer == "תשובה"
        assert result.sources == []
        assert result.confidence_score == 0.0
        assert result.faithfulness_score == 0.0
        assert result.is_insufficient is False
        assert result.low_confidence_warning is False

    def test_with_all_fields(self):
        result = GenerationResult(
            title="כותרת",
            answer="תשובה",
            sources=["doc1.pdf", "doc2.docx"],
            confidence_score=0.85,
            faithfulness_score=0.92,
            is_insufficient=False,
            low_confidence_warning=False,
        )
        assert result.sources == ["doc1.pdf", "doc2.docx"]
        assert result.confidence_score == 0.85
        assert result.faithfulness_score == 0.92


class TestIndexingManifest:
    def test_default_values(self):
        manifest = IndexingManifest()
        assert manifest.successful_files == []
        assert manifest.failed_files == []
        assert manifest.total_chunks == 0
        assert manifest.timestamp == ""

    def test_with_populated_data(self):
        manifest = IndexingManifest(
            successful_files=[{"name": "doc.pdf", "chunk_count": 10}],
            failed_files=[{"name": "bad.docx", "error": "corrupt file"}],
            total_chunks=10,
            timestamp="2024-01-15T12:00:00Z",
        )
        assert len(manifest.successful_files) == 1
        assert manifest.successful_files[0]["name"] == "doc.pdf"
        assert manifest.successful_files[0]["chunk_count"] == 10
        assert len(manifest.failed_files) == 1
        assert manifest.failed_files[0]["error"] == "corrupt file"
        assert manifest.total_chunks == 10
        assert manifest.timestamp == "2024-01-15T12:00:00Z"
