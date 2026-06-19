"""Unit tests for the semantic chunking engine."""

import pytest
from chunking.semantic_chunker import SemanticChunker
from config import RAGConfig
from models import Chunk


class TestSemanticChunkerBasic:
    """Test basic chunking behavior."""

    def test_short_document_single_chunk(self):
        """A document shorter than max_size should produce one chunk."""
        chunker = SemanticChunker()
        text = "This is a short document with enough content to pass minimum size. " * 3
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        assert len(chunks) == 1
        assert text.strip() in chunks[0].content or chunks[0].content in text

    def test_empty_document_returns_empty(self):
        """An empty document should return empty list."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document("", "/tmp/test.txt")
        assert chunks == []

    def test_metadata_contains_required_fields(self):
        """Each chunk must have source, chunk_index, section_title in metadata."""
        chunker = SemanticChunker()
        text = "A" * 200  # enough content
        chunks = chunker.chunk_document(text, "/tmp/doc.pdf")
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "section_title" in chunk.metadata
            assert chunk.metadata["source"] == "/tmp/doc.pdf"

    def test_chunk_index_is_zero_based_sequential(self):
        """Chunk indices should be 0, 1, 2, ..."""
        config = RAGConfig(max_chunk_size=200, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = "A" * 500 + "\n\n" + "B" * 500
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


class TestMaxChunkSize:
    """Test that no chunk exceeds max_chunk_size."""

    def test_all_chunks_within_max_size(self):
        """Every chunk content must be <= max_chunk_size."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = "Word " * 200  # ~1000 chars
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chunk_size

    def test_long_paragraph_is_split(self):
        """A single paragraph exceeding max_size should be split."""
        config = RAGConfig(max_chunk_size=200, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = "A" * 500  # single paragraph, exceeds max
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chunk_size


class TestMinChunkSize:
    """Test minimum chunk size enforcement."""

    def test_no_chunk_below_min_size_except_short_doc(self):
        """No chunk should be below min_size unless the whole doc is shorter."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        # Create text with a trailing fragment < 100 chars
        text = "A" * 250 + "\n\n" + "B" * 250 + "\n\n" + "C" * 50
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        # The trailing "C" * 50 should be merged into the previous chunk
        for chunk in chunks:
            assert len(chunk.content) >= config.min_chunk_size

    def test_trailing_fragment_merged(self):
        """Content < min_size at end of document is merged into preceding chunk."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = "A" * 250 + "\n\n" + "B" * 30  # trailing fragment < 100
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        # Should result in single chunk (fragment merged)
        assert len(chunks) == 1
        assert "B" * 30 in chunks[0].content

    def test_document_shorter_than_min_size(self):
        """A document shorter than min_size should still produce one chunk or empty."""
        chunker = SemanticChunker()
        text = "Short"  # 5 chars, below min_size
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        # Very short doc - may produce one chunk or empty, but no crash
        # If produced, it's acceptable to have < min_size for the entire doc
        assert len(chunks) <= 1


class TestSectionBoundaries:
    """Test section-aware splitting."""

    def test_splits_at_markdown_headings(self):
        """Should prefer splitting at markdown headings."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = (
            "# Introduction\n\n"
            + "A" * 200
            + "\n\n"
            + "# Methods\n\n"
            + "B" * 200
        )
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        assert len(chunks) >= 2
        # Check section titles in metadata
        titles = [c.metadata["section_title"] for c in chunks]
        assert "Introduction" in titles or "# Introduction" in titles

    def test_splits_at_numbered_headings(self):
        """Should recognize numbered headings."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = (
            "1. First Section\n\n"
            + "A" * 200
            + "\n\n"
            + "2. Second Section\n\n"
            + "B" * 200
        )
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        assert len(chunks) >= 2

    def test_section_title_in_metadata(self):
        """Chunks should have the nearest section title in metadata."""
        config = RAGConfig(max_chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = (
            "# First\n\n"
            + "Content of first section. " * 5
            + "\n\n"
            + "# Second\n\n"
            + "Content of second section. " * 5
        )
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        # At least one chunk should have "First" or "Second" as section_title
        titles = [c.metadata["section_title"] for c in chunks]
        assert any("First" in t for t in titles) or any("Second" in t for t in titles)


class TestParagraphBoundaries:
    """Test paragraph-level splitting."""

    def test_splits_at_paragraphs_within_section(self):
        """Within a section that exceeds max_size, split at paragraph boundaries."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = (
            "# Section\n\n"
            + "Paragraph one content here. " * 8
            + "\n\n"
            + "Paragraph two content here. " * 8
            + "\n\n"
            + "Paragraph three content here. " * 8
        )
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        assert len(chunks) >= 2


class TestOverlap:
    """Test overlap behavior between consecutive chunks."""

    def test_overlap_applied_between_chunks(self):
        """Consecutive chunks should share overlap characters."""
        config = RAGConfig(max_chunk_size=200, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = "A" * 150 + "\n\n" + "B" * 150 + "\n\n" + "C" * 150
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        if len(chunks) > 1:
            # The beginning of chunk[1] should contain some chars from end of chunk[0]
            # (overlap from previous chunk)
            end_of_first = chunks[0].content[-50:]
            # Check some overlap exists (at least partial)
            assert any(c in chunks[1].content[:60] for c in end_of_first[:20])


class TestEdgeCases:
    """Test edge cases."""

    def test_only_whitespace_returns_empty(self):
        """Document with only whitespace should return empty."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document("   \n\n   \n\n   ", "/tmp/test.txt")
        assert chunks == []

    def test_single_heading_no_content(self):
        """A heading with no substantive content."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_document("# Title\n\n", "/tmp/test.txt")
        # Either empty or single very short chunk
        assert len(chunks) <= 1

    def test_hebrew_text_chunking(self):
        """Hebrew text should be chunked correctly."""
        config = RAGConfig(max_chunk_size=300, chunk_overlap=50, min_chunk_size=100)
        chunker = SemanticChunker(config)
        text = "זוהי פסקה בעברית שמכילה מספיק תווים כדי לעבור את הגודל המינימלי. " * 20
        chunks = chunker.chunk_document(text, "/tmp/test.txt")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chunk_size
