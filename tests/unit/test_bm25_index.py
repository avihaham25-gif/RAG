"""Unit tests for BM25 Index."""

import os
import pytest

from indexing.bm25_index import BM25Index
from models import Chunk
from config import RAGConfig


def _make_chunks(texts: list[str], source: str = "test.txt") -> list[Chunk]:
    """Helper to create Chunk objects from text list."""
    return [
        Chunk(
            content=text,
            metadata={"source": source, "chunk_index": i, "section_title": ""},
        )
        for i, text in enumerate(texts)
    ]


class TestBM25IndexTokenizer:
    """Test Hebrew-aware tokenization."""

    def test_simple_hebrew_tokenization(self):
        """Should tokenize Hebrew text on whitespace."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("שלום עולם")
        assert tokens == ["שלום", "עולם"]

    def test_splits_on_maqaf(self):
        """Should split on Hebrew maqaf ־ (U+05BE)."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("בית\u05BEספר")
        assert "בית" in tokens
        assert "ספר" in tokens

    def test_splits_on_geresh(self):
        """Should split on geresh ׳ (U+05F3)."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("ג\u05F3ון")
        assert len(tokens) >= 1
        # The geresh should be used as a split point
        assert "ג" in tokens
        assert "ון" in tokens

    def test_splits_on_gershayim(self):
        """Should split on gershayim ״ (U+05F4)."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("צה\u05F4ל")
        assert len(tokens) >= 1
        assert "צה" in tokens
        assert "ל" in tokens

    def test_preserves_niqqud(self):
        """Should preserve niqqud (diacritics) as part of the token."""
        index = BM25Index()
        # שָׁלוֹם - shalom with niqqud
        text_with_niqqud = "\u05E9\u05B8\u05C1\u05DC\u05D5\u05B9\u05DD"
        tokens = index.tokenize_hebrew(text_with_niqqud)
        assert len(tokens) == 1
        # The niqqud should be part of the token, not split off
        assert tokens[0] == text_with_niqqud.lower()

    def test_splits_on_standard_punctuation(self):
        """Should split on standard punctuation marks."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("hello, world! test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_empty_string_returns_empty(self):
        """Empty string should return empty list."""
        index = BM25Index()
        assert index.tokenize_hebrew("") == []

    def test_only_punctuation_returns_empty(self):
        """String of only punctuation should return empty list."""
        index = BM25Index()
        assert index.tokenize_hebrew(".,!?;:") == []

    def test_mixed_hebrew_english(self):
        """Should handle mixed Hebrew and English text."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("שלום hello עולם world")
        assert "שלום" in tokens
        assert "hello" in tokens
        assert "עולם" in tokens
        assert "world" in tokens

    def test_lowercase_english_tokens(self):
        """English tokens should be lowercased."""
        index = BM25Index()
        tokens = index.tokenize_hebrew("Hello World")
        assert "hello" in tokens
        assert "world" in tokens


class TestBM25IndexBuild:
    """Test index building."""

    def test_build_creates_index(self, tmp_path):
        """Building should create a searchable index."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks(["שלום עולם", "בוקר טוב", "ערב טוב"])
        index.build(chunks)

        assert index.is_built
        assert index.document_count == 3

    def test_build_persists_to_disk(self, tmp_path):
        """Building should save the index to disk."""
        index_path = tmp_path / "bm25.pkl"
        config = RAGConfig(bm25_index_path=str(index_path))
        index = BM25Index(config)

        chunks = _make_chunks(["content one", "content two"])
        index.build(chunks)

        assert index_path.exists()

    def test_build_empty_chunks(self, tmp_path):
        """Building with empty list should not crash."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        index.build([])
        assert not index.is_built


class TestBM25IndexSearch:
    """Test search functionality."""

    def test_search_hebrew_returns_relevant(self, tmp_path):
        """Searching for a Hebrew word should return the relevant chunk."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks(
            [
                "שלום עולם זהו מסמך ראשון",
                "בוקר טוב זהו מסמך שני",
                "ערב טוב זהו מסמך שלישי",
            ]
        )
        index.build(chunks)

        results = index.search("שלום")
        assert len(results) > 0
        # First result should be the chunk containing "שלום"
        assert "שלום" in results[0][0].content

    def test_search_returns_scores(self, tmp_path):
        """Results should include BM25 scores."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks(["apple banana", "banana cherry", "cherry date"])
        index.build(chunks)

        results = index.search("banana")
        assert len(results) > 0
        for chunk, score in results:
            assert isinstance(score, float)
            assert score > 0.0

    def test_search_results_sorted_by_score(self, tmp_path):
        """Results should be sorted by descending score."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks(
            [
                "apple apple apple",  # Most relevant for "apple"
                "apple banana cherry",  # Somewhat relevant
                "banana cherry date",  # Not relevant
            ]
        )
        index.build(chunks)

        results = index.search("apple")
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_respects_k_limit(self, tmp_path):
        """Should return at most k results."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks([f"document {i} word" for i in range(20)])
        index.build(chunks)

        results = index.search("word", k=5)
        assert len(results) <= 5

    def test_search_empty_query_returns_empty(self, tmp_path):
        """Empty query should return empty results."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks(["content here"])
        index.build(chunks)

        results = index.search("")
        assert results == []

    def test_search_no_match_returns_empty(self, tmp_path):
        """Query with no matching terms should return empty."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "bm25.pkl"))
        index = BM25Index(config)

        chunks = _make_chunks(["apple banana", "cherry date"])
        index.build(chunks)

        results = index.search("elephant")
        assert results == []

    def test_search_without_build_returns_empty(self):
        """Searching without building should return empty."""
        index = BM25Index()
        results = index.search("test")
        assert results == []


class TestBM25IndexPersistence:
    """Test save/load functionality."""

    def test_load_restores_index(self, tmp_path):
        """Loading should restore a previously built index."""
        index_path = str(tmp_path / "bm25.pkl")
        config = RAGConfig(bm25_index_path=index_path)

        # Build and save
        index1 = BM25Index(config)
        chunks = _make_chunks(["שלום עולם", "בוקר טוב"])
        index1.build(chunks)

        # Load in a new instance
        index2 = BM25Index(config)
        assert index2.load() is True
        assert index2.is_built
        assert index2.document_count == 2

    def test_loaded_index_searchable(self, tmp_path):
        """A loaded index should be searchable."""
        index_path = str(tmp_path / "bm25.pkl")
        config = RAGConfig(bm25_index_path=index_path)

        # Build and save
        index1 = BM25Index(config)
        chunks = _make_chunks(["python programming", "java development"])
        index1.build(chunks)

        # Load and search
        index2 = BM25Index(config)
        index2.load()
        results = index2.search("python")

        assert len(results) > 0
        assert "python" in results[0][0].content

    def test_load_nonexistent_returns_false(self, tmp_path):
        """Loading from non-existent path should return False."""
        config = RAGConfig(bm25_index_path=str(tmp_path / "nonexistent.pkl"))
        index = BM25Index(config)
        assert index.load() is False

    def test_load_corrupt_file_returns_false(self, tmp_path):
        """Loading from corrupt file should return False."""
        index_path = tmp_path / "corrupt.pkl"
        index_path.write_bytes(b"not a valid pickle file")

        config = RAGConfig(bm25_index_path=str(index_path))
        index = BM25Index(config)
        assert index.load() is False

    def test_save_creates_directory(self, tmp_path):
        """Save should create parent directory if missing."""
        index_path = str(tmp_path / "subdir" / "bm25.pkl")
        config = RAGConfig(bm25_index_path=index_path)
        index = BM25Index(config)

        chunks = _make_chunks(["test content"])
        index.build(chunks)

        assert os.path.exists(index_path)


class TestBM25IndexIntegrationWithParsers:
    """Test BM25 index works with the parser/chunker output."""

    def test_index_chunks_from_chunker_output(self, tmp_path):
        """Should index Chunk objects as produced by SemanticChunker."""
        from chunking.semantic_chunker import SemanticChunker

        config = RAGConfig(
            bm25_index_path=str(tmp_path / "bm25.pkl"),
            max_chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=50,
        )
        chunker = SemanticChunker(config)

        # Simulate document text
        text = (
            "# מבוא\n\n"
            "זהו מסמך בדיקה בעברית שמכיל מספיק תוכן כדי ליצור מספר חלקים. "
            "המסמך מדבר על תכנות ובינה מלאכותית.\n\n"
            "# שיטות\n\n"
            "בפרק זה נדון בשיטות שונות לעיבוד שפה טבעית. "
            "נלמד על מודלים של למידה עמוקה ורשתות נוירונים."
        )

        chunks = chunker.chunk_document(text, "/tmp/hebrew_doc.txt")

        # Build BM25 index from chunker output
        bm25 = BM25Index(config)
        bm25.build(chunks)

        assert bm25.is_built
        assert bm25.document_count == len(chunks)

        # Search should work
        results = bm25.search("בינה מלאכותית")
        assert len(results) > 0
