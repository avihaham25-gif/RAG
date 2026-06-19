"""BM25 sparse index for keyword-based retrieval.

Provides a BM25Okapi index over document chunks with Hebrew-aware tokenization.
The index can be persisted to disk and loaded for query-time search.
"""

import math
import os
import pickle
import re
from models import Chunk
from config import RAGConfig


try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:

    class _BM25Okapi:
        """Minimal BM25Okapi implementation as fallback when rank_bm25 is not installed."""

        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
            self.corpus_size = len(corpus)
            self.avgdl = (
                sum(len(doc) for doc in corpus) / self.corpus_size
                if self.corpus_size
                else 0
            )
            self.k1 = k1
            self.b = b
            self.corpus = corpus

            # Document frequencies
            self.df = {}
            for doc in corpus:
                seen = set()
                for token in doc:
                    if token not in seen:
                        self.df[token] = self.df.get(token, 0) + 1
                        seen.add(token)

            # IDF values
            self.idf = {}
            for word, freq in self.df.items():
                self.idf[word] = math.log(
                    (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
                )

        def get_scores(self, query: list[str]) -> list[float]:
            scores = []
            for doc in self.corpus:
                score = 0.0
                doc_len = len(doc)
                # Term frequency in document
                tf_map = {}
                for token in doc:
                    tf_map[token] = tf_map.get(token, 0) + 1

                for term in query:
                    if term in tf_map:
                        tf = tf_map[term]
                        idf = self.idf.get(term, 0)
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (
                            1 - self.b + self.b * doc_len / self.avgdl
                        )
                        score += idf * numerator / denominator
                scores.append(score)
            return scores


class BM25Index:
    """BM25 sparse keyword index with Hebrew-aware tokenization.

    Uses BM25Okapi algorithm for scoring document relevance based on
    term frequency. Supports persistence to disk via pickle serialization.

    The tokenizer handles Hebrew-specific punctuation:
    - Maqaf (־) U+05BE: Hebrew hyphen
    - Geresh (׳) U+05F3: Hebrew punctuation
    - Gershayim (״) U+05F4: Hebrew punctuation
    - Standard Unicode punctuation categories

    Niqqud (Hebrew diacritics, U+0591-U+05BD, U+05BF, U+05C1-U+05C2, U+05C4-U+05C5, U+05C7)
    are preserved as part of the token.
    """

    # Hebrew-specific punctuation to split on (in addition to standard punctuation)
    # Maqaf ־ (U+05BE), Geresh ׳ (U+05F3), Gershayim ״ (U+05F4)
    HEBREW_PUNCTUATION = "\u05BE\u05F3\u05F4"

    # Regex pattern for tokenization: split on whitespace and punctuation
    # but preserve niqqud (U+0591-U+05BD, U+05BF, U+05C1-U+05C2, U+05C4-U+05C5, U+05C7)
    # These are combining marks that attach to the preceding letter
    SPLIT_PATTERN = re.compile(
        r"[\s"  # whitespace
        r"\u05BE\u05F3\u05F4"  # Hebrew punctuation (maqaf, geresh, gershayim)
        r"!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"  # ASCII punctuation
        r"\u2000-\u206F"  # General punctuation block
        r"\u2E00-\u2E7F"  # Supplemental punctuation
        r"\uFE50-\uFE6F"  # Small form variants
        r"\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65"  # Fullwidth punctuation
        r"]+"
    )

    def __init__(self, config: RAGConfig = None):
        """Initialize BM25 index.

        Args:
            config: RAGConfig instance for index path configuration.
        """
        if config is None:
            config = RAGConfig()
        self.index_path = config.bm25_index_path
        self._index = None  # BM25Okapi instance
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks and persist to disk.

        Tokenizes all chunk content using Hebrew-aware tokenization
        and constructs a BM25Okapi index.

        Args:
            chunks: List of Chunk objects to index.
        """
        self._chunks = chunks
        self._tokenized_corpus = [
            self.tokenize_hebrew(chunk.content) for chunk in chunks
        ]

        if self._tokenized_corpus:
            self._index = _BM25Okapi(self._tokenized_corpus)
        else:
            self._index = None

        # Persist to disk
        self.save()

    def search(self, query: str, k: int = 50) -> list[tuple[Chunk, float]]:
        """Search BM25 index for relevant chunks.

        Args:
            query: The search query string.
            k: Maximum number of results to return.

        Returns:
            List of (Chunk, score) tuples sorted by descending BM25 score.
            Returns empty list if index is not built or query is empty.
        """
        if self._index is None or not self._chunks:
            return []

        query_tokens = self.tokenize_hebrew(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Get top-k indices sorted by score (descending)
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        # Filter out zero-score results
        results = [
            (self._chunks[idx], float(score))
            for idx, score in scored_indices
            if score > 0.0
        ]

        return results

    def tokenize_hebrew(self, text: str) -> list[str]:
        """Tokenize text with Hebrew-aware splitting.

        Splits on whitespace and Unicode punctuation characters
        (including maqaf ־, geresh ׳, and gershayim ״) while
        preserving niqqud (diacritics) as part of the token.

        Args:
            text: Text to tokenize.

        Returns:
            List of non-empty lowercase tokens.
        """
        if not text:
            return []

        # Split using the pattern
        tokens = self.SPLIT_PATTERN.split(text)

        # Filter empty tokens and convert to lowercase
        # (lowercase for Latin chars; Hebrew has no case)
        return [token.lower() for token in tokens if token.strip()]

    def save(self) -> None:
        """Persist the BM25 index to disk.

        Saves the index, chunks, and tokenized corpus as a pickle file.
        Creates the parent directory if it doesn't exist.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        data = {
            "index": self._index,
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """Load a persisted BM25 index from disk.

        Returns:
            True if index was loaded successfully, False otherwise.
        """
        if not os.path.exists(self.index_path):
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)

            self._index = data["index"]
            self._chunks = data["chunks"]
            self._tokenized_corpus = data["tokenized_corpus"]
            return True
        except (pickle.UnpicklingError, KeyError, EOFError, Exception):
            return False

    @property
    def is_built(self) -> bool:
        """Check if the index has been built or loaded."""
        return self._index is not None and len(self._chunks) > 0

    @property
    def document_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
