"""Shared data models used across the RAG system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A semantically coherent text segment ready for indexing."""

    content: str
    metadata: dict = field(default_factory=dict)
    # metadata contains:
    #   "source": str - file path
    #   "chunk_index": int - zero-based index within document
    #   "section_title": str - nearest preceding section header or ""


@dataclass
class ScoredChunk:
    """A chunk with a retrieval/reranking relevance score."""

    chunk: Chunk
    score: float  # 0.0 to 1.0
    source_method: str = ""  # "dense", "bm25", "rrf", "reranker"


@dataclass
class GenerationResult:
    """Complete result of the generation pipeline."""

    title: str  # Hebrew title (<=10 words)
    answer: str  # Answer paragraph
    sources: list[str] = field(default_factory=list)  # Source document file names
    confidence_score: float = 0.0  # Average reranker score (0.0-1.0)
    faithfulness_score: float = 0.0  # Faithfulness metric (0.0-1.0)
    is_insufficient: bool = False  # True if no relevant context found
    low_confidence_warning: bool = False  # True if faithfulness < 0.7


@dataclass
class IndexingManifest:
    """Record of an indexing batch run."""

    successful_files: list[dict] = field(
        default_factory=list
    )  # [{"name": str, "chunk_count": int}]
    failed_files: list[dict] = field(
        default_factory=list
    )  # [{"name": str, "error": str}]
    total_chunks: int = 0
    timestamp: str = ""  # ISO 8601 UTC
