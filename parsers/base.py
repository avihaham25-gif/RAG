"""Base parser interface and data models for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TextSegment:
    """Raw text extracted from a document by the parser."""

    content: str
    metadata: dict = field(default_factory=dict)
    # metadata always contains: {"source": "/path/to/file.ext"}


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> list[TextSegment]:
        """Extract text segments from a document file.

        Returns a list of TextSegment objects, each containing extracted text
        and metadata including the source file path.
        """
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if this parser can handle the given file extension."""
        ...
