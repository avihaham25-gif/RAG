"""Parser registry for dispatching files to appropriate parsers."""

from parsers.base import BaseParser, TextSegment
from parsers.txt_parser import TxtParser
from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PdfParser


class ParserRegistry:
    """Registry that selects the appropriate parser for a given file.
    
    Maintains a list of registered parsers and dispatches files to the
    first parser whose `supports()` method returns True for the given path.
    """
    
    def __init__(self):
        """Initialize with the default set of parsers."""
        self._parsers: list[BaseParser] = [
            TxtParser(),
            DocxParser(),
            PdfParser(),
        ]
    
    def get_parser(self, file_path: str) -> BaseParser | None:
        """Get the appropriate parser for a given file path.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            The first parser that supports the file extension,
            or None if no parser supports it.
        """
        for parser in self._parsers:
            if parser.supports(file_path):
                return parser
        return None
    
    def parse_file(self, file_path: str) -> list[TextSegment]:
        """Parse a single file using the appropriate parser.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of TextSegment objects, or empty list if file type
            is unsupported or parsing fails.
        """
        parser = self.get_parser(file_path)
        if parser is None:
            # Req 1.6: skip unsupported extensions silently
            return []
        return parser.parse(file_path)
    
    def parse_batch(self, file_paths: list[str]) -> list[TextSegment]:
        """Parse a batch of files, skipping failures gracefully.
        
        Iterates through all provided file paths, parsing each with the
        appropriate parser. Files with unsupported extensions are skipped
        silently. Files that fail to parse are skipped with an error message
        printed to stdout (handled by individual parsers).
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            Aggregated list of TextSegment objects from all successfully
            parsed files.
        """
        all_segments: list[TextSegment] = []
        for file_path in file_paths:
            segments = self.parse_file(file_path)
            all_segments.extend(segments)
        return all_segments
    
    def register_parser(self, parser: BaseParser) -> None:
        """Register an additional parser.
        
        Args:
            parser: A parser implementing BaseParser to add to the registry.
        """
        self._parsers.append(parser)
    
    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions for documentation."""
        return [".txt", ".docx", ".pdf"]
