"""TXT document parser for plain text files."""

from parsers.base import BaseParser, TextSegment


class TxtParser(BaseParser):
    """Parser for plain text (.txt) documents.
    
    Reads the full file content using UTF-8 encoding (primary),
    with fallback to latin-1 and cp1255 for Hebrew compatibility.
    """
    
    # Encoding fallback chain: try UTF-8 first, then Hebrew-common encodings
    ENCODINGS = ["utf-8", "cp1255", "latin-1"]
    
    def supports(self, file_path: str) -> bool:
        """Check if file has .txt extension."""
        return file_path.lower().endswith(".txt")
    
    def parse(self, file_path: str) -> list[TextSegment]:
        """Read full content of a text file.
        
        Tries UTF-8 first, falls back to cp1255 (Hebrew Windows encoding)
        and latin-1 if UTF-8 fails.
        
        Args:
            file_path: Path to the .txt file
            
        Returns:
            List containing one TextSegment with the file content,
            or empty list if file fails to load.
        """
        content = None
        
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break  # Success, stop trying encodings
            except UnicodeDecodeError:
                continue  # Try next encoding
            except (FileNotFoundError, OSError, PermissionError) as e:
                print(f"Error loading {file_path}: {e}")
                return []
        
        if content is None:
            # All encodings failed
            print(f"Error loading {file_path}: unable to decode file with any supported encoding (utf-8, cp1255, latin-1)")
            return []
        
        # Requirement 1.7: Must produce at least one non-empty text segment
        if not content.strip():
            return []
        
        return [TextSegment(
            content=content,
            metadata={"source": file_path}
        )]
