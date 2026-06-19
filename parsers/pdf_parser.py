"""PDF document parser for extracting text from PDF files.

Supports pypdf and PyPDF2 libraries when available, with a fallback
basic PDF text extractor using Python's standard library for environments
where no PDF library is installed.
"""

import re

from parsers.base import BaseParser, TextSegment


class PdfParser(BaseParser):
    """Parser for PDF documents.

    Extracts text from PDF files page by page, separating paragraphs
    with double newlines. Uses pypdf if available, falls back to PyPDF2,
    or uses a basic standard library extractor as a last resort.
    """

    def supports(self, file_path: str) -> bool:
        """Check if file has .pdf extension."""
        return file_path.lower().endswith(".pdf")

    def parse(self, file_path: str) -> list[TextSegment]:
        """Extract text from a PDF file.

        Reads each page and extracts text content, separating pages
        and paragraphs with double newlines.

        Args:
            file_path: Path to the .pdf file

        Returns:
            List containing one TextSegment with the full extracted text,
            or empty list if file fails to load or contains no text.
        """
        # Try to import PDF library
        result = self._get_pdf_content(file_path)
        return result

    def _get_pdf_content(self, file_path: str) -> list[TextSegment]:
        """Try to read PDF using available libraries or fallback."""
        # Strategy 1: Try pypdf
        try:
            from pypdf import PdfReader
            return self._parse_with_pypdf(file_path, PdfReader)
        except ImportError:
            pass

        # Strategy 2: Try PyPDF2
        try:
            from PyPDF2 import PdfReader
            return self._parse_with_pypdf(file_path, PdfReader)
        except ImportError:
            pass

        # Strategy 3: Basic fallback using standard library
        return self._parse_basic(file_path)

    def _parse_with_pypdf(self, file_path: str, PdfReader) -> list[TextSegment]:
        """Parse PDF using pypdf or PyPDF2 library."""
        try:
            reader = PdfReader(file_path)
        except FileNotFoundError:
            print(f"Error loading {file_path}: file not found")
            return []
        except Exception as e:
            error_msg = str(e).lower()
            if "encrypt" in error_msg or "password" in error_msg:
                print(f"Error loading {file_path}: PDF is password-protected or encrypted")
            else:
                print(f"Error loading {file_path}: {e}")
            return []

        # Check if encrypted
        if hasattr(reader, "is_encrypted") and reader.is_encrypted:
            print(f"Error loading {file_path}: PDF is password-protected or encrypted")
            return []

        # Extract text from all pages
        page_texts = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text and text.strip():
                    page_texts.append(text.strip())
            except Exception:
                # Skip problematic pages silently
                continue

        if not page_texts:
            print(
                f"Error loading {file_path}: no extractable text found "
                f"(possibly a scanned/image-only PDF)"
            )
            return []

        # Join pages with double newlines for paragraph separation
        full_text = "\n\n".join(page_texts)

        return [TextSegment(content=full_text, metadata={"source": file_path})]

    def _parse_basic(self, file_path: str) -> list[TextSegment]:
        """Basic PDF text extraction fallback using standard library.

        This is a simplified parser that extracts text streams from PDF files.
        It handles basic text content but may miss complex formatting.
        """
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            print(f"Error loading {file_path}: file not found")
            return []
        except (OSError, PermissionError) as e:
            print(f"Error loading {file_path}: {e}")
            return []

        # Verify PDF magic bytes
        if not data.startswith(b"%PDF-"):
            print(f"Error loading {file_path}: not a valid PDF file")
            return []

        # Check for encryption
        if b"/Encrypt" in data:
            print(f"Error loading {file_path}: PDF is password-protected or encrypted")
            return []

        # Extract text from PDF streams
        text_parts = self._extract_text_from_streams(data)

        if not text_parts:
            print(
                f"Error loading {file_path}: no extractable text found "
                f"(possibly a scanned/image-only PDF)"
            )
            return []

        # Join with double newlines for paragraph separation
        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            return []

        return [TextSegment(content=full_text, metadata={"source": file_path})]

    def _extract_text_from_streams(self, data: bytes) -> list[str]:
        """Extract text content from PDF stream objects."""
        import zlib

        text_parts = []

        # Find all stream content
        stream_pattern = re.compile(b"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)

        for match in stream_pattern.finditer(data):
            stream_data = match.group(1)

            # Try to decompress if FlateDecode
            try:
                decompressed = zlib.decompress(stream_data)
                stream_data = decompressed
            except Exception:
                pass  # Not compressed or different compression

            # Extract text between BT and ET markers
            bt_et_pattern = re.compile(b"BT(.*?)ET", re.DOTALL)
            for text_block in bt_et_pattern.finditer(stream_data):
                block = text_block.group(1)
                block_texts = self._extract_text_operators(block)
                if block_texts:
                    text_parts.append(block_texts)

        return text_parts

    def _extract_text_operators(self, block: bytes) -> str:
        """Extract text from PDF text operators (Tj and TJ)."""
        parts = []

        # Extract text from Tj operator: (text) Tj
        tj_pattern = re.compile(rb"\((.*?)\)\s*Tj", re.DOTALL)
        for tj_match in tj_pattern.finditer(block):
            text_bytes = tj_match.group(1)
            decoded = self._decode_pdf_string(text_bytes)
            if decoded.strip():
                parts.append(decoded.strip())

        # Extract text from TJ operator: [(text) num (text) ...] TJ
        tj_array_pattern = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
        for tj_arr_match in tj_array_pattern.finditer(block):
            arr_content = tj_arr_match.group(1)
            # Extract all text strings from the array
            str_pattern = re.compile(rb"\((.*?)\)")
            arr_parts = []
            for str_match in str_pattern.finditer(arr_content):
                text_bytes = str_match.group(1)
                decoded = self._decode_pdf_string(text_bytes)
                arr_parts.append(decoded)
            combined = "".join(arr_parts)
            if combined.strip():
                parts.append(combined.strip())

        return " ".join(parts) if parts else ""

    def _decode_pdf_string(self, text_bytes: bytes) -> str:
        """Decode a PDF string, handling escape sequences."""
        # Handle PDF escape sequences
        text_bytes = text_bytes.replace(b"\\(", b"(")
        text_bytes = text_bytes.replace(b"\\)", b")")
        text_bytes = text_bytes.replace(b"\\\\", b"\\")
        text_bytes = text_bytes.replace(b"\\n", b"\n")
        text_bytes = text_bytes.replace(b"\\r", b"\r")
        text_bytes = text_bytes.replace(b"\\t", b"\t")

        try:
            return text_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""
