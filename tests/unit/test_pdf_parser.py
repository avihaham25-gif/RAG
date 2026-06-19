"""Unit tests for the PDF parser."""

import pytest

from parsers.pdf_parser import PdfParser
from parsers.base import TextSegment


def _create_minimal_pdf(text: str = "Hello World") -> bytes:
    """Create a minimal valid PDF file with the given text.

    This creates a bare-minimum PDF 1.0 file with a single page
    containing uncompressed text using the Tj operator.
    """
    # Encode text for PDF (basic ASCII/Latin-1)
    text_bytes = text.encode("latin-1", errors="replace")

    # Build the content stream (page content)
    content_stream = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"100 700 Td\n"
        b"(" + text_bytes + b") Tj\n"
        b"ET\n"
    )
    stream_length = len(content_stream)

    # Build PDF structure
    pdf = b"%PDF-1.0\n"

    # Object 1: Catalog
    obj1_offset = len(pdf)
    pdf += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    # Object 2: Pages
    obj2_offset = len(pdf)
    pdf += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    # Object 3: Page
    obj3_offset = len(pdf)
    pdf += (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    # Object 4: Content stream
    obj4_offset = len(pdf)
    pdf += (
        b"4 0 obj\n"
        b"<< /Length " + str(stream_length).encode() + b" >>\n"
        b"stream\n"
        + content_stream
        + b"\nendstream\n"
        b"endobj\n"
    )

    # Object 5: Font
    obj5_offset = len(pdf)
    pdf += (
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )

    # Cross-reference table
    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += b"0 6\n"
    pdf += b"0000000000 65535 f \n"
    pdf += f"{obj1_offset:010d} 00000 n \n".encode()
    pdf += f"{obj2_offset:010d} 00000 n \n".encode()
    pdf += f"{obj3_offset:010d} 00000 n \n".encode()
    pdf += f"{obj4_offset:010d} 00000 n \n".encode()
    pdf += f"{obj5_offset:010d} 00000 n \n".encode()

    # Trailer
    pdf += (
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode()
        + b"\n"
        b"%%EOF\n"
    )

    return pdf


def _create_multi_text_pdf(texts: list[str]) -> bytes:
    """Create a PDF with multiple text blocks (simulating multiple paragraphs).

    Each text gets its own BT/ET block within the same content stream,
    which the parser should extract as separate paragraph blocks.
    """
    # Build the content stream with multiple BT/ET blocks
    content_stream = b""
    y_position = 700
    for text in texts:
        text_bytes = text.encode("latin-1", errors="replace")
        content_stream += (
            b"BT\n"
            b"/F1 12 Tf\n"
            + f"100 {y_position} Td\n".encode()
            + b"(" + text_bytes + b") Tj\n"
            b"ET\n"
        )
        y_position -= 50

    stream_length = len(content_stream)

    # Build PDF structure
    pdf = b"%PDF-1.0\n"

    obj1_offset = len(pdf)
    pdf += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    obj2_offset = len(pdf)
    pdf += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    obj3_offset = len(pdf)
    pdf += (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    obj4_offset = len(pdf)
    pdf += (
        b"4 0 obj\n"
        b"<< /Length " + str(stream_length).encode() + b" >>\n"
        b"stream\n"
        + content_stream
        + b"\nendstream\n"
        b"endobj\n"
    )

    obj5_offset = len(pdf)
    pdf += (
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += b"0 6\n"
    pdf += b"0000000000 65535 f \n"
    pdf += f"{obj1_offset:010d} 00000 n \n".encode()
    pdf += f"{obj2_offset:010d} 00000 n \n".encode()
    pdf += f"{obj3_offset:010d} 00000 n \n".encode()
    pdf += f"{obj4_offset:010d} 00000 n \n".encode()
    pdf += f"{obj5_offset:010d} 00000 n \n".encode()

    pdf += (
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode()
        + b"\n"
        b"%%EOF\n"
    )

    return pdf


def _create_empty_pdf() -> bytes:
    """Create a valid PDF with a page but no text content."""
    pdf = b"%PDF-1.0\n"

    obj1_offset = len(pdf)
    pdf += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    obj2_offset = len(pdf)
    pdf += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    obj3_offset = len(pdf)
    pdf += (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
        b"endobj\n"
    )

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += b"0 4\n"
    pdf += b"0000000000 65535 f \n"
    pdf += f"{obj1_offset:010d} 00000 n \n".encode()
    pdf += f"{obj2_offset:010d} 00000 n \n".encode()
    pdf += f"{obj3_offset:010d} 00000 n \n".encode()

    pdf += (
        b"trailer\n"
        b"<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode()
        + b"\n"
        b"%%EOF\n"
    )

    return pdf


def _create_encrypted_pdf() -> bytes:
    """Create a PDF that appears encrypted (contains /Encrypt dictionary)."""
    pdf = b"%PDF-1.0\n"

    obj1_offset = len(pdf)
    pdf += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    obj2_offset = len(pdf)
    pdf += b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"

    obj3_offset = len(pdf)
    pdf += b"3 0 obj\n<< /Type /Encrypt /Filter /Standard /V 1 >>\nendobj\n"

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += b"0 4\n"
    pdf += b"0000000000 65535 f \n"
    pdf += f"{obj1_offset:010d} 00000 n \n".encode()
    pdf += f"{obj2_offset:010d} 00000 n \n".encode()
    pdf += f"{obj3_offset:010d} 00000 n \n".encode()

    pdf += (
        b"trailer\n"
        b"<< /Size 4 /Root 1 0 R /Encrypt 3 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode()
        + b"\n"
        b"%%EOF\n"
    )

    return pdf


def _create_tj_array_pdf(text: str = "Hello Array World") -> bytes:
    """Create a PDF that uses the TJ array operator for text rendering."""
    text_bytes = text.encode("latin-1", errors="replace")

    # TJ operator uses an array: [(text) kerning (text) ...] TJ
    content_stream = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"100 700 Td\n"
        b"[(" + text_bytes + b") -10] TJ\n"
        b"ET\n"
    )
    stream_length = len(content_stream)

    pdf = b"%PDF-1.0\n"

    obj1_offset = len(pdf)
    pdf += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    obj2_offset = len(pdf)
    pdf += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    obj3_offset = len(pdf)
    pdf += (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    obj4_offset = len(pdf)
    pdf += (
        b"4 0 obj\n"
        b"<< /Length " + str(stream_length).encode() + b" >>\n"
        b"stream\n"
        + content_stream
        + b"\nendstream\n"
        b"endobj\n"
    )

    obj5_offset = len(pdf)
    pdf += (
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += b"0 6\n"
    pdf += b"0000000000 65535 f \n"
    pdf += f"{obj1_offset:010d} 00000 n \n".encode()
    pdf += f"{obj2_offset:010d} 00000 n \n".encode()
    pdf += f"{obj3_offset:010d} 00000 n \n".encode()
    pdf += f"{obj4_offset:010d} 00000 n \n".encode()
    pdf += f"{obj5_offset:010d} 00000 n \n".encode()

    pdf += (
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode()
        + b"\n"
        b"%%EOF\n"
    )

    return pdf


class TestPdfParserSupports:
    """Test file extension support."""

    def test_supports_pdf_extension(self):
        parser = PdfParser()
        assert parser.supports("document.pdf") is True

    def test_supports_uppercase_pdf(self):
        parser = PdfParser()
        assert parser.supports("DOCUMENT.PDF") is True

    def test_supports_mixed_case_pdf(self):
        parser = PdfParser()
        assert parser.supports("Document.Pdf") is True

    def test_does_not_support_docx(self):
        parser = PdfParser()
        assert parser.supports("document.docx") is False

    def test_does_not_support_txt(self):
        parser = PdfParser()
        assert parser.supports("document.txt") is False

    def test_does_not_support_no_extension(self):
        parser = PdfParser()
        assert parser.supports("document") is False

    def test_does_not_support_pdf_in_name_without_extension(self):
        parser = PdfParser()
        assert parser.supports("pdf_document.txt") is False

    def test_supports_path_with_directories(self):
        parser = PdfParser()
        assert parser.supports("/some/path/to/file.pdf") is True


class TestPdfParserParse:
    """Test PDF parsing functionality."""

    def test_parse_simple_pdf(self, tmp_path):
        """Parse a simple PDF with text using Tj operator."""
        file_path = tmp_path / "simple.pdf"
        file_path.write_bytes(_create_minimal_pdf("Hello World"))

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        assert isinstance(result[0], TextSegment)
        assert "Hello World" in result[0].content

    def test_parse_preserves_source_metadata(self, tmp_path):
        """Requirement 1.5: source file path in metadata."""
        file_path = tmp_path / "meta.pdf"
        file_path.write_bytes(_create_minimal_pdf("Test content"))

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        assert result[0].metadata["source"] == str(file_path)

    def test_parse_produces_non_empty_content(self, tmp_path):
        """Requirement 1.7: at least one non-empty segment."""
        file_path = tmp_path / "nonempty.pdf"
        file_path.write_bytes(_create_minimal_pdf("Some text content here"))

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        assert len(result[0].content.strip()) > 0

    def test_parse_empty_pdf_returns_empty(self, tmp_path):
        """A PDF with no text content should return empty."""
        file_path = tmp_path / "empty.pdf"
        file_path.write_bytes(_create_empty_pdf())

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert result == []

    def test_parse_returns_single_segment(self, tmp_path):
        """PDF parser should produce a single TextSegment."""
        file_path = tmp_path / "single.pdf"
        file_path.write_bytes(_create_minimal_pdf("Page content"))

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1

    def test_parse_tj_array_operator(self, tmp_path):
        """Parse a PDF using the TJ array operator."""
        file_path = tmp_path / "tj_array.pdf"
        file_path.write_bytes(_create_tj_array_pdf("Array Text"))

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        assert "Array Text" in result[0].content


class TestPdfParserErrorHandling:
    """Test error handling for corrupt/missing/locked files."""

    def test_nonexistent_file_returns_empty(self, capsys):
        """Requirement 1.4: handle missing files gracefully."""
        parser = PdfParser()
        result = parser.parse("/nonexistent/path/file.pdf")

        assert result == []
        captured = capsys.readouterr()
        assert "/nonexistent/path/file.pdf" in captured.out

    def test_corrupt_file_returns_empty(self, tmp_path, capsys):
        """Requirement 1.4: handle corrupt files gracefully."""
        corrupt_file = tmp_path / "corrupt.pdf"
        corrupt_file.write_bytes(b"this is not a valid pdf file at all")

        parser = PdfParser()
        result = parser.parse(str(corrupt_file))

        assert result == []
        captured = capsys.readouterr()
        assert "corrupt.pdf" in captured.out

    def test_encrypted_pdf_returns_empty(self, tmp_path, capsys):
        """Handle encrypted/password-protected PDFs."""
        file_path = tmp_path / "encrypted.pdf"
        file_path.write_bytes(_create_encrypted_pdf())

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert result == []
        captured = capsys.readouterr()
        assert "encrypted.pdf" in captured.out

    def test_binary_garbage_returns_empty(self, tmp_path, capsys):
        """Random binary data should be handled gracefully."""
        file_path = tmp_path / "garbage.pdf"
        file_path.write_bytes(bytes(range(256)) * 10)

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert result == []
        captured = capsys.readouterr()
        assert "garbage.pdf" in captured.out

    def test_error_does_not_raise_exception(self, tmp_path):
        """Parser should never raise exceptions."""
        corrupt_file = tmp_path / "bad.pdf"
        corrupt_file.write_bytes(b"\x00\x01\x02\x03")

        parser = PdfParser()
        result = parser.parse(str(corrupt_file))
        assert isinstance(result, list)

    def test_scanned_pdf_no_text(self, tmp_path, capsys):
        """A PDF that's image-only (simulated by empty text content)."""
        file_path = tmp_path / "scanned.pdf"
        file_path.write_bytes(_create_empty_pdf())

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert result == []
        # Should print a message about no text found
        captured = capsys.readouterr()
        assert "scanned.pdf" in captured.out

    def test_pdf_magic_bytes_but_no_streams(self, tmp_path, capsys):
        """A file starting with %PDF- but with no valid content."""
        file_path = tmp_path / "fake.pdf"
        file_path.write_bytes(b"%PDF-1.0\n%garbage content here\n%%EOF\n")

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert result == []
        captured = capsys.readouterr()
        assert "fake.pdf" in captured.out


class TestPdfParserParagraphSeparation:
    """Test paragraph separation with double newlines (Req 1.2)."""

    def test_multiple_text_blocks_separated_by_double_newline(self, tmp_path):
        """Text from different BT/ET blocks should be separated by \\n\\n."""
        file_path = tmp_path / "paragraphs.pdf"
        file_path.write_bytes(
            _create_multi_text_pdf(["First paragraph", "Second paragraph"])
        )

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        content = result[0].content
        # Should contain double newline separating the blocks
        assert "\n\n" in content
        assert "First paragraph" in content
        assert "Second paragraph" in content

    def test_three_paragraphs_separated(self, tmp_path):
        """Three distinct text blocks should produce two \\n\\n separators."""
        file_path = tmp_path / "three_paras.pdf"
        file_path.write_bytes(
            _create_multi_text_pdf(["Para one", "Para two", "Para three"])
        )

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        content = result[0].content
        assert "Para one" in content
        assert "Para two" in content
        assert "Para three" in content
        # Check double newlines are used as separators
        parts = content.split("\n\n")
        assert len(parts) >= 3

    def test_single_paragraph_no_extra_newlines(self, tmp_path):
        """A single text block should not have leading/trailing double newlines."""
        file_path = tmp_path / "single_para.pdf"
        file_path.write_bytes(_create_minimal_pdf("Just one paragraph"))

        parser = PdfParser()
        result = parser.parse(str(file_path))

        assert len(result) == 1
        content = result[0].content
        assert "Just one paragraph" in content
        # Single paragraph should not have double newlines
        assert "\n\n" not in content
