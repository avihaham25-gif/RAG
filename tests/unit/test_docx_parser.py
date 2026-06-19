"""Unit tests for the DOCX parser."""

import os
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import pytest

from parsers.docx_parser import DocxParser
from parsers.base import TextSegment


# Helper functions to create DOCX test files using standard library
# (since python-docx may not be available in the test environment)

_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


def _create_content_types_xml() -> str:
    """Create [Content_Types].xml for a minimal DOCX."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )


def _create_rels_xml() -> str:
    """Create _rels/.rels for a minimal DOCX."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        '</Relationships>'
    )


def _create_word_rels_xml() -> str:
    """Create word/_rels/document.xml.rels."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )


def _create_styles_xml(heading_styles: list[tuple[str, str]] | None = None) -> str:
    """Create word/styles.xml with heading style definitions.

    Args:
        heading_styles: List of (style_id, style_name) tuples, e.g. [("Heading1", "heading 1")]
    """
    if heading_styles is None:
        heading_styles = [
            ("Heading1", "heading 1"),
            ("Heading2", "heading 2"),
            ("Heading3", "heading 3"),
        ]

    styles_content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles xmlns:w="{_WORD_NS}">'
    )
    for style_id, style_name in heading_styles:
        styles_content += (
            f'<w:style w:type="paragraph" w:styleId="{style_id}">'
            f'<w:name w:val="{style_name}"/>'
            '</w:style>'
        )
    styles_content += '</w:styles>'
    return styles_content


def _make_paragraph_xml(text: str, style_id: str | None = None) -> str:
    """Create XML for a single paragraph with optional style."""
    ppr = ""
    if style_id:
        ppr = f'<w:pPr><w:pStyle w:val="{style_id}"/></w:pPr>'
    return (
        f'<w:p>{ppr}'
        f'<w:r><w:t>{text}</w:t></w:r>'
        '</w:p>'
    )


def _make_table_xml(rows: list[list[str]]) -> str:
    """Create XML for a table with given row data."""
    table_xml = '<w:tbl>'
    for row in rows:
        table_xml += '<w:tr>'
        for cell_text in row:
            table_xml += (
                '<w:tc>'
                f'<w:p><w:r><w:t>{cell_text}</w:t></w:r></w:p>'
                '</w:tc>'
            )
        table_xml += '</w:tr>'
    table_xml += '</w:tbl>'
    return table_xml


def _create_document_xml(body_content: str) -> str:
    """Create word/document.xml with given body content."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_WORD_NS}">'
        f'<w:body>{body_content}</w:body>'
        '</w:document>'
    )


def _write_docx(file_path: str, body_content: str, styles: list[tuple[str, str]] | None = None) -> None:
    """Write a minimal valid DOCX file with the given body content."""
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _create_content_types_xml())
        zf.writestr("_rels/.rels", _create_rels_xml())
        zf.writestr("word/_rels/document.xml.rels", _create_word_rels_xml())
        zf.writestr("word/document.xml", _create_document_xml(body_content))
        zf.writestr("word/styles.xml", _create_styles_xml(styles))


class TestDocxParserSupports:
    """Test file extension support."""

    def test_supports_docx_extension(self):
        parser = DocxParser()
        assert parser.supports("document.docx") is True

    def test_supports_uppercase_docx(self):
        parser = DocxParser()
        assert parser.supports("DOCUMENT.DOCX") is True

    def test_supports_mixed_case_docx(self):
        parser = DocxParser()
        assert parser.supports("Document.Docx") is True

    def test_does_not_support_pdf(self):
        parser = DocxParser()
        assert parser.supports("document.pdf") is False

    def test_does_not_support_txt(self):
        parser = DocxParser()
        assert parser.supports("document.txt") is False

    def test_does_not_support_doc(self):
        parser = DocxParser()
        assert parser.supports("document.doc") is False

    def test_does_not_support_no_extension(self):
        parser = DocxParser()
        assert parser.supports("document") is False


class TestDocxParserParse:
    """Test DOCX parsing functionality."""

    @pytest.fixture
    def sample_docx(self, tmp_path):
        """Create a sample DOCX file with headings, paragraphs, and a table."""
        body = (
            _make_paragraph_xml("Test Document", style_id="Heading1")
            + _make_paragraph_xml(
                "This is the first paragraph with enough content to be meaningful in our testing scenario."
            )
            + _make_paragraph_xml("Second Section", style_id="Heading2")
            + _make_paragraph_xml(
                "This is the second paragraph that belongs to the second section of our test document."
            )
            + _make_table_xml([
                ["Header 1", "Header 2"],
                ["Value 1", "Value 2"],
            ])
        )
        file_path = str(tmp_path / "test_document.docx")
        _write_docx(file_path, body)
        return file_path

    @pytest.fixture
    def empty_docx(self, tmp_path):
        """Create an empty DOCX file (body with no content)."""
        file_path = str(tmp_path / "empty.docx")
        _write_docx(file_path, "")
        return file_path

    @pytest.fixture
    def whitespace_only_docx(self, tmp_path):
        """Create a DOCX file with only whitespace content."""
        body = _make_paragraph_xml("   ")
        file_path = str(tmp_path / "whitespace.docx")
        _write_docx(file_path, body)
        return file_path

    def test_parse_returns_text_segments(self, sample_docx):
        parser = DocxParser()
        result = parser.parse(sample_docx)
        assert len(result) >= 1
        assert all(isinstance(s, TextSegment) for s in result)

    def test_parse_preserves_source_in_metadata(self, sample_docx):
        """Requirement 1.5: source file path in metadata."""
        parser = DocxParser()
        result = parser.parse(sample_docx)
        for segment in result:
            assert "source" in segment.metadata
            assert segment.metadata["source"] == sample_docx

    def test_parse_extracts_headings(self, sample_docx):
        parser = DocxParser()
        result = parser.parse(sample_docx)
        full_text = " ".join(s.content for s in result)
        assert "Test Document" in full_text
        assert "Second Section" in full_text

    def test_parse_extracts_paragraphs(self, sample_docx):
        parser = DocxParser()
        result = parser.parse(sample_docx)
        full_text = " ".join(s.content for s in result)
        assert "first paragraph" in full_text
        assert "second paragraph" in full_text

    def test_parse_extracts_table_content(self, sample_docx):
        parser = DocxParser()
        result = parser.parse(sample_docx)
        full_text = " ".join(s.content for s in result)
        assert "Header 1" in full_text
        assert "Value 1" in full_text

    def test_parse_produces_non_empty_content(self, sample_docx):
        """Requirement 1.7: at least one non-empty segment."""
        parser = DocxParser()
        result = parser.parse(sample_docx)
        assert len(result) >= 1
        assert any(len(s.content.strip()) > 0 for s in result)

    def test_parse_empty_docx_returns_empty(self, empty_docx):
        parser = DocxParser()
        result = parser.parse(empty_docx)
        assert result == []

    def test_parse_whitespace_only_returns_empty(self, whitespace_only_docx):
        parser = DocxParser()
        result = parser.parse(whitespace_only_docx)
        assert result == []

    def test_parse_headings_formatted_with_markdown(self, sample_docx):
        """Headings should be formatted so SemanticChunker recognizes them."""
        parser = DocxParser()
        result = parser.parse(sample_docx)
        full_text = " ".join(s.content for s in result)
        # Should contain markdown-style heading markers
        assert "# Test Document" in full_text

    def test_parse_heading_levels(self, tmp_path):
        """Different heading levels produce different markdown prefix depths."""
        body = (
            _make_paragraph_xml("Level One", style_id="Heading1")
            + _make_paragraph_xml("Level Two", style_id="Heading2")
            + _make_paragraph_xml("Level Three", style_id="Heading3")
        )
        file_path = str(tmp_path / "headings.docx")
        _write_docx(file_path, body)

        parser = DocxParser()
        result = parser.parse(file_path)
        full_text = result[0].content

        assert "# Level One" in full_text
        assert "## Level Two" in full_text
        assert "### Level Three" in full_text

    def test_parse_table_row_separation(self, tmp_path):
        """Table rows should produce pipe-separated cell content."""
        body = _make_table_xml([
            ["Name", "Age", "City"],
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ])
        file_path = str(tmp_path / "table.docx")
        _write_docx(file_path, body)

        parser = DocxParser()
        result = parser.parse(file_path)
        full_text = result[0].content

        assert "Name | Age | City" in full_text
        assert "Alice | 30 | NYC" in full_text
        assert "Bob | 25 | LA" in full_text

    def test_parse_reading_order_preserved(self, tmp_path):
        """Content should appear in the same order as in the document."""
        body = (
            _make_paragraph_xml("First paragraph")
            + _make_table_xml([["Table content"]])
            + _make_paragraph_xml("Last paragraph")
        )
        file_path = str(tmp_path / "order.docx")
        _write_docx(file_path, body)

        parser = DocxParser()
        result = parser.parse(file_path)
        full_text = result[0].content

        first_idx = full_text.index("First paragraph")
        table_idx = full_text.index("Table content")
        last_idx = full_text.index("Last paragraph")

        assert first_idx < table_idx < last_idx


class TestDocxParserErrorHandling:
    """Test error handling for corrupt/missing files."""

    def test_nonexistent_file_returns_empty(self, capsys):
        """Requirement 1.4: handle missing files gracefully."""
        parser = DocxParser()
        result = parser.parse("/nonexistent/path/file.docx")
        assert result == []
        captured = capsys.readouterr()
        assert "/nonexistent/path/file.docx" in captured.out

    def test_corrupt_file_returns_empty(self, tmp_path, capsys):
        """Requirement 1.4: handle corrupt files gracefully."""
        corrupt_file = tmp_path / "corrupt.docx"
        corrupt_file.write_bytes(b"this is not a valid docx file")

        parser = DocxParser()
        result = parser.parse(str(corrupt_file))
        assert result == []
        captured = capsys.readouterr()
        assert "corrupt.docx" in captured.out

    def test_error_does_not_raise_exception(self, tmp_path):
        """Parser should never raise exceptions to the caller."""
        corrupt_file = tmp_path / "bad.docx"
        corrupt_file.write_bytes(b"\x00\x01\x02\x03")

        parser = DocxParser()
        # Should not raise
        result = parser.parse(str(corrupt_file))
        assert isinstance(result, list)

    def test_zip_without_document_xml_returns_empty(self, tmp_path, capsys):
        """A valid ZIP that's not a DOCX should be handled gracefully."""
        fake_docx = tmp_path / "not_docx.docx"
        with zipfile.ZipFile(str(fake_docx), "w") as zf:
            zf.writestr("random.txt", "not a docx")

        parser = DocxParser()
        result = parser.parse(str(fake_docx))
        assert result == []
        captured = capsys.readouterr()
        assert "not_docx.docx" in captured.out

    def test_malformed_xml_in_docx_returns_empty(self, tmp_path, capsys):
        """A DOCX with invalid XML should be handled gracefully."""
        bad_xml_docx = tmp_path / "bad_xml.docx"
        with zipfile.ZipFile(str(bad_xml_docx), "w") as zf:
            zf.writestr("[Content_Types].xml", _create_content_types_xml())
            zf.writestr("_rels/.rels", _create_rels_xml())
            zf.writestr("word/document.xml", "<this is not valid xml<<<")

        parser = DocxParser()
        result = parser.parse(str(bad_xml_docx))
        assert result == []
        captured = capsys.readouterr()
        assert "bad_xml.docx" in captured.out
