"""Unit tests for the TXT parser."""

import pytest

from parsers.txt_parser import TxtParser
from parsers.base import TextSegment


class TestTxtParserSupports:
    """Test file extension support."""
    
    def test_supports_txt_extension(self):
        parser = TxtParser()
        assert parser.supports("document.txt") is True
    
    def test_supports_uppercase_txt(self):
        parser = TxtParser()
        assert parser.supports("DOCUMENT.TXT") is True
    
    def test_supports_mixed_case_txt(self):
        parser = TxtParser()
        assert parser.supports("Document.Txt") is True
    
    def test_does_not_support_docx(self):
        parser = TxtParser()
        assert parser.supports("document.docx") is False
    
    def test_does_not_support_pdf(self):
        parser = TxtParser()
        assert parser.supports("document.pdf") is False
    
    def test_does_not_support_no_extension(self):
        parser = TxtParser()
        assert parser.supports("document") is False
    
    def test_does_not_support_text_extension(self):
        """Only .txt, not .text"""
        parser = TxtParser()
        assert parser.supports("document.text") is False


class TestTxtParserParse:
    """Test TXT parsing functionality."""
    
    def test_parse_simple_text(self, tmp_path):
        """Parse a simple UTF-8 text file."""
        file_path = tmp_path / "simple.txt"
        file_path.write_text("Hello, this is a simple text file.", encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
        assert isinstance(result[0], TextSegment)
        assert result[0].content == "Hello, this is a simple text file."
    
    def test_parse_preserves_source_metadata(self, tmp_path):
        """Requirement 1.5: source file path in metadata."""
        file_path = tmp_path / "meta_test.txt"
        file_path.write_text("Content here", encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert result[0].metadata["source"] == str(file_path)
    
    def test_parse_hebrew_utf8(self, tmp_path):
        """Parse a Hebrew text file in UTF-8."""
        hebrew_text = "שלום עולם! זהו קובץ טקסט בעברית.\nשורה שנייה בעברית."
        file_path = tmp_path / "hebrew.txt"
        file_path.write_text(hebrew_text, encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
        assert "שלום עולם" in result[0].content
        assert "שורה שנייה" in result[0].content
    
    def test_parse_hebrew_cp1255_fallback(self, tmp_path):
        """Parse a Hebrew text file encoded in cp1255 (Windows Hebrew)."""
        hebrew_text = "שלום עולם"
        file_path = tmp_path / "hebrew_win.txt"
        file_path.write_bytes(hebrew_text.encode("cp1255"))
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
        assert "שלום עולם" in result[0].content
    
    def test_parse_latin1_fallback(self, tmp_path):
        """Parse a file with latin-1 encoding."""
        text = "Café résumé naïve"
        file_path = tmp_path / "latin.txt"
        file_path.write_bytes(text.encode("latin-1"))
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
        # latin-1 decodes any byte sequence, so content should be present
        assert len(result[0].content) > 0
    
    def test_parse_multiline_text(self, tmp_path):
        """Parse a multi-line text file preserving newlines."""
        text = "Line 1\nLine 2\nLine 3\n\nParagraph 2"
        file_path = tmp_path / "multiline.txt"
        file_path.write_text(text, encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
        assert "Line 1" in result[0].content
        assert "Paragraph 2" in result[0].content
    
    def test_parse_large_text(self, tmp_path):
        """Parse a larger text file."""
        text = "This is a sentence. " * 500  # ~10KB
        file_path = tmp_path / "large.txt"
        file_path.write_text(text, encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
        assert len(result[0].content) == len(text)
    
    def test_parse_produces_single_segment(self, tmp_path):
        """TXT parser should always produce exactly one segment (Req 1.3)."""
        file_path = tmp_path / "single.txt"
        file_path.write_text("Content", encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert len(result) == 1
    
    def test_parse_empty_file_returns_empty(self, tmp_path):
        """An empty file should return empty list."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert result == []
    
    def test_parse_whitespace_only_returns_empty(self, tmp_path):
        """A file with only whitespace should return empty list."""
        file_path = tmp_path / "whitespace.txt"
        file_path.write_text("   \n\n   \t   ", encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert result == []
    
    def test_parse_preserves_full_content(self, tmp_path):
        """The full file content should be in the segment."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        file_path = tmp_path / "full.txt"
        file_path.write_text(text, encoding="utf-8")
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert result[0].content == text


class TestTxtParserErrorHandling:
    """Test error handling."""
    
    def test_nonexistent_file_returns_empty(self, capsys):
        """Requirement 1.4: handle missing files gracefully."""
        parser = TxtParser()
        result = parser.parse("/nonexistent/path/file.txt")
        
        assert result == []
        captured = capsys.readouterr()
        assert "/nonexistent/path/file.txt" in captured.out
    
    @pytest.mark.skipif(
        __import__("os").geteuid() == 0,
        reason="Root user bypasses file permissions"
    )
    def test_permission_denied_returns_empty(self, tmp_path, capsys):
        """Handle permission errors gracefully."""
        import os
        file_path = tmp_path / "noperm.txt"
        file_path.write_text("content", encoding="utf-8")
        os.chmod(str(file_path), 0o000)
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        assert result == []
        captured = capsys.readouterr()
        assert "noperm.txt" in captured.out
        
        # Restore permissions for cleanup
        os.chmod(str(file_path), 0o644)
    
    def test_binary_file_uses_fallback(self, tmp_path):
        """A binary file that's not valid in any encoding should be handled."""
        file_path = tmp_path / "binary.txt"
        # Write bytes that are valid in latin-1 (since latin-1 accepts any byte)
        file_path.write_bytes(bytes(range(128, 256)))
        
        parser = TxtParser()
        result = parser.parse(str(file_path))
        
        # latin-1 accepts all bytes, so it should succeed with fallback
        # Result may be non-empty since latin-1 can decode anything
        assert isinstance(result, list)
    
    def test_error_does_not_raise_exception(self):
        """Parser should never raise exceptions to the caller."""
        parser = TxtParser()
        # Should not raise, even with a path that doesn't exist
        result = parser.parse("/some/impossible/path/that/does/not/exist.txt")
        assert isinstance(result, list)
