"""Unit tests for the Parser Registry."""

import pytest

from parsers.registry import ParserRegistry
from parsers.base import BaseParser, TextSegment
from parsers.txt_parser import TxtParser
from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PdfParser


class TestParserRegistryGetParser:
    """Test get_parser dispatching logic."""
    
    def test_txt_file_returns_txt_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("document.txt")
        assert isinstance(parser, TxtParser)
    
    def test_docx_file_returns_docx_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("document.docx")
        assert isinstance(parser, DocxParser)
    
    def test_pdf_file_returns_pdf_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("report.pdf")
        assert isinstance(parser, PdfParser)
    
    def test_uppercase_extensions_work(self):
        registry = ParserRegistry()
        assert isinstance(registry.get_parser("FILE.TXT"), TxtParser)
        assert isinstance(registry.get_parser("FILE.DOCX"), DocxParser)
        assert isinstance(registry.get_parser("FILE.PDF"), PdfParser)
    
    def test_unsupported_extension_returns_none(self):
        registry = ParserRegistry()
        assert registry.get_parser("image.png") is None
        assert registry.get_parser("data.csv") is None
        assert registry.get_parser("archive.zip") is None
    
    def test_no_extension_returns_none(self):
        registry = ParserRegistry()
        assert registry.get_parser("README") is None
    
    def test_path_with_directories(self):
        registry = ParserRegistry()
        parser = registry.get_parser("/home/user/docs/report.pdf")
        assert isinstance(parser, PdfParser)
    
    def test_dotfile_returns_none(self):
        registry = ParserRegistry()
        assert registry.get_parser(".gitignore") is None


class TestParserRegistryParseFile:
    """Test parse_file method."""
    
    def test_parse_txt_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello World", encoding="utf-8")
        
        registry = ParserRegistry()
        result = registry.parse_file(str(file_path))
        
        assert len(result) == 1
        assert result[0].content == "Hello World"
        assert result[0].metadata["source"] == str(file_path)
    
    def test_parse_unsupported_returns_empty(self):
        registry = ParserRegistry()
        result = registry.parse_file("image.png")
        assert result == []
    
    def test_parse_nonexistent_file_returns_empty(self):
        registry = ParserRegistry()
        result = registry.parse_file("/nonexistent/file.txt")
        assert result == []


class TestParserRegistryParseBatch:
    """Test batch processing."""
    
    def test_batch_multiple_txt_files(self, tmp_path):
        # Create test files
        file1 = tmp_path / "doc1.txt"
        file1.write_text("Content one", encoding="utf-8")
        file2 = tmp_path / "doc2.txt"
        file2.write_text("Content two", encoding="utf-8")
        
        registry = ParserRegistry()
        result = registry.parse_batch([str(file1), str(file2)])
        
        assert len(result) == 2
        contents = [s.content for s in result]
        assert "Content one" in contents
        assert "Content two" in contents
    
    def test_batch_mixed_formats(self, tmp_path):
        # Create a TXT file
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Text content", encoding="utf-8")
        
        registry = ParserRegistry()
        result = registry.parse_batch([str(txt_file)])
        
        assert len(result) >= 1
        assert any("Text content" in s.content for s in result)
    
    def test_batch_skips_unsupported_silently(self, tmp_path):
        """Req 1.6: unsupported extensions are skipped without error."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Valid content", encoding="utf-8")
        png_file = tmp_path / "image.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        
        registry = ParserRegistry()
        result = registry.parse_batch([str(txt_file), str(png_file)])
        
        # Only the TXT file should produce output
        assert len(result) == 1
        assert result[0].content == "Valid content"
    
    def test_batch_continues_after_failure(self, tmp_path, capsys):
        """Req 1.4: batch continues after individual file failures."""
        # Create a valid file and a non-existent file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("Valid content here", encoding="utf-8")
        
        registry = ParserRegistry()
        result = registry.parse_batch([
            "/nonexistent/bad.txt",
            str(valid_file),
        ])
        
        # Should still get the valid file's content
        assert len(result) == 1
        assert result[0].content == "Valid content here"
        
        # Error should have been printed for the bad file
        captured = capsys.readouterr()
        assert "bad.txt" in captured.out
    
    def test_batch_empty_list(self):
        registry = ParserRegistry()
        result = registry.parse_batch([])
        assert result == []
    
    def test_batch_all_unsupported(self, tmp_path):
        """All unsupported files should produce empty result."""
        registry = ParserRegistry()
        result = registry.parse_batch(["file.png", "file.csv", "file.zip"])
        assert result == []


class TestParserRegistryRegisterParser:
    """Test custom parser registration."""
    
    def test_register_custom_parser(self):
        class MarkdownParser(BaseParser):
            def supports(self, file_path: str) -> bool:
                return file_path.lower().endswith(".md")
            
            def parse(self, file_path: str) -> list[TextSegment]:
                return [TextSegment(content="markdown", metadata={"source": file_path})]
        
        registry = ParserRegistry()
        registry.register_parser(MarkdownParser())
        
        parser = registry.get_parser("README.md")
        assert parser is not None
        assert isinstance(parser, MarkdownParser)
    
    def test_registered_parser_used_in_batch(self, tmp_path):
        class MarkdownParser(BaseParser):
            def supports(self, file_path: str) -> bool:
                return file_path.lower().endswith(".md")
            
            def parse(self, file_path: str) -> list[TextSegment]:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    return [TextSegment(content=content, metadata={"source": file_path})]
                except Exception:
                    return []
        
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Hello", encoding="utf-8")
        
        registry = ParserRegistry()
        registry.register_parser(MarkdownParser())
        
        result = registry.parse_batch([str(md_file)])
        assert len(result) == 1
        assert "# Hello" in result[0].content


class TestParserRegistrySupportedExtensions:
    """Test supported_extensions property."""
    
    def test_default_supported_extensions(self):
        registry = ParserRegistry()
        extensions = registry.supported_extensions
        assert ".txt" in extensions
        assert ".docx" in extensions
        assert ".pdf" in extensions
    
    def test_supported_extensions_count(self):
        registry = ParserRegistry()
        assert len(registry.supported_extensions) == 3
