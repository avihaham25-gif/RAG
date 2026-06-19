"""Parsers package for multi-format document loading (DOCX, PDF, TXT)."""

from parsers.base import BaseParser, TextSegment
from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PdfParser
from parsers.txt_parser import TxtParser
from parsers.registry import ParserRegistry

__all__ = ["BaseParser", "TextSegment", "DocxParser", "PdfParser", "TxtParser", "ParserRegistry"]
