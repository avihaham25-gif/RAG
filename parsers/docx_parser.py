"""DOCX document parser using standard library zipfile and xml.etree.

DOCX files are ZIP archives containing XML files. The main document body
is stored in word/document.xml. This parser extracts text from paragraphs,
headings, and table cells in reading order without requiring python-docx.
"""

import zipfile
import xml.etree.ElementTree as ET

from parsers.base import BaseParser, TextSegment


# Word XML namespace
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS = {"w": _WORD_NS}


class DocxParser(BaseParser):
    """Parser for Microsoft Word .docx documents.

    Extracts text from paragraphs, headings, and table cells
    in reading order using Python's built-in zipfile and xml modules.
    """

    def supports(self, file_path: str) -> bool:
        """Check if file has .docx extension."""
        return file_path.lower().endswith(".docx")

    def parse(self, file_path: str) -> list[TextSegment]:
        """Extract text segments from a DOCX file.

        Walks through the document body in order, extracting:
        - Headings (preserved with markdown-style heading markers)
        - Paragraphs (body text)
        - Table cell text (row by row, cell by cell)

        All content is returned in reading order as a single TextSegment
        with the full document text, preserving structure via newlines.

        Args:
            file_path: Path to the .docx file

        Returns:
            List containing one TextSegment with the full extracted text,
            or empty list if file fails to load.
        """
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if "word/document.xml" not in zf.namelist():
                    print(f"Error loading {file_path}: not a valid DOCX file (missing word/document.xml)")
                    return []
                xml_content = zf.read("word/document.xml")
                # Also try to read styles for heading detection
                styles_map = self._load_styles(zf)
        except (zipfile.BadZipFile, FileNotFoundError, OSError, KeyError) as e:
            print(f"Error loading {file_path}: {e}")
            return []
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"Error loading {file_path}: {e}")
            return []

        # Find the document body
        body = root.find(f"{{{_WORD_NS}}}body")
        if body is None:
            print(f"Error loading {file_path}: no document body found")
            return []

        # Extract content in reading order
        content_parts = self._extract_body_content(body, styles_map)

        # Join all parts with double newlines to create paragraph separation
        full_text = "\n\n".join(content_parts)

        # Requirement 1.7: Must produce at least one non-empty text segment
        if not full_text.strip():
            return []

        return [TextSegment(
            content=full_text,
            metadata={"source": file_path}
        )]

    def _load_styles(self, zf: zipfile.ZipFile) -> dict[str, str]:
        """Load style definitions to identify heading styles.

        Returns a mapping from style ID to style name.
        """
        styles_map: dict[str, str] = {}
        try:
            if "word/styles.xml" in zf.namelist():
                styles_xml = zf.read("word/styles.xml")
                styles_root = ET.fromstring(styles_xml)
                for style_elem in styles_root.findall(f"{{{_WORD_NS}}}style", _NS):
                    style_id = style_elem.get(f"{{{_WORD_NS}}}styleId", "")
                    name_elem = style_elem.find(f"{{{_WORD_NS}}}name", _NS)
                    if name_elem is not None:
                        style_name = name_elem.get(f"{{{_WORD_NS}}}val", "")
                        styles_map[style_id] = style_name
        except Exception:
            pass  # If styles can't be loaded, we'll still extract text
        return styles_map

    def _extract_body_content(
        self, body: ET.Element, styles_map: dict[str, str]
    ) -> list[str]:
        """Extract text content from body elements in reading order."""
        content_parts: list[str] = []

        for element in body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # It's a paragraph
                text = self._extract_paragraph_text(element)
                if text.strip():
                    heading_level = self._get_heading_level(element, styles_map)
                    if heading_level > 0:
                        prefix = "#" * min(heading_level, 6)
                        content_parts.append(f"{prefix} {text}")
                    else:
                        content_parts.append(text)

            elif tag == "tbl":
                # It's a table
                table_text = self._extract_table_text(element)
                if table_text:
                    content_parts.extend(table_text)

        return content_parts

    def _extract_paragraph_text(self, para_elem: ET.Element) -> str:
        """Extract all text from a paragraph element, including nested runs."""
        texts: list[str] = []
        # Iterate through all 'r' (run) elements within the paragraph
        for run in para_elem.iter(f"{{{_WORD_NS}}}r"):
            for text_elem in run.iter(f"{{{_WORD_NS}}}t"):
                if text_elem.text:
                    texts.append(text_elem.text)
        return "".join(texts)

    def _get_heading_level(
        self, para_elem: ET.Element, styles_map: dict[str, str]
    ) -> int:
        """Determine if a paragraph is a heading and return its level (0 = not a heading)."""
        # Check paragraph properties for style reference
        ppr = para_elem.find(f"{{{_WORD_NS}}}pPr")
        if ppr is None:
            return 0

        style_ref = ppr.find(f"{{{_WORD_NS}}}pStyle")
        if style_ref is None:
            return 0

        style_id = style_ref.get(f"{{{_WORD_NS}}}val", "")

        # Check if it's a heading style by ID pattern (e.g., "Heading1", "heading1")
        # Common patterns: "Heading1", "Heading2", ..., "heading 1", etc.
        style_name = styles_map.get(style_id, style_id)

        # Check style name patterns
        for name in [style_name, style_id]:
            name_lower = name.lower().replace(" ", "")
            if name_lower.startswith("heading"):
                level_str = name_lower.replace("heading", "")
                try:
                    level = int(level_str)
                    return level
                except ValueError:
                    return 1

        return 0

    def _extract_table_text(self, table_elem: ET.Element) -> list[str]:
        """Extract text from a table element, row by row."""
        rows_text: list[str] = []

        for row in table_elem.iter(f"{{{_WORD_NS}}}tr"):
            cell_texts: list[str] = []
            for cell in row.iter(f"{{{_WORD_NS}}}tc"):
                # Extract all paragraph text within the cell
                cell_content_parts: list[str] = []
                for para in cell.iter(f"{{{_WORD_NS}}}p"):
                    para_text = self._extract_paragraph_text(para)
                    if para_text.strip():
                        cell_content_parts.append(para_text.strip())
                cell_text = " ".join(cell_content_parts)
                if cell_text:
                    cell_texts.append(cell_text)
            if cell_texts:
                rows_text.append(" | ".join(cell_texts))

        return rows_text
