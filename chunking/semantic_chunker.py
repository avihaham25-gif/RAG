"""Semantic chunking engine for section-aware document splitting."""

import re
from models import Chunk
from config import RAGConfig


class SemanticChunker:
    """Splits documents into semantically coherent chunks.

    Strategy (in priority order):
    1. Split at section boundaries (headings)
    2. Split at paragraph boundaries (double newlines)
    3. Split at character-length limits within paragraphs
    """

    # Markdown headings: lines starting with # (one or more)
    MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+.+$")
    # Numbered headings: lines starting with digits followed by dot/parenthesis
    NUMBERED_HEADING_RE = re.compile(r"^\d+[\.\)]\s+.+$")

    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.max_size = config.max_chunk_size
        self.overlap = config.chunk_overlap
        self.min_size = config.min_chunk_size

    def chunk_document(self, text: str, source: str) -> list[Chunk]:
        """Split document text into semantically coherent chunks.

        Args:
            text: The full document text to split
            source: Source file path for metadata

        Returns:
            List of Chunk objects with content and metadata
        """
        # Handle empty or whitespace-only documents
        if not text or not text.strip():
            return []

        # Step 1: Detect sections
        sections = self._detect_sections(text)

        # Step 2: Process sections into raw chunk content strings
        raw_chunks = []  # list of (content, section_title)
        for section_title, section_content in sections:
            content = section_content.strip()
            if not content:
                continue

            if len(content) <= self.max_size:
                # Section fits in one chunk
                raw_chunks.append((content, section_title))
            else:
                # Split at paragraph boundaries within the section
                paragraphs = self._split_section_into_paragraphs(section_content)
                accumulated = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue

                    if len(para) > self.max_size:
                        # Flush accumulated content first
                        if accumulated.strip():
                            raw_chunks.append((accumulated.strip(), section_title))
                            accumulated = ""
                        # Split the large paragraph by character limit
                        sub_pieces = self._split_by_character_limit(para)
                        for piece in sub_pieces:
                            raw_chunks.append((piece.strip(), section_title))
                    elif accumulated and (len(accumulated) + len("\n\n") + len(para)) > self.max_size:
                        # Adding this paragraph would exceed max; flush
                        raw_chunks.append((accumulated.strip(), section_title))
                        accumulated = para
                    else:
                        if accumulated:
                            accumulated += "\n\n" + para
                        else:
                            accumulated = para

                # Flush remaining accumulated content
                if accumulated.strip():
                    raw_chunks.append((accumulated.strip(), section_title))

        # If no content was produced, return empty
        if not raw_chunks:
            return []

        # Step 3: Apply overlap
        contents = [c for c, _ in raw_chunks]
        titles = [t for _, t in raw_chunks]
        contents = self._apply_overlap(contents)

        # Step 4: Create Chunk objects with metadata
        chunks = []
        for i, (content, title) in enumerate(zip(contents, titles)):
            chunk = Chunk(
                content=content,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "section_title": title,
                },
            )
            chunks.append(chunk)

        # Step 5: Merge trailing fragment if < min_size
        chunks = self._merge_trailing_fragment(chunks)

        # Step 6: Filter out any chunks that are entirely empty after processing
        chunks = [c for c in chunks if c.content.strip()]

        # Re-index after merge
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks

    def _detect_sections(self, text: str) -> list[tuple[str, str]]:
        """Identify section boundaries and split text into sections.

        Returns list of (section_title, section_content) tuples.
        A section_title is "" for content before the first heading.
        """
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            is_heading = False

            # Check markdown heading
            if self.MARKDOWN_HEADING_RE.match(line):
                is_heading = True
            # Check numbered heading
            elif self.NUMBERED_HEADING_RE.match(line):
                is_heading = True
            # Check short line (< 80 chars) followed by an empty line
            elif len(line.strip()) > 0 and len(line) < 80:
                # Look ahead for empty line
                if i + 1 < len(lines) and lines[i + 1].strip() == "":
                    # Only treat as heading if it looks like a title (not a regular short line)
                    # We'll be more conservative: only if the line doesn't end with
                    # sentence-ending punctuation and is reasonably short
                    stripped = line.strip()
                    if (
                        stripped
                        and not stripped.endswith(".")
                        and not stripped.endswith(",")
                        and not stripped.endswith(";")
                        and not stripped.endswith(":")
                        and len(stripped) < 80
                        and len(stripped) > 0
                    ):
                        is_heading = True

            if is_heading:
                # Save previous section
                if current_lines or current_title:
                    content = "\n".join(current_lines)
                    sections.append((current_title, content))
                # Extract title text
                stripped = line.strip()
                if self.MARKDOWN_HEADING_RE.match(stripped):
                    # Remove leading # symbols
                    current_title = re.sub(r"^#+\s*", "", stripped)
                elif self.NUMBERED_HEADING_RE.match(stripped):
                    current_title = stripped
                else:
                    current_title = stripped
                current_lines = []
            else:
                current_lines.append(line)

            i += 1

        # Don't forget the last section
        if current_lines or current_title:
            content = "\n".join(current_lines)
            sections.append((current_title, content))

        return sections

    def _split_section_into_paragraphs(self, section_text: str) -> list[str]:
        """Split a section into paragraphs at double-newline boundaries."""
        # Split at sequences of two or more consecutive newline characters
        paragraphs = re.split(r"\n{2,}", section_text)
        return [p for p in paragraphs if p.strip()]

    def _split_by_character_limit(self, text: str) -> list[str]:
        """Split text that exceeds max_size into pieces at character limits.
        Tries to split at sentence boundaries (., !, ?) first, then whitespace."""
        pieces = []
        remaining = text

        while len(remaining) > self.max_size:
            # Try to find a sentence boundary within max_size
            chunk_candidate = remaining[: self.max_size]

            # Look for last sentence-ending punctuation followed by space
            split_pos = -1
            for punct in [".", "!", "?"]:
                pos = chunk_candidate.rfind(punct + " ")
                if pos > split_pos:
                    split_pos = pos + 1  # include the punctuation

            # If we found a sentence boundary and it gives us enough content
            if split_pos > self.min_size:
                pieces.append(remaining[:split_pos].strip())
                remaining = remaining[split_pos:].strip()
            else:
                # Try splitting at whitespace
                space_pos = chunk_candidate.rfind(" ")
                if space_pos > self.min_size:
                    pieces.append(remaining[:space_pos].strip())
                    remaining = remaining[space_pos:].strip()
                else:
                    # Hard split at max_size
                    pieces.append(remaining[: self.max_size].strip())
                    remaining = remaining[self.max_size :].strip()

        if remaining.strip():
            pieces.append(remaining.strip())

        return pieces

    def _apply_overlap(self, chunks_content: list[str]) -> list[str]:
        """Apply overlap between consecutive chunks by prepending overlap chars from previous chunk."""
        if len(chunks_content) <= 1:
            return chunks_content

        result = [chunks_content[0]]
        for i in range(1, len(chunks_content)):
            prev = chunks_content[i - 1]
            current = chunks_content[i]

            # Get the last `overlap` characters from previous chunk
            overlap_text = prev[-self.overlap :] if len(prev) >= self.overlap else prev

            # Prepend overlap to current chunk
            overlapped = overlap_text + " " + current
            # Ensure the overlapped chunk doesn't exceed max_size
            if len(overlapped) > self.max_size:
                # Trim overlap to fit
                available = self.max_size - len(current) - 1  # -1 for space
                if available > 0:
                    overlap_text = prev[-available:]
                    overlapped = overlap_text + " " + current
                else:
                    overlapped = current

            result.append(overlapped)

        return result

    def _merge_trailing_fragment(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge final chunk into preceding if < min_size chars of content."""
        if len(chunks) <= 1:
            return chunks

        last_chunk = chunks[-1]
        # Calculate content length excluding overlap
        # For simplicity, we check the total content length
        if len(last_chunk.content) < self.min_size:
            # Merge into preceding chunk
            preceding = chunks[-2]
            merged_content = preceding.content + "\n\n" + last_chunk.content
            preceding.content = merged_content
            return chunks[:-1]

        return chunks
