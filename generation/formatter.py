"""Structured response formatter for the Generation Pipeline.

Ensures LLM responses conform to the required format:
Line 1: Hebrew title (≤10 words)
Lines 2+: Answer paragraph
Final line: Source citations (comma-separated file names)
"""

import re
from config import RAGConfig


# Preamble patterns to remove (Req 9.4)
# These are phrases that reference the retrieval process or documents
PREAMBLE_PATTERNS = [
    r'^על\s*פי\s*המסמכים[,:]?\s*',
    r'^בהתבסס\s*על\s*המסמכים[,:]?\s*',
    r'^לפי\s*המסמכים[,:]?\s*',
    r'^מתוך\s*המסמכים[,:]?\s*',
    r'^על\s*סמך\s*המידע[,:]?\s*',
    r'^בהתאם\s*למסמכים[,:]?\s*',
    r'^המסמכים\s*מציינים[,:]?\s*',
    r'^according\s*to\s*the\s*(documents|sources)[,:]?\s*',
    r'^based\s*on\s*the\s*(documents|sources|retrieved\s*information)[,:]?\s*',
    r'^from\s*the\s*(retrieved\s*information|documents|sources)[,:]?\s*',
    r'^להלן\s*התשובה[,:]?\s*',
    r'^הנה\s*התשובה[,:]?\s*',
    r'^התשובה\s*היא[,:]?\s*',
]

# Compiled preamble patterns (case-insensitive)
_PREAMBLE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PREAMBLE_PATTERNS]


class ResponseFormatter:
    """Formats LLM responses into the required structure.
    
    Required structure:
    - Line 1: Hebrew title (≤10 words)
    - Body: Single answer paragraph
    - Final line: Source citations (מקורות: file1.txt, file2.pdf)
    
    If the LLM output doesn't conform, the formatter auto-corrects it.
    """
    
    # Citation line prefix
    CITATIONS_PREFIX = "מקורות:"
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.config = config
    
    def format(self, raw_response: str, sources: list[str]) -> str:
        """Format a raw LLM response into the required structure.
        
        Args:
            raw_response: The raw text from the LLM.
            sources: List of source document file names that contributed.
            
        Returns:
            Formatted response with title, answer paragraph, and citations.
        """
        if not raw_response or not raw_response.strip():
            return self._build_formatted("", "", sources)
        
        # Step 1: Remove preambles (Req 9.4)
        cleaned = self._remove_preambles(raw_response.strip())
        
        # Step 2: Try to parse existing structure
        title, body = self._extract_title_and_body(cleaned)
        
        # Step 3: Remove any existing citation lines from body (we'll add our own)
        body = self._remove_existing_citations(body)
        
        # Step 4: Validate title (≤10 words)
        title = self._validate_title(title, body)
        
        # Step 5: Clean up body paragraph
        body = self._clean_body(body)
        
        # Step 6: Build the final formatted output
        return self._build_formatted(title, body, sources)
    
    def _remove_preambles(self, text: str) -> str:
        """Remove introductory preambles that reference documents/sources.
        
        Applies regex patterns to strip common preamble phrases.
        """
        result = text
        for pattern in _PREAMBLE_RE:
            result = pattern.sub('', result)
        return result.strip()
    
    def _extract_title_and_body(self, text: str) -> tuple[str, str]:
        """Extract title and body from text.
        
        Strategies:
        1. If first line is short (≤10 words), use it as title
        2. If text has no natural title, generate one from the first sentence
        """
        lines = text.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        if not lines:
            return "", ""
        
        first_line = lines[0]
        
        # Check if first line looks like a title (≤10 words, no period at end)
        first_line_words = first_line.split()
        if len(first_line_words) <= 10 and not first_line.endswith('.'):
            title = first_line
            body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        elif len(first_line_words) <= 10 and first_line.endswith(':'):
            # Title ending with colon
            title = first_line.rstrip(':')
            body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        else:
            # No clear title — extract from first few words
            title = self._generate_title(first_line)
            body = text
        
        return title, body
    
    def _generate_title(self, text: str) -> str:
        """Generate a short title from the first part of text.
        
        Takes up to 7 words from the start as a title.
        """
        words = text.split()
        title_words = words[:7]
        title = ' '.join(title_words)
        
        # Remove trailing punctuation from title
        title = title.rstrip('.,;:!?')
        
        return title
    
    def _validate_title(self, title: str, body: str) -> str:
        """Ensure title is ≤10 words. Truncate if needed."""
        if not title:
            # Generate from body
            return self._generate_title(body) if body else ""
        
        words = title.split()
        if len(words) > 10:
            return ' '.join(words[:10])
        return title
    
    def _remove_existing_citations(self, text: str) -> str:
        """Remove any existing citation/source lines from the text.
        
        Looks for lines starting with 'מקורות:' or 'Sources:' and removes them.
        """
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('מקורות:') or stripped.startswith('מקורות :'):
                continue
            if stripped.lower().startswith('sources:'):
                continue
            # Also remove lines that are just file names in brackets
            if re.match(r'^\[.*\]$', stripped):
                continue
            filtered.append(line)
        
        return '\n'.join(filtered).strip()
    
    def _clean_body(self, body: str) -> str:
        """Clean and consolidate the body into a single paragraph.
        
        - Remove extra whitespace and newlines
        - Merge into one paragraph
        """
        if not body:
            return ""
        
        # Remove preambles from body as well
        body = self._remove_preambles(body)
        
        # Replace multiple newlines with single space (merge into one paragraph)
        body = re.sub(r'\n+', ' ', body)
        
        # Collapse multiple spaces
        body = re.sub(r'\s+', ' ', body)
        
        return body.strip()
    
    def _build_formatted(self, title: str, body: str, sources: list[str]) -> str:
        """Build the final formatted response string.
        
        Format:
        {title}
        {body paragraph}
        מקורות: source1.txt, source2.pdf
        """
        parts = []
        
        if title:
            parts.append(title)
        
        if body:
            parts.append(body)
        
        # Build citations line (Req 9.2, 9.3)
        if sources:
            # Deduplicate while preserving order
            seen = set()
            unique_sources = []
            for s in sources:
                if s not in seen:
                    seen.add(s)
                    unique_sources.append(s)
            
            citations = f"{self.CITATIONS_PREFIX} {', '.join(unique_sources)}"
            parts.append(citations)
        
        return '\n'.join(parts)
    
    def is_well_formatted(self, response: str) -> bool:
        """Check if a response already conforms to the required structure.
        
        Returns True if:
        - Has a title line (≤10 words)
        - Has a body paragraph
        - Has a citations line starting with 'מקורות:'
        - Does not start with preamble phrases
        """
        lines = response.strip().split('\n')
        if len(lines) < 3:
            return False
        
        # Check title (≤10 words)
        title_words = lines[0].split()
        if len(title_words) > 10:
            return False
        
        # Check citations line
        last_line = lines[-1].strip()
        if not last_line.startswith(self.CITATIONS_PREFIX):
            return False
        
        # Check no preambles
        for pattern in _PREAMBLE_RE:
            if pattern.search(response):
                return False
        
        return True
