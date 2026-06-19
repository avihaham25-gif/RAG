"""Hebrew response validator for language compliance checking.

Validates that LLM responses are in Hebrew and conform to structural requirements.
"""

import re
from dataclasses import dataclass, field
from generation.formatter import ResponseFormatter
from config import RAGConfig


@dataclass
class ValidationResult:
    """Result of response validation.
    
    Attributes:
        is_valid: True if response passes all validation checks
        errors: List of error descriptions (empty if valid)
        has_language_violation: True if Hebrew language requirement is violated
        has_structure_violation: True if response format is wrong
    """
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    has_language_violation: bool = False
    has_structure_violation: bool = False


class ResponseValidator:
    """Validates Hebrew language compliance and response structure.
    
    Checks:
    1. Hebrew character ratio (at least 50% of alphabetic chars should be Hebrew)
    2. No sequences of >2 consecutive Latin-alphabet words (unless proper nouns)
    3. Response structure conforms to format requirements
    
    When validation fails, the caller should re-prompt the LLM with
    strengthened Hebrew-only instructions (Req 6.6).
    """
    
    # Hebrew character range: U+0590-U+05FF (includes letters, niqqud, etc.)
    HEBREW_LETTER_PATTERN = re.compile(r'[\u05D0-\u05EA]')  # Hebrew letters only
    
    # Latin letter pattern
    LATIN_LETTER_PATTERN = re.compile(r'[a-zA-Z]')
    
    # Pattern to find sequences of consecutive Latin words
    # A "Latin word" is a sequence of Latin letters (possibly with apostrophes/hyphens)
    LATIN_WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")
    
    # Common proper nouns and technical terms that are allowed in Latin
    # These are examples; in production this would be a larger list or ML-based
    ALLOWED_LATIN_TERMS = {
        # Technologies
        "python", "javascript", "typescript", "java", "c++", "c#",
        "html", "css", "sql", "api", "rest", "graphql",
        "docker", "kubernetes", "linux", "windows", "macos",
        "git", "github", "gitlab",
        # AI/ML terms commonly used
        "ai", "ml", "nlp", "llm", "gpt", "bert", "transformer",
        "rag", "bm25", "rrf", "tf-idf",
        # Companies/Products
        "google", "microsoft", "amazon", "openai", "meta",
        "chatgpt", "ollama", "chromadb", "langchain",
        "streamlit", "fastapi", "flask", "django",
        # Standards
        "iso", "ieee", "utf-8", "json", "xml", "yaml",
        "http", "https", "tcp", "ip",
    }
    
    # Minimum Hebrew character ratio (of all alphabetic characters)
    MIN_HEBREW_RATIO = 0.5  # At least 50% Hebrew
    
    # Maximum consecutive Latin words allowed
    MAX_CONSECUTIVE_LATIN = 2
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.config = config
        self._formatter = ResponseFormatter(config)
    
    def validate(self, response: str, sources: list[str] = None) -> ValidationResult:
        """Validate a response for Hebrew compliance and structure.
        
        Args:
            response: The LLM-generated response text.
            sources: Optional list of source files (for structure validation).
            
        Returns:
            ValidationResult with is_valid flag and error details.
        """
        if sources is None:
            sources = []
        
        errors = []
        has_language_violation = False
        has_structure_violation = False
        
        # Check 1: Hebrew character ratio
        ratio_ok, ratio_error = self._check_hebrew_ratio(response)
        if not ratio_ok:
            errors.append(ratio_error)
            has_language_violation = True
        
        # Check 2: Consecutive Latin words (Req 6.6)
        latin_ok, latin_error = self._check_consecutive_latin(response)
        if not latin_ok:
            errors.append(latin_error)
            has_language_violation = True
        
        # Check 3: Response structure (only if sources provided)
        if sources:
            # Format the response first, then check structure
            formatted = self._formatter.format(response, sources)
            structure_ok = self._formatter.is_well_formatted(formatted)
            if not structure_ok:
                errors.append("Response does not conform to required structure (title + body + citations)")
                has_structure_violation = True
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            has_language_violation=has_language_violation,
            has_structure_violation=has_structure_violation,
        )
    
    def validate_hebrew_only(self, response: str) -> ValidationResult:
        """Validate only Hebrew language compliance (skip structure check).
        
        Useful for checking raw LLM output before formatting.
        
        Args:
            response: The text to validate.
            
        Returns:
            ValidationResult focused on language checks only.
        """
        errors = []
        has_language_violation = False
        
        ratio_ok, ratio_error = self._check_hebrew_ratio(response)
        if not ratio_ok:
            errors.append(ratio_error)
            has_language_violation = True
        
        latin_ok, latin_error = self._check_consecutive_latin(response)
        if not latin_ok:
            errors.append(latin_error)
            has_language_violation = True
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            has_language_violation=has_language_violation,
            has_structure_violation=False,
        )
    
    def _check_hebrew_ratio(self, text: str) -> tuple[bool, str]:
        """Check that Hebrew characters make up at least MIN_HEBREW_RATIO of alphabetic chars.
        
        Returns:
            Tuple of (is_ok, error_message). error_message is empty if ok.
        """
        if not text or not text.strip():
            return True, ""  # Empty text passes (nothing to check)
        
        hebrew_count = len(self.HEBREW_LETTER_PATTERN.findall(text))
        latin_count = len(self.LATIN_LETTER_PATTERN.findall(text))
        
        total_alpha = hebrew_count + latin_count
        if total_alpha == 0:
            return True, ""  # No alphabetic characters (numbers/punctuation only)
        
        hebrew_ratio = hebrew_count / total_alpha
        
        if hebrew_ratio < self.MIN_HEBREW_RATIO:
            return False, (
                f"Hebrew character ratio too low: {hebrew_ratio:.1%} "
                f"(minimum: {self.MIN_HEBREW_RATIO:.0%}). "
                f"Response appears to be primarily in a non-Hebrew language."
            )
        
        return True, ""
    
    def _check_consecutive_latin(self, text: str) -> tuple[bool, str]:
        """Check for sequences of >2 consecutive Latin words (Req 6.6).
        
        Allows known proper nouns and technical terms.
        
        Returns:
            Tuple of (is_ok, error_message). error_message is empty if ok.
        """
        if not text:
            return True, ""
        
        # Find all Latin words in the text
        words = self.LATIN_WORD_PATTERN.findall(text)
        
        if not words:
            return True, ""
        
        # Walk through the text and find consecutive Latin word sequences
        # We need position-aware checking, so use finditer
        matches = list(self.LATIN_WORD_PATTERN.finditer(text))
        
        consecutive_count = 0
        consecutive_words = []
        prev_end = -1
        
        for match in matches:
            word = match.group()
            start = match.start()
            
            # Check if this word immediately follows the previous one
            # (only whitespace/punctuation between them, no Hebrew)
            if prev_end >= 0:
                between = text[prev_end:start]
                # If there's Hebrew text between, reset counter
                if self.HEBREW_LETTER_PATTERN.search(between):
                    consecutive_count = 1
                    consecutive_words = [word]
                else:
                    consecutive_count += 1
                    consecutive_words.append(word)
            else:
                consecutive_count = 1
                consecutive_words = [word]
            
            prev_end = match.end()
            
            # Check if we've exceeded the limit with non-allowed terms
            if consecutive_count > self.MAX_CONSECUTIVE_LATIN:
                # Check if all consecutive words are allowed terms
                non_allowed = [
                    w for w in consecutive_words
                    if w.lower() not in self.ALLOWED_LATIN_TERMS
                ]
                if len(non_allowed) > self.MAX_CONSECUTIVE_LATIN:
                    return False, (
                        f"Found {consecutive_count} consecutive Latin words: "
                        f"'{' '.join(consecutive_words)}'. "
                        f"Maximum allowed is {self.MAX_CONSECUTIVE_LATIN} "
                        f"(excluding proper nouns and technical terms)."
                    )
        
        return True, ""
