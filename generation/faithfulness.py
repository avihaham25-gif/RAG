"""Faithfulness scorer for grounding verification.

Checks if the generated response is supported by the retrieved context chunks.
Decomposes the response into individual claims and verifies each against context.
"""

import re
from dataclasses import dataclass, field
from models import Chunk, ScoredChunk
from config import RAGConfig


@dataclass
class FaithfulnessResult:
    """Result of faithfulness assessment.
    
    Attributes:
        score: Faithfulness score (0.0-1.0) = supported_claims / total_claims
        total_claims: Number of claims extracted from the response
        supported_claims: Number of claims supported by context
        unsupported_claims: List of claims not found in context
        low_confidence_warning: True if score < faithfulness_threshold (0.7)
    """
    score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: list[str] = field(default_factory=list)
    low_confidence_warning: bool = False


class FaithfulnessScorer:
    """Scores response faithfulness against retrieved context.
    
    Algorithm:
    1. Extract individual factual claims from the response
    2. For each claim, check if it's supported by at least one context chunk
    3. Compute score = supported_count / total_count
    4. Flag as low-confidence if score < threshold (default 0.7)
    
    Claim extraction uses sentence-level decomposition with heuristics
    to identify factual statements (as opposed to connectives, questions, etc.).
    
    Claim verification uses token overlap / keyword matching as a 
    deterministic approach (no LLM dependency for scoring itself).
    """
    
    # Sentence-ending punctuation for splitting
    SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?。])\s+')
    
    # Minimum tokens for a sentence to be considered a "claim"
    MIN_CLAIM_TOKENS = 3
    
    # Non-claim patterns (connectives, questions, etc.)
    NON_CLAIM_PATTERNS = [
        re.compile(r'^\s*$'),                      # Empty
        re.compile(r'^[?؟]\s*$'),                   # Just a question mark
        re.compile(r'^\s*מקורות\s*:'),              # Citations line
        re.compile(r'^\s*\['),                      # Bracket references
    ]
    
    # Minimum overlap ratio for a claim to be "supported"
    SUPPORT_THRESHOLD = 0.4  # 40% of claim tokens must appear in context
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.faithfulness_threshold = config.faithfulness_threshold
    
    def score(self, response: str, context_chunks: list[ScoredChunk]) -> FaithfulnessResult:
        """Compute faithfulness score for a response against context.
        
        Args:
            response: The generated response text.
            context_chunks: The chunks that were used as context for generation.
            
        Returns:
            FaithfulnessResult with score, claim counts, and warning flag.
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Handle edge case: no claims (Req 7.4: score = 1.0 when N=0)
        if not claims:
            return FaithfulnessResult(
                score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
                low_confidence_warning=False,
            )
        
        # Build combined context text for verification
        context_text = self._build_context_text(context_chunks)
        
        # Verify each claim
        supported_count = 0
        unsupported = []
        
        for claim in claims:
            if self._verify_claim(claim, context_text):
                supported_count += 1
            else:
                unsupported.append(claim)
        
        # Compute score (Req 7.4: M/N)
        total = len(claims)
        score = supported_count / total if total > 0 else 1.0
        
        # Round to 2 decimal places
        score = round(score, 2)
        
        # Check low-confidence threshold (Req 7.5)
        low_confidence = score < self.faithfulness_threshold
        
        return FaithfulnessResult(
            score=score,
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported,
            low_confidence_warning=low_confidence,
        )
    
    def _extract_claims(self, response: str) -> list[str]:
        """Decompose response into individual factual claims.
        
        Strategy:
        1. Split by sentence-ending punctuation
        2. Filter out non-claims (citations, empty lines, connectives)
        3. Keep sentences with at least MIN_CLAIM_TOKENS words
        
        Args:
            response: The response text to decompose.
            
        Returns:
            List of claim strings.
        """
        if not response or not response.strip():
            return []
        
        # Split into sentences
        sentences = self.SENTENCE_SPLIT_PATTERN.split(response.strip())
        
        # If no sentence boundaries found, try splitting by newlines
        if len(sentences) <= 1:
            sentences = response.strip().split('\n')
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip non-claims
            if self._is_non_claim(sentence):
                continue
            
            # Must have minimum tokens to be a claim
            tokens = sentence.split()
            if len(tokens) < self.MIN_CLAIM_TOKENS:
                continue
            
            claims.append(sentence)
        
        return claims
    
    def _is_non_claim(self, text: str) -> bool:
        """Check if a sentence is a non-claim (citation, question, etc.)."""
        for pattern in self.NON_CLAIM_PATTERNS:
            if pattern.search(text):
                return True
        return False
    
    def _build_context_text(self, chunks: list[ScoredChunk]) -> str:
        """Combine all context chunks into a single searchable text."""
        parts = []
        for scored_chunk in chunks:
            parts.append(scored_chunk.chunk.content.lower())
        return ' '.join(parts)
    
    def _verify_claim(self, claim: str, context_text: str) -> bool:
        """Verify if a claim is supported by the context.
        
        Uses token overlap approach:
        - Tokenize the claim into meaningful words
        - Check what proportion of claim tokens appear in the context
        - If overlap >= SUPPORT_THRESHOLD, consider it supported
        
        Args:
            claim: The claim to verify.
            context_text: Combined lowercase context text.
            
        Returns:
            True if the claim is supported by the context.
        """
        # Tokenize the claim (lowercase, remove punctuation)
        claim_tokens = self._tokenize(claim)
        
        if not claim_tokens:
            return True  # Empty claim is trivially supported
        
        # Count how many claim tokens appear in context
        supported_tokens = 0
        for token in claim_tokens:
            if token in context_text:
                supported_tokens += 1
        
        # Compute overlap ratio
        overlap_ratio = supported_tokens / len(claim_tokens)
        
        return overlap_ratio >= self.SUPPORT_THRESHOLD
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase meaningful words.
        
        Removes punctuation and short words (<=2 chars) that are
        likely stopwords or particles.
        """
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = cleaned.split()
        
        # Filter short tokens (likely stopwords)
        # Keep tokens with 3+ characters
        return [t for t in tokens if len(t) >= 3]
