"""Confidence scoring for retrieval quality assessment."""

from dataclasses import dataclass
from models import ScoredChunk
from config import RAGConfig


# Predefined Hebrew messages for insufficient context
INSUFFICIENT_CONTEXT_MESSAGE = "המידע אינו קיים במסמכים."
REPHRASE_SUGGESTION = "ניתן לנסות לנסח מחדש את השאלה או להוסיף מסמכים רלוונטיים למאגר."


@dataclass
class ConfidenceResult:
    """Result of confidence assessment.

    Attributes:
        score: Confidence score (0.0-1.0), rounded to 2 decimal places
        is_insufficient: True if context is insufficient for answering
        message: Hebrew message if insufficient, None otherwise
    """
    score: float
    is_insufficient: bool
    message: str | None = None


class ConfidenceScorer:
    """Assesses retrieval confidence and detects insufficient context.

    Computes confidence as the arithmetic mean of reranker relevance scores.
    Classifies queries as insufficient when:
    - No chunks are returned (Req 8.3)
    - Maximum relevance score is below threshold (Req 8.1)
    """

    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.threshold = config.confidence_threshold

    def assess(self, chunks: list[ScoredChunk]) -> ConfidenceResult:
        """Assess confidence based on retrieved chunk scores.

        Args:
            chunks: List of ScoredChunk from reranker (with relevance scores).

        Returns:
            ConfidenceResult with score, insufficiency flag, and optional message.
        """
        # Req 8.3: Zero chunks -> insufficient
        if not chunks:
            return ConfidenceResult(
                score=0.0,
                is_insufficient=True,
                message=f"{INSUFFICIENT_CONTEXT_MESSAGE}\n{REPHRASE_SUGGESTION}",
            )

        # Extract scores
        scores = [chunk.score for chunk in chunks]

        # Req 8.5: Confidence = arithmetic mean of relevance scores
        confidence_score = round(sum(scores) / len(scores), 2)

        # Req 8.1: Max score below threshold -> insufficient
        max_score = max(scores)
        if max_score < self.threshold:
            return ConfidenceResult(
                score=confidence_score,
                is_insufficient=True,
                message=f"{INSUFFICIENT_CONTEXT_MESSAGE}\n{REPHRASE_SUGGESTION}",
            )

        # Sufficient context
        return ConfidenceResult(
            score=confidence_score,
            is_insufficient=False,
            message=None,
        )
