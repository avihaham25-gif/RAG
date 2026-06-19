"""Unit tests for the Generation Layer: ConfidenceScorer and HebrewPromptBuilder."""

import pytest
from models import Chunk, ScoredChunk
from config import RAGConfig
from generation.scorer import (
    ConfidenceScorer,
    ConfidenceResult,
    INSUFFICIENT_CONTEXT_MESSAGE,
    REPHRASE_SUGGESTION,
)
from generation.prompt_builder import HebrewPromptBuilder


def _make_scored_chunks(scores: list[float], contents: list[str] = None) -> list[ScoredChunk]:
    """Helper to create ScoredChunks with given scores."""
    if contents is None:
        contents = [f"chunk content {i}" for i in range(len(scores))]
    return [
        ScoredChunk(
            chunk=Chunk(
                content=content,
                metadata={"source": f"/docs/doc{i}.txt", "chunk_index": i, "section_title": ""},
            ),
            score=score,
            source_method="reranker",
        )
        for i, (score, content) in enumerate(zip(scores, contents))
    ]


class TestConfidenceScorer:
    """Test confidence score computation."""

    def test_compute_mean_of_scores(self):
        """Req 8.5: Confidence = arithmetic mean of relevance scores."""
        scorer = ConfidenceScorer()
        chunks = _make_scored_chunks([0.8, 0.6, 0.4])

        result = scorer.assess(chunks)

        # Mean of 0.8, 0.6, 0.4 = 0.6
        assert result.score == 0.6
        assert result.is_insufficient is False

    def test_score_rounded_to_two_decimals(self):
        """Req 8.6: Score rounded to 2 decimal places."""
        scorer = ConfidenceScorer()
        chunks = _make_scored_chunks([0.333, 0.667, 0.5])

        result = scorer.assess(chunks)

        # Mean = (0.333 + 0.667 + 0.5) / 3 = 0.5
        assert result.score == round((0.333 + 0.667 + 0.5) / 3, 2)

    def test_empty_chunks_insufficient(self):
        """Req 8.3: Zero chunks -> insufficient context."""
        scorer = ConfidenceScorer()
        result = scorer.assess([])

        assert result.is_insufficient is True
        assert result.score == 0.0
        assert result.message is not None

    def test_max_score_below_threshold_insufficient(self):
        """Req 8.1: Max score below threshold -> insufficient."""
        config = RAGConfig(confidence_threshold=0.3)
        scorer = ConfidenceScorer(config)

        chunks = _make_scored_chunks([0.1, 0.2, 0.15])  # All below 0.3
        result = scorer.assess(chunks)

        assert result.is_insufficient is True

    def test_max_score_at_threshold_sufficient(self):
        """Score exactly at threshold should be sufficient."""
        config = RAGConfig(confidence_threshold=0.3)
        scorer = ConfidenceScorer(config)

        chunks = _make_scored_chunks([0.3, 0.1, 0.1])  # Max = 0.3, not < 0.3
        result = scorer.assess(chunks)

        assert result.is_insufficient is False

    def test_max_score_above_threshold_sufficient(self):
        """Score above threshold -> sufficient context."""
        config = RAGConfig(confidence_threshold=0.3)
        scorer = ConfidenceScorer(config)

        chunks = _make_scored_chunks([0.9, 0.7, 0.5])
        result = scorer.assess(chunks)

        assert result.is_insufficient is False
        assert result.message is None

    def test_insufficient_message_contains_hebrew(self):
        """Req 8.2, 8.4: Message should contain Hebrew not-found text + suggestion."""
        scorer = ConfidenceScorer()
        result = scorer.assess([])

        assert INSUFFICIENT_CONTEXT_MESSAGE in result.message
        assert REPHRASE_SUGGESTION in result.message

    def test_single_chunk_above_threshold(self):
        """Single chunk with score above threshold -> sufficient."""
        scorer = ConfidenceScorer()
        chunks = _make_scored_chunks([0.85])

        result = scorer.assess(chunks)

        assert result.is_insufficient is False
        assert result.score == 0.85

    def test_custom_threshold(self):
        """Should respect custom threshold from config."""
        config = RAGConfig(confidence_threshold=0.7)
        scorer = ConfidenceScorer(config)

        # Max = 0.6, below threshold of 0.7
        chunks = _make_scored_chunks([0.6, 0.5, 0.4])
        result = scorer.assess(chunks)

        assert result.is_insufficient is True

    def test_confidence_result_dataclass(self):
        """ConfidenceResult should have expected fields."""
        result = ConfidenceResult(score=0.75, is_insufficient=False)
        assert result.score == 0.75
        assert result.is_insufficient is False
        assert result.message is None


class TestHebrewPromptBuilder:
    """Test Hebrew prompt construction."""

    def test_system_prompt_contains_hebrew_enforcement(self):
        """Req 6.5: System prompt should enforce Hebrew."""
        builder = HebrewPromptBuilder()
        system = builder.build_system_prompt()

        assert "עברית" in system
        assert "בעברית" in system or "בעברית בלבד" in system

    def test_system_prompt_contains_role(self):
        """System prompt should define the assistant's role."""
        builder = HebrewPromptBuilder()
        system = builder.build_system_prompt()

        assert "עוזר" in system or "תפקידך" in system

    def test_system_prompt_forbids_hallucination(self):
        """Req 7.1: System prompt should forbid making up information."""
        builder = HebrewPromptBuilder()
        system = builder.build_system_prompt()

        assert "תמציא" in system or "אינו מופיע" in system

    def test_system_prompt_requires_citations(self):
        """Req 7.2: Should require citing source documents."""
        builder = HebrewPromptBuilder()
        system = builder.build_system_prompt()

        assert "מסמך" in system or "ציין" in system

    def test_system_prompt_handles_technical_terms(self):
        """Req 6.2: Should mention transliteration of technical terms."""
        builder = HebrewPromptBuilder()
        system = builder.build_system_prompt()

        assert "תרגם" in system or "תעתיק" in system

    def test_user_prompt_contains_query(self):
        """User prompt should include the user's question."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["some context text"])

        prompt = builder.build_user_prompt("מה זה בינה מלאכותית?", chunks)

        assert "מה זה בינה מלאכותית?" in prompt

    def test_user_prompt_contains_context(self):
        """User prompt should include chunk content."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["זהו תוכן ממסמך מקורי"])

        prompt = builder.build_user_prompt("שאלה?", chunks)

        assert "זהו תוכן ממסמך מקורי" in prompt

    def test_user_prompt_contains_source_attribution(self):
        """Context should include source file names for citation."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["content here"])

        prompt = builder.build_user_prompt("query?", chunks)

        # Should contain the filename
        assert "doc0.txt" in prompt

    def test_user_prompt_has_hebrew_enforcement(self):
        """Req 6.5: User prompt also enforces Hebrew (dual-level)."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["content"])

        prompt = builder.build_user_prompt("query?", chunks)

        assert "בעברית" in prompt

    def test_user_prompt_forbids_preambles(self):
        """Req 9.4: Should instruct not to use preambles."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["content"])

        prompt = builder.build_user_prompt("query?", chunks)

        assert "על פי המסמכים" in prompt or "אל תפתח" in prompt

    def test_user_prompt_requires_structure(self):
        """Should instruct response format: title + paragraph + citations."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["content"])

        prompt = builder.build_user_prompt("query?", chunks)

        assert "כותרת" in prompt
        assert "פסקה" in prompt
        assert "מקורות" in prompt

    def test_build_messages_format(self):
        """build_messages should return proper chat format."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9], ["content"])

        messages = builder.build_messages("query?", chunks)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "human"
        assert len(messages[0]["content"]) > 0
        assert len(messages[1]["content"]) > 0

    def test_multiple_chunks_all_included(self):
        """All chunks should appear in the context."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks(
            [0.9, 0.8, 0.7],
            ["chunk one content", "chunk two content", "chunk three content"]
        )

        prompt = builder.build_user_prompt("query?", chunks)

        assert "chunk one content" in prompt
        assert "chunk two content" in prompt
        assert "chunk three content" in prompt

    def test_empty_chunks_produces_empty_context(self):
        """No chunks should produce empty context section."""
        builder = HebrewPromptBuilder()
        prompt = builder.build_user_prompt("query?", [])

        # Should still contain the query and instructions
        assert "query?" in prompt

    def test_context_format_labels_each_source(self):
        """Each chunk in context should be labeled with its source."""
        builder = HebrewPromptBuilder()
        chunks = _make_scored_chunks([0.9, 0.8], ["content A", "content B"])
        # chunks have sources: doc0.txt, doc1.txt

        prompt = builder.build_user_prompt("query?", chunks)

        assert "מסמך" in prompt  # Hebrew label for document
        assert "doc0.txt" in prompt
        assert "doc1.txt" in prompt
