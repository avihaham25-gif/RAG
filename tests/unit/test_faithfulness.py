"""Unit tests for the Faithfulness Scorer."""

import pytest
from generation.faithfulness import FaithfulnessScorer, FaithfulnessResult
from models import Chunk, ScoredChunk
from config import RAGConfig


def _make_context_chunks(texts: list[str]) -> list[ScoredChunk]:
    """Helper to create context ScoredChunks."""
    return [
        ScoredChunk(
            chunk=Chunk(
                content=text,
                metadata={"source": f"doc{i}.txt", "chunk_index": i, "section_title": ""},
            ),
            score=0.9,
            source_method="reranker",
        )
        for i, text in enumerate(texts)
    ]


class TestFaithfulnessScoreComputation:
    """Test faithfulness score calculation (Req 7.4)."""
    
    def test_fully_supported_response_score_1(self):
        """Response fully grounded in context should get score 1.0."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks([
            "בינה מלאכותית היא תחום מחקר במדעי המחשב.",
            "למידה עמוקה משתמשת ברשתות נוירונים מרובות שכבות.",
        ])
        # Response uses same words as context
        response = "בינה מלאכותית היא תחום מחקר במדעי המחשב. למידה עמוקה משתמשת ברשתות נוירונים."
        
        result = scorer.score(response, context)
        
        assert result.score >= 0.8  # High faithfulness
        assert result.low_confidence_warning is False
    
    def test_hallucinated_response_low_score(self):
        """Response with fabricated information should get low score."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks([
            "פייתון היא שפת תכנות פופולרית.",
            "פייתון נוצרה בשנת 1991 על ידי גואידו ואן רוסום.",
        ])
        # Response contains claims not in context
        response = "פייתון היא השפה המהירה ביותר בעולם. היא משמשת לפיתוח משחקי מחשב תלת-ממדיים בלבד."
        
        result = scorer.score(response, context)
        
        assert result.score < 0.7  # Low faithfulness
        assert result.low_confidence_warning is True
    
    def test_partially_supported_response(self):
        """Mix of supported and unsupported claims."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks([
            "בינה מלאכותית היא תחום מחקר חשוב.",
        ])
        # First claim is supported, second is fabricated
        response = "בינה מלאכותית היא תחום מחקר חשוב. היא הומצאה בשנת 1800 בצרפת."
        
        result = scorer.score(response, context)
        
        assert 0.0 < result.score < 1.0
        assert result.total_claims == 2
        assert result.supported_claims >= 1
    
    def test_score_is_ratio_of_supported_to_total(self):
        """Score should be M/N (supported/total claims)."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks(["תוכן מסמך כאן."])
        response = "תוכן מסמך כאן. טענה שנייה שלא קיימת. טענה שלישית ממוצאת."
        
        result = scorer.score(response, context)
        
        # Verify score = supported / total
        if result.total_claims > 0:
            expected = result.supported_claims / result.total_claims
            assert abs(result.score - round(expected, 2)) < 0.01
    
    def test_no_claims_returns_score_1(self):
        """Empty response (no claims) should return score 1.0 (Req: N=0 -> 1.0)."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks(["some context"])
        
        result = scorer.score("", context)
        
        assert result.score == 1.0
        assert result.total_claims == 0
        assert result.low_confidence_warning is False
    
    def test_score_rounded_to_2_decimals(self):
        """Score should be rounded to 2 decimal places."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks(["context text content here"])
        response = "claim one here. claim two here. claim three different."
        
        result = scorer.score(response, context)
        
        # Check that score has at most 2 decimal places
        assert result.score == round(result.score, 2)


class TestFaithfulnessLowConfidenceWarning:
    """Test low-confidence warning (Req 7.5)."""
    
    def test_score_above_threshold_no_warning(self):
        """Score >= 0.7 should NOT trigger warning."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks([
            "בינה מלאכותית היא תחום מחקר חשוב במדעי המחשב שעוסק ביצירת מערכות חכמות.",
        ])
        response = "בינה מלאכותית היא תחום מחקר חשוב במדעי המחשב."
        
        result = scorer.score(response, context)
        
        if result.score >= 0.7:
            assert result.low_confidence_warning is False
    
    def test_score_below_threshold_triggers_warning(self):
        """Score < 0.7 should trigger warning."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks(["פייתון היא שפת תכנות."])
        # Mostly fabricated response
        response = "פייתון נוצרה בירח. היא רצה רק על מחשבי קוונטום. רק חייזרים משתמשים בה."
        
        result = scorer.score(response, context)
        
        assert result.score < 0.7
        assert result.low_confidence_warning is True
    
    def test_custom_threshold(self):
        """Should respect custom faithfulness threshold from config."""
        config = RAGConfig(faithfulness_threshold=0.9)
        scorer = FaithfulnessScorer(config)
        context = _make_context_chunks(["תוכן כלשהו כאן בהקשר."])
        response = "תוכן כלשהו כאן בהקשר. וגם משהו נוסף שלא."
        
        result = scorer.score(response, context)
        
        # With high threshold (0.9), partial support may trigger warning
        if result.score < 0.9:
            assert result.low_confidence_warning is True


class TestFaithfulnessClaimExtraction:
    """Test claim decomposition."""
    
    def test_single_sentence_one_claim(self):
        """Single sentence should produce one claim."""
        scorer = FaithfulnessScorer()
        claims = scorer._extract_claims("בינה מלאכותית היא תחום מחקר חשוב.")
        
        assert len(claims) >= 1
    
    def test_multiple_sentences_multiple_claims(self):
        """Multiple sentences should produce multiple claims."""
        scorer = FaithfulnessScorer()
        claims = scorer._extract_claims(
            "טענה ראשונה בנושא חשוב. טענה שנייה על נושא אחר. טענה שלישית נוספת כאן."
        )
        
        assert len(claims) >= 2
    
    def test_citations_line_excluded(self):
        """Citations line should not be counted as a claim."""
        scorer = FaithfulnessScorer()
        claims = scorer._extract_claims("תוכן התשובה כאן בפסקה.\nמקורות: doc1.txt, doc2.pdf")
        
        # The citations line should be excluded
        assert not any("מקורות:" in c for c in claims)
    
    def test_empty_response_no_claims(self):
        """Empty response should produce no claims."""
        scorer = FaithfulnessScorer()
        claims = scorer._extract_claims("")
        
        assert claims == []
    
    def test_short_fragments_excluded(self):
        """Very short text fragments (< 3 words) should be excluded."""
        scorer = FaithfulnessScorer()
        claims = scorer._extract_claims("כן. לא. אולי. זוהי טענה ארוכה יותר שצריכה להיכלל.")
        
        # Short fragments should be excluded, long one included
        assert any("טענה ארוכה" in c for c in claims)


class TestFaithfulnessClaimVerification:
    """Test claim verification against context."""
    
    def test_exact_match_supported(self):
        """Claim that appears verbatim in context should be supported."""
        scorer = FaithfulnessScorer()
        context_text = "בינה מלאכותית היא תחום מחקר במדעי המחשב"
        
        result = scorer._verify_claim(
            "בינה מלאכותית היא תחום מחקר במדעי המחשב.",
            context_text,
        )
        
        assert result is True
    
    def test_fabricated_claim_not_supported(self):
        """Claim with words not in context should not be supported."""
        scorer = FaithfulnessScorer()
        context_text = "פייתון היא שפת תכנות פופולרית"
        
        result = scorer._verify_claim(
            "חייזרים יצרו את השפה בכוכב מאדים בשנת אלף.",
            context_text,
        )
        
        assert result is False
    
    def test_partial_overlap_threshold(self):
        """Claim with partial overlap depends on threshold."""
        scorer = FaithfulnessScorer()
        context_text = "בינה מלאכותית משתמשת ברשתות נוירונים ללמידה עמוקה"
        
        # Has some overlap (בינה, מלאכותית, רשתות, נוירונים) but also new words
        result = scorer._verify_claim(
            "בינה מלאכותית ורשתות נוירונים הם כלים חזקים.",
            context_text,
        )
        
        # Should pass because enough tokens overlap
        assert result is True


class TestFaithfulnessResultDataclass:
    """Test FaithfulnessResult dataclass."""
    
    def test_default_values(self):
        """Should have proper defaults."""
        result = FaithfulnessResult(
            score=0.85,
            total_claims=5,
            supported_claims=4,
        )
        assert result.score == 0.85
        assert result.total_claims == 5
        assert result.supported_claims == 4
        assert result.unsupported_claims == []
        assert result.low_confidence_warning is False
    
    def test_with_unsupported_claims(self):
        """Should store unsupported claim texts."""
        result = FaithfulnessResult(
            score=0.5,
            total_claims=4,
            supported_claims=2,
            unsupported_claims=["fake claim 1", "fake claim 2"],
            low_confidence_warning=True,
        )
        assert len(result.unsupported_claims) == 2
        assert result.low_confidence_warning is True


class TestFaithfulnessEdgeCases:
    """Test edge cases."""
    
    def test_empty_context(self):
        """Empty context should result in low/zero score for any claims."""
        scorer = FaithfulnessScorer()
        context = []
        response = "טענה כלשהי שאין לה מקור בהקשר."
        
        result = scorer.score(response, context)
        
        # With no context, claims can't be supported
        assert result.score <= 0.5
    
    def test_very_long_response(self):
        """Long response with many claims should be handled."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks([
            "בינה מלאכותית היא תחום חשוב. למידה עמוקה היא שיטה מרכזית. "
            "רשתות נוירונים מבצעות חישובים מורכבים. פייתון משמשת לפיתוח."
        ])
        response = (
            "בינה מלאכותית היא תחום חשוב. "
            "למידה עמוקה היא שיטה מרכזית. "
            "רשתות נוירונים מבצעות חישובים מורכבים. "
            "פייתון משמשת לפיתוח."
        )
        
        result = scorer.score(response, context)
        
        assert result.score >= 0.7
        assert result.total_claims >= 3
    
    def test_response_with_only_citations(self):
        """Response that's only a citations line should have no claims."""
        scorer = FaithfulnessScorer()
        context = _make_context_chunks(["context"])
        
        result = scorer.score("מקורות: doc.txt", context)
        
        assert result.total_claims == 0
        assert result.score == 1.0
