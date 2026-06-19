"""Unit tests for the Hebrew Response Validator."""

import pytest
from generation.validator import ResponseValidator, ValidationResult


class TestResponseValidatorHebrewRatio:
    """Test Hebrew character ratio checking."""
    
    def test_pure_hebrew_passes(self):
        """100% Hebrew text should pass."""
        validator = ResponseValidator()
        result = validator.validate_hebrew_only("שלום עולם, זוהי תשובה בעברית בלבד.")
        
        assert result.is_valid is True
        assert result.has_language_violation is False
    
    def test_mostly_hebrew_passes(self):
        """Hebrew with some English terms should pass if ratio > 50%."""
        validator = ResponseValidator()
        text = "בינה מלאכותית (AI) היא תחום מחקר חשוב מאוד במדעי המחשב."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
    
    def test_pure_english_fails(self):
        """100% English text should fail Hebrew ratio."""
        validator = ResponseValidator()
        result = validator.validate_hebrew_only(
            "This is a completely English response with no Hebrew at all."
        )
        
        assert result.is_valid is False
        assert result.has_language_violation is True
        assert len(result.errors) > 0
    
    def test_mostly_english_fails(self):
        """Mostly English text (>50% Latin) should fail."""
        validator = ResponseValidator()
        text = "The answer is that this technology works well. מעט עברית."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is False
        assert result.has_language_violation is True
    
    def test_empty_text_passes(self):
        """Empty text should pass (nothing to validate)."""
        validator = ResponseValidator()
        result = validator.validate_hebrew_only("")
        
        assert result.is_valid is True
    
    def test_numbers_only_passes(self):
        """Text with only numbers/punctuation should pass."""
        validator = ResponseValidator()
        result = validator.validate_hebrew_only("123, 456! 789?")
        
        assert result.is_valid is True
    
    def test_50_percent_hebrew_passes(self):
        """Exactly 50% Hebrew should pass (>= threshold)."""
        validator = ResponseValidator()
        # 4 Hebrew letters + 4 Latin letters = 50%
        text = "אבגד abcd"
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True


class TestResponseValidatorConsecutiveLatin:
    """Test consecutive Latin word detection (Req 6.6)."""
    
    def test_no_latin_passes(self):
        """Pure Hebrew should pass."""
        validator = ResponseValidator()
        result = validator.validate_hebrew_only("זוהי תשובה בעברית בלבד ללא מילים באנגלית.")
        
        assert result.is_valid is True
    
    def test_single_latin_word_passes(self):
        """One Latin word is fine."""
        validator = ResponseValidator()
        text = "שפת Python היא שפה פופולרית."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
    
    def test_two_consecutive_latin_passes(self):
        """Two consecutive Latin words is allowed (<=2)."""
        validator = ResponseValidator()
        text = "טכנולוגיית Machine Learning היא חשובה מאוד בתחום."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
    
    def test_three_consecutive_latin_fails(self):
        """Three+ consecutive non-allowed Latin words should fail."""
        validator = ResponseValidator()
        text = "הנושא הוא very important topic שצריך לדון בו."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is False
        assert result.has_language_violation is True
        assert "consecutive Latin words" in result.errors[0]
    
    def test_allowed_technical_terms_pass(self):
        """Known technical terms don't count against the limit."""
        validator = ResponseValidator()
        # "Python API REST" - all are allowed terms
        text = "שימוש ב-Python API REST הוא נפוץ בפיתוח."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
    
    def test_mixed_allowed_and_not_allowed(self):
        """Mix of allowed and non-allowed consecutive words."""
        validator = ResponseValidator()
        # "Python is great" - Python is allowed, "is great" are not
        # Added enough Hebrew to pass the ratio check (this test focuses on consecutive Latin logic)
        text = "שפת התכנות הפופולרית Python is great והיא נמצאת בשימוש נרחב בתעשייה."
        result = validator.validate_hebrew_only(text)
        
        # "Python is great" = 3 consecutive, but Python is allowed
        # So effectively "is great" after Python = 2 non-allowed -> passes
        assert result.is_valid is True
    
    def test_long_english_sentence_fails(self):
        """A full English sentence embedded in Hebrew should fail."""
        validator = ResponseValidator()
        text = "ההקדמה: The system uses advanced algorithms for processing data. וזה מעניין."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is False
        assert result.has_language_violation is True


class TestResponseValidatorFullValidation:
    """Test full validation including structure check."""
    
    def test_valid_response_passes(self):
        """Well-formed Hebrew response with sources should pass."""
        validator = ResponseValidator()
        response = "כותרת קצרה\nזוהי תשובה בעברית על הנושא המבוקש.\nמקורות: doc.txt"
        result = validator.validate(response, sources=["doc.txt"])
        
        # Note: structure check uses formatter.is_well_formatted on the formatted result
        assert result.has_language_violation is False
    
    def test_english_response_fails_language(self):
        """English response should fail language check."""
        validator = ResponseValidator()
        response = "This is a title\nThis is an English paragraph about the topic.\nSources: doc.txt"
        result = validator.validate(response, sources=["doc.txt"])
        
        assert result.is_valid is False
        assert result.has_language_violation is True
    
    def test_no_sources_skips_structure(self):
        """Without sources, structure check is skipped."""
        validator = ResponseValidator()
        response = "תשובה בעברית ללא בדיקת מבנה."
        result = validator.validate(response)
        
        # Only language checks apply
        assert result.has_structure_violation is False


class TestValidationResultDataclass:
    """Test ValidationResult dataclass."""
    
    def test_default_values(self):
        """Default ValidationResult should have empty errors."""
        result = ValidationResult(is_valid=True)
        assert result.errors == []
        assert result.has_language_violation is False
        assert result.has_structure_violation is False
    
    def test_with_errors(self):
        """Should store error messages."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            has_language_violation=True,
        )
        assert len(result.errors) == 2
        assert result.has_language_violation is True


class TestResponseValidatorEdgeCases:
    """Test edge cases."""
    
    def test_hebrew_with_numbers_passes(self):
        """Hebrew text with numbers should pass."""
        validator = ResponseValidator()
        text = "בשנת 2024 התפתחו 15 מודלים חדשים."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
    
    def test_hebrew_with_punctuation_passes(self):
        """Hebrew with various punctuation should pass."""
        validator = ResponseValidator()
        text = 'שאלה: "מה זה?" - תשובה (חשובה!) [ראה מסמך].'
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
    
    def test_single_character_response(self):
        """Very short responses should be handled."""
        validator = ResponseValidator()
        result = validator.validate_hebrew_only("כ")
        assert result.is_valid is True
    
    def test_mixed_scripts_within_threshold(self):
        """Hebrew-dominant mixed text should pass."""
        validator = ResponseValidator()
        text = "בינה מלאכותית היא תחום של AI שמשתמש ב-ML כדי ליצור מודלים חכמים."
        result = validator.validate_hebrew_only(text)
        
        assert result.is_valid is True
