"""Unit tests for the Structured Response Formatter."""

import pytest
from generation.formatter import ResponseFormatter


class TestResponseFormatterBasic:
    """Test basic formatting functionality."""
    
    def test_well_formatted_input_preserved(self):
        """Already well-formatted response should remain structured."""
        formatter = ResponseFormatter()
        raw = "כותרת קצרה\nזוהי תשובה מפורטת על הנושא המבוקש."
        sources = ["doc1.txt"]
        
        result = formatter.format(raw, sources)
        
        assert "כותרת קצרה" in result
        assert "זוהי תשובה מפורטת" in result
        assert "מקורות: doc1.txt" in result
    
    def test_empty_response(self):
        """Empty response should still produce citations."""
        formatter = ResponseFormatter()
        result = formatter.format("", ["doc.txt"])
        
        assert "מקורות: doc.txt" in result
    
    def test_whitespace_only_response(self):
        """Whitespace-only response should still produce citations."""
        formatter = ResponseFormatter()
        result = formatter.format("   \n\n   ", ["doc.txt"])
        
        assert "מקורות: doc.txt" in result
    
    def test_sources_appended(self):
        """Sources should appear on the last line."""
        formatter = ResponseFormatter()
        result = formatter.format("כותרת\nתוכן", ["file1.txt", "file2.pdf"])
        
        lines = result.strip().split('\n')
        assert lines[-1].startswith("מקורות:")
        assert "file1.txt" in lines[-1]
        assert "file2.pdf" in lines[-1]
    
    def test_multiple_sources_comma_separated(self):
        """Req 9.3: Multiple sources separated by commas."""
        formatter = ResponseFormatter()
        result = formatter.format("כותרת\nתוכן", ["a.txt", "b.pdf", "c.docx"])
        
        lines = result.strip().split('\n')
        last_line = lines[-1]
        assert "a.txt, b.pdf, c.docx" in last_line
    
    def test_duplicate_sources_deduplicated(self):
        """Duplicate source names should be removed."""
        formatter = ResponseFormatter()
        result = formatter.format("כותרת\nתוכן", ["doc.txt", "doc.txt", "other.pdf"])
        
        lines = result.strip().split('\n')
        last_line = lines[-1]
        # Should only appear once
        assert last_line.count("doc.txt") == 1


class TestResponseFormatterPreambleRemoval:
    """Test removal of introductory preambles (Req 9.4)."""
    
    def test_remove_hebrew_preamble_al_pi(self):
        """Should remove 'על פי המסמכים'."""
        formatter = ResponseFormatter()
        raw = "על פי המסמכים, הנושא מרכזי."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "על פי המסמכים" not in result
        assert "הנושא מרכזי" in result
    
    def test_remove_hebrew_preamble_behitbasesut(self):
        """Should remove 'בהתבסס על המסמכים'."""
        formatter = ResponseFormatter()
        raw = "בהתבסס על המסמכים, התשובה היא כדלקמן."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "בהתבסס על המסמכים" not in result
    
    def test_remove_hebrew_preamble_lehallan(self):
        """Should remove 'להלן התשובה'."""
        formatter = ResponseFormatter()
        raw = "להלן התשובה: הנושא חשוב מאוד."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "להלן התשובה" not in result
        assert "הנושא חשוב" in result
    
    def test_remove_english_preamble_based_on(self):
        """Should remove 'Based on the documents'."""
        formatter = ResponseFormatter()
        raw = "Based on the documents, the answer is clear."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "Based on the documents" not in result
    
    def test_remove_english_preamble_according_to(self):
        """Should remove 'According to the sources'."""
        formatter = ResponseFormatter()
        raw = "According to the sources, this is important."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "According to the sources" not in result
    
    def test_remove_preamble_with_colon(self):
        """Should handle preambles followed by colon."""
        formatter = ResponseFormatter()
        raw = "על פי המסמכים: זוהי התשובה החשובה."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "על פי המסמכים" not in result
        assert "התשובה החשובה" in result
    
    def test_no_false_positive_removal(self):
        """Should not remove text that isn't a preamble."""
        formatter = ResponseFormatter()
        raw = "כותרת\nהמסמכים שלנו עוסקים בנושא חשוב."
        result = formatter.format(raw, ["doc.txt"])
        
        # This is content, not a preamble (doesn't start with the pattern)
        assert "המסמכים שלנו" in result


class TestResponseFormatterTitleValidation:
    """Test title extraction and validation (Req 9.1)."""
    
    def test_title_extracted_from_first_line(self):
        """Short first line should become the title."""
        formatter = ResponseFormatter()
        raw = "בינה מלאכותית\nזהו תחום מחקר מרכזי במדעי המחשב שעוסק ביצירת מערכות חכמות."
        result = formatter.format(raw, ["doc.txt"])
        
        lines = result.strip().split('\n')
        assert lines[0] == "בינה מלאכותית"
    
    def test_title_max_10_words(self):
        """Title should be at most 10 words."""
        formatter = ResponseFormatter()
        raw = "מילה אחת שתיים שלוש ארבע חמש שש שבע שמונה\nתוכן."
        result = formatter.format(raw, ["doc.txt"])
        
        lines = result.strip().split('\n')
        title_words = lines[0].split()
        assert len(title_words) <= 10
    
    def test_long_title_truncated(self):
        """Title longer than 10 words should be truncated."""
        formatter = ResponseFormatter()
        long_title = "מילה " * 15  # 15 words
        raw = f"{long_title.strip()}\nתוכן הפסקה כאן."
        result = formatter.format(raw, ["doc.txt"])
        
        lines = result.strip().split('\n')
        title_words = lines[0].split()
        assert len(title_words) <= 10
    
    def test_no_title_auto_generated(self):
        """If no natural title, one should be generated from content."""
        formatter = ResponseFormatter()
        raw = "זהו טקסט ארוך מאוד שמהווה פסקה שלמה ולא כותרת כי הוא ארוך מדי."
        result = formatter.format(raw, ["doc.txt"])
        
        lines = result.strip().split('\n')
        # First line should be a generated title (≤10 words)
        title_words = lines[0].split()
        assert len(title_words) <= 10


class TestResponseFormatterBodyCleaning:
    """Test body paragraph cleaning."""
    
    def test_body_merged_into_single_paragraph(self):
        """Multiple paragraphs should be merged into one."""
        formatter = ResponseFormatter()
        raw = "כותרת\nפסקה ראשונה.\n\nפסקה שנייה.\n\nפסקה שלישית."
        result = formatter.format(raw, ["doc.txt"])
        
        lines = result.strip().split('\n')
        # Should have: title, body (1 line), citations
        # Body should be merged (no double newlines in middle)
        body_lines = [l for l in lines[1:-1] if l.strip()]
        assert len(body_lines) == 1  # single paragraph
        assert "פסקה ראשונה" in body_lines[0]
        assert "פסקה שנייה" in body_lines[0]
    
    def test_existing_citations_removed_from_body(self):
        """LLM-generated citation lines in body should be replaced by our own."""
        formatter = ResponseFormatter()
        raw = "כותרת\nתוכן התשובה.\nמקורות: old_source.txt"
        result = formatter.format(raw, ["new_source.pdf"])
        
        # Should use our sources, not LLM's
        assert "new_source.pdf" in result
        assert "old_source.txt" not in result
    
    def test_extra_whitespace_collapsed(self):
        """Multiple spaces and newlines should be collapsed."""
        formatter = ResponseFormatter()
        raw = "כותרת\nתוכן   עם    רווחים     מרובים."
        result = formatter.format(raw, ["doc.txt"])
        
        assert "  " not in result.split('\n')[1]  # No double spaces in body


class TestResponseFormatterAutoReformat:
    """Test auto-reformatting of non-conforming responses (Req 9.5)."""
    
    def test_reformat_missing_title(self):
        """Response without title gets one auto-generated."""
        formatter = ResponseFormatter()
        raw = "זוהי תשובה ארוכה ומפורטת שלא כוללת כותרת נפרדת אלא רק פסקה אחת גדולה שצריכה לעבור עיבוד."
        result = formatter.format(raw, ["doc.txt"])
        
        lines = result.strip().split('\n')
        # Should have at least title + citations
        assert len(lines) >= 2
        # Title should be short
        title_words = lines[0].split()
        assert len(title_words) <= 10
    
    def test_reformat_with_preamble_and_no_title(self):
        """Dirty input: preamble + no title → should be cleaned and restructured."""
        formatter = ResponseFormatter()
        raw = "על פי המסמכים, בינה מלאכותית היא תחום מחקר במדעי המחשב שעוסק בפיתוח מערכות שיכולות לחקות חשיבה אנושית."
        result = formatter.format(raw, ["ai.pdf", "cs.docx"])
        
        # Preamble removed
        assert "על פי המסמכים" not in result
        # Content preserved
        assert "בינה מלאכותית" in result
        # Citations present
        assert "מקורות:" in result
        assert "ai.pdf" in result
        assert "cs.docx" in result
    
    def test_reformat_all_components_present(self):
        """Reformatted output should always have title + body + citations."""
        formatter = ResponseFormatter()
        raw = "בהתבסס על המסמכים: הנושא מורכב ודורש התייחסות מעמיקה לכל ההיבטים."
        result = formatter.format(raw, ["doc1.txt", "doc2.txt"])
        
        lines = result.strip().split('\n')
        assert len(lines) >= 2  # At minimum: title/body + citations
        assert lines[-1].startswith("מקורות:")


class TestResponseFormatterIsWellFormatted:
    """Test the format validation check."""
    
    def test_well_formatted_passes(self):
        """Properly formatted response should pass validation."""
        formatter = ResponseFormatter()
        response = "כותרת קצרה\nתוכן התשובה המפורטת כאן.\nמקורות: doc.txt"
        
        assert formatter.is_well_formatted(response) is True
    
    def test_missing_citations_fails(self):
        """Response without citations line should fail."""
        formatter = ResponseFormatter()
        response = "כותרת\nתוכן"
        
        assert formatter.is_well_formatted(response) is False
    
    def test_too_long_title_fails(self):
        """Response with >10 word first line should fail."""
        formatter = ResponseFormatter()
        response = "אחת שתיים שלוש ארבע חמש שש שבע שמונה תשע עשר אחת עשרה\nתוכן\nמקורות: doc.txt"
        
        assert formatter.is_well_formatted(response) is False
    
    def test_preamble_present_fails(self):
        """Response starting with preamble should fail."""
        formatter = ResponseFormatter()
        response = "על פי המסמכים, כותרת\nתוכן\nמקורות: doc.txt"
        
        assert formatter.is_well_formatted(response) is False
    
    def test_too_short_fails(self):
        """Response with fewer than 3 lines should fail."""
        formatter = ResponseFormatter()
        response = "just one line"
        
        assert formatter.is_well_formatted(response) is False
