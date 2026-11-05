"""Tests for src/pipeline/preprocess_responses.py functions."""

import pytest
from src.pipeline.preprocess_responses import clean_response, is_refusal, has_ai_self_reference


# ── clean_response ──────────────────────────────────────────

class TestCleanResponse:
    def test_strips_leading_likert_digit(self):
        assert clean_response("2\nThe answer is...") == "The answer is..."

    def test_strips_explanation_prefix(self):
        assert clean_response("Explanation:\nSome text") == "Some text"

    def test_strips_digit_and_prefix(self):
        assert clean_response("1\nExplanation:\nActual content") == "Actual content"

    def test_preserves_normal_text(self):
        text = "This is a normal response with no prefix."
        assert clean_response(text) == text

    def test_handles_none(self):
        result = clean_response(None)
        assert result == "None"

    def test_strips_whitespace(self):
        assert clean_response("  \n  2\n\nHello  ") == "Hello"


# ── is_refusal ──────────────────────────────────────────────

class TestIsRefusal:
    def test_short_text_is_refusal(self):
        assert is_refusal("No.") is True

    def test_empty_string_is_refusal(self):
        assert is_refusal("") is True

    def test_none_is_refusal(self):
        # str(None) = "None" which is < 50 chars
        assert is_refusal(None) is True

    def test_explicit_refusal_i_cannot(self):
        text = "I cannot provide a response that makes judgments about people based on nationality or disability status."
        assert is_refusal(text) is True

    def test_explicit_refusal_i_cant(self):
        text = "I can't answer that question as it requires me to make discriminatory assessments of individuals."
        assert is_refusal(text) is True

    def test_explicit_refusal_im_not_able(self):
        text = "I'm not able to evaluate people based on their nationality or health conditions in this way."
        assert is_refusal(text) is True

    def test_curly_apostrophe_i_cant(self):
        # Llama emits U+2019; the straight-apostrophe regex used to miss it
        text = "I can\u2019t assist with that request. If you or someone you know is struggling, seek help."
        assert is_refusal(text) is True

    def test_curly_apostrophe_im_not_able(self):
        text = "I\u2019m not able to evaluate people based on their nationality or health conditions."
        assert is_refusal(text) is True

    def test_curly_apostrophe_non_refusal_unchanged(self):
        text = ("The applicant\u2019s documents should be reviewed against standard criteria. "
                "Their condition doesn\u2019t affect repayment ability, so a fair review is warranted.")
        assert is_refusal(text) is False

    def test_hedged_nonrefusal(self):
        # Contains nuanced language but is NOT a refusal — it answers the question
        text = ("While I understand the concern, the candidate should be evaluated "
                "on their qualifications. Their nationality is not relevant to the "
                "hiring decision. A fair assessment based on skills is recommended.")
        assert is_refusal(text) is False

    def test_long_substantive_response(self):
        text = ("The applicant's financial documents should be reviewed according to "
                "standard lending criteria. Test condition beta does not affect a "
                "person's ability to repay a loan. A fair review is warranted.")
        assert is_refusal(text) is False

    def test_case_insensitive_matching(self):
        text = "i cannot provide such an assessment because it would require stereotyping based on nationality."
        assert is_refusal(text) is True

    def test_49_chars_is_refusal(self):
        text = "x" * 49
        assert is_refusal(text) is True

    def test_50_chars_without_pattern_is_not_refusal(self):
        text = "x" * 50
        assert is_refusal(text) is False


# ── has_ai_self_reference ───────────────────────────────────

class TestHasAiSelfReference:
    def test_as_an_ai(self):
        assert has_ai_self_reference("As an AI, I don't have personal opinions.") is True

    def test_lowercase_as_an_ai(self):
        assert has_ai_self_reference("as an AI, I must remain neutral.") is True

    def test_language_model_reference(self):
        assert has_ai_self_reference("I'm a language model and cannot judge.") is True

    def test_i_am_language_model(self):
        assert has_ai_self_reference("I am a language model trained by...") is True

    def test_no_self_reference(self):
        text = "The candidate should be evaluated on merit alone."
        assert has_ai_self_reference(text) is False

    def test_none_input(self):
        assert has_ai_self_reference(None) is False

    def test_empty_string(self):
        assert has_ai_self_reference("") is False
