"""Tests for src/dictionary/lexicon_analysis.py functions."""

import pytest
from unittest.mock import patch
from src.dictionary.lexicon_analysis import tokenize, score_text, _load_lexicon, BIAS_WORDS
import src.dictionary.lexicon_analysis as lexicon_mod


# ── tokenize ────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert tokenize("Hello, world! How are you?") == ["hello", "world", "how", "are", "you"]

    def test_casing(self):
        assert tokenize("UPPER lower MiXeD") == ["upper", "lower", "mixed"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_none_input(self):
        assert tokenize(None) == ["none"]

    def test_numbers_excluded(self):
        # \b[a-z]+\b only matches alpha
        tokens = tokenize("test123 hello 42 world")
        assert "hello" in tokens
        assert "world" in tokens
        assert "42" not in tokens
        assert "test123" not in tokens

    def test_hyphenated_words_split(self):
        tokens = tokenize("well-known fact")
        assert tokens == ["well", "known", "fact"]

    def test_substring_not_matched(self):
        # "unreliable" should tokenize as one token, not match "un" + "reliable"
        tokens = tokenize("unreliable")
        assert tokens == ["unreliable"]


# ── score_text ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_lexicon():
    """Inject a small test lexicon instead of loading from HuggingFace."""
    test_words = {"unreliable", "dangerous", "inferior", "incompetent", "biased"}
    lexicon_mod.BIAS_WORDS = test_words
    yield
    lexicon_mod.BIAS_WORDS = None


class TestScoreText:
    def test_no_bias_words(self):
        result = score_text("The candidate is qualified and experienced.")
        assert result["bias_count"] == 0
        assert result["bias_label"] == 0
        assert result["severity"] == 0
        assert result["matched_terms"] == []

    def test_single_bias_word(self):
        result = score_text("The candidate is unreliable.")
        assert result["bias_count"] == 1
        assert result["bias_label"] == 1
        assert result["severity"] == 1
        assert "unreliable" in result["matched_terms"]

    def test_multiple_bias_words(self):
        result = score_text("They are unreliable and dangerous and inferior.")
        assert result["bias_count"] == 3
        assert result["severity"] == 2  # 3 <= 5 -> severity 2

    def test_severity_boundaries(self):
        # 0 terms -> 0
        assert score_text("clean text here nothing bad")["severity"] == 0
        # 1 term -> 1
        assert score_text("this is dangerous")["severity"] == 1
        # 2 terms -> 1
        assert score_text("dangerous and unreliable")["severity"] == 1
        # 3 terms -> 2
        assert score_text("dangerous unreliable inferior")["severity"] == 2

    def test_case_insensitive(self):
        result = score_text("UNRELIABLE person")
        assert result["bias_count"] == 1

    def test_bias_score_norm(self):
        result = score_text("one unreliable word here")
        # 1 match out of 4 tokens
        assert result["bias_score_norm"] == round(1 / 4, 4)

    def test_empty_string(self):
        result = score_text("")
        assert result["bias_count"] == 0
        assert result["bias_score_norm"] == 0

    def test_sentence_breakdown(self):
        result = score_text("This is clean. This is dangerous.")
        assert result["num_sentences"] == 2
        sents = result["sentence_scores"]
        assert sents[0]["bias_count"] == 0
        assert sents[1]["bias_count"] == 1

    def test_substring_no_false_positive(self):
        # "biased" is in lexicon, but "unbiased" should not match "biased"
        # because tokenize splits on word boundaries
        result = score_text("The review was unbiased and fair.")
        # "unbiased" tokenizes as single token "unbiased", not "biased"
        assert "biased" not in result["matched_terms"]


# ── _load_lexicon guard ─────────────────────────────────────

class TestLoadLexiconGuard:
    def test_empty_lexicon_raises(self):
        """If load_dataset returns no words, _load_lexicon must raise."""
        lexicon_mod.BIAS_WORDS = None  # force reload
        with patch("src.dictionary.lexicon_analysis.load_dataset") as mock_ld:
            mock_ld.return_value = {"word": []}
            with pytest.raises(RuntimeError, match="Bias lexicon loaded only 0 terms"):
                _load_lexicon()
        lexicon_mod.BIAS_WORDS = None
