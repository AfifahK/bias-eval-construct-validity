"""Validate that example fixture CSVs have the expected schema and are non-empty."""

from pathlib import Path
import pandas as pd
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "example"


@pytest.fixture
def prompts():
    return pd.read_csv(EXAMPLE_DIR / "prompts.csv")


@pytest.fixture
def responses():
    return pd.read_csv(EXAMPLE_DIR / "responses.csv")


@pytest.fixture
def dict_scores():
    return pd.read_csv(EXAMPLE_DIR / "dict_scores.csv")


@pytest.fixture
def judge_scores():
    return pd.read_csv(EXAMPLE_DIR / "judge_scores.csv")


# ---- prompts.csv ----

def test_prompts_columns(prompts):
    required = {
        "prompt_id", "nationality", "disability", "disability_type",
        "scenario_id", "scenario", "prompt_text_regular", "prompt_text_flipped",
    }
    assert required.issubset(prompts.columns), f"Missing: {required - set(prompts.columns)}"


def test_prompts_nonempty(prompts):
    assert len(prompts) > 0


# ---- responses.csv ----

def test_responses_columns(responses):
    required = {
        "prompt_id", "nationality", "disability", "disability_type",
        "scenario_id", "scenario", "model", "scale", "prompt_text",
        "response", "step", "response_clean", "is_refusal", "ai_self_reference",
    }
    assert required.issubset(responses.columns), f"Missing: {required - set(responses.columns)}"


def test_responses_nonempty(responses):
    assert len(responses) > 0


# ---- dict_scores.csv ----

def test_dict_scores_columns(dict_scores):
    required = {
        "prompt_id", "model", "scale", "step",
        "bias_count", "bias_label", "severity",
    }
    assert required.issubset(dict_scores.columns), f"Missing: {required - set(dict_scores.columns)}"


def test_dict_scores_nonempty(dict_scores):
    assert len(dict_scores) > 0


# ---- judge_scores.csv ----

def test_judge_scores_columns(judge_scores):
    required = {
        "prompt_id", "model", "scale", "step",
        "bias_label", "severity", "reasoning",
    }
    assert required.issubset(judge_scores.columns), f"Missing: {required - set(judge_scores.columns)}"


def test_judge_scores_nonempty(judge_scores):
    assert len(judge_scores) > 0
