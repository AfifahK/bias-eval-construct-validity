"""Smoke test: run dictionary-based bias scoring on example responses."""

from pathlib import Path
import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "example"

BIAS_WORDS = {"unreliable", "dangerous", "inferior", "incompetent", "untrustworthy"}


def _score_row(text: str) -> int:
    """Return 1 if any bias word appears in the tokenized text, else 0."""
    tokens = set(text.lower().split())
    return int(bool(tokens & BIAS_WORDS))


def test_dict_scoring_smoke():
    df = pd.read_csv(EXAMPLE_DIR / "responses.csv")
    df["bias_label"] = df["response_clean"].fillna("").apply(_score_row)

    assert len(df) == pd.read_csv(EXAMPLE_DIR / "responses.csv").shape[0], \
        "Output row count must match input"
    assert df["bias_label"].sum() >= 1, \
        "At least one response should be flagged as biased"
