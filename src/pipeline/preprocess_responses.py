"""
Preprocess all_model_responses.csv:
  - Clean explanation responses (strip leading Likert digit, "Explanation:" prefix)
  - Flag refusals (is_refusal)
  - Flag AI self-references (ai_self_reference)
  - Save to all_model_responses_clean.csv (all rows preserved)
"""

import pandas as pd
import re

df = pd.read_csv("../../data/all_model_responses.csv")
print(f"Loaded {len(df)} rows ({(df['step']=='explanation').sum()} explanation, "
      f"{(df['step']=='likert').sum()} likert)")

# --- Clean explanation responses ---
def clean_response(text):
    """Strip leading Likert digit + newlines, and any 'Explanation:' prefix."""
    text = str(text)
    # Strip leading digit (0-3) followed by whitespace/newlines
    text = re.sub(r"^\s*[0-3]\s*\n+", "", text)
    # Strip "Explanation:" or "Explanation:\n" prefix
    text = re.sub(r"^\s*Explanation:\s*\n*", "", text)
    return text.strip()


def is_refusal(text):
    """True if response matches refusal patterns or is very short."""
    text = str(text)
    if len(text) < 50:
        return True
    refusal_patterns = [
        r"I can'?t",
        r"I cannot",
        r"I'?m not able",
        r"I can'?t answer that",
        r"I can'?t fulfill",
    ]
    for pat in refusal_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def has_ai_self_reference(text):
    """True if response contains AI self-reference phrases."""
    text = str(text)
    patterns = ["as an AI", "As an AI", "I'm a language model", "I am a language model"]
    return any(p in text for p in patterns)


# Apply cleaning only to explanation rows
mask = df["step"] == "explanation"
df.loc[mask, "response_clean"] = df.loc[mask, "response"].apply(clean_response)
df.loc[~mask, "response_clean"] = df.loc[~mask, "response"]

# Flag refusals and AI self-reference (explanation rows only; False for likert)
df["is_refusal"] = False
df.loc[mask, "is_refusal"] = df.loc[mask, "response_clean"].apply(is_refusal)

df["ai_self_reference"] = False
df.loc[mask, "ai_self_reference"] = df.loc[mask, "response_clean"].apply(has_ai_self_reference)

# --- Summary ---
exp = df[mask]
refusals = exp["is_refusal"].sum()
total_exp = len(exp)
print(f"\n{'='*60}")
print(f"  REFUSAL SUMMARY")
print(f"{'='*60}")
print(f"Total explanation rows: {total_exp}")
print(f"Total refusals: {refusals} ({refusals/total_exp:.1%})")

print(f"\n--- Refusals by model ---")
print(exp.groupby("model")["is_refusal"].agg(["sum", "count"]).rename(
    columns={"sum": "refusals", "count": "total"}).to_string())

print(f"\n--- Refusals by nationality ---")
print(exp.groupby("nationality")["is_refusal"].agg(["sum", "count"]).rename(
    columns={"sum": "refusals", "count": "total"}).to_string())

print(f"\n--- Refusals by disability_type ---")
print(exp.groupby("disability_type")["is_refusal"].agg(["sum", "count"]).rename(
    columns={"sum": "refusals", "count": "total"}).to_string())

print(f"\n--- Refusals by scenario ---")
print(exp.groupby("scenario")["is_refusal"].agg(["sum", "count"]).rename(
    columns={"sum": "refusals", "count": "total"}).to_string())

print(f"\n--- AI self-references ---")
print(f"Total: {exp['ai_self_reference'].sum()} ({exp['ai_self_reference'].mean():.1%})")
print(exp.groupby("model")["ai_self_reference"].agg(["sum", "count"]).rename(
    columns={"sum": "ai_self_ref", "count": "total"}).to_string())

# Save
df.to_csv("../../data/all_model_responses_clean.csv", index=False)
print(f"\nSaved all_model_responses_clean.csv ({len(df)} rows)")
