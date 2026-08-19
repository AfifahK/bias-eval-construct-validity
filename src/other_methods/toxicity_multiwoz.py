"""
Toxicity and sentiment analysis of MultiWOZ 500-dialogue subsample.
Uses HuggingFace pipelines:
  - unitary/toxic-bert for toxicity
  - cardiffnlp/twitter-roberta-base-sentiment-latest for sentiment
Computes agreement with dict and judge bias labels where available.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import cohen_kappa_score

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("../../multiwoz/multiwoz_sample_500.csv")
print(f"Loaded {len(df)} turns from {df['dialogue_id'].nunique()} dialogues")

# ── Load models ──────────────────────────────────────────────
print("Loading toxicity model...")
toxicity_pipe = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    truncation=True,
    max_length=512,
)

print("Loading sentiment model...")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512,
)

# ── Score responses ──────────────────────────────────────────
print("Scoring utterances...")
texts = df["utterance"].astype(str).tolist()

# Batch inference
tox_results = toxicity_pipe(texts, batch_size=32)
sent_results = sentiment_pipe(texts, batch_size=32)

df["toxicity_score"] = [r["score"] if r["label"] == "toxic" else 1 - r["score"] for r in tox_results]
df["toxicity_label"] = (df["toxicity_score"] > 0.5).astype(int)
df["sentiment_label"] = [r["label"] for r in sent_results]
df["sentiment_score"] = [round(r["score"], 4) for r in sent_results]

# ── Summary stats ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TOXICITY & SENTIMENT SUMMARY (MultiWOZ 500)")
print(f"{'='*60}")
print(f"Toxic utterances (score > 0.5): {df['toxicity_label'].sum()} / {len(df)} "
      f"({df['toxicity_label'].mean():.1%})")
print(f"Mean toxicity score: {df['toxicity_score'].mean():.4f}")
print(f"\nSentiment distribution:")
print(df["sentiment_label"].value_counts().to_string())

for label, col in [("speaker", "speaker"), ("domain", "domain")]:
    print(f"\n--- By {label} ---")
    agg = df.groupby(col).agg(
        mean_toxicity=("toxicity_score", "mean"),
        pct_toxic=("toxicity_label", "mean"),
        n=("toxicity_label", "count"),
    ).round(4)
    print(agg.to_string())

    # Sentiment breakdown
    sent_ct = pd.crosstab(df[col], df["sentiment_label"], normalize="index").round(3)
    print(f"  Sentiment proportions:")
    print(f"  {sent_ct.to_string()}")

# ── Conversation-level aggregation ───────────────────────────
conv_tox = df.groupby("dialogue_id").agg(
    max_toxicity=("toxicity_score", "max"),
    mean_toxicity=("toxicity_score", "mean"),
    any_toxic=("toxicity_label", "max"),
    n_turns=("toxicity_label", "count"),
).reset_index()

print(f"\n--- Conversation-level ---")
print(f"Dialogues with any toxic utterance: {conv_tox['any_toxic'].sum()}/{len(conv_tox)} "
      f"({conv_tox['any_toxic'].mean():.1%})")
print(f"Mean max-toxicity per dialogue: {conv_tox['max_toxicity'].mean():.4f}")

# ── Save scores ──────────────────────────────────────────────
out_cols = ["dialogue_id", "turn_index", "speaker", "domain",
            "toxicity_score", "toxicity_label", "sentiment_label", "sentiment_score"]
df[out_cols].to_csv("../../multiwoz/multiwoz_toxicity_scores_500.csv", index=False)
print(f"\nSaved multiwoz_toxicity_scores_500.csv ({len(df)} rows)")

# ── Agreement with dict scores (if available) ────────────────
print(f"\n{'='*60}")
print(f"  AGREEMENT: TOXICITY vs DICT BIAS LABELS")
print(f"{'='*60}")

merge_keys = ["dialogue_id", "turn_index"]

try:
    dict_scores = pd.read_csv("../../multiwoz/multiwoz_dict_scores.csv")
    # Filter to 500-sample dialogues
    sample_ids = df["dialogue_id"].unique()
    dict_scores = dict_scores[dict_scores["dialogue_id"].isin(sample_ids)]
    dict_scores = dict_scores[merge_keys + ["bias_label", "hurtlex_label"]].copy()

    merged = df[merge_keys + ["toxicity_label"]].merge(dict_scores, on=merge_keys, how="inner")
    print(f"Merged: {len(merged)} rows")

    # Toxicity vs Bias-Lexicon
    try:
        kappa_lex = cohen_kappa_score(merged["toxicity_label"], merged["bias_label"])
    except ValueError:
        kappa_lex = np.nan
    print(f"\nToxicity vs Bias-Lexicon:  Cohen's kappa = {kappa_lex:.4f}  (n={len(merged)})")

    # Toxicity vs HurtLex
    try:
        kappa_hurt = cohen_kappa_score(merged["toxicity_label"], merged["hurtlex_label"])
    except ValueError:
        kappa_hurt = np.nan
    print(f"Toxicity vs HurtLex:      Cohen's kappa = {kappa_hurt:.4f}  (n={len(merged)})")

    # Cross-tabs
    print(f"\n--- Toxicity × Bias-Lexicon ---")
    print(pd.crosstab(merged["toxicity_label"], merged["bias_label"]).to_string())
    print(f"\n--- Toxicity × HurtLex ---")
    print(pd.crosstab(merged["toxicity_label"], merged["hurtlex_label"]).to_string())

except FileNotFoundError:
    print("Dict scores not found — skipping agreement analysis")

# ── Agreement with judge scores (if available) ───────────────
for judge_name, judge_file in [
    ("Gemma 4.3B", "multiwoz_judge_scores_500_gemma3.csv"),
    ("Llama 3.1", "multiwoz_judge_scores_500_llama3_1.csv"),
    ("Mistral", "multiwoz_judge_scores_500_mistral.csv"),
    ("Gemma 1B", "multiwoz_judge_scores_500_gemma3_1b.csv"),
]:
    try:
        judge_df = pd.read_csv(f"../../multiwoz/{judge_file}")
        judge_df = judge_df[judge_df["bias_label"] != -1]
        judge_df = judge_df[merge_keys + ["bias_label"]].rename(
            columns={"bias_label": "judge_bias_label"}
        )
        merged_j = df[merge_keys + ["toxicity_label"]].merge(judge_df, on=merge_keys, how="inner")
        try:
            kappa_j = cohen_kappa_score(merged_j["toxicity_label"], merged_j["judge_bias_label"])
        except ValueError:
            kappa_j = np.nan
        print(f"\nToxicity vs {judge_name} judge: Cohen's kappa = {kappa_j:.4f}  (n={len(merged_j)})")
        print(f"  Judge bias rate: {merged_j['judge_bias_label'].mean():.1%}")
    except FileNotFoundError:
        print(f"\n{judge_file} not found — skipping {judge_name} agreement")
