"""
Sentence-level dictionary scoring (Bias-Lexicon, HurtLex, toxic-bert) on MultiWOZ 500-dialogue subsample.
Segments each turn into sentences, scores each sentence independently.
"""

import pandas as pd
import numpy as np
import nltk
import re
import os

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv("../../multiwoz/multiwoz_sample_500.csv")
print(f"Loaded {len(df)} turns from {df['dialogue_id'].nunique()} dialogues")

# ── Sentence segmentation ─────────────────────────────────
sent_rows = []
for _, row in df.iterrows():
    sentences = nltk.sent_tokenize(str(row["utterance"]))
    for idx, sent in enumerate(sentences):
        sent_rows.append({
            "dialogue_id": row["dialogue_id"],
            "turn_index": row["turn_index"],
            "sentence_index": idx,
            "speaker": row["speaker"],
            "domain": row["domain"],
            "sentence_text": sent,
            "sentence_id": f"{row['dialogue_id']}_{row['turn_index']}_{idx}",
        })

sent_df = pd.DataFrame(sent_rows)
print(f"Total sentences: {len(sent_df)}")

# ── Tokenizer ─────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"\b[a-z]+\b", str(text).lower())

def severity_from_count(n):
    if n == 0: return 0
    elif n <= 2: return 1
    elif n <= 5: return 2
    elif n <= 10: return 3
    elif n <= 20: return 4
    else: return 5

# ── Load Bias-Lexicon ─────────────────────────────────────
from datasets import load_dataset
print("Loading bias lexicon...")
lexicon_ds = load_dataset("mediabiasgroup/bias-lexicon", split="train")
BIAS_WORDS = set(w.lower() for w in lexicon_ds["word"])
print(f"  Bias lexicon: {len(BIAS_WORDS)} terms")

# ── Load HurtLex ──────────────────────────────────────────
print("Loading HurtLex...")
HURTLEX_TSV = "../../external/hurtlex_repo/lexica/EN/1.2/hurtlex_EN.tsv"
hurtlex_raw = pd.read_csv(HURTLEX_TSV, sep="\t")
HURTLEX = {}
for _, row in hurtlex_raw.iterrows():
    lemma = str(row["lemma"]).lower().strip()
    if lemma not in HURTLEX:
        HURTLEX[lemma] = row["category"]
print(f"  HurtLex: {len(HURTLEX)} unique lemmas")

# ── Score sentences ───────────────────────────────────────
print("Scoring with Bias-Lexicon and HurtLex...")
bl_labels = []
bl_counts = []
bl_norms = []
bl_sevs = []
bl_terms = []
hl_labels = []
hl_counts = []
hl_norms = []
hl_sevs = []
hl_terms = []
hl_cats = []

for text in sent_df["sentence_text"]:
    tokens = tokenize(text)
    # Bias-Lexicon
    matched = [t for t in tokens if t in BIAS_WORDS]
    bl_counts.append(len(matched))
    bl_norms.append(round(len(matched) / len(tokens), 4) if tokens else 0)
    bl_labels.append(1 if len(matched) > 0 else 0)
    bl_sevs.append(severity_from_count(len(matched)))
    bl_terms.append(", ".join(matched))
    # HurtLex
    h_matched = [(t, HURTLEX[t]) for t in tokens if t in HURTLEX]
    hl_counts.append(len(h_matched))
    hl_norms.append(round(len(h_matched) / len(tokens), 4) if tokens else 0)
    hl_labels.append(1 if len(h_matched) > 0 else 0)
    hl_sevs.append(severity_from_count(len(h_matched)))
    hl_terms.append(", ".join(t for t, _ in h_matched))
    hl_cats.append(", ".join(c for _, c in h_matched))

sent_df["bias_count"] = bl_counts
sent_df["bias_score_norm"] = bl_norms
sent_df["bias_label"] = bl_labels
sent_df["severity"] = bl_sevs
sent_df["matched_terms"] = bl_terms
sent_df["hurtlex_count"] = hl_counts
sent_df["hurtlex_norm"] = hl_norms
sent_df["hurtlex_label"] = hl_labels
sent_df["hurtlex_severity"] = hl_sevs
sent_df["hurtlex_terms"] = hl_terms
sent_df["hurtlex_cats"] = hl_cats

# ── Toxicity ──────────────────────────────────────────────
print("Loading toxic-bert...")
from transformers import pipeline as hf_pipeline
toxicity_pipe = hf_pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    truncation=True,
    max_length=512,
)
print("Scoring toxicity...")
texts = sent_df["sentence_text"].tolist()
tox_results = toxicity_pipe(texts, batch_size=32)
sent_df["toxicity_score"] = [
    r["score"] if r["label"] == "toxic" else 1 - r["score"]
    for r in tox_results
]
sent_df["toxicity_label"] = (sent_df["toxicity_score"] > 0.5).astype(int)

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SENTENCE-LEVEL RESULTS ({len(sent_df)} sentences)")
print(f"{'='*60}")
print(f"Bias-Lexicon:  {sent_df['bias_label'].sum()} flagged ({sent_df['bias_label'].mean():.2%})")
print(f"HurtLex:       {sent_df['hurtlex_label'].sum()} flagged ({sent_df['hurtlex_label'].mean():.2%})")
print(f"toxic-bert:    {sent_df['toxicity_label'].sum()} flagged ({sent_df['toxicity_label'].mean():.2%})")

# ── Save ──────────────────────────────────────────────────
sent_df.to_csv("../../multiwoz/multiwoz_sentence_dict_scores.csv", index=False)
print(f"\nSaved multiwoz_sentence_dict_scores.csv ({len(sent_df)} rows)")
