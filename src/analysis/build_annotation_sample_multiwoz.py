#!/usr/bin/env python3
"""
Build a stratified 100-case annotation sample from MultiWOZ subsample disagreements.
Samples from 5 categories to cover the key questions:
  A: Gemma-only flags (is Gemma over-flagging benign dialogue?)
  B: HurtLex-only, no judges flag (HurtLex false positives on dialogue)
  C: 2+ judges flag (most likely to be real bias — if any exists)
  D: All agree no bias (control group — sanity check)
  E: Random from full subsample (unbiased prevalence estimate)

Output: data/annotation_sample_multiwoz.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42

# Load all scores on 500-dialogue subsample
sample = pd.read_csv(ROOT / "multiwoz" / "multiwoz_sample_500.csv")
sample_ids = set(sample["dialogue_id"].unique())

dict_scores = pd.read_csv(ROOT / "multiwoz" / "multiwoz_dict_scores.csv")
dict_sub = dict_scores[dict_scores["dialogue_id"].isin(sample_ids)]

gemma = pd.read_csv(ROOT / "multiwoz" / "multiwoz_judge_scores_500_gemma3.csv")
llama = pd.read_csv(ROOT / "multiwoz" / "multiwoz_judge_scores_500_llama3_1.csv")
mistral = pd.read_csv(ROOT / "multiwoz" / "multiwoz_judge_scores_500_mistral.csv")

mk = ["dialogue_id", "turn_index"]

# Build unified table
m = dict_sub[mk + ["speaker", "domain", "bias_label", "hurtlex_label"]].rename(
    columns={"bias_label": "biaslex", "hurtlex_label": "hurtlex"})
m = m.merge(sample[mk + ["utterance"]], on=mk, how="left")

for name, df in [("gemma", gemma), ("llama", llama), ("mistral", mistral)]:
    valid = df[df["bias_label"] != -1][mk + ["bias_label"]].rename(columns={"bias_label": name})
    m = m.merge(valid, on=mk, how="inner")

judges = ["gemma", "llama", "mistral"]
methods = ["biaslex", "hurtlex", "gemma", "llama", "mistral"]
m["n_judge_flag"] = m[judges].sum(axis=1)
m["n_all_flag"] = m[methods].sum(axis=1)

# Define categories
cat_a = m[(m["gemma"] == 1) & (m["llama"] == 0) & (m["mistral"] == 0)].copy()
cat_a["category"] = "A_gemma_only"

cat_b = m[(m["hurtlex"] == 1) & (m["gemma"] == 0) & (m["llama"] == 0) & (m["mistral"] == 0) & (m["biaslex"] == 0)].copy()
cat_b["category"] = "B_hurtlex_only"

cat_c = m[m["n_judge_flag"] >= 2].copy()
cat_c["category"] = "C_multi_judge"

cat_d = m[m["n_all_flag"] == 0].copy()
cat_d["category"] = "D_all_agree_no_bias"

cat_e = m.copy()
cat_e["category"] = "E_random"

# Allocation: 100 total
# A:25, B:25, C:20, D:15, E:15
allocs = [
    (cat_a, 25, "A_gemma_only"),
    (cat_b, 25, "B_hurtlex_only"),
    (cat_c, 20, "C_multi_judge"),
    (cat_d, 15, "D_all_agree_no_bias"),
    (cat_e, 15, "E_random"),
]

print(f"Category sizes:")
sampled = []
already_sampled = set()
for pool, n, label in allocs:
    available = pool[~pool.set_index(mk).index.isin(already_sampled)]
    n_actual = min(n, len(available))
    s = available.sample(n=n_actual, random_state=RANDOM_STATE)
    sampled.append(s)
    already_sampled.update(s.set_index(mk).index)
    print(f"  {label:<25} pool={len(pool):>5}  sampled={n_actual}")

result = pd.concat(sampled, ignore_index=True)
result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Add annotation columns
result["human_bias_label"] = ""
result["bias_type"] = ""
result["brief_rationale"] = ""

# Select columns
out_cols = [
    "dialogue_id", "turn_index", "speaker", "domain", "category",
    "biaslex", "hurtlex", "gemma", "llama", "mistral",
    "utterance",
    "human_bias_label", "bias_type", "brief_rationale",
]
result = result[out_cols]

out_path = ROOT / "data" / "annotation_sample_multiwoz.csv"
result.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Total rows: {len(result)}")
print(f"\nCategory distribution:")
print(result["category"].value_counts().to_string())
