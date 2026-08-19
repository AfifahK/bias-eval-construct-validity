#!/usr/bin/env python3
"""
Build a stratified 100-case annotation sample from the 679 divergence cases.
Includes full response text so the annotator can judge in context.

Output: data/annotation_sample.csv — open in Excel/Sheets, fill columns, save.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Load divergence cases
div = pd.read_csv(ROOT / "results" / "tables" / "table_divergence_cases.csv")

# Load response text
master = pd.read_csv(ROOT / "data" / "all_model_responses_clean.csv")
expl = master[(master["step"] == "explanation") & (~master["is_refusal"])].copy()

# Merge response text into divergence cases
mk = ["prompt_id", "model", "scale"]
div = div.merge(
    expl[mk + ["response"]].drop_duplicates(mk),
    on=mk, how="left"
)

print(f"Total divergence cases: {len(div)}")
print(f"With response text: {div['response'].notna().sum()}")

# --- Stratified sampling ---
# Strategy: sample proportionally by (divergence_type × judge), with minimum 5 per cell
TARGET = 100
RANDOM_STATE = 42

# Group and compute proportional allocation
groups = div.groupby(["divergence_type", "judge"]).size().reset_index(name="count")
groups["proportion"] = groups["count"] / groups["count"].sum()
groups["alloc"] = (groups["proportion"] * TARGET).round().astype(int)

# Ensure minimum 3 per non-empty cell
groups["alloc"] = groups["alloc"].clip(lower=3)

# Adjust to hit target
diff = TARGET - groups["alloc"].sum()
if diff > 0:
    # Add to largest groups
    largest = groups.nlargest(abs(diff), "count").index
    groups.loc[largest, "alloc"] += 1
elif diff < 0:
    # Remove from largest groups
    largest = groups.nlargest(abs(diff), "count").index
    groups.loc[largest, "alloc"] -= 1

print(f"\nSampling allocation:")
print(groups.to_string(index=False))
print(f"Total target: {groups['alloc'].sum()}")

# Sample
sampled = []
for _, row in groups.iterrows():
    pool = div[(div["divergence_type"] == row["divergence_type"]) &
               (div["judge"] == row["judge"])]
    n = min(row["alloc"], len(pool))
    s = pool.sample(n=n, random_state=RANDOM_STATE)
    sampled.append(s)

sample = pd.concat(sampled, ignore_index=True)

# Shuffle so annotator doesn't see patterns
sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Add annotation columns (empty — annotator fills these)
sample["human_bias_label"] = ""      # 0, 1, or "unclear"
sample["bias_type"] = ""             # representational / allocative / both / none
sample["brief_rationale"] = ""       # 1 sentence max

# Select and order columns for annotation
out_cols = [
    "prompt_id", "model", "scale", "nationality", "disability_type", "scenario",
    "divergence_type", "judge", "dict_label", "judge_label",
    "response",
    "human_bias_label", "bias_type", "brief_rationale",
]
sample = sample[out_cols]

# Save
out_path = ROOT / "data" / "annotation_sample.csv"
sample.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Rows: {len(sample)}")
print(f"\nDistribution check:")
print(sample.groupby(["divergence_type", "judge"]).size().to_string())
