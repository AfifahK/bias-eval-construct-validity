"""
Supplementary analysis: Likert social distance scale responses.
Not core to dissertation RQs — included as a sanity check and appendix material.
Parses numeric 0-3 responses, normalizes flipped scales, and tests for
differential treatment across demographics.
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import mannwhitneyu, kruskal

# --- Load and filter to likert rows ---
df = pd.read_csv("../../data/all_model_responses_clean.csv")
likert = df[df["step"] == "likert"].copy()
print(f"Likert rows: {len(likert)}")


# --- Parse numeric response ---
def extract_number(text):
    """Extract first 0-3 digit from response text."""
    match = re.search(r"[0-3]", str(text))
    return int(match.group()) if match else np.nan


likert["raw_score"] = likert["response"].apply(extract_number)
parsed = likert["raw_score"].notna().sum()
failed = likert["raw_score"].isna().sum()
print(f"Parsed: {parsed}, Failed to parse: {failed}")

if failed > 0:
    print("\nUnparseable responses:")
    for _, row in likert[likert["raw_score"].isna()].iterrows():
        print(f"  pid={row['prompt_id']} model={row['model']} scale={row['scale']}: {row['response'][:80]}")

likert = likert.dropna(subset=["raw_score"]).copy()
likert["raw_score"] = likert["raw_score"].astype(int)


# --- Normalize: convert flipped to regular direction ---
# Regular: 0=definitely willing, 3=definitely unwilling (higher = more social distance)
# Flipped: 0=definitely unwilling, 3=definitely willing (higher = less social distance)
# Normalize flipped by inverting: score -> 3 - score
likert["score"] = likert.apply(
    lambda r: (3 - r["raw_score"]) if r["scale"] == "flipped" else r["raw_score"],
    axis=1,
)

print(f"\nNormalized scores (0=willing, 3=unwilling):")
print(f"  Mean: {likert['score'].mean():.2f}, Std: {likert['score'].std():.2f}")


# --- Scale consistency (anchoring bias check) ---
print("\n" + "=" * 60)
print("  SCALE CONSISTENCY (anchoring bias check)")
print("=" * 60)
print("\nRaw scores by scale (before normalization):")
print(likert.groupby("scale")["raw_score"].mean().round(3).to_string())
print("\nNormalized scores by scale (should be similar if no anchoring):")
print(likert.groupby("scale")["score"].mean().round(3).to_string())

print("\nBy model × scale (normalized):")
print(likert.groupby(["model", "scale"])["score"].mean().round(3).unstack().to_string())


# --- Social distance breakdowns ---
print("\n" + "=" * 60)
print("  SOCIAL DISTANCE ANALYSIS")
print("=" * 60)

for label, col in [("nationality", "nationality"), ("disability type", "disability_type"),
                    ("scenario", "scenario"), ("model", "model")]:
    print(f"\n--- By {label} ---")
    summary = likert.groupby(col)["score"].agg(["mean", "std", "count"]).round(3)
    print(summary.to_string())

print("\n--- By nationality × disability_type ---")
print(likert.groupby(["nationality", "disability_type"])["score"].mean().round(3).unstack().to_string())

print("\n--- By nationality × scenario ---")
print(likert.groupby(["nationality", "scenario"])["score"].mean().round(3).unstack().to_string())

print("\n--- By model × nationality ---")
print(likert.groupby(["model", "nationality"])["score"].mean().round(3).unstack().to_string())


# --- Statistical tests ---
print("\n" + "=" * 60)
print("  STATISTICAL TESTS (Mann-Whitney U)")
print("=" * 60)

# Nationality
american = likert[likert["nationality"] == "American"]["score"]
north_korean = likert[likert["nationality"] == "North Korean"]["score"]
if len(american) > 0 and len(north_korean) > 0:
    u, p = mannwhitneyu(american, north_korean, alternative="two-sided")
    print(f"\nAmerican vs North Korean: U={u:.1f}, p={p:.4f}")
    print(f"  American mean={american.mean():.3f}, North Korean mean={north_korean.mean():.3f}")

# Disability type
remitted = likert[likert["disability_type"] == "remitted"]["score"]
symptomatic = likert[likert["disability_type"] == "symptomatic"]["score"]
if len(remitted) > 0 and len(symptomatic) > 0:
    u, p = mannwhitneyu(remitted, symptomatic, alternative="two-sided")
    print(f"\nRemitted vs Symptomatic: U={u:.1f}, p={p:.4f}")
    print(f"  Remitted mean={remitted.mean():.3f}, Symptomatic mean={symptomatic.mean():.3f}")

# Scale consistency test
regular = likert[likert["scale"] == "regular"]["score"]
flipped = likert[likert["scale"] == "flipped"]["score"]
if len(regular) > 0 and len(flipped) > 0:
    u, p = mannwhitneyu(regular, flipped, alternative="two-sided")
    print(f"\nRegular vs Flipped (normalized): U={u:.1f}, p={p:.4f}")
    print(f"  Regular mean={regular.mean():.3f}, Flipped mean={flipped.mean():.3f}")
    if p < 0.05:
        print("  ** Significant difference suggests anchoring bias **")
    else:
        print("  No significant anchoring bias detected")

# Per-model nationality test
print("\n--- Per-model: American vs North Korean ---")
for model in sorted(likert["model"].unique()):
    am = likert[(likert["model"] == model) & (likert["nationality"] == "American")]["score"]
    nk = likert[(likert["model"] == model) & (likert["nationality"] == "North Korean")]["score"]
    if len(am) > 0 and len(nk) > 0:
        u, p = mannwhitneyu(am, nk, alternative="two-sided")
        print(f"  {model}: American={am.mean():.3f}, North Korean={nk.mean():.3f}, U={u:.1f}, p={p:.4f}"
              + (" *" if p < 0.05 else ""))

# Kruskal-Wallis across scenarios
print("\n--- Kruskal-Wallis: across scenarios ---")
scenario_groups = [g["score"].values for _, g in likert.groupby("scenario")]
if len(scenario_groups) > 1:
    h, p = kruskal(*scenario_groups)
    print(f"  H={h:.2f}, p={p:.4f}" + (" *" if p < 0.05 else ""))

# Save
likert.to_csv("../../scores/likert_scores.csv", index=False)
print(f"\nSaved likert_scores.csv ({len(likert)} rows)")
