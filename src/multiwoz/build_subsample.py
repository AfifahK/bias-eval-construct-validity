"""
Build a stratified subsample of MultiWOZ dialogues for judge evaluation.
Stratifies by primary domain with proportional allocation.
"""

import pandas as pd
import numpy as np

INPUT = "../../multiwoz/multiwoz_sample.csv"
OUTPUT = "../../multiwoz/multiwoz_sample_500.csv"
TARGET_DIALOGUES = 500
RANDOM_STATE = 42

df = pd.read_csv(INPUT)
print(f"Full dataset: {len(df)} turns, {df['dialogue_id'].nunique()} dialogues")

# Each dialogue can span multiple domains. Use the primary (most frequent) domain per dialogue.
dlg_domain = (
    df.groupby("dialogue_id")["domain"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"domain": "primary_domain"})
)

# Stratify by primary_domain
dlg_ids = dlg_domain.copy()
n_total = len(dlg_ids)
target = min(TARGET_DIALOGUES, n_total)

# Proportional allocation
domain_counts = dlg_ids["primary_domain"].value_counts()
domain_alloc = (domain_counts / n_total * target).round().astype(int)

# Adjust rounding to hit exact target
diff = target - domain_alloc.sum()
if diff > 0:
    for dom in domain_alloc.index[:diff]:
        domain_alloc[dom] += 1
elif diff < 0:
    for dom in domain_alloc.index[:abs(diff)]:
        domain_alloc[dom] -= 1

print(f"\nTarget: {target} dialogues")
print(f"\nStratification (by primary domain):")
sampled_ids = []
for domain, n_sample in domain_alloc.items():
    pool = dlg_ids[dlg_ids["primary_domain"] == domain]["dialogue_id"]
    n_sample = min(n_sample, len(pool))
    chosen = pool.sample(n=n_sample, random_state=RANDOM_STATE)
    sampled_ids.extend(chosen.tolist())
    print(f"  {domain:<40} {n_sample:>4} / {len(pool):>5} dialogues")

# Filter to sampled dialogues
sample_df = df[df["dialogue_id"].isin(sampled_ids)].copy()

print(f"\nSubsample: {sample_df['dialogue_id'].nunique()} dialogues, {len(sample_df)} turns")
print(f"Avg turns/dialogue: {len(sample_df) / sample_df['dialogue_id'].nunique():.1f}")

# Speaker breakdown
print(f"\nSpeaker distribution:")
print(sample_df["speaker"].value_counts().to_string())

# Domain × speaker cross-tab
print(f"\nDomain × speaker turn counts:")
ct = pd.crosstab(sample_df["domain"], sample_df["speaker"])
print(ct.to_string())

sample_df.to_csv(OUTPUT, index=False)
print(f"\nSaved {OUTPUT}")
