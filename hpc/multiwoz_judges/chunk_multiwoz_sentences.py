"""
Sentence-segment the 500-dialogue MultiWOZ subsample and write chunks
for HPC sentence-level judge array jobs.
Pre-computes dialogue history so each chunk is self-contained.
"""

import os
import math
import nltk
import pandas as pd

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

INPUT = os.path.join(os.path.dirname(__file__), "..", "multiwoz", "multiwoz_sample_500.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sentence_chunks")
CHUNK_SIZE = 3000

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} turns from {df['dialogue_id'].nunique()} dialogues")

# Pre-compute history for each turn
def build_history(dialogue_df, up_to_turn):
    prior = dialogue_df[dialogue_df["turn_index"] < up_to_turn]
    if len(prior) == 0:
        return "(start of dialogue)"
    lines = []
    for _, t in prior.iterrows():
        lines.append(f"{t['speaker'].upper()}: {t['utterance']}")
    return "\n".join(lines)

# Sentence-segment all turns with history
rows = []
for dlg_id, dlg_df in df.groupby("dialogue_id", sort=False):
    dlg_df = dlg_df.sort_values("turn_index")
    for _, row in dlg_df.iterrows():
        history = build_history(dlg_df, row["turn_index"])
        sentences = nltk.sent_tokenize(str(row["utterance"]))
        for idx, sent in enumerate(sentences):
            rows.append({
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "sentence_index": idx,
                "sentence_id": f"{row['dialogue_id']}_{row['turn_index']}_{idx}",
                "speaker": row["speaker"],
                "domain": row["domain"],
                "utterance": row["utterance"],
                "sentence_text": sent,
                "history": history,
            })

sent_df = pd.DataFrame(rows)
total = len(sent_df)
n_chunks = math.ceil(total / CHUNK_SIZE)
pad = len(str(n_chunks - 1))

print(f"Total sentences: {total}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunks to create: {n_chunks}")

for i in range(n_chunks):
    start = i * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, total)
    chunk = sent_df.iloc[start:end]
    fname = f"sentence_chunk_{str(i).zfill(2)}.csv"
    chunk.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  {fname}: {len(chunk)} sentences")

print(f"\nDone. {n_chunks} chunks covering {total} sentences.")
