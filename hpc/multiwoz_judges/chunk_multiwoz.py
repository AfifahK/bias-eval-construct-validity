"""
Split the full MultiWOZ turn-level CSV into chunks for HPC array jobs.
Writes to hpc_run/chunks/multiwoz_chunk_NN.csv (zero-padded).
"""

import os
import math
import pandas as pd

INPUT = os.path.join(os.path.dirname(__file__), "..", "multiwoz", "multiwoz_sample.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "chunks")
CHUNK_SIZE = 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)
total_turns = len(df)
n_chunks = math.ceil(total_turns / CHUNK_SIZE)
pad = len(str(n_chunks - 1))

print(f"Total turns: {total_turns}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunks to create: {n_chunks}")

for i in range(n_chunks):
    start = i * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, total_turns)
    chunk = df.iloc[start:end]
    fname = f"multiwoz_chunk_{str(i).zfill(2)}.csv"
    chunk.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  {fname}: {len(chunk)} turns")

print(f"\nDone. {n_chunks} chunks covering {total_turns} turns.")
