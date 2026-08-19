"""
Generate task_config.csv mapping SLURM array task IDs to (judge, chunk, level) combos.
Covers both turn-level and sentence-level jobs.
"""

import os
import glob
import csv

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "chunks")
SENT_CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "sentence_chunks")
OUTPUT = os.path.join(os.path.dirname(__file__), "task_config.csv")

JUDGES = [
    "gemma3:latest",
    "gemma3:1b",
    "llama3.1:latest",
    "mistral:latest",
]

# Turn-level chunks
turn_chunks = sorted(
    os.path.basename(f)
    for f in glob.glob(os.path.join(CHUNKS_DIR, "multiwoz_chunk_*.csv"))
)

# Sentence-level chunks
sent_chunks = sorted(
    os.path.basename(f)
    for f in glob.glob(os.path.join(SENT_CHUNKS_DIR, "sentence_chunk_*.csv"))
)

if not turn_chunks:
    print("WARNING: No turn chunks found. Run chunk_multiwoz.py first.")
if not sent_chunks:
    print("WARNING: No sentence chunks found. Run chunk_multiwoz_sentences.py first.")

rows = []
task_id = 0

# Turn-level tasks
for judge in JUDGES:
    for chunk in turn_chunks:
        rows.append({
            "task_id": task_id,
            "judge": judge,
            "chunk": chunk,
            "level": "turn",
        })
        task_id += 1

# Sentence-level tasks (disabled — running locally on subsample instead)
# for judge in JUDGES:
#     for chunk in sent_chunks:
#         rows.append({
#             "task_id": task_id,
#             "judge": judge,
#             "chunk": chunk,
#             "level": "sentence",
#         })
#         task_id += 1

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task_id", "judge", "chunk", "level"])
    writer.writeheader()
    writer.writerows(rows)

n_turn = len(JUDGES) * len(turn_chunks)
n_sent = len(JUDGES) * len(sent_chunks)
print(f"Turn-level tasks: {n_turn} ({len(JUDGES)} judges x {len(turn_chunks)} chunks)")
print(f"Sentence-level tasks: {n_sent} ({len(JUDGES)} judges x {len(sent_chunks)} chunks)")
print(f"Total tasks: {len(rows)}")
print(f"SLURM array range: 0-{len(rows)-1}")
print(f"Saved to {OUTPUT}")
