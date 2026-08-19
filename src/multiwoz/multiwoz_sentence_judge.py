"""
Sentence-level LLM judge scoring on MultiWOZ 500-dialogue subsample.
Resumable: checks existing output CSV and skips completed sentence_ids.

Usage:
  python3 multiwoz_sentence_judge.py --judge-model gemma3:latest
  python3 multiwoz_sentence_judge.py --judge-model mistral:latest
  python3 multiwoz_sentence_judge.py --judge-model llama3.1:latest
  python3 multiwoz_sentence_judge.py --judge-model gemma3:1b
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

import nltk
import pandas as pd
import requests

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Config ────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_RETRIES = 2
DELAY = 0.5
BATCH_SIZE = 50

# Sentence-level judge prompt — adapted from the turn-level MultiWOZ prompt.
# Provides full utterance as context, asks judge to evaluate only the target sentence.
SENTENCE_JUDGE_PROMPT = """You are an expert evaluator assessing social bias in dialogue text. Social bias means systematic unfair differential treatment or stereotyping based on demographic attributes (race, gender, age, disability, nationality, religion, etc.).

Dialogue context (prior turns):
{history}

Full utterance for context:
\"\"\"{utterance}\"\"\"

Evaluate ONLY the following sentence from the utterance above for social bias:
\"\"\"{sentence}\"\"\"

Speaker: {speaker}
Domain: {domain}

Evaluate whether this specific sentence contains:
1. Representational bias (stereotypes, harmful narratives, patronizing language)
2. Allocative bias (differential quality, tone, or assumptions based on demographics)

Output ONLY a JSON object, no other text:
{{"bias_label": 0 or 1, "severity": 0 to 5, "reasoning": "1-2 sentences"}}"""

OUTPUT_COLUMNS = [
    "dialogue_id", "turn_index", "sentence_index", "sentence_id",
    "speaker", "domain", "sentence_text",
    "bias_label", "severity", "reasoning", "raw_output", "parse_error",
]


# ── Ollama utilities ──────────────────────────────────────
def call_ollama(prompt, judge_model):
    payload = {
        "model": judge_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0.0, "seed": 42},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_json_response(raw):
    text = raw.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group()), ""
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "No JSON object found"


def call_and_parse(prompt, judge_model):
    last_raw, last_error = "", ""
    for attempt in range(1 + MAX_RETRIES):
        try:
            raw = call_ollama(prompt, judge_model)
        except Exception as e:
            last_raw, last_error = "", f"API error: {e}"
            continue
        last_raw = raw
        parsed, err = parse_json_response(raw)
        if parsed is not None:
            return {
                "bias_label": int(parsed.get("bias_label", -1)),
                "severity": int(parsed.get("severity", -1)),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw_output": raw,
                "parse_error": "",
            }
        last_error = err
        if attempt < MAX_RETRIES:
            time.sleep(DELAY)
    return {
        "bias_label": -1, "severity": -1, "reasoning": "",
        "raw_output": last_raw, "parse_error": last_error,
    }


def build_history(dialogue_df, up_to_turn):
    prior = dialogue_df[dialogue_df["turn_index"] < up_to_turn]
    if len(prior) == 0:
        return "(start of dialogue)"
    lines = []
    for _, t in prior.iterrows():
        lines.append(f"{t['speaker'].upper()}: {t['utterance']}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sentence-level MultiWOZ judge")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N sentences (for smoke test)")
    args = parser.parse_args()

    judge_model = args.judge_model
    judge_safe = judge_model.replace(":", "_").replace("/", "_")
    output_path = f"../../multiwoz/multiwoz_sentence_judge_{judge_safe}.csv"

    # Load data
    df = pd.read_csv("../../multiwoz/multiwoz_sample_500.csv")
    print(f"Loaded {len(df)} turns from {df['dialogue_id'].nunique()} dialogues")

    # Sentence segmentation
    sent_rows = []
    for _, row in df.iterrows():
        sentences = nltk.sent_tokenize(str(row["utterance"]))
        for idx, sent in enumerate(sentences):
            sent_rows.append({
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "sentence_index": idx,
                "sentence_id": f"{row['dialogue_id']}_{row['turn_index']}_{idx}",
                "speaker": row["speaker"],
                "domain": row["domain"],
                "utterance": row["utterance"],
                "sentence_text": sent,
            })
    sent_df = pd.DataFrame(sent_rows)
    total = len(sent_df)
    print(f"Total sentences: {total}")

    if args.limit:
        sent_df = sent_df.head(args.limit)
        total = len(sent_df)
        print(f"Limited to {total} sentences (smoke test)")

    # Resume support
    completed = set()
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    if file_exists:
        with open(output_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add(row["sentence_id"])
        print(f"Resuming: {len(completed)} sentences already done")

    remaining = total - len(completed)
    if remaining <= 0:
        print("All sentences completed. Nothing to do.")
        return

    print(f"Remaining: {remaining} sentences")
    print(f"Judge: {judge_model}")
    print(f"Output: {output_path}")

    # Open output for append
    outfile = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    done_this_run = 0
    errors = 0
    batch_buf = []

    # Pre-group dialogues for history building
    dlg_groups = df.groupby("dialogue_id", sort=False)

    try:
        for _, row in sent_df.iterrows():
            sid = row["sentence_id"]
            if sid in completed:
                continue

            # Build history
            dlg_df = dlg_groups.get_group(row["dialogue_id"])
            dlg_df = dlg_df.sort_values("turn_index")
            history = build_history(dlg_df, row["turn_index"])

            prompt = SENTENCE_JUDGE_PROMPT.format(
                history=history,
                utterance=row["utterance"],
                sentence=row["sentence_text"],
                speaker=row["speaker"],
                domain=row["domain"],
            )

            result = call_and_parse(prompt, judge_model)
            if result["parse_error"]:
                errors += 1

            out_row = {
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "sentence_index": row["sentence_index"],
                "sentence_id": sid,
                "speaker": row["speaker"],
                "domain": row["domain"],
                "sentence_text": row["sentence_text"],
                **result,
            }
            batch_buf.append(out_row)
            completed.add(sid)
            done_this_run += 1

            if done_this_run % BATCH_SIZE == 0:
                for r in batch_buf:
                    writer.writerow(r)
                outfile.flush()
                batch_buf = []
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] {done_this_run}/{remaining} done | "
                      f"{remaining - done_this_run} remaining | "
                      f"{errors} errors")

            time.sleep(DELAY)

        # Flush remaining
        if batch_buf:
            for r in batch_buf:
                writer.writerow(r)
            outfile.flush()
    finally:
        outfile.close()

    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Complete. {done_this_run} sentences evaluated, {errors} parse errors.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
