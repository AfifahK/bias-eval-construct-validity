"""
HPC-adapted sentence-level MultiWOZ judge evaluation.
Chunk-aware, resumable, single-judge-per-run.
Preserves exact prompt template structure and output schema.
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests


# ── Sentence-level judge prompt ───────────────────────────
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


# ── Configuration ─────────────────────────────────────────
MAX_RETRIES = 2
DELAY = 0.5

OUTPUT_COLUMNS = [
    "dialogue_id", "turn_index", "sentence_index", "sentence_id",
    "speaker", "domain", "sentence_text",
    "bias_label", "severity", "reasoning", "raw_output", "parse_error",
]


# ── Ollama utilities ──────────────────────────────────────
def call_ollama(prompt, judge_model, ollama_url):
    payload = {
        "model": judge_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0.0, "seed": 42},
    }
    resp = requests.post(ollama_url, json=payload, timeout=120)
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


def call_and_parse(prompt, judge_model, ollama_url):
    last_raw, last_error = "", ""
    for attempt in range(1 + MAX_RETRIES):
        try:
            raw = call_ollama(prompt, judge_model, ollama_url)
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


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HPC sentence-level MultiWOZ judge")
    parser.add_argument("--input", required=True, help="Sentence chunk CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    ollama_url = f"{args.ollama_host}/api/generate"

    # Load input chunk (pre-segmented sentences with history)
    df = pd.read_csv(args.input)
    total = len(df)
    print(f"Input: {total} sentences from {args.input}")
    print(f"Judge: {args.judge_model} | Batch: {args.batch_size}")

    # Resume support
    completed = set()
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    if file_exists:
        with open(args.output, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add(row["sentence_id"])
        print(f"Resuming: {len(completed)} sentences already done")

    remaining = total - len(completed)
    if remaining <= 0:
        print("All sentences completed. Nothing to do.")
        return

    print(f"Remaining: {remaining} sentences")

    # Open output for append
    outfile = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    done_this_run = 0
    errors = 0
    batch_buf = []

    try:
        for _, row in df.iterrows():
            sid = row["sentence_id"]
            if sid in completed:
                continue

            prompt = SENTENCE_JUDGE_PROMPT.format(
                history=row["history"],
                utterance=row["utterance"],
                sentence=row["sentence_text"],
                speaker=row["speaker"],
                domain=row["domain"],
            )

            result = call_and_parse(prompt, args.judge_model, ollama_url)
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

            if done_this_run % args.batch_size == 0:
                for r in batch_buf:
                    writer.writerow(r)
                outfile.flush()
                batch_buf = []
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] {done_this_run}/{remaining} done | "
                      f"{remaining - done_this_run} remaining | {errors} errors")

            time.sleep(DELAY)

        if batch_buf:
            for r in batch_buf:
                writer.writerow(r)
            outfile.flush()
    finally:
        outfile.close()

    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Complete. {done_this_run} sentences, {errors} parse errors.")


if __name__ == "__main__":
    main()
