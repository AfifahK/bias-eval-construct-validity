"""
Generate LLM responses for operational prompts via Ollama.

Reads ../prompts/operational_prompts.csv, queries three models, and writes
per-model and combined CSVs to ../responses/.

Resumable: already-completed (prompt_id, model, scale_direction) tuples are
skipped on restart.
"""

import argparse
import csv
import os
import time

import requests

# ── Configuration ────────────────────────────────────────

MODELS = ["gemma3", "llama3.1", "mistral"]
DELAY = 0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "..", "prompts", "operational_prompts.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "responses")

OUTPUT_COLUMNS = [
    "prompt_id",
    "nationality",
    "mental_health_status",
    "mental_health_state",
    "scenario",
    "model",
    "scale_direction",
    "prompt_text",
    "response",
]


# ── Ollama call ──────────────────────────────────────────

def call_ollama(model, prompt, ollama_url):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 300,
            "temperature": 1.0,
            "num_ctx": 32768,
        },
    }
    resp = requests.post(ollama_url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


# ── Resume support ───────────────────────────────────────

def load_completed(output_path):
    """Return set of (prompt_id, model, scale_direction) already completed."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["prompt_id"], row["model"], row["scale_direction"])
            completed.add(key)
    return completed


# ── File writers ─────────────────────────────────────────

def open_csv_append(path):
    """Open a CSV for appending; write header if the file is new/empty."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
    if not file_exists:
        writer.writeheader()
    return fh, writer


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate operational responses via Ollama"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint URL (default: http://localhost:11434/api/generate)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N prompts (for testing)",
    )
    args = parser.parse_args()

    ollama_url = args.ollama_host
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load prompts
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))

    if args.limit is not None:
        prompts = prompts[: args.limit]

    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompt rows from {INPUT_CSV}")

    # Per-model and combined output paths
    combined_path = os.path.join(OUTPUT_DIR, "operational_combined.csv")

    # Load combined resume state
    combined_completed = load_completed(combined_path)
    print(f"Found {len(combined_completed)} completed tasks in combined CSV")

    # Per-model resume states and writers
    model_completed = {}
    model_writers = {}
    model_files = {}

    for model in MODELS:
        model_path = os.path.join(OUTPUT_DIR, f"operational_{model}.csv")
        model_completed[model] = load_completed(model_path)
        fh, writer = open_csv_append(model_path)
        model_files[model] = fh
        model_writers[model] = writer

    combined_fh, combined_writer = open_csv_append(combined_path)

    total_tasks = total_prompts * len(MODELS)
    done = sum(len(v) for v in model_completed.values())
    print(f"Total tasks: {total_tasks} | Already done: {done}")

    try:
        for prompt_row in prompts:
            pid = prompt_row["prompt_id"]
            direction = prompt_row["scale_direction"]
            prompt_text = prompt_row["prompt_text"]

            for model in MODELS:
                key = (pid, model, direction)

                # Skip if already in per-model output
                if key in model_completed[model]:
                    continue

                done += 1
                print(
                    f"[{done}/{total_tasks}] prompt_id={pid} model={model} "
                    f"direction={direction}"
                )

                try:
                    response = call_ollama(model, prompt_text, ollama_url)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    response = f"ERROR: {e}"

                out_row = {
                    "prompt_id": pid,
                    "nationality": prompt_row["nationality"],
                    "mental_health_status": prompt_row["mental_health_status"],
                    "mental_health_state": prompt_row["mental_health_state"],
                    "scenario": prompt_row["scenario"],
                    "model": model,
                    "scale_direction": direction,
                    "prompt_text": prompt_text,
                    "response": response,
                }

                # Write to per-model CSV
                model_writers[model].writerow(out_row)
                model_files[model].flush()

                # Write to combined CSV if not already there
                if key not in combined_completed:
                    combined_writer.writerow(out_row)
                    combined_fh.flush()
                    combined_completed.add(key)

                model_completed[model].add(key)
                time.sleep(DELAY)

    finally:
        for fh in model_files.values():
            fh.close()
        combined_fh.close()

    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
