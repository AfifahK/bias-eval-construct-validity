import argparse
import csv
import requests
import time
import os

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["gemma3", "llama3.1", "mistral"]
INPUT_CSV = "../../data/prompts_dataset.csv"
OUTPUT_CSV = "../../data/all_model_responses.csv"
DELAY = 0.5

OUTPUT_COLUMNS = [
    "prompt_id", "nationality", "disability", "disability_type",
    "scenario_id", "scenario", "model", "scale", "prompt_text",
    "response", "step",
]

STEPS = {
    "likert": {"num_predict": 5, "temperature": 1.0, "suffix": ""},
    "explanation": {"num_predict": 300, "temperature": 1.0, "suffix": " Please explain why."},
}


def call_ollama(model, prompt, num_predict, temperature):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["response"]


def load_completed(output_path):
    """Load already-completed (prompt_id, model, scale, step) tuples."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["prompt_id"], row["model"], row["scale"], row["step"])
            completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser(description="Generate responses via Ollama")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N prompt_ids")
    args = parser.parse_args()

    # Load prompts
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))

    if args.limit is not None:
        prompts = prompts[:args.limit]

    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts from {INPUT_CSV}")

    # Load resume state
    completed = load_completed(OUTPUT_CSV)
    print(f"Found {len(completed)} completed tasks — will skip those")

    # Total work: prompts × models × scales × steps
    total_tasks = total_prompts * len(MODELS) * 2 * len(STEPS)
    done = len(completed)

    # Open output in append mode
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    outfile = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    try:
        for i, prompt_row in enumerate(prompts):
            pid = prompt_row["prompt_id"]
            for model in MODELS:
                for scale, prompt_col in [
                    ("regular", "prompt_text_regular"),
                    ("flipped", "prompt_text_flipped"),
                ]:
                    base_text = prompt_row[prompt_col]
                    for step_name, step_cfg in STEPS.items():
                        key = (pid, model, scale, step_name)
                        if key in completed:
                            continue

                        prompt_text = base_text + step_cfg["suffix"]
                        done += 1
                        print(
                            f"[{done}/{total_tasks}] prompt_id={pid} model={model} "
                            f"scale={scale} step={step_name}"
                        )

                        try:
                            response = call_ollama(
                                model,
                                prompt_text,
                                step_cfg["num_predict"],
                                step_cfg["temperature"],
                            )
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            response = f"ERROR: {e}"

                        writer.writerow({
                            "prompt_id": pid,
                            "nationality": prompt_row["nationality"],
                            "disability": prompt_row["disability"],
                            "disability_type": prompt_row["disability_type"],
                            "scenario_id": prompt_row["scenario_id"],
                            "scenario": prompt_row["scenario"],
                            "model": model,
                            "scale": scale,
                            "prompt_text": prompt_text,
                            "response": response,
                            "step": step_name,
                        })
                        outfile.flush()

                        time.sleep(DELAY)
    finally:
        outfile.close()

    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
