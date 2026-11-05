"""Judge the 158 refusals + 40-row sanity slice with 4 judges (prompt v1 and v2, response level only,
T=0, seed 42 -- both hardcoded in the imported call_ollama functions). v1 = with the "Model evaluated"
identity line, v2 = identity line removed. v1 is run for all judges first, then v2.

Uses the unmodified judge code copied into src/:
  src/decl_llm_judge.py  (from src/llm_judge/llm_judge.py)         -> declarative rows
  src/op_run_judges.py   (from hpc/operational/scripts/run_judges.py) -> operational rows
--ollama-host is the BASE url (http://127.0.0.1:PORT); /api/generate is appended exactly once.
Output (existing judge-score formats, one file per slice x dataset x judge):
  results/{refusal,sanity}_{declarative,operational}_<tag>_{v1,v2}.csv
Resumable: rows already present in an output file are skipped.
"""
import argparse
import csv
import importlib.util
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGES = ["gemma3:latest", "gemma3:1b", "llama3.1:latest", "mistral:latest"]
VERSIONS = ["v1", "v2"]


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "src", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434",
                    help="base URL, no /api/generate")
    ap.add_argument("--input", default=os.path.join(HERE, "judge_input.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--versions", default=",".join(VERSIONS), help="prompt versions, run in this order")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per slice+dataset (smoke test)")
    a = ap.parse_args()
    base = a.ollama_host.rstrip("/")
    if base.endswith("/api/generate"):
        raise SystemExit("--ollama-host must be the base URL (no /api/generate)")

    decl = load("decl_judge", "decl_llm_judge.py")
    op = load("op_judge", "op_run_judges.py")
    decl.OLLAMA_URL = base + "/api/generate"   # decl call_ollama reads this module global

    os.makedirs(a.outdir, exist_ok=True)
    with open(a.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault((r["slice"], r["dataset"]), []).append(r)
    if a.limit:
        groups = {k: v[:a.limit] for k, v in groups.items()}

    for VERSION in a.versions.split(","):
        for tag in a.judges.split(","):
            safe = tag.replace(":", "_").replace("/", "_")
            for (sl, ds), grp in sorted(groups.items()):
                path = os.path.join(a.outdir, f"{sl}_{ds}_{safe}_{VERSION}.csv")
                cols = decl.OUTPUT_COLUMNS if ds == "declarative" else op.OUTPUT_COLUMNS
                scale_col = "scale" if ds == "declarative" else "scale_direction"
                done = set()
                if os.path.exists(path):
                    with open(path, newline="", encoding="utf-8") as f:
                        done = {(x["prompt_id"], x["model"], x[scale_col]) for x in csv.DictReader(f)}
                new = not os.path.exists(path) or os.path.getsize(path) == 0
                with open(path, "a", newline="", encoding="utf-8") as out:
                    w = csv.DictWriter(out, fieldnames=cols)
                    if new:
                        w.writeheader()
                    for i, r in enumerate(grp, 1):
                        if (r["prompt_id"], r["model"], r["scale"]) in done:
                            continue
                        if ds == "declarative":
                            row = dict(r, response=r["judge_text"])
                            res, _ = decl.judge_row(row, tag, VERSION)
                            rec = {"prompt_id": r["prompt_id"], "nationality": r["nationality"],
                                   "disability": r["disability"], "disability_type": r["disability_type"],
                                   "scenario_id": r["scenario_id"], "scenario": r["scenario"],
                                   "model": r["model"], "scale": r["scale"], "step": r["step"], **res}
                        else:
                            row = dict(r, response=r["judge_text"], response_clean=r["judge_text"],
                                       scale_direction=r["scale"])
                            res = op.judge_row(row, tag, base, VERSION)
                            rec = {"prompt_id": r["prompt_id"], "nationality": r["nationality"],
                                   "mental_health_status": r["mental_health_status"],
                                   "mental_health_state": r["mental_health_state"],
                                   "scenario": r["scenario"], "model": r["model"],
                                   "scale_direction": r["scale"], **res}
                        w.writerow(rec)
                        out.flush()
                        print(f"[{VERSION} {tag}] {sl}/{ds} {i}/{len(grp)} pid={r['prompt_id']} "
                              f"label={res['bias_label']} err={res['parse_error'][:40]!r}", flush=True)
                        time.sleep(0.25)
    print("ALL DONE")


if __name__ == "__main__":
    main()
