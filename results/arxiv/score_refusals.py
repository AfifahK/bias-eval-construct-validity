"""Score the 138 refusals (67 declarative, 71 operational) with all 7 instruments.

Usage (from repo root):
  /usr/bin/python3 results/arxiv/score_refusals.py validate   # re-score a few existing rows, compare
  /usr/bin/python3 results/arxiv/score_refusals.py score      # score refusals -> results/arxiv/refusal_scores.csv

Existing functions are imported, not copied:
  declarative dictionaries  src.dictionary.lexicon_analysis / hurtlex_analysis
  declarative judges        src.llm_judge.llm_judge.judge_row (prompt v1, response-level)
  operational judges        hpc/operational/scripts/run_judges.py judge_row (prompt v1)
hpc/operational/scripts/run_dictionary.py and src/other_methods/toxicity_analysis.py run at
import time (and overwrite existing results), so they cannot be imported; operational
dictionaries use the src/ functions (equivalence checked in `validate`) and toxic-bert uses the
same call each dataset's original script made.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "arxiv"
CACHE = OUT / "judge_cache.jsonl"
HURTLEX_TSV = os.environ.get("HURTLEX_TSV")
if not HURTLEX_TSV:
    raise SystemExit("Set HURTLEX_TSV=/path/to/hurtlex_EN.tsv (HurtLex EN 1.2 lexicon, not included in this repo).")

from src.dictionary import lexicon_analysis as lex          # noqa: E402
from src.dictionary import hurtlex_analysis as hl           # noqa: E402
from src.llm_judge import llm_judge as decl_judge           # noqa: E402

spec = importlib.util.spec_from_file_location(
    "op_run_judges", ROOT / "hpc" / "operational" / "scripts" / "run_judges.py")
op_judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op_judge)

hl._load_hurtlex(HURTLEX_TSV)

JUDGES = {  # column prefix -> ollama tag
    "gemma3_4b": "gemma3:latest",
    "gemma3_1b": "gemma3:1b",
    "llama31_8b": "llama3.1:latest",
    "mistral_7b": "mistral:latest",
}
PROMPT_VERSION = "v1"
OLLAMA_HOST = "http://localhost:11434"

DECL_CSV = ROOT / "data" / "all_model_responses_clean.csv"
OP_REFUSALS = ROOT / "hpc" / "operational" / "responses" / "operational_refusals.csv"
OP_NONREF = ROOT / "hpc" / "operational" / "responses" / "operational_non_refusals.csv"


# ---------- toxic-bert (same call as each dataset's original script) ----------
_pipe = _detox = None


def tox_decl(text):
    """src/other_methods/toxicity_analysis.py: transformers pipeline on raw response."""
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text-classification", model="unitary/toxic-bert",
                         truncation=True, max_length=512)
    r = _pipe([str(text)], batch_size=16)[0]
    return r["score"] if r["label"] == "toxic" else 1 - r["score"]


def tox_op(text):
    """hpc/operational/scripts/run_dictionary.py: Detoxify('original') on response_clean."""
    global _detox
    if _detox is None:
        from detoxify import Detoxify
        _detox = Detoxify("original")
    return round(_detox.predict(str(text))["toxicity"], 4)


# ---------- dictionaries ----------
def dict_scores(text):
    b = lex.score_text(text)
    h = hl.score_text(text)
    return {"bias_lexicon_count": b["bias_count"], "bias_lexicon_norm": b["bias_score_norm"],
            "bias_lexicon_flag": b["bias_label"],
            "bias_lexicon_terms": ", ".join(b["matched_terms"]),
            "hurtlex_count": h["bias_count"], "hurtlex_norm": h["bias_score_norm"],
            "hurtlex_flag": h["bias_label"],
            "hurtlex_terms": ", ".join(t for t, _ in h["matched_terms"])}


# ---------- judges ----------
def cache_load():
    c = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            c[tuple(r["key"])] = r["result"]
    return c


def run_judge(dataset, row, tag, cache):
    key = (dataset, str(row["prompt_id"]), row["model"], row["scale"], tag)
    if key in cache:
        return cache[key]
    if dataset == "declarative":
        res, _ = decl_judge.judge_row(row, tag, PROMPT_VERSION)
    else:
        res = op_judge.judge_row(row, tag, OLLAMA_HOST, PROMPT_VERSION)
    cache[key] = res
    with open(CACHE, "a") as f:
        f.write(json.dumps({"key": list(key), "result": res}) + "\n")
    time.sleep(decl_judge.DELAY)
    return res


# ---------- data ----------
def load_refusals():
    d = pd.read_csv(DECL_CSV)
    dec = d[(d.step == "explanation") & d.is_refusal].copy()
    dec["dataset"] = "declarative"
    dec["condition"] = dec["disability"]
    dec["text_for_dict"] = dec["response"]  # main pipeline scored `response`
    op = pd.read_csv(OP_REFUSALS)
    op["dataset"] = "operational"
    op = op.rename(columns={"scale_direction": "scale"})
    op["scale"] = op["scale"]
    op["condition"] = op["mental_health_state"] + " " + op["mental_health_status"]
    op["text_for_dict"] = op["response_clean"]
    assert len(dec) == 67 and len(op) == 71
    return dec, op


def op_row_for_judge(r):
    """run_judges.judge_row expects the original operational column names."""
    r = r.copy()
    r["scale_direction"] = r["scale"]
    return r


def main_score():
    dec, op = load_refusals()
    cache = cache_load()
    rows = []
    for ds, df in (("declarative", dec), ("operational", op)):
        for i, (_, r) in enumerate(df.iterrows(), 1):
            r = r.to_dict()
            out = {"dataset": ds, "prompt_id": r["prompt_id"], "responder": r["model"],
                   "scale": r["scale"], "nationality": r["nationality"],
                   "condition": r["condition"], "scenario": r["scenario"],
                   "response": r["response"], "scored_text": r["text_for_dict"]}
            out.update(dict_scores(r["text_for_dict"]))
            out["toxicity_score"] = tox_decl(r["response"]) if ds == "declarative" \
                else tox_op(r["response_clean"])
            out["toxicity_flag"] = int(out["toxicity_score"] > 0.5)
            jrow = r if ds == "declarative" else op_row_for_judge(r)
            for pre, tag in JUDGES.items():
                res = run_judge(ds, jrow, tag, cache)
                out[f"{pre}_flag"] = res["bias_label"]
                out[f"{pre}_severity"] = res["severity"]
                out[f"{pre}_rationale"] = res["reasoning"]
                out[f"{pre}_raw"] = res["raw_output"]
                out[f"{pre}_parse_error"] = res["parse_error"]
            rows.append(out)
            print(f"[{ds} {i}/{len(df)}] done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "refusal_scores.csv", index=False)
    print("saved refusal_scores.csv", len(rows))


def main_validate():
    cache = {}
    res = []
    # declarative dictionaries / toxicity vs existing scores
    d = pd.read_csv(DECL_CSV)
    e = d[(d.step == "explanation") & ~d.is_refusal].head(40)
    k = ["prompt_id", "model", "scale"]
    ds = pd.read_csv(ROOT / "scores/dict_scores.csv")
    hs = pd.read_csv(ROOT / "scores/hurtlex_scores.csv")
    ts = pd.read_csv(ROOT / "scores/toxicity_scores.csv")
    m_b = m_h = 0
    tdiff = []
    for _, r in e.iterrows():
        s = dict_scores(r["response"])
        ex = ds[(ds.prompt_id == r.prompt_id) & (ds.model == r.model) & (ds.scale == r.scale)
                & (ds.step == "explanation")].iloc[0]
        m_b += int(s["bias_lexicon_flag"] == ex.bias_label)
        eh = hs[(hs.prompt_id == r.prompt_id) & (hs.model == r.model) & (hs.scale == r.scale)]
        m_h += int(len(eh) == 1 and s["hurtlex_flag"] == eh.iloc[0].bias_label)
        et = ts[(ts.prompt_id == r.prompt_id) & (ts.model == r.model) & (ts.scale == r.scale)].iloc[0]
        tdiff.append(abs(tox_decl(r["response"]) - et.toxicity_score))
    print(f"DECL n={len(e)} bias-lexicon flag match {m_b}, hurtlex flag match {m_h}, "
          f"toxic-bert max|diff| {max(tdiff):.2e}")
    # operational dictionaries / toxicity
    nr = pd.read_csv(OP_NONREF).head(40)
    od = pd.read_csv(ROOT / "hpc/operational/evaluations/dictionary_operational.csv")
    mb = mh = 0
    td = []
    for _, r in nr.iterrows():
        s = dict_scores(r["response_clean"])
        ex = od[(od.prompt_id == r.prompt_id) & (od.model == r.model)
                & (od.scale_direction == r.scale_direction)].iloc[0]
        mb += int(s["bias_lexicon_flag"] == ex.bias_label)
        mh += int(s["hurtlex_flag"] == ex.hurtlex_label)
        td.append(abs(tox_op(r["response_clean"]) - ex.toxicity_score))
    print(f"OP n={len(nr)} bias-lexicon match {mb}, hurtlex match {mh}, detoxify max|diff| {max(td):.2e}")
    # judge reproducibility: rerun v1 on 4 operational + 4 declarative existing rows, compare
    nr4 = nr.head(4)
    for pre, tag in (("gemma3_4b", "gemma3:latest"), ("llama31_8b", "llama3.1:latest")):
        safe = tag.replace(":", "_")
        v1 = pd.read_csv(ROOT / f"hpc/operational/evaluations/judge_operational_{safe}_v1.csv")
        v2 = pd.read_csv(ROOT / f"hpc/operational/evaluations/judge_operational_{safe}_v2.csv")
        for _, r in nr4.iterrows():
            row = r.to_dict()
            row["scale"] = row["scale_direction"]
            out = op_judge.judge_row(row, tag, OLLAMA_HOST, "v1")
            sel = lambda df: df[(df.prompt_id == r.prompt_id) & (df.model == r.model)
                                & (df.scale_direction == r.scale_direction)].iloc[0]
            print(f"OP judge {tag} pid={r.prompt_id}: rerun-v1 raw == existing v1 raw: "
                  f"{out['raw_output'] == sel(v1).raw_output} | == existing v2 raw: "
                  f"{out['raw_output'] == sel(v2).raw_output}", flush=True)
    dj = pd.read_csv(ROOT / "scores/llm_judge_scores_llama.csv")
    for _, r in e.head(4).iterrows():
        out, _ = decl_judge.judge_row(r.to_dict(), "llama3.1:latest", "v1")
        ex = dj[(dj.prompt_id == r.prompt_id) & (dj.model == r.model) & (dj.scale == r.scale)
                & (dj.step == "explanation")].iloc[0]
        print(f"DECL judge llama pid={r.prompt_id}: raw match {out['raw_output'] == ex.raw_output}, "
              f"label {out['bias_label']} vs {ex.bias_label}", flush=True)


if __name__ == "__main__":
    {"validate": main_validate, "score": main_score}[sys.argv[1]]()
