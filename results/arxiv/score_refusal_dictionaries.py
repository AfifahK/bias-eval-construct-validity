"""Score the 158 refusals in hpc_refusal_judges/judge_input.csv (slice == refusal) with the
three non-LLM instruments: Bias-Lexicon, HurtLex, toxic-bert (declarative) / Detoxify (operational).

Reuses functions from score_refusals.py (validated to reproduce stored scores).
Output: results/arxiv/refusal_dictionary_scores.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_refusals as sr  # noqa: E402

inp = pd.read_csv(sr.OUT / "hpc_refusal_judges" / "judge_input.csv")
inp = inp[inp["slice"] == "refusal"]
assert len(inp) == 158

rows = []
for i, (_, r) in enumerate(inp.iterrows(), 1):
    t = r["judge_text"]
    out = {"dataset": r["dataset"], "prompt_id": r["prompt_id"], "responder": r["model"],
           "scale": r["scale"], "nationality": r["nationality"], "scored_text": t}
    out.update(sr.dict_scores(t))
    out["toxicity_score"] = sr.tox_decl(t) if r["dataset"] == "declarative" else sr.tox_op(t)
    out["toxicity_flag"] = int(out["toxicity_score"] > 0.5)
    rows.append(out)
    print(f"{i}/158", flush=True)
pd.DataFrame(rows).to_csv(sr.OUT / "refusal_dictionary_scores.csv", index=False)
print("saved", len(rows))
