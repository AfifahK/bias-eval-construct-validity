"""Run locally. Builds judge_input.csv (158 refusals + 40 sanity rows)."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 42
COLS = ["slice", "dataset", "prompt_id", "nationality", "disability", "disability_type", "scenario_id",
        "scenario", "model", "scale", "step", "mental_health_status", "mental_health_state", "judge_text"]

d = pd.read_csv(ROOT / "data/all_model_responses_clean.csv")
dexp = d[d.step == "explanation"].copy()
dexp["dataset"] = "declarative"
dexp["judge_text"] = dexp["response"]            # declarative main pipeline judged raw `response`
dexp["mental_health_status"] = ""; dexp["mental_health_state"] = ""

# corrected operational split (curly-apostrophe fix): 91 refusals / 269 non-refusals
o_ref = pd.read_csv(ROOT / "results/refusal_audit/operational_refusals_corrected.csv")
o_non = pd.read_csv(ROOT / "results/refusal_audit/operational_non_refusals_corrected.csv")
for o in (o_ref, o_non):
    o["dataset"] = "operational"; o["scale"] = o["scale_direction"]
    o["judge_text"] = o["response_clean"]        # operational judges used response_clean
    o["disability"] = ""; o["disability_type"] = ""; o["scenario_id"] = ""; o["step"] = ""

dec_ref = dexp[dexp.is_refusal]
dec_non = dexp[~dexp.is_refusal]
# declarative stays at the original 67 (the regex fix would add 1 mid-answer false positive)
assert len(dec_ref) == 67 and len(o_ref) == 91

sanity = pd.concat([dec_non.sample(20, random_state=SEED), o_non.sample(20, random_state=SEED)])
parts = []
for name, df in (("refusal", pd.concat([dec_ref, o_ref])), ("sanity", sanity)):
    df = df.copy(); df["slice"] = name; parts.append(df[COLS])
inp = pd.concat(parts, ignore_index=True)
inp.to_csv(OUT / "judge_input.csv", index=False)
print(inp.groupby(["slice", "dataset"]).size())
