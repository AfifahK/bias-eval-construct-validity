"""Run locally. Adds stored v2 labels for the 40 sanity rows to sanity_expected.csv
(columns expected_label_v2__<tag>, expected_severity_v2__<tag>). judge_input.csv is not touched.
Stored v2 files: declarative Gemma 4.3B / Llama / Mistral (no Gemma 1B v2); operational all four.
Rows/judges without a stored v2 label stay empty and are skipped by compare_sanity.py."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
exp = pd.read_csv(OUT / "sanity_expected.csv")
exp = exp[[c for c in exp.columns if "_v2__" not in c]]
DECL_V2 = {"gemma3:latest": "llm_judge_scores_v2", "llama3.1:latest": "llm_judge_scores_llama3_1_v2",
           "mistral:latest": "llm_judge_scores_mistral_v2"}
k = ["prompt_id", "model", "scale"]
for tag in ["gemma3:latest", "gemma3:1b", "llama3.1:latest", "mistral:latest"]:
    safe = tag.replace(":", "_")
    L, S = f"expected_label_v2__{safe}", f"expected_severity_v2__{safe}"
    parts = []
    if tag in DECL_V2:
        s = pd.read_csv(ROOT / f"scores/{DECL_V2[tag]}.csv"); s = s[s.step == "explanation"]
        parts.append(s.rename(columns={"bias_label": L, "severity": S})[k + [L, S]].assign(dataset="declarative"))
    s2 = pd.read_csv(ROOT / f"hpc/operational/evaluations/judge_operational_{safe}_v2.csv")
    parts.append(s2.rename(columns={"scale_direction": "scale", "bias_label": L, "severity": S})[k + [L, S]]
                 .assign(dataset="operational"))
    exp = exp.merge(pd.concat(parts), on=["dataset"] + k, how="left")
exp.to_csv(OUT / "sanity_expected.csv", index=False)
print(exp.filter(like="_v2__").notna().sum().to_string())
