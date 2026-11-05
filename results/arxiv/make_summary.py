"""Build results/arxiv/refusal_scoring_summary.md from refusal_scores.csv + existing scores.

Scored-response flags come from existing files (not modified), restricted to the 293 scored
declarative explanation rows (non-refusal). Operational scored flags come from
hpc/operational/evaluations (v1, response-level) joined to operational_non_refusals.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "arxiv"
ref = pd.read_csv(OUT / "refusal_scores.csv")

INSTR = {  # name -> (flag col in refusal_scores, severity/score col)
    "Bias-Lexicon": ("bias_lexicon_flag", "bias_lexicon_count"),
    "HurtLex": ("hurtlex_flag", "hurtlex_count"),
    "toxic-bert >0.5": ("toxicity_flag", "toxicity_score"),
    "Gemma 3 4.3B": ("gemma3_4b_flag", "gemma3_4b_severity"),
    "Gemma 3 1B": ("gemma3_1b_flag", "gemma3_1b_severity"),
    "Llama 3.1 8B": ("llama31_8b_flag", "llama31_8b_severity"),
    "Mistral 7B": ("mistral_7b_flag", "mistral_7b_severity"),
}
NK, US = "North Korean", "American"
key = ["prompt_id", "model", "scale"]


def decl_scored():
    d = pd.read_csv(ROOT / "data/all_model_responses_clean.csv")
    ex = d[(d.step == "explanation") & ~d.is_refusal][key + ["nationality"]]
    out = {}
    s = pd.read_csv(ROOT / "scores/dict_scores.csv"); s = s[s.step == "explanation"]
    out["Bias-Lexicon"] = ex.merge(s[key + ["bias_label"]], on=key).rename(columns={"bias_label": "flag"})
    s = pd.read_csv(ROOT / "scores/hurtlex_scores.csv")
    out["HurtLex"] = ex.merge(s[key + ["bias_label"]], on=key).rename(columns={"bias_label": "flag"})
    s = pd.read_csv(ROOT / "scores/toxicity_scores.csv")
    out["toxic-bert >0.5"] = ex.merge(s[key + ["toxicity_label"]], on=key).rename(columns={"toxicity_label": "flag"})
    for name, f in [("Gemma 3 4.3B", "llm_judge_scores"), ("Gemma 3 1B", "llm_judge_scores_1b"),
                    ("Llama 3.1 8B", "llm_judge_scores_llama"), ("Mistral 7B", "llm_judge_scores_mistral")]:
        s = pd.read_csv(ROOT / f"scores/{f}.csv"); s = s[s.step == "explanation"]
        m = ex.merge(s[key + ["bias_label"]], on=key).rename(columns={"bias_label": "flag"})
        out[name] = m[m.flag >= 0]
    return out


def op_scored():
    nr = pd.read_csv(ROOT / "hpc/operational/responses/operational_non_refusals.csv")
    nr = nr.rename(columns={"scale_direction": "scale"})[key + ["nationality"]]
    ev = ROOT / "hpc/operational/evaluations"
    out = {}
    dd = pd.read_csv(ev / "dictionary_operational.csv").rename(columns={"scale_direction": "scale"})
    for name, col in [("Bias-Lexicon", "bias_label"), ("HurtLex", "hurtlex_label"), ("toxic-bert >0.5", "toxicity_label")]:
        out[name] = nr.merge(dd[key + [col]], on=key).rename(columns={col: "flag"})
    for name, tag in [("Gemma 3 4.3B", "gemma3_latest"), ("Gemma 3 1B", "gemma3_1b"),
                      ("Llama 3.1 8B", "llama3.1_latest"), ("Mistral 7B", "mistral_latest")]:
        s = pd.read_csv(ev / f"judge_operational_{tag}_v1.csv").rename(columns={"scale_direction": "scale"})
        m = nr.merge(s[key + ["bias_label"]], on=key).rename(columns={"bias_label": "flag"})
        out[name] = m[m.flag >= 0]
    return out


def rate(df, nat):
    x = df[df.nationality == nat].flag
    return x.mean() if len(x) else np.nan


def ratio(df):
    u = rate(df, US)
    return rate(df, NK) / u if u and not np.isnan(u) else np.nan


def f(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def tables(dataset, scored):
    r = ref[ref.dataset == dataset]
    lines = ["| Instrument | Refusals n | Flag rate (refusals) | Mean score (refusals) | "
             "Scored n (US/NK) | NK/US scored only | Scored+refusals n (US/NK) | NK/US scored+refusals |",
             "|---|---|---|---|---|---|---|---|"]
    for name, (fc, sc) in INSTR.items():
        rr = r[r[fc] >= 0]
        sdf = scored[name]
        rf = rr[["nationality", fc]].rename(columns={fc: "flag"})
        comb = pd.concat([sdf[["nationality", "flag"]], rf])
        n = lambda d: f"{(d.nationality == US).sum()}/{(d.nationality == NK).sum()}"
        lines.append(f"| {name} | {len(rr)} | {f(rr[fc].mean())} | {f(rr[sc].mean())} | {n(sdf)} | "
                     f"{f(ratio(sdf), 2)} | {n(comb)} | {f(ratio(comb), 2)} |")
    return "\n".join(lines)


def refusal_nat_rates(dataset):
    r = ref[ref.dataset == dataset]
    lines = ["| Instrument | US flag rate (refusals) | NK flag rate (refusals) |", "|---|---|---|"]
    for name, (fc, _) in INSTR.items():
        rr = r[r[fc] >= 0]
        lines.append(f"| {name} | {f(rr[rr.nationality == US][fc].mean())} (n={(rr.nationality == US).sum()}) | "
                     f"{f(rr[rr.nationality == NK][fc].mean())} (n={(rr.nationality == NK).sum()}) |")
    return "\n".join(lines)


NOP = len(pd.read_csv(ROOT / 'hpc/operational/responses/operational_non_refusals.csv'))
perr = {k: int((ref[f"{p}_flag"] < 0).sum()) for k, p in
        [("Gemma 3 4.3B", "gemma3_4b"), ("Gemma 3 1B", "gemma3_1b"),
         ("Llama 3.1 8B", "llama31_8b"), ("Mistral 7B", "mistral_7b")]}

md = f"""# Refusal scoring summary

Instruments run on the 138 refusals (67 declarative, 71 operational). Judges use prompt **v1**
(includes the `Model evaluated: {{model}}` identity line), response level, T=0, seed 42,
`num_predict` 300. v1 response-level is the version/level that reproduces the paper's section 3.2
NK/US ratios on scored declarative responses (Bias-Lexicon 2.69, Llama 2.41, Mistral 5.11).

Definitions. Flag rate = share of refusals with flag=1. Mean score = mean raw count for
Bias-Lexicon/HurtLex, mean probability for toxic-bert, mean severity (0-5) for judges. NK/US =
(NK flag rate)/(US flag rate). "Scored only" = non-refusal responses from the existing results
files (declarative: the 293 explanation rows; operational: the {NOP} operational non-refusals
joined from `hpc/operational/evaluations`, v1). "Scored + refusals" appends the refusals scored
here as additional rows. Rows where a judge output could not be parsed (flag -1) are dropped
from that judge's rates; parse failures on refusals: {perr}.

## Declarative (67 refusals)

{tables("declarative", decl_scored())}

Flag rate on refusals by nationality:

{refusal_nat_rates("declarative")}

## Operational (71 refusals)

{tables("operational", op_scored())}

Flag rate on refusals by nationality:

{refusal_nat_rates("operational")}

## Method notes

- Prompt version per dataset: declarative judges use `src/llm_judge/llm_judge.py::judge_row(..., "v1")`
  on the raw `response`; operational judges use `hpc/operational/scripts/run_judges.py::judge_row(..., "v1")`
  (that script's default) on `response_clean` with `mental_health_status` as the condition, which is how
  the existing `judge_operational_*_v1.csv` files were produced (the HPC run executed both v1 and v2).
- Dictionaries: `src/dictionary` functions for both datasets. Re-scoring 40 existing rows per dataset
  reproduced the stored Bias-Lexicon and HurtLex flags 40/40. HurtLex 1.2 TSV read from the private repo copy.
- toxic-bert: declarative uses the transformers `unitary/toxic-bert` pipeline on `response` (as in
  `toxicity_analysis.py`); operational uses `Detoxify("original")` on `response_clean` (as in
  `run_dictionary.py`). Re-scoring existing rows reproduced stored scores to <1.5e-9. Those two scripts
  execute on import, so the call was reproduced rather than imported.
- Judge reproducibility: local Ollama output is not byte-identical to the stored HPC outputs. On 6
  operational rows, Gemma 3 4.3B labels matched the stored v1 labels 6/6 and Llama 3.1 matched 3/6
  (the stored v2 labels also matched 3/6 on the same rows); on 4 declarative rows Llama matched 4/4.
  Judge raw outputs and rationales are saved verbatim in `refusal_scores.csv`.
- Judge models: `gemma3:latest` (4.3B), `gemma3:1b`, `llama3.1:latest`, `mistral:latest`, all present locally.
"""
(OUT / "refusal_scoring_summary.md").write_text(md)
print(md)
