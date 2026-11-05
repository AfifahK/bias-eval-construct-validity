"""Mean dictionary/classifier score on refusals vs scored (non-refusal) responses, by dataset x
nationality (+ pooled), with prompt-clustered bootstrap CIs.

Decision 30: refusals = hand-audited 63 declarative + 91 operational (154); the 4 audit false
positives are scored responses (declarative scored = 297, scores computed with the repo functions,
validated to reproduce stored scores). Operational scored = 269 corrected non-refusals. Scores for scored rows are the stored ones (scores/*, hpc/operational/evaluations/);
refusal scores come from refusal_dictionary_scores.csv. Output: refusal_mean_comparison.csv/.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED, B = 42, 10_000
INSTR = {"bias_lexicon_norm": "Bias-Lexicon (norm)", "hurtlex_norm": "HurtLex (norm)",
         "toxicity_score": "toxic-bert / Detoxify"}
NK, US = "North Korean", "American"

sys.path.insert(0, str(HERE))
import refusal_sets  # noqa: E402

ref = pd.read_csv(HERE / "refusal_dictionary_scores.csv").rename(columns={"responder": "model"})
fp_ids = refusal_sets.audit_fp_ids()
is_fp = (ref.dataset == "declarative") & (ref.prompt_id.astype(str) + "_" + ref.model + "_" + ref.scale).isin(fp_ids)
moved = ref[is_fp][["nationality", "prompt_id", "model", "scale", "bias_lexicon_norm", "hurtlex_norm", "toxicity_score"]]
ref = ref[~is_fp]
assert len(moved) == 4 and len(ref) == 154

# ---- scored declarative ----
d = refusal_sets.declarative_explanation()
d = d[~d.is_refusal_regex]      # the 293 regex-scored rows with stored scores; the 4 moved rows added below
k = ["prompt_id", "model", "scale"]
b = pd.read_csv(ROOT / "scores/dict_scores.csv"); b = b[b.step == "explanation"][k + ["bias_score_norm"]]
h = pd.read_csv(ROOT / "scores/hurtlex_scores.csv"); h = h[h.step == "explanation"][k + ["bias_score_norm"]]
t = pd.read_csv(ROOT / "scores/toxicity_scores.csv")[k + ["toxicity_score"]]
for n, x in (("b", b), ("h", h), ("t", t)):
    assert not x.duplicated(k).any(), n
sd = (d[k + ["nationality"]].merge(b.rename(columns={"bias_score_norm": "bias_lexicon_norm"}), on=k, how="left")
      .merge(h.rename(columns={"bias_score_norm": "hurtlex_norm"}), on=k, how="left")
      .merge(t, on=k, how="left"))
sd = pd.concat([sd, moved], ignore_index=True)
sd["dataset"] = "declarative"

# ---- scored operational (corrected non-refusals, stored scores by key) ----
nr = pd.read_csv(HERE / "operational_non_refusals_corrected.csv")
od = pd.read_csv(ROOT / "hpc/operational/evaluations/dictionary_operational.csv")
ko = ["prompt_id", "model", "scale_direction"]
assert not od.duplicated(ko).any()
so = nr[ko + ["nationality"]].merge(
    od[ko + ["bias_score_norm", "hurtlex_norm", "toxicity_score"]].rename(
        columns={"bias_score_norm": "bias_lexicon_norm"}), on=ko, how="left")
so["dataset"] = "operational"
sc = pd.concat([sd, so], ignore_index=True)
assert len(sd) == 297 and len(so) == 269, (len(sd), len(so))
assert sc[list(INSTR)].notna().all().all(), sc[list(INSTR)].isna().sum()

ref["grp"], sc["grp"] = "refusal", "scored"
allr = pd.concat([ref, sc], ignore_index=True)


def boot(df, col, strata, seed=SEED):
    """df: rows with grp, nationality, prompt_id, col. Resample prompts within each stratum
    (prompt = cluster, rows of both groups move together). Returns means + diff CIs."""
    rng = np.random.default_rng(seed)
    sums = {g: np.zeros(B) for g in ("refusal", "scored")}
    cnts = {g: np.zeros(B) for g in ("refusal", "scored")}
    for s in strata:
        x = df[df.nationality == s]
        per = x.pivot_table(index="prompt_id", columns="grp", values=col,
                            aggfunc=["sum", "count"], fill_value=0)
        idx = rng.integers(0, len(per), size=(B, len(per)))
        for g in sums:
            sums[g] += per[("sum", g)].values[idx].sum(1) if ("sum", g) in per else 0
            cnts[g] += per[("count", g)].values[idx].sum(1) if ("count", g) in per else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        m = {g: sums[g] / cnts[g] for g in sums}
    ok = np.isfinite(m["refusal"]) & np.isfinite(m["scored"])
    out = {}
    for name, v in (("refusal", m["refusal"]), ("scored", m["scored"]),
                    ("diff", m["refusal"] - m["scored"])):
        out[name] = np.percentile(v[ok], [2.5, 97.5])
    out["dropped"] = int((~ok).sum())
    return out


rows = []
for ds in ("declarative", "operational"):
    for nat in (US, NK, "pooled"):
        sub = allr[allr.dataset == ds]
        strata = [US, NK] if nat == "pooled" else [nat]
        sub = sub[sub.nationality.isin(strata)]
        for col, label in INSTR.items():
            mr = sub[sub.grp == "refusal"][col].mean()
            ms = sub[sub.grp == "scored"][col].mean()
            bt = boot(sub, col, strata)
            rows.append({"dataset": ds, "nationality": nat, "instrument": label,
                         "n_refusal": int((sub.grp == "refusal").sum()),
                         "n_scored": int((sub.grp == "scored").sum()),
                         "mean_refusal": mr, "refusal_lo": bt["refusal"][0], "refusal_hi": bt["refusal"][1],
                         "mean_scored": ms, "scored_lo": bt["scored"][0], "scored_hi": bt["scored"][1],
                         "diff": mr - ms, "diff_lo": bt["diff"][0], "diff_hi": bt["diff"][1],
                         "dropped_resamples": bt["dropped"]})
res = pd.DataFrame(rows)
res.to_csv(HERE / "refusal_mean_comparison.csv", index=False)

f = lambda v: f"{v:.4f}"
lines = ["| dataset | nationality | instrument | n ref / scored | mean refusals [95% CI] | mean scored [95% CI] | diff (ref - scored) [95% CI] |",
         "|---|---|---|---|---|---|---|"]
for r in res.itertuples():
    lines.append(f"| {r.dataset} | {r.nationality} | {r.instrument} | {r.n_refusal} / {r.n_scored} | "
                 f"{f(r.mean_refusal)} [{f(r.refusal_lo)}, {f(r.refusal_hi)}] | "
                 f"{f(r.mean_scored)} [{f(r.scored_lo)}, {f(r.scored_hi)}] | "
                 f"{f(r.diff)} [{f(r.diff_lo)}, {f(r.diff_hi)}] |")
(HERE / "refusal_mean_comparison.md").write_text("\n".join(lines) + "\n")
HEAD = """# Refusal scoring summary

Dictionary / classifier instruments only. LLM-judge columns are pending the HPC run.

## Mean score on refusals vs scored responses

Refusals: 63 declarative (hand-audited; the 4 audit false positives are scored), 91 operational (corrected); `refusal_dictionary_scores.csv`. Scored: 297 declarative, 269 corrected operational (stored scores; the 4 moved declarative rows scored with the repo functions). Score = the instrument's length-normalised score (`bias_score_norm`, `hurtlex_norm`) or toxic-bert (declarative) / Detoxify (operational) toxicity probability. CIs: 10,000 bootstrap resamples of prompts (a prompt's refusal and scored rows move together), within nationality (pooled rows stratified by nationality), percentile 95%, seed 42; no resamples dropped. Diff = refusals minus scored. Source: `refusal_mean_comparison.py` -> `refusal_mean_comparison.csv`.

"""
(HERE / "refusal_scoring_summary.md").write_text(HEAD + "\n".join(lines) + "\n")
print("\n".join(lines)); print("max dropped", res.dropped_resamples.max())
