"""Aggregate the HPC refusal-judge outputs (hpc_refusal_judges/results_hpc/).

Per judge x prompt version (v1 with model-identity line, v2 without) x dataset, response level:
  * flag rate on the audited refusals (declarative 63, operational 91), overall and by nationality
  * NK/US flag ratio on scored responses only, and on scored responses + refusals (prompt-level
    bootstrap within nationality, 10,000 resamples, seed 42, same as boot.py)
Scored set: declarative 297 (293 stored + the 4 audit false positives, labelled by the HPC run),
operational 269 corrected non-refusals. Parse failures (bias_label -1) are dropped per judge.
Stored labels are never modified. Descriptive only.

Writes: judge_refusal_flags.csv, judge_ratio_comparison.csv, judge_refusal_rationales.csv,
        judge_refusal_summary.md
Run:  python3 results/refusal_audit/score_refusal_judges.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from boot import boot_ratio, NK, US, SEED, B  # noqa: E402
import refusal_sets  # noqa: E402

ROOT = HERE.parents[1]
HPC = HERE / "hpc_refusal_judges" / "results_hpc"
JUDGES = [  # (display, hpc/op tag, declarative stored files {version: file})
    ("Gemma 3 (4.3B) judge", "gemma3_latest", {"v1": "llm_judge_scores", "v2": "llm_judge_scores_v2"}),
    ("Gemma 3 (1B) judge", "gemma3_1b", {"v1": "llm_judge_scores_1b"}),
    ("Llama 3.1 judge", "llama3.1_latest", {"v1": "llm_judge_scores_llama", "v2": "llm_judge_scores_llama3_1_v2"}),
    ("Mistral judge", "mistral_latest", {"v1": "llm_judge_scores_mistral", "v2": "llm_judge_scores_mistral_v2"}),
]
KD = ["prompt_id", "model", "scale"]
KO = ["prompt_id", "model", "scale_direction"]

exp = refusal_sets.declarative_explanation()
ref_key_d = exp[exp.is_refusal][KD]
scored_key_d = exp[~exp.is_refusal][KD]
op_ref = pd.read_csv(HERE / "operational_refusals_corrected.csv")
op_scored = pd.read_csv(HERE / "operational_non_refusals_corrected.csv")
nat_d = exp[KD + ["nationality"]]


def boot_rate(df, seed=SEED):
    """Pooled flag rate, prompts resampled with replacement within nationality."""
    per = df.groupby(["nationality", "prompt_id"]).flag.agg(["sum", "count"]).reset_index()
    rng = np.random.default_rng(seed)
    s = np.zeros(B)
    c = np.zeros(B)
    for nat in per.nationality.unique():
        p = per[per.nationality == nat]
        idx = rng.integers(0, len(p), size=(B, len(p)))
        s += p["sum"].values[idx].sum(1)
        c += p["count"].values[idx].sum(1)
    lo, hi = np.percentile(s / c, [2.5, 97.5])
    return lo, hi


def hpc(ds, tag, v):
    p = HPC / f"refusal_{ds}_{tag}_{v}.csv"
    return pd.read_csv(p) if p.exists() else None


def stored_decl(f):
    s = pd.read_csv(ROOT / f"scores/{f}.csv")
    return s[s.step == "explanation"][KD + ["bias_label"]]


def stored_op(tag, v):
    return pd.read_csv(ROOT / f"hpc/operational/evaluations/judge_operational_{tag}_{v}.csv")[KO + ["bias_label"]]


flag_rows, ratio_rows, rat_rows = [], [], []
for disp, tag, dfiles in JUDGES:
    for v in ("v1", "v2"):
        for ds in ("declarative", "operational"):
            h = hpc(ds, tag, v)
            if h is None:
                continue
            K = KD if ds == "declarative" else KO
            if ds == "declarative":
                refl = h.merge(ref_key_d, on=KD)                       # audited 63
                fp = h.merge(exp[exp.audit_fp][KD], on=KD)             # 4 false positives -> scored
                st = None
                if v in dfiles:
                    st = stored_decl(dfiles[v]).merge(scored_key_d, on=KD)
                    st = pd.concat([st, fp[KD + ["bias_label"]]]).drop_duplicates(KD)
                scored_lab = st
                nat = nat_d
            else:
                refl = h.merge(op_ref[KO], on=KO)
                scored_lab = stored_op(tag, v).merge(op_scored[KO], on=KO)
                nat = pd.concat([op_ref, op_scored])[KO + ["nationality"]]
            # rationales (all 158-row HPC output for this judge/version/dataset), verbatim
            r = h.copy()
            r["audited_refusal"] = r.merge(refl[K].assign(_x=1), on=K, how="left")["_x"].fillna(0).astype(int).values
            r.insert(0, "judge", disp)
            r.insert(1, "version", v)
            r.insert(2, "dataset", ds)
            rat_rows.append(r[["judge", "version", "dataset", "audited_refusal", "prompt_id", "nationality", "model",
                               K[2], "bias_label", "severity", "reasoning", "raw_output", "parse_error"]]
                            .rename(columns={K[2]: "scale"}))
            n_all = len(refl)
            fl = refl[refl.bias_label >= 0].assign(flag=lambda x: x.bias_label)
            lo, hi = boot_rate(fl) if len(fl) else (np.nan, np.nan)
            row = {"judge": disp, "version": v, "dataset": ds, "n_refusals": n_all,
                   "n_parse_fail": int((refl.bias_label < 0).sum()), "n_flagged": int(fl.flag.sum()),
                   "refusal_flag_rate": fl.flag.mean(), "rate_lo": lo, "rate_hi": hi}
            for nm, nt in (("US", US), ("NK", NK)):
                x = fl[fl.nationality == nt]
                row[f"refusal_flag_rate_{nm}"] = x.flag.mean()
                row[f"n_{nm}"] = len(x)
            if scored_lab is not None:
                sl = scored_lab[scored_lab.bias_label >= 0].merge(nat.drop_duplicates(K), on=K)
                sl = sl.assign(flag=sl.bias_label)
                row.update({"n_scored": len(sl), "scored_flag_rate": sl.flag.mean()})
                both = pd.concat([sl[["prompt_id", "nationality", "flag"]], fl[["prompt_id", "nationality", "flag"]]])
                # prompt_id clusters are shared across scored and refusal rows of the same prompt
                for st_name, d in (("scored only", sl), ("scored + refusals", both)):
                    b = boot_ratio(d)
                    ratio_rows.append({"instrument": f"{disp} ({v})", "dataset": ds, "set": st_name,
                                       "ratio": b["point"], "lo": b["lo"], "hi": b["hi"],
                                       "n_us": b["n_us"], "n_nk": b["n_nk"],
                                       "dropped_resamples": b["dropped_resamples"]})
            flag_rows.append(row)

flags = pd.DataFrame(flag_rows)
ratios = pd.DataFrame(ratio_rows)
rat = pd.concat(rat_rows, ignore_index=True)
flags.to_csv(HERE / "judge_refusal_flags.csv", index=False)
ratios.to_csv(HERE / "judge_ratio_comparison.csv", index=False)
rat.to_csv(HERE / "judge_refusal_rationales.csv", index=False)

# ---------------------------------------------------------------- summary
f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
ci = lambda p, lo, hi: "--" if pd.isna(p) else f"{p:.2f} [{lo:.2f}, {hi:.2f}]"
L = ["# Judge outputs on refusals (HPC run, response level, T=0, seed 42)", "",
     "Descriptive numbers only. v1 = prompt with the model-identity line, v2 = without. "
     "Refusals: declarative 63 (hand-audited), operational 91. Scored: declarative 297, operational 269. "
     "Parse failures (bias_label -1) are dropped per judge. Bootstrap: prompts resampled within nationality, "
     "10,000 resamples, seed 42.", "", "## 1. Flag rate on refusals", "",
     "| Judge | Ver | Dataset | n refusals | parse fail | flagged | flag rate [95% CI] | US | NK |",
     "|---|---|---|---|---|---|---|---|---|"]
for _, r in flags.iterrows():
    L.append(f"| {r.judge} | {r.version} | {r.dataset} | {r.n_refusals} | {r.n_parse_fail} | {r.n_flagged} | "
             f"{ci(r.refusal_flag_rate, r.rate_lo, r.rate_hi)} | {f2(r.refusal_flag_rate_US)} (n={r.n_US}) | "
             f"{f2(r.refusal_flag_rate_NK)} (n={r.n_NK}) |")
L += ["", "## 2. NK/US flag ratio, scored only vs scored + refusals", "",
      "| Judge | Dataset | Scored only [95% CI] | n US / NK | Scored + refusals [95% CI] | n US / NK |",
      "|---|---|---|---|---|---|"]
for (inst, ds), g in ratios.groupby(["instrument", "dataset"], sort=False):
    a = g[g.set == "scored only"].iloc[0]
    b = g[g.set == "scored + refusals"].iloc[0]
    L.append(f"| {inst} | {ds} | {ci(a.ratio, a.lo, a.hi)} | {a.n_us} / {a.n_nk} | "
             f"{ci(b.ratio, b.lo, b.hi)} | {b.n_us} / {b.n_nk} |")
L += ["", "Gemma 3 (1B) has no stored declarative v2 labels, so no declarative v2 ratio is computed for it "
      "(its v2 refusal flag rate is in section 1). Dropped bootstrap resamples (zero US rate) are in "
      "`judge_ratio_comparison.csv`.", "",
      "Verbatim rationales for every judged row: `judge_refusal_rationales.csv` "
      "(`audited_refusal` = 1 for the 154 analysed refusals, 0 for the 4 declarative audit false positives)."]
(HERE / "judge_refusal_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
