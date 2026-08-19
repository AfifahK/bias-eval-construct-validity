#!/usr/bin/env python3
"""
v1 vs v2 prompt ablation analysis.

Runs:
  Part 1: McNemar paired test per judge (overall flip rate v1 → v2)
  Part 2: McNemar split own-model vs other-model (does the self-leniency
          population flip differently from the cross-model population?)
  Part 3: Self-leniency gap table per judge × version (confirms report numbers)
  Part 4: Length-controlled logistic regression
          - Model A: bias_label ~ judge + responder + version + own + scale + step
          - Model B: same + log_length
          - Likelihood ratio test on length
          - Coefficient stability of judge / own effects before vs after control

Usage:
    python src/analysis/v1v2_ablation.py

Background:
    nohup python src/analysis/v1v2_ablation.py > v1v2_results.log 2>&1 &
    tail -f v1v2_results.log

Expected runtime: ~30s on the 293-response dataset. Long enough to background,
short enough to babysit.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import logit
from statsmodels.stats.contingency_tables import mcnemar

# ----- Paths -----------------------------------------------------------------
# Assumes script lives at src/analysis/v1v2_ablation.py
ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "scores"
DATA = ROOT / "data"
MASTER_PATH = DATA / "all_model_responses_clean.csv"

JUDGES = {
    "gemma":   ("llm_judge_scores.csv",         "llm_judge_scores_v2.csv"),
    "llama":   ("llm_judge_scores_llama.csv",    "llm_judge_scores_llama3_1_v2.csv"),
    "mistral": ("llm_judge_scores_mistral.csv",  "llm_judge_scores_mistral_v2.csv"),
}

CANDIDATE_KEYS = ["prompt_id", "responder_model", "model", "scale", "step"]


# ----- Helpers ---------------------------------------------------------------
def log(msg=""):
    print(msg, flush=True)

def hr(title=""):
    log("\n" + "=" * 72)
    if title:
        log(title)
    log("=" * 72)

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_bias_label(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.map({
            "0": 0, "1": 1, 0: 0, 1: 1,
            "biased": 1, "not_biased": 0, "unbiased": 0,
            True: 1, False: 0, "True": 1, "False": 0,
        })
    return s


def normalize(df, judge_name, version, refusal_keys=None):
    """Standardize columns, drop parse failures (-1) and refusals, tag with judge + version."""
    out = df.copy()
    # responder model column → 'responder_model'
    rm = find_col(out, ["responder_model", "model"])
    if rm and rm != "responder_model":
        out = out.rename(columns={rm: "responder_model"})
    out["bias_label"] = coerce_bias_label(out["bias_label"])
    out = out[out["bias_label"].isin([0, 1])].copy()
    out["bias_label"] = out["bias_label"].astype(int)
    # Filter refusals if keys provided
    if refusal_keys is not None and len(refusal_keys) > 0:
        merge_cols = [c for c in ["prompt_id", "responder_model", "scale"] if c in out.columns]
        before = len(out)
        out = out.merge(refusal_keys[merge_cols], on=merge_cols, how="inner")
        log(f"    refusal filter: {before} → {len(out)} rows")
    out["judge_model"] = judge_name
    out["version"] = version
    return out


def mcnemar_paired(v1, v2, keys, label=""):
    paired = v1.merge(v2, on=keys, suffixes=("_v1", "_v2"))
    if len(paired) == 0:
        log(f"  {label}: no paired rows on {keys}")
        return None
    a = int(((paired["bias_label_v1"] == 0) & (paired["bias_label_v2"] == 0)).sum())
    b = int(((paired["bias_label_v1"] == 0) & (paired["bias_label_v2"] == 1)).sum())
    c = int(((paired["bias_label_v1"] == 1) & (paired["bias_label_v2"] == 0)).sum())
    d = int(((paired["bias_label_v1"] == 1) & (paired["bias_label_v2"] == 1)).sum())
    n = a + b + c + d
    exact = (b + c) < 25
    try:
        res = mcnemar([[a, b], [c, d]], exact=exact, correction=(not exact))
        p = res.pvalue
    except Exception as e:
        log(f"  {label}: mcnemar failed → {e}")
        return None
    log(
        f"  {label:<6} n={n:<4} | 00={a:<4} 01={b:<3} 10={c:<3} 11={d:<4} | "
        f"flip={(b+c)/n:6.2%} | McNemar p={p:.4f} (exact={exact})"
    )
    return dict(n=n, a=a, b=b, c=c, d=d, p=p, exact=exact)


def join_response_text(judges_df, master):
    # join keys present in both
    keys = [k for k in ["prompt_id", "responder_model", "scale", "step"]
            if k in master.columns and k in judges_df.columns]
    resp_col = find_col(master, ["response", "response_text", "explanation", "text"])
    if resp_col is None:
        raise RuntimeError("No response text column found on master")
    log(f"  join keys = {keys}; response col = {resp_col}")
    m_slim = (master[keys + [resp_col]]
              .rename(columns={resp_col: "response"})
              .drop_duplicates(keys))
    out = judges_df.merge(m_slim, on=keys, how="left")
    n_missing = out["response"].isna().sum()
    if n_missing:
        log(f"  WARNING: {n_missing} rows missing response after join — dropping")
    out = out.dropna(subset=["response"]).copy()
    out["response_length"] = out["response"].astype(str).str.len()
    out["log_length"] = np.log1p(out["response_length"])
    return out


# ----- Main ------------------------------------------------------------------
def main():
    hr("v1 vs v2 ablation — McNemar + length-controlled regression")
    log(f"Repo root: {ROOT}")

    if not MASTER_PATH.exists():
        log(f"ERROR: master not found at {MASTER_PATH}")
        sys.exit(1)
    master = pd.read_csv(MASTER_PATH)
    log(f"Master: {len(master)} rows; columns: {list(master.columns)}")
    # standardize on master too
    rm = find_col(master, ["responder_model", "model"])
    if rm and rm != "responder_model":
        master = master.rename(columns={rm: "responder_model"})

    # Build non-refusal explanation keys for consistent filtering
    expl = master[(master["step"] == "explanation") & (~master["is_refusal"])].copy()
    nr_keys = expl[["prompt_id", "responder_model", "scale"]].drop_duplicates()
    log(f"Non-refusal explanations: {len(nr_keys)} unique (prompt_id, model, scale) keys")

    all_rows = []
    paired_per_judge = {}

    # ---- Part 1: per-judge overall McNemar ---------------------------------
    hr("PART 1 — McNemar paired (overall) per judge")
    for jname in JUDGES:
        log(f"\n[{jname.upper()}]")
        v1f, v2f = JUDGES[jname]
        try:
            v1_raw = pd.read_csv(SCORES / v1f)
            v2_raw = pd.read_csv(SCORES / v2f)
        except FileNotFoundError as e:
            log(f"  SKIP: {e}")
            continue
        v1 = normalize(v1_raw, jname, "v1", refusal_keys=nr_keys)
        v2 = normalize(v2_raw, jname, "v2", refusal_keys=nr_keys)
        keys = [k for k in CANDIDATE_KEYS if k in v1.columns and k in v2.columns]
        v1 = v1.drop_duplicates(subset=keys)
        v2 = v2.drop_duplicates(subset=keys)
        log(f"  v1 n={len(v1)}, v2 n={len(v2)}, join keys={keys}")
        res = mcnemar_paired(v1, v2, keys, label="all")
        paired_per_judge[jname] = (v1, v2, keys, res)
        all_rows.extend([v1, v2])

    # ---- Part 2: split own vs other ----------------------------------------
    hr("PART 2 — McNemar split: own-model vs other-model")
    for jname, (v1, v2, keys, _) in paired_per_judge.items():
        log(f"\n[{jname.upper()}]")
        own_v1 = v1[v1["responder_model"].astype(str).str.lower().str.contains(jname)]
        oth_v1 = v1[~v1["responder_model"].astype(str).str.lower().str.contains(jname)]
        own_v2 = v2[v2["responder_model"].astype(str).str.lower().str.contains(jname)]
        oth_v2 = v2[~v2["responder_model"].astype(str).str.lower().str.contains(jname)]
        mcnemar_paired(own_v1, own_v2, keys, label="own")
        mcnemar_paired(oth_v1, oth_v2, keys, label="other")

    # ---- Part 3: self-leniency table ---------------------------------------
    hr("PART 3 — Self-leniency gap per judge × version")
    log(f"{'judge':<10}{'version':<8}{'own_rate':>10}{'oth_rate':>10}"
        f"{'gap_pp':>10}{'n_own':>8}{'n_oth':>8}")
    for jname, (v1, v2, _, _) in paired_per_judge.items():
        for ver, df in [("v1", v1), ("v2", v2)]:
            own = df[df["responder_model"].astype(str).str.lower().str.contains(jname)]
            oth = df[~df["responder_model"].astype(str).str.lower().str.contains(jname)]
            if len(own) == 0 or len(oth) == 0:
                continue
            gap_pp = (oth["bias_label"].mean() - own["bias_label"].mean()) * 100
            log(f"{jname:<10}{ver:<8}"
                f"{own['bias_label'].mean():>10.3f}{oth['bias_label'].mean():>10.3f}"
                f"{gap_pp:>10.2f}{len(own):>8}{len(oth):>8}")

    # ---- Part 4: length-controlled regression ------------------------------
    hr("PART 4 — Length-controlled logistic regression")
    if not all_rows:
        log("No judge data loaded; skipping regression.")
        return
    stacked = pd.concat(all_rows, ignore_index=True)
    log(f"Stacked rows: {len(stacked)}")
    full = join_response_text(stacked, master)
    log(f"After joining response text: {len(full)}")
    log(
        f"Response length (chars): "
        f"mean={full['response_length'].mean():.0f} "
        f"median={full['response_length'].median():.0f} "
        f"sd={full['response_length'].std():.0f} "
        f"min={full['response_length'].min()} "
        f"max={full['response_length'].max()}"
    )
    log("\nMean response length by responder_model:")
    log(full.groupby("responder_model")["response_length"]
        .agg(["mean", "median", "count"]).round(1).to_string())

    # own-judge indicator (judge name appears in responder model string)
    full["own_judge"] = full.apply(
        lambda r: str(r["judge_model"]).lower() in str(r["responder_model"]).lower(),
        axis=1,
    ).astype(int)

    base = "bias_label ~ C(judge_model) + C(responder_model) + C(version) + own_judge"
    if "scale" in full.columns:
        base += " + C(scale)"
    if "step" in full.columns:
        base += " + C(step)"

    log(f"\nModel A (no length):\n  {base}")
    m_a = logit(base, data=full).fit(disp=0)
    log(m_a.summary().as_text())

    f_b = base + " + log_length"
    log(f"\nModel B (with log_length):\n  {f_b}")
    m_b = logit(f_b, data=full).fit(disp=0)
    log(m_b.summary().as_text())

    # likelihood ratio test
    lr = 2 * (m_b.llf - m_a.llf)
    p_lr = stats.chi2.sf(lr, df=1)
    log(f"\nLikelihood ratio test (length vs no-length):")
    log(f"  LR={lr:.3f}, df=1, p={p_lr:.4g}")
    log(f"  AIC: A={m_a.aic:.1f}, B={m_b.aic:.1f}  (lower = better)")
    if "log_length" in m_b.params.index:
        log(f"  log_length: beta={m_b.params['log_length']:.4f} "
            f"(SE={m_b.bse['log_length']:.4f}, p={m_b.pvalues['log_length']:.4g})")

    # coefficient stability for judge / own effects
    log("\nJudge & own_judge coefficient stability (Model A → Model B):")
    log(f"{'term':<40}{'beta_A':>10}{'p_A':>10}{'beta_B':>10}{'p_B':>10}")
    for k in m_a.params.index:
        if k.startswith("C(judge_model)") or k == "own_judge":
            ba, pa = m_a.params[k], m_a.pvalues[k]
            bb = m_b.params.get(k, np.nan)
            pb = m_b.pvalues.get(k, np.nan)
            log(f"{k:<40}{ba:>10.3f}{pa:>10.4g}{bb:>10.3f}{pb:>10.4g}")

    log("\nInterpretation guide:")
    log("  - If LR p < 0.05 AND judge/own coefs shrink substantially → length matters.")
    log("  - If judge/own coefs stable across A→B and stay significant → main findings")
    log("    survive length control. Report length as confound, not as driver.")

    # ---- One-line summary for quick verification ----------------------------
    mcnemar_ps = {jname: res["p"] if res else float("nan")
                  for jname, (_, _, _, res) in paired_per_judge.items()}
    mc_str = ", ".join(f"{j}={p:.4g}" for j, p in mcnemar_ps.items())

    own_coef = m_b.params.get("own_judge", float("nan"))
    own_p = m_b.pvalues.get("own_judge", float("nan"))
    ll_coef = m_b.params.get("log_length", float("nan"))
    ll_p = m_b.pvalues.get("log_length", float("nan"))

    hr("ONE-LINE SUMMARY")
    log(f"McNemar p: {mc_str}; own_judge β={own_coef:.3f}, p={own_p:.4g}; "
        f"length log-LR p={p_lr:.4g}; AIC A={m_a.aic:.1f} B={m_b.aic:.1f}")

    hr("DONE")

if __name__ == "__main__":
    main()
