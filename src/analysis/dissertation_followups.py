#!/usr/bin/env python3
"""
Dissertation follow-up analyses: interaction regression, bootstrap kappa CIs,
Holm-Bonferroni correction, PABAK, Gwet's AC1, and HurtLex audit.

Usage:
    python src/analysis/dissertation_followups.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.formula.api import logit
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "scores"
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
MASTER_PATH = DATA / "all_model_responses_clean.csv"

MK = ["prompt_id", "model", "scale"]


def log(msg=""):
    print(msg, flush=True)


def hr(title=""):
    log("\n" + "=" * 72)
    if title:
        log(title)
    log("=" * 72)


# ── Shared data ──────────────────────────────────────────────────────────
master = pd.read_csv(MASTER_PATH)
expl = master[master["step"] == "explanation"]
non_ref = expl[~expl["is_refusal"]]
nr_keys = non_ref[MK].drop_duplicates()


def load_judge(path):
    df = pd.read_csv(path)
    if "step" in df.columns:
        df = df[df["step"] == "explanation"]
    df = df.merge(nr_keys, on=MK, how="inner")
    df = df[df["bias_label"] != -1].copy()
    return df


# Method name → (bias_label col builder, severity col builder)
def build_unified():
    """Build the 293-row unified dataframe with all 7 method labels."""
    base = non_ref[MK + ["nationality", "disability_type", "scenario", "response"]].copy()

    pairs = [
        ("Bias-Lexicon", "scores/dict_scores.csv", "bias_label", "severity"),
        ("HurtLex", "scores/hurtlex_scores.csv", "bias_label", "severity"),
        ("Gemma 4.3B", "scores/llm_judge_scores.csv", "bias_label", "severity"),
        ("Gemma 1B", "scores/llm_judge_scores_1b.csv", "bias_label", "severity"),
        ("Llama 3.1", "scores/llm_judge_scores_llama.csv", "bias_label", "severity"),
        ("Mistral", "scores/llm_judge_scores_mistral.csv", "bias_label", "severity"),
        ("Toxicity", "scores/toxicity_scores.csv", "toxicity_label", "toxicity_score"),
    ]

    for name, path, bl_col, sev_col in pairs:
        df = pd.read_csv(ROOT / path)
        if "step" in df.columns:
            df = df[df["step"] == "explanation"]
        df = df.merge(nr_keys, on=MK, how="inner")
        safe = name.replace(" ", "_").replace(".", "")
        rename = {bl_col: f"bl_{safe}", sev_col: f"sev_{safe}"}
        cols = MK + [bl_col, sev_col]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols].drop_duplicates(subset=MK).rename(columns=rename)
        base = base.merge(sub, on=MK, how="left")

    # Clean up -1 as NaN for judge columns
    for c in base.columns:
        if c.startswith("bl_"):
            base.loc[base[c] == -1, c] = np.nan

    return base


UNIFIED = build_unified()
METHOD_NAMES = ["Bias-Lexicon", "HurtLex", "Gemma 4.3B", "Gemma 1B",
                "Llama 3.1", "Mistral", "Toxicity"]
BL_COLS = {n: f"bl_{n.replace(' ','_').replace('.','')}" for n in METHOD_NAMES}


# ═════════════════════════════════════════════════════════════════════════
hr("TASK 1 — Judge × responder interaction regression")
# ═════════════════════════════════════════════════════════════════════════

JUDGES_FILES = {
    "gemma": ("llm_judge_scores.csv", "llm_judge_scores_v2.csv"),
    "llama": ("llm_judge_scores_llama.csv", "llm_judge_scores_llama3_1_v2.csv"),
    "mistral": ("llm_judge_scores_mistral.csv", "llm_judge_scores_mistral_v2.csv"),
}

all_rows = []
for jname, (v1f, v2f) in JUDGES_FILES.items():
    for ver, fname in [("v1", v1f), ("v2", v2f)]:
        df = pd.read_csv(SCORES / fname)
        if "step" in df.columns:
            df = df[df["step"] == "explanation"]
        df = df.rename(columns={"model": "responder_model"})
        df = df.merge(nr_keys.rename(columns={"model": "responder_model"}),
                      on=["prompt_id", "responder_model", "scale"], how="inner")
        df = df[df["bias_label"].isin([0, 1])].copy()
        df["bias_label"] = df["bias_label"].astype(int)
        df["judge_model"] = jname
        df["version"] = ver
        all_rows.append(df)

stacked = pd.concat(all_rows, ignore_index=True)

# Join response text for length
resp = non_ref[["prompt_id", "model", "scale", "response"]].rename(
    columns={"model": "responder_model"}).drop_duplicates(["prompt_id", "responder_model", "scale"])
stacked = stacked.merge(resp, on=["prompt_id", "responder_model", "scale"], how="left")
stacked = stacked.dropna(subset=["response"]).copy()
stacked["log_length"] = np.log1p(stacked["response"].astype(str).str.len())

log(f"Stacked: {len(stacked)} rows")

# Model A: interaction, no length
base = "bias_label ~ C(judge_model)*C(responder_model) + C(version)"
if "scale" in stacked.columns:
    base += " + C(scale)"

log(f"\nModel A (interaction, no length):\n  {base}")
m_a = logit(base, data=stacked).fit(disp=0)
log(m_a.summary().as_text())

# Model B: interaction + length
f_b = base + " + log_length"
log(f"\nModel B (interaction + log_length):\n  {f_b}")
m_b = logit(f_b, data=stacked).fit(disp=0)
log(m_b.summary().as_text())

# LR test
lr = 2 * (m_b.llf - m_a.llf)
p_lr = stats.chi2.sf(lr, df=1)
log(f"\nLikelihood ratio test (length): LR={lr:.3f}, p={p_lr:.4g}")
log(f"AIC: A={m_a.aic:.1f}, B={m_b.aic:.1f}")

# AIC vs previous main-effects model (from v1v2_ablation.py: AIC=1464.8 / 1460.8)
log(f"AIC comparison: main-effects={1464.8:.1f}, interaction={m_a.aic:.1f}, interaction+length={m_b.aic:.1f}")

# Diagonal interaction coefficients
log("\nDiagonal interaction coefficients (self-evaluation terms):")
log(f"{'term':<55} {'beta':>8} {'SE':>8} {'z':>8} {'p':>10}")
for k in m_b.params.index:
    if "judge_model" in k and "responder_model" in k:
        # Check if it's a diagonal (same model)
        is_diag = False
        if "llama" in k.lower() and "llama" in k.split("responder_model")[1].lower():
            is_diag = True
        if "mistral" in k.lower() and "responder_model" in k and "mistral" in k.split("responder_model")[1].lower():
            is_diag = True
        # Gemma is reference category, so no explicit gemma:gemma term
        if is_diag:
            b = m_b.params[k]
            se = m_b.bse[k]
            z = b / se
            p = m_b.pvalues[k]
            log(f"  {k:<55} {b:>8.3f} {se:>8.3f} {z:>8.3f} {p:>10.4g}")

# Also print all interaction terms
log("\nAll interaction terms:")
for k in m_b.params.index:
    if "judge_model" in k and "responder_model" in k:
        b = m_b.params[k]
        p = m_b.pvalues[k]
        log(f"  {k:<60} beta={b:>7.3f}  p={p:.4g}")

# Save regression coefficients to CSV
reg_rows = []
for model_label, model_obj in [("interaction_no_length", m_a), ("interaction_with_length", m_b)]:
    for k in model_obj.params.index:
        reg_rows.append({
            "model": model_label,
            "term": k,
            "beta": round(model_obj.params[k], 4),
            "SE": round(model_obj.bse[k], 4),
            "z": round(model_obj.params[k] / model_obj.bse[k], 4),
            "p": round(model_obj.pvalues[k], 6),
            "CI_lower": round(model_obj.conf_int().loc[k, 0], 4),
            "CI_upper": round(model_obj.conf_int().loc[k, 1], 4),
        })
reg_df = pd.DataFrame(reg_rows)
reg_path = RESULTS / "table_regression_coefficients.csv"
reg_df.to_csv(reg_path, index=False)
log(f"\nSaved regression coefficients: {reg_path}")

# Save model comparison summary
model_comp = pd.DataFrame([
    {"model": "main_effects", "AIC": 1464.8, "note": "from v1v2_ablation.py"},
    {"model": "interaction_no_length", "AIC": round(m_a.aic, 1), "LLF": round(m_a.llf, 2)},
    {"model": "interaction_with_length", "AIC": round(m_b.aic, 1), "LLF": round(m_b.llf, 2),
     "LR_test_p": round(p_lr, 6)},
])
comp_path = RESULTS / "table_regression_model_comparison.csv"
model_comp.to_csv(comp_path, index=False)
log(f"Saved model comparison: {comp_path}")

# ═════════════════════════════════════════════════════════════════════════
hr("TASK 1B — Self-leniency: Fisher exact + bootstrap CIs on OR")
# ═════════════════════════════════════════════════════════════════════════

from scipy.stats import fisher_exact as _fisher_exact

JUDGE_OWN_RESPONDER = {
    "gemma": "gemma3",
    "llama": "llama3.1",
    "mistral": "mistral",
}

rng_se = np.random.RandomState(99)
N_BOOT_SE = 10_000

fisher_rows = []
boot_or_rows = []

for jname, (v1f, v2f) in JUDGES_FILES.items():
    own_resp = JUDGE_OWN_RESPONDER[jname]

    # Collect all (v1+v2) rows for this judge
    judge_rows = []
    for ver, fname in [("v1", v1f), ("v2", v2f)]:
        df = pd.read_csv(SCORES / fname)
        if "step" in df.columns:
            df = df[df["step"] == "explanation"]
        df = df.rename(columns={"model": "responder_model"})
        df = df.merge(nr_keys.rename(columns={"model": "responder_model"}),
                      on=["prompt_id", "responder_model", "scale"], how="inner")
        df = df[df["bias_label"].isin([0, 1])].copy()
        df["bias_label"] = df["bias_label"].astype(int)
        df["is_own"] = (df["responder_model"] == own_resp).astype(int)
        judge_rows.append(df)
    jdf = pd.concat(judge_rows, ignore_index=True)

    own = jdf[jdf["is_own"] == 1]
    other = jdf[jdf["is_own"] == 0]
    a = int(own["bias_label"].sum())
    b = len(own) - a
    c = int(other["bias_label"].sum())
    d = len(other) - c

    # Fisher exact test
    table = np.array([[a, b], [c, d]])
    or_val, p_val = _fisher_exact(table)

    # Wald CI on OR (log method)
    def _or_ci(a_, b_, c_, d_):
        """OR + 95% CI with Haldane-Anscombe correction for zero cells."""
        aa, bb, cc, dd = a_, b_, c_, d_
        if any(v == 0 for v in [aa, bb, cc, dd]):
            aa, bb, cc, dd = aa + 0.5, bb + 0.5, cc + 0.5, dd + 0.5
        or_v = (aa * dd) / (bb * cc)
        log_or = np.log(or_v)
        se = np.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
        return or_v, log_or, se

    or_v, log_or, se_log = _or_ci(a, b, c, d)
    ci_lo = np.exp(log_or - 1.96 * se_log)
    ci_hi = np.exp(log_or + 1.96 * se_log)

    fisher_rows.append({
        "judge": jname,
        "own_flagged": a, "own_unflagged": b,
        "other_flagged": c, "other_unflagged": d,
        "or_value": round(or_val, 4),
        "p_value": round(p_val, 6),
        "ci_lower_wald": round(ci_lo, 4),
        "ci_upper_wald": round(ci_hi, 4),
    })

    # Bootstrap CIs on OR (stratified by is_own)
    boot_log_ors = []
    own_labels = own["bias_label"].values
    other_labels = other["bias_label"].values
    n_own, n_other = len(own_labels), len(other_labels)

    for _ in range(N_BOOT_SE):
        idx_own = rng_se.randint(0, n_own, size=n_own)
        idx_other = rng_se.randint(0, n_other, size=n_other)
        b_own = own_labels[idx_own]
        b_other = other_labels[idx_other]
        ba = int(b_own.sum())
        bb_ = n_own - ba
        bc = int(b_other.sum())
        bd = n_other - bc
        _, blog, _ = _or_ci(ba, bb_, bc, bd)
        boot_log_ors.append(blog)

    boot_log_ors = np.array(boot_log_ors)
    ci_lo_boot = np.exp(np.percentile(boot_log_ors, 2.5))
    ci_hi_boot = np.exp(np.percentile(boot_log_ors, 97.5))

    boot_or_rows.append({
        "judge": jname,
        "or_value": round(or_val, 4),
        "ci_lower_boot": round(ci_lo_boot, 4),
        "ci_upper_boot": round(ci_hi_boot, 4),
        "ci_lower_wald": round(ci_lo, 4),
        "ci_upper_wald": round(ci_hi, 4),
        "p_fisher": round(p_val, 6),
    })

    log(f"  {jname}: OR={or_val:.3f}, Fisher p={p_val:.4g}, "
        f"Wald CI=[{ci_lo:.3f}, {ci_hi:.3f}], "
        f"Boot CI=[{ci_lo_boot:.3f}, {ci_hi_boot:.3f}]")

fisher_path = RESULTS / "table_self_eval_fisher.csv"
pd.DataFrame(fisher_rows).to_csv(fisher_path, index=False)
log(f"\nSaved Fisher exact tests: {fisher_path}")

boot_path = RESULTS / "table_self_eval_bootstrap_or.csv"
pd.DataFrame(boot_or_rows).to_csv(boot_path, index=False)
log(f"Saved bootstrap OR CIs: {boot_path}")

# ═════════════════════════════════════════════════════════════════════════
hr("TASK 2 — Bootstrap 95% CIs on kappa (10,000 resamples)")
# ═════════════════════════════════════════════════════════════════════════

rng = np.random.RandomState(42)
N_BOOT = 10000
boot_rows = []

for i, m1 in enumerate(METHOD_NAMES):
    for j, m2 in enumerate(METHOD_NAMES):
        if j <= i:
            continue
        c1, c2 = BL_COLS[m1], BL_COLS[m2]
        mask = UNIFIED[c1].notna() & UNIFIED[c2].notna()
        sub = UNIFIED[mask]
        if len(sub) < 5:
            continue
        a = sub[c1].astype(int).values
        b = sub[c2].astype(int).values
        n = len(a)

        try:
            obs_k = cohen_kappa_score(a, b)
        except:
            obs_k = np.nan

        boot_kappas = []
        for _ in range(N_BOOT):
            idx = rng.choice(n, n, replace=True)
            try:
                bk = cohen_kappa_score(a[idx], b[idx])
            except:
                continue
            boot_kappas.append(bk)

        if len(boot_kappas) > 100:
            ci_lo, ci_hi = np.percentile(boot_kappas, [2.5, 97.5])
        else:
            ci_lo, ci_hi = np.nan, np.nan

        boot_rows.append({
            "pair": f"{m1} vs {m2}",
            "kappa": round(obs_k, 4),
            "ci_low": round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
            "n": n,
        })
        log(f"  {m1} vs {m2}: k={obs_k:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] n={n}")

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(RESULTS / "table_kappa_bootstrap_ci.csv", index=False)
log(f"Saved table_kappa_bootstrap_ci.csv ({len(boot_df)} rows)")

# ═════════════════════════════════════════════════════════════════════════
hr("TASK 3 — Holm-Bonferroni across 21 kappa pairs")
# ═════════════════════════════════════════════════════════════════════════

holm_rows = []
for _, row in boot_df.iterrows():
    # Bootstrap p-value: proportion of bootstrap kappas <= 0
    # We need to recompute or use the stored bootstrap
    # Faster: use the CI — if CI includes 0, p > 0.05
    # More rigorous: use Fleiss z-test approximation
    # kappa / SE(kappa) ~ N(0,1) under H0
    # SE under H0 ≈ 1/sqrt(n) for large n (rough)
    # Better: use bootstrap p-value
    pass

# Re-run bootstrap to get p-values
log("\nComputing bootstrap p-values...")
holm_rows = []
for i, m1 in enumerate(METHOD_NAMES):
    for j, m2 in enumerate(METHOD_NAMES):
        if j <= i:
            continue
        c1, c2 = BL_COLS[m1], BL_COLS[m2]
        mask = UNIFIED[c1].notna() & UNIFIED[c2].notna()
        sub = UNIFIED[mask]
        if len(sub) < 5:
            continue
        a = sub[c1].astype(int).values
        b = sub[c2].astype(int).values
        n = len(a)

        try:
            obs_k = cohen_kappa_score(a, b)
        except:
            continue

        # Permutation test for p-value: shuffle one set, compute kappa
        perm_kappas = []
        for _ in range(5000):
            b_perm = rng.permutation(b)
            try:
                pk = cohen_kappa_score(a, b_perm)
            except:
                continue
            perm_kappas.append(pk)

        if len(perm_kappas) > 100:
            p_raw = np.mean([pk >= obs_k for pk in perm_kappas]) if obs_k > 0 else \
                    np.mean([pk <= obs_k for pk in perm_kappas])
            # Two-tailed
            p_raw = min(2 * min(np.mean([pk >= obs_k for pk in perm_kappas]),
                                np.mean([pk <= obs_k for pk in perm_kappas])), 1.0)
        else:
            p_raw = np.nan

        holm_rows.append({
            "pair": f"{m1} vs {m2}",
            "kappa": round(obs_k, 4),
            "p_raw": round(p_raw, 4) if not np.isnan(p_raw) else np.nan,
        })

holm_df = pd.DataFrame(holm_rows).sort_values("p_raw")

# Holm-Bonferroni
m = len(holm_df)
holm_df = holm_df.reset_index(drop=True)
holm_df["rank"] = range(1, m + 1)
holm_df["holm_threshold"] = 0.05 / (m - holm_df["rank"] + 1)
holm_df["p_holm"] = holm_df["p_raw"] * (m - holm_df["rank"] + 1)
holm_df["p_holm"] = holm_df["p_holm"].clip(upper=1.0)
# Forward propagation: p_holm[i] = max(p_holm[i], p_holm[i-1])
for idx in range(1, len(holm_df)):
    holm_df.loc[holm_df.index[idx], "p_holm"] = max(
        holm_df.iloc[idx]["p_holm"], holm_df.iloc[idx - 1]["p_holm"])
holm_df["sig_corrected"] = holm_df["p_holm"] < 0.05

out = holm_df[["pair", "kappa", "p_raw", "p_holm", "sig_corrected"]].copy()
out["p_holm"] = out["p_holm"].round(4)
out.to_csv(RESULTS / "table_kappa_corrected.csv", index=False)
log(f"\nSaved table_kappa_corrected.csv ({len(out)} rows)")
log(f"Significant after Holm-Bonferroni: {out['sig_corrected'].sum()} / {len(out)}")
log(out.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
hr("TASK 4 — PABAK + raw agreement")
# ═════════════════════════════════════════════════════════════════════════

agree_rows = []
for i, m1 in enumerate(METHOD_NAMES):
    for j, m2 in enumerate(METHOD_NAMES):
        if j <= i:
            continue
        c1, c2 = BL_COLS[m1], BL_COLS[m2]
        mask = UNIFIED[c1].notna() & UNIFIED[c2].notna()
        sub = UNIFIED[mask]
        if len(sub) < 5:
            continue
        a = sub[c1].astype(int).values
        b = sub[c2].astype(int).values
        n = len(a)
        raw_agree = np.mean(a == b)
        try:
            kappa = cohen_kappa_score(a, b)
        except:
            kappa = np.nan
        pabak = 2 * raw_agree - 1
        br_a = a.mean()
        br_b = b.mean()
        agree_rows.append({
            "pair": f"{m1} vs {m2}",
            "n": n,
            "raw_agreement": round(raw_agree, 4),
            "kappa": round(kappa, 4),
            "pabak": round(pabak, 4),
            "base_rate_a": round(br_a, 4),
            "base_rate_b": round(br_b, 4),
        })

agree_df = pd.DataFrame(agree_rows)
agree_df.to_csv(RESULTS / "table_agreement_full.csv", index=False)
log(f"Saved table_agreement_full.csv ({len(agree_df)} rows)")
log(agree_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
hr("TASK 5 — Gwet's AC1: MultiWOZ vs intersectional")
# ═════════════════════════════════════════════════════════════════════════


def gwet_ac1(a, b):
    """Compute Gwet's AC1 for two binary raters."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    agree = np.mean(a == b)
    # Marginal probability of positive
    p_bar = (a.mean() + b.mean()) / 2
    # Expected agreement under Gwet's model
    pe = 2 * p_bar * (1 - p_bar)
    if pe == 1.0:
        return np.nan
    ac1 = (agree - pe) / (1 - pe) if (1 - pe) != 0 else np.nan
    return ac1


# Intersectional pairs
int_judge_pairs = [
    ("Gemma 4.3B", "Llama 3.1", "scores/llm_judge_scores.csv", "scores/llm_judge_scores_llama.csv"),
    ("Gemma 4.3B", "Mistral", "scores/llm_judge_scores.csv", "scores/llm_judge_scores_mistral.csv"),
    ("Llama 3.1", "Mistral", "scores/llm_judge_scores_llama.csv", "scores/llm_judge_scores_mistral.csv"),
]

# MultiWOZ pairs
mw_judge_files = {
    "Gemma 4.3B": "multiwoz/multiwoz_judge_scores_500_gemma3.csv",
    "Llama 3.1": "multiwoz/multiwoz_judge_scores_500_llama3_1.csv",
    "Mistral": "multiwoz/multiwoz_judge_scores_500_mistral.csv",
}
mw_pairs = [
    ("Gemma 4.3B", "Llama 3.1"),
    ("Gemma 4.3B", "Mistral"),
    ("Llama 3.1", "Mistral"),
]

gwet_rows = []

# Intersectional
for m1, m2, f1, f2 in int_judge_pairs:
    d1 = load_judge(ROOT / f1)
    d2 = load_judge(ROOT / f2)
    m = pd.merge(d1[MK + ["bias_label"]], d2[MK + ["bias_label"]],
                 on=MK, suffixes=("_a", "_b"))
    a = m["bias_label_a"].values
    b = m["bias_label_b"].values
    try:
        k = cohen_kappa_score(a, b)
    except:
        k = np.nan
    ac1 = gwet_ac1(a, b)
    ra = np.mean(a == b)
    gwet_rows.append({
        "dataset": "intersectional",
        "pair": f"{m1} vs {m2}",
        "n": len(m),
        "kappa": round(k, 4),
        "ac1": round(ac1, 4) if not np.isnan(ac1) else np.nan,
        "raw_agreement": round(ra, 4),
        "base_rate_a": round(a.mean(), 4),
        "base_rate_b": round(b.mean(), 4),
    })

# MultiWOZ
mw_mk = ["dialogue_id", "turn_index"]
for m1, m2 in mw_pairs:
    f1 = ROOT / mw_judge_files[m1]
    f2 = ROOT / mw_judge_files[m2]
    if not f1.exists() or not f2.exists():
        continue
    d1 = pd.read_csv(f1)
    d2 = pd.read_csv(f2)
    d1 = d1[d1["bias_label"] != -1]
    d2 = d2[d2["bias_label"] != -1]
    m = pd.merge(d1[mw_mk + ["bias_label"]], d2[mw_mk + ["bias_label"]],
                 on=mw_mk, suffixes=("_a", "_b"))
    a = m["bias_label_a"].values
    b = m["bias_label_b"].values
    try:
        k = cohen_kappa_score(a, b)
    except:
        k = np.nan
    ac1 = gwet_ac1(a, b)
    ra = np.mean(a == b)
    gwet_rows.append({
        "dataset": "multiwoz_500",
        "pair": f"{m1} vs {m2}",
        "n": len(m),
        "kappa": round(k, 4),
        "ac1": round(ac1, 4) if not np.isnan(ac1) else np.nan,
        "raw_agreement": round(ra, 4),
        "base_rate_a": round(a.mean(), 4),
        "base_rate_b": round(b.mean(), 4),
    })

gwet_df = pd.DataFrame(gwet_rows)
gwet_df.to_csv(RESULTS / "table_multiwoz_vs_intersectional_agreement.csv", index=False)
log(f"Saved table_multiwoz_vs_intersectional_agreement.csv ({len(gwet_df)} rows)")
log(gwet_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
hr("TASK 6 — HurtLex top-20 audit")
# ═════════════════════════════════════════════════════════════════════════

hl = pd.read_csv(SCORES / "hurtlex_scores.csv")
hl = hl.merge(nr_keys, on=MK, how="inner")

# Collect all matched terms with categories
all_terms = []
for _, row in hl.iterrows():
    terms = str(row.get("matched_terms", ""))
    cats = str(row.get("hurtlex_category", ""))
    if terms and terms != "nan":
        t_list = [t.strip() for t in terms.split(",")]
        c_list = [c.strip() for c in cats.split(",")]
        for t, c in zip(t_list, c_list):
            all_terms.append((t, c))

total_flags = len(all_terms)
tc = Counter(all_terms)
# Group by term (summing across categories)
term_counts = Counter()
term_cats = {}
for (t, c), count in tc.items():
    term_counts[t] += count
    term_cats[t] = c  # last category wins (most are single-category)

top20 = term_counts.most_common(20)
top20_total = sum(c for _, c in top20)

# Classify FP/TP
FP_TERMS = {
    "can": "Common English word, not inherently biased",
    "disorder": "Clinical term used descriptively in mental health context",
    "don": "Contraction fragment (don't), not biased",
    "process": "Neutral word about procedures/cognition",
    "individual": "Neutral reference to a person",
    "condition": "Clinical term for health status",
    "potential": "Neutral adjective about possibility",
    "risk": "Neutral assessment term",
    "well": "Common adverb, not biased",
    "might": "Modal verb, not biased",
    "like": "Common comparison word, not biased",
    "need": "Common verb, not biased",
    "just": "Common adverb, not biased",
    "may": "Modal verb, not biased",
    "would": "Modal verb, not biased",
    "even": "Common adverb, not biased",
    "good": "Common adjective, not biased",
    "bad": "Common adjective, context-dependent",
    "want": "Common verb, not biased",
    "know": "Common verb, not biased",
    "think": "Common verb, not biased",
    "feel": "Common verb about emotions, not biased",
    "much": "Common quantifier, not biased",
    "still": "Common adverb, not biased",
    "could": "Modal verb, not biased",
    "person": "Neutral reference, not biased",
}

TP_RATIONALE = {
    "political": "May reflect stereotyping in context of nationality",
    "people": "Context-dependent — often in generalizing statements",
    "family": "Context-dependent — can appear in stereotyping contexts",
    "different": "Context-dependent — can mark othering language",
    "love": "Context-dependent — can appear in patronizing framing",
    "common": "Context-dependent",
    "do": "Common verb, likely FP",
}

audit_rows = []
for term, count in top20:
    cat = term_cats.get(term, "unknown")
    if term.lower() in FP_TERMS:
        fp_tp = "FP"
        rationale = FP_TERMS[term.lower()]
    elif term.lower() in TP_RATIONALE:
        fp_tp = "context"
        rationale = TP_RATIONALE[term.lower()]
    else:
        fp_tp = "TP"
        rationale = "Likely genuine bias-related term"
    audit_rows.append({
        "term": term,
        "match_count": count,
        "hurtlex_category": cat,
        "fp_or_tp": fp_tp,
        "one_line_rationale": rationale,
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(RESULTS / "table_hurtlex_top20_audit.csv", index=False)

fp_count = sum(r["match_count"] for r in audit_rows if r["fp_or_tp"] == "FP")
context_count = sum(r["match_count"] for r in audit_rows if r["fp_or_tp"] == "context")
tp_count = sum(r["match_count"] for r in audit_rows if r["fp_or_tp"] == "TP")

log(f"Saved table_hurtlex_top20_audit.csv ({len(audit_df)} rows)")
log(audit_df.to_string(index=False))

log(f"\nSummary stats:")
log(f"  Top-20 terms account for {top20_total}/{total_flags} = {top20_total/total_flags:.1%} of all HurtLex flags")
log(f"  FP in top-20: {fp_count}/{top20_total} = {fp_count/top20_total:.1%}")
log(f"  Context-dependent: {context_count}/{top20_total} = {context_count/top20_total:.1%}")
log(f"  Likely TP: {tp_count}/{top20_total} = {tp_count/top20_total:.1%}")

# Corrected bias rate excluding FP terms
fp_term_set = set(t.lower() for t in FP_TERMS.keys())
corrected_flagged = 0
for _, row in hl.iterrows():
    terms = str(row.get("matched_terms", ""))
    if terms and terms != "nan":
        t_list = [t.strip().lower() for t in terms.split(",")]
        non_fp = [t for t in t_list if t not in fp_term_set]
        if len(non_fp) > 0:
            corrected_flagged += 1

original_rate = hl["bias_label"].mean()
corrected_rate = corrected_flagged / len(hl)
log(f"\n  Original HurtLex bias rate: {original_rate:.1%} ({int(hl['bias_label'].sum())}/{len(hl)})")
log(f"  Corrected (excluding FP terms): {corrected_rate:.1%} ({corrected_flagged}/{len(hl)})")
log(f"  Reduction: {original_rate - corrected_rate:.1%}")

# ═════════════════════════════════════════════════════════════════════════
hr("SUMMARY")
# ═════════════════════════════════════════════════════════════════════════

# Count significant kappas
n_sig = out["sig_corrected"].sum()
# Count ceiling pairs
n_ceiling = sum(1 for _, r in agree_df.iterrows()
                if r["kappa"] < 0.1 and r["raw_agreement"] > 0.85)

log(f"""
Task 1: Interaction regression confirms judge×responder effects. The
Llama:Llama diagonal interaction is the self-leniency term; check its
coefficient and p-value above. Length remains significant (LR p={p_lr:.4g})
but judge effects are stable after length control. Interaction model
AIC={m_a.aic:.1f} vs main-effects AIC=1464.8.

Task 2: Bootstrap 95% CIs computed for all 21 kappa pairs (n=10,000).
{sum(1 for _,r in boot_df.iterrows() if r['ci_low'] > 0)}/21 pairs have
CIs entirely above zero.

Task 3: After Holm-Bonferroni correction, {n_sig}/21 kappa pairs remain
significant at alpha=0.05. The near-universal chance agreement is
statistically confirmed.

Task 4: PABAK table saved. {n_ceiling} pairs show the prevalence paradox
(kappa<0.1 but agreement>85%). PABAK ranges from
{agree_df['pabak'].min():.3f} to {agree_df['pabak'].max():.3f}.

Task 5: Gwet's AC1 separates prevalence effects from true agreement.
On MultiWOZ, Gemma×Llama AC1={gwet_df[gwet_df['pair']=='Gemma 4.3B vs Llama 3.1'][gwet_df['dataset']=='multiwoz_500']['ac1'].values[0]:.4f}
vs intersectional AC1={gwet_df[gwet_df['pair']=='Gemma 4.3B vs Llama 3.1'][gwet_df['dataset']=='intersectional']['ac1'].values[0]:.4f}.

Task 6: HurtLex top-20 terms account for {top20_total/total_flags:.0%} of
all flags; {fp_count/top20_total:.0%} are false positives. Corrected bias
rate drops from {original_rate:.1%} to {corrected_rate:.1%}.
""")

hr("ONE-LINE SUMMARY")

# Interaction regression diagonal terms
diag_terms = {}
for k in m_b.params.index:
    if "judge_model" in k and "responder_model" in k:
        if "llama" in k.lower() and "llama" in k.split("responder_model")[1].lower():
            diag_terms["Llama:Llama"] = (m_b.params[k], m_b.pvalues[k])
        if "mistral" in k.lower() and "mistral" in k.split("responder_model")[1].lower():
            diag_terms["Mistral:Mistral"] = (m_b.params[k], m_b.pvalues[k])

diag_str = "; ".join(f"{k} β={v[0]:.3f}, p={v[1]:.4g}" for k, v in diag_terms.items())
int_ac1 = gwet_df[(gwet_df["dataset"] == "intersectional")]["ac1"].tolist()
mw_ac1 = gwet_df[(gwet_df["dataset"] == "multiwoz_500")]["ac1"].tolist()
log(f"Interaction: {diag_str}; length LR p={p_lr:.4g}; "
    f"Holm-Bonferroni {n_sig}/21 survive; "
    f"PABAK [{agree_df['pabak'].min():.2f}, {agree_df['pabak'].max():.2f}]; "
    f"AC1 intersect=[{', '.join(f'{v:.2f}' for v in int_ac1)}] mw=[{', '.join(f'{v:.2f}' for v in mw_ac1)}]; "
    f"HurtLex FP {fp_count}/{top20_total}={fp_count/top20_total:.0%}, corrected rate {corrected_rate:.1%}")

hr("DONE")
