"""
compute_statistics.py — Produce statistical CSVs from operational evaluation outputs.

Reads evaluation CSVs from ../evaluations/ and response CSVs from ../responses/.
Outputs all tables to ../statistics/operational/.

Usage:
    python scripts/compute_statistics.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.formula.api import logit

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
EVAL_DIR = os.path.join(BASE_DIR, "evaluations")
RESP_DIR = os.path.join(BASE_DIR, "responses")
OUT_DIR = os.path.join(BASE_DIR, "statistics", "operational")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Judge/version combinations ───────────────────────────
JUDGES = ["gemma3_1b", "gemma3_latest", "llama3.1_latest", "mistral_latest"]
VERSIONS = ["v1", "v2"]

# Human-readable names
JUDGE_DISPLAY = {
    "gemma3_1b": "Gemma 1B",
    "gemma3_latest": "Gemma 4.3B",
    "llama3.1_latest": "Llama 3.1",
    "mistral_latest": "Mistral",
}

# Map judge model name to the responder model it "owns"
JUDGE_OWN_MODEL = {
    "gemma3_1b": "gemma3",
    "gemma3_latest": "gemma3",
    "llama3.1_latest": "llama3.1",
    "mistral_latest": "mistral",
}

# Dictionary methods
DICT_METHODS = {
    "Bias-Lexicon": "bias_label",
    "HurtLex": "hurtlex_label",
    "Toxicity": "toxicity_label",
}


# ── Helpers ──────────────────────────────────────────────

def load_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  SKIP: {path} not found")
    return None


def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  Saved {name} ({len(df)} rows)")


# ── Load data ────────────────────────────────────────────
print("Loading evaluation data...")

# Dictionary response-level
dict_resp = load_if_exists(os.path.join(EVAL_DIR, "dictionary_operational.csv"))

# Dictionary sentence-level
dict_sent = load_if_exists(os.path.join(EVAL_DIR, "dictionary_operational_sentences.csv"))

# Judge evaluations: load all available judge/version combos for v1
judge_v1 = {}
judge_all = {}
for judge in JUDGES:
    for version in VERSIONS:
        path = os.path.join(EVAL_DIR, f"judge_operational_{judge}_{version}.csv")
        df = load_if_exists(path)
        if df is not None:
            # Drop parse failures
            df = df[df["bias_label"] != -1].copy()
            judge_all[(judge, version)] = df
            if version == "v1":
                judge_v1[judge] = df

# Refusals and non-refusals
refusals = load_if_exists(os.path.join(RESP_DIR, "operational_refusals.csv"))
non_refusals = load_if_exists(os.path.join(RESP_DIR, "operational_non_refusals.csv"))

# Combined responses (for total counts)
combined = load_if_exists(os.path.join(RESP_DIR, "operational_combined.csv"))

# ── Build unified label frame ────────────────────────────
# Merge keys for operational data
MERGE_KEYS = ["prompt_id", "model", "scale_direction"]

if non_refusals is not None:
    base = non_refusals[
        ["prompt_id", "nationality", "mental_health_status", "scenario",
         "model", "scale_direction", "response"]
    ].copy()
    n_base = len(base)
    print(f"Non-refusal responses: {n_base}")
else:
    base = pd.DataFrame()
    n_base = 0
    print("WARNING: No non-refusal responses found")


def merge_labels(base_df, source_df, label_col, target_col, merge_keys=MERGE_KEYS):
    """Merge a label column from source into base."""
    if source_df is None or base_df is None or len(base_df) == 0:
        return base_df
    cols = merge_keys + [label_col]
    available = [c for c in cols if c in source_df.columns]
    if label_col not in available:
        return base_df
    sub = source_df[available].drop_duplicates(subset=merge_keys)
    sub = sub.rename(columns={label_col: target_col})
    return base_df.merge(sub, on=merge_keys, how="left")


# Merge dictionary labels
if dict_resp is not None:
    for method_name, col in DICT_METHODS.items():
        if col in dict_resp.columns:
            base = merge_labels(base, dict_resp, col, f"bl_{method_name}")

# Merge judge v1 labels
for judge, df in judge_v1.items():
    display = JUDGE_DISPLAY[judge]
    base = merge_labels(base, df, "bias_label", f"bl_{display}")

# Also store severity columns
if dict_resp is not None:
    for sev_col_name in ["severity", "hurtlex_severity", "toxicity_score"]:
        if sev_col_name in dict_resp.columns:
            base = merge_labels(base, dict_resp, sev_col_name, f"sev_{sev_col_name}")

for judge, df in judge_v1.items():
    display = JUDGE_DISPLAY[judge]
    if "severity" in df.columns:
        base = merge_labels(base, df, "severity", f"sev_{display}")

# Collect all method names (dict + v1 judges)
ALL_METHODS = []
for m in DICT_METHODS.keys():
    col = f"bl_{m}"
    if col in base.columns:
        ALL_METHODS.append(m)
for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    col = f"bl_{display}"
    if col in base.columns:
        ALL_METHODS.append(display)

print(f"Methods available: {ALL_METHODS}")

# ══════════════════════════════════════════════════════════
# 1. flagging_rates.csv
# ══════════════════════════════════════════════════════════
print("\n[1] Flagging rates...")
fr_rows = []
for method in ALL_METHODS:
    col = f"bl_{method}"
    if col not in base.columns:
        continue
    valid = base[col].dropna()
    n = len(valid)
    flagged = int((valid == 1).sum())
    rate = round(flagged / n, 4) if n > 0 else 0.0
    fr_rows.append({"method": method, "n": n, "flagged": flagged, "bias_rate": rate})

flagging_rates = pd.DataFrame(fr_rows).sort_values("bias_rate")
save_csv(flagging_rates, "flagging_rates.csv")


# ══════════════════════════════════════════════════════════
# 2. pairwise_kappa.csv (with bootstrap CI + Holm-Bonferroni)
# ══════════════════════════════════════════════════════════
print("\n[2] Pairwise kappa with bootstrap CI...")


def bootstrap_kappa_ci(labels_a, labels_b, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for Cohen's kappa."""
    rng = np.random.RandomState(seed)
    n = len(labels_a)
    kappas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        a_boot = labels_a[idx]
        b_boot = labels_b[idx]
        try:
            kappas[i] = cohen_kappa_score(a_boot, b_boot)
        except ValueError:
            kappas[i] = np.nan
    kappas = kappas[~np.isnan(kappas)]
    if len(kappas) == 0:
        return np.nan, np.nan
    lo = np.percentile(kappas, 100 * alpha / 2)
    hi = np.percentile(kappas, 100 * (1 - alpha / 2))
    return lo, hi


pk_rows = []
for i, m1 in enumerate(ALL_METHODS):
    for j, m2 in enumerate(ALL_METHODS):
        if j <= i:
            continue
        c1, c2 = f"bl_{m1}", f"bl_{m2}"
        if c1 not in base.columns or c2 not in base.columns:
            continue
        mask = base[c1].notna() & base[c2].notna()
        sub = base[mask]
        if len(sub) < 5:
            continue
        a = sub[c1].astype(int).values
        b = sub[c2].astype(int).values
        try:
            k = cohen_kappa_score(a, b)
        except ValueError:
            continue
        ci_lo, ci_hi = bootstrap_kappa_ci(a, b)
        # p-value: test if kappa significantly different from 0
        # Use bootstrap: proportion of bootstrap samples crossing 0
        pk_rows.append({
            "method_a": m1, "method_b": m2,
            "cohens_kappa": round(k, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "raw_p": np.nan,  # placeholder, filled below
        })

# Bootstrap p-value: proportion of resamples with kappa <= 0 (two-sided)
for row in pk_rows:
    c1 = f"bl_{row['method_a']}"
    c2 = f"bl_{row['method_b']}"
    mask = base[c1].notna() & base[c2].notna()
    sub = base[mask]
    a = sub[c1].astype(int).values
    b = sub[c2].astype(int).values
    rng = np.random.RandomState(42)
    n = len(a)
    count_le_zero = 0
    n_boot = 10000
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            k_boot = cohen_kappa_score(a[idx], b[idx])
        except ValueError:
            continue
        if k_boot <= 0:
            count_le_zero += 1
    row["raw_p"] = round(count_le_zero / n_boot, 6)

# Holm-Bonferroni correction
pk_df = pd.DataFrame(pk_rows)
if len(pk_df) > 0:
    pk_df = pk_df.sort_values("raw_p").reset_index(drop=True)
    m_tests = len(pk_df)
    adjusted = []
    for rank_idx in range(m_tests):
        adj_p = min(pk_df.loc[rank_idx, "raw_p"] * (m_tests - rank_idx), 1.0)
        adjusted.append(round(adj_p, 6))
    pk_df["holm_adjusted_p"] = adjusted
    pk_df = pk_df.drop(columns=["raw_p"])
save_csv(pk_df, "pairwise_kappa.csv")


# ══════════════════════════════════════════════════════════
# 3. pairwise_ac1_pabak.csv
# ══════════════════════════════════════════════════════════
print("\n[3] Pairwise AC1 and PABAK...")
ac1_rows = []
for i, m1 in enumerate(ALL_METHODS):
    for j, m2 in enumerate(ALL_METHODS):
        if j <= i:
            continue
        c1, c2 = f"bl_{m1}", f"bl_{m2}"
        if c1 not in base.columns or c2 not in base.columns:
            continue
        mask = base[c1].notna() & base[c2].notna()
        sub = base[mask]
        if len(sub) < 5:
            continue
        a = sub[c1].astype(int).values
        b = sub[c2].astype(int).values
        # Raw agreement
        pa = np.mean(a == b)
        # Cohen's kappa
        try:
            kappa = cohen_kappa_score(a, b)
        except ValueError:
            kappa = np.nan
        # Gwet's AC1
        pbar = (np.mean(a) + np.mean(b)) / 2
        pe_ac1 = 2 * pbar * (1 - pbar)
        if pe_ac1 < 1.0:
            ac1 = (pa - pe_ac1) / (1 - pe_ac1)
        else:
            ac1 = np.nan
        # PABAK
        pabak = 2 * pa - 1

        ac1_rows.append({
            "method_a": m1, "method_b": m2,
            "raw_agreement": round(pa, 4),
            "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else np.nan,
            "gwets_ac1": round(ac1, 4) if not np.isnan(ac1) else np.nan,
            "pabak": round(pabak, 4),
        })

save_csv(pd.DataFrame(ac1_rows), "pairwise_ac1_pabak.csv")


# ══════════════════════════════════════════════════════════
# 4. severity_spearman.csv
# ══════════════════════════════════════════════════════════
print("\n[4] Severity Spearman correlations...")

# Collect all severity columns available
sev_map = {}
for method in ALL_METHODS:
    # Try various severity column names
    for candidate in [f"sev_{method}", f"sev_severity", f"sev_hurtlex_severity",
                      f"sev_toxicity_score"]:
        if candidate in base.columns:
            sev_map[method] = candidate
            break

sp_rows = []
for i, m1 in enumerate(ALL_METHODS):
    for j, m2 in enumerate(ALL_METHODS):
        if j <= i:
            continue
        s1 = sev_map.get(m1)
        s2 = sev_map.get(m2)
        if s1 is None or s2 is None:
            continue
        if s1 not in base.columns or s2 not in base.columns:
            continue
        mask = base[s1].notna() & base[s2].notna()
        sub = base[mask]
        if len(sub) < 5:
            continue
        if sub[s1].nunique() < 2 and sub[s2].nunique() < 2:
            continue
        rho, p = spearmanr(sub[s1], sub[s2])
        sp_rows.append({
            "method_a": m1, "method_b": m2,
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 6),
        })

save_csv(pd.DataFrame(sp_rows), "severity_spearman.csv")


# ══════════════════════════════════════════════════════════
# 5. granularity_inter.csv
# ══════════════════════════════════════════════════════════
print("\n[5] Granularity (sentence vs response inflation)...")
gran_rows = []

if dict_sent is not None and dict_resp is not None:
    for method_name, bl_col in DICT_METHODS.items():
        # Sentence-level rate
        if bl_col in dict_sent.columns:
            sent_valid = dict_sent[bl_col].dropna()
            sent_rate = sent_valid.mean() if len(sent_valid) > 0 else np.nan
        else:
            sent_rate = np.nan

        # Response-level rate
        resp_col = f"bl_{method_name}"
        if resp_col in base.columns:
            resp_valid = base[resp_col].dropna()
            resp_rate = resp_valid.mean() if len(resp_valid) > 0 else np.nan
        else:
            resp_rate = np.nan

        inflation = round(resp_rate / sent_rate, 4) if (
            not np.isnan(sent_rate) and sent_rate > 0
        ) else np.nan

        gran_rows.append({
            "method": method_name,
            "sentence_rate": round(sent_rate, 4) if not np.isnan(sent_rate) else np.nan,
            "response_rate": round(resp_rate, 4) if not np.isnan(resp_rate) else np.nan,
            "inflation": inflation,
        })

# Judges: sentence-level would require sentence-level judge evals
# For now, only dictionary methods have sentence-level data
save_csv(pd.DataFrame(gran_rows), "granularity_inter.csv")


# ══════════════════════════════════════════════════════════
# 6. refusal_rates.csv
# ══════════════════════════════════════════════════════════
print("\n[6] Refusal rates...")
ref_rows = []

if combined is not None:
    combined["is_refusal"] = False
    if refusals is not None and len(refusals) > 0:
        refusal_keys = set(
            zip(refusals["prompt_id"], refusals["model"], refusals["scale_direction"])
        )
        combined["is_refusal"] = combined.apply(
            lambda r: (r["prompt_id"], r["model"], r["scale_direction"]) in refusal_keys,
            axis=1,
        )

    for breakdown, col in [("model", "model"),
                           ("nationality", "nationality"),
                           ("mental_health_status", "mental_health_status"),
                           ("scenario", "scenario")]:
        if col not in combined.columns:
            continue
        for grp in sorted(combined[col].unique()):
            sub = combined[combined[col] == grp]
            n = len(sub)
            refs = int(sub["is_refusal"].sum())
            ref_rows.append({
                "breakdown": breakdown, "group": grp, "n": n,
                "refusals": refs,
                "refusal_rate": round(refs / n, 4) if n > 0 else 0,
            })

save_csv(pd.DataFrame(ref_rows), "refusal_rates.csv")


# ══════════════════════════════════════════════════════════
# 7. self_eval_rates.csv
# ══════════════════════════════════════════════════════════
print("\n[7] Self-evaluation rates...")
se_rows = []

for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    bl_col = f"bl_{display}"
    if bl_col not in base.columns:
        continue
    own_responder = JUDGE_OWN_MODEL[judge]
    for model in sorted(base["model"].unique()):
        sub = base[base["model"] == model]
        valid = sub[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        se_rows.append({
            "judge": display,
            "evaluated_model": model,
            "own_model": model == own_responder,
            "n": n,
            "flagged": flagged,
            "bias_rate": round(flagged / n, 4) if n > 0 else 0,
        })

save_csv(pd.DataFrame(se_rows), "self_eval_rates.csv")


# ══════════════════════════════════════════════════════════
# 8. self_eval_fisher.csv
# ══════════════════════════════════════════════════════════
print("\n[8] Self-evaluation Fisher exact tests...")
fisher_rows = []

se_df = pd.DataFrame(se_rows)
for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    sub = se_df[se_df["judge"] == display]
    if len(sub) == 0:
        continue
    own = sub[sub["own_model"] == True]
    other = sub[sub["own_model"] == False]
    if len(own) == 0 or len(other) == 0:
        continue
    # Aggregate across evaluated models
    own_flagged = own["flagged"].sum()
    own_n = own["n"].sum()
    own_unflagged = own_n - own_flagged
    other_flagged = other["flagged"].sum()
    other_n = other["n"].sum()
    other_unflagged = other_n - other_flagged

    # 2x2 table: [[own_flagged, own_unflagged], [other_flagged, other_unflagged]]
    table = np.array([[own_flagged, own_unflagged],
                      [other_flagged, other_unflagged]])
    or_val, p_val = fisher_exact(table)
    # Confidence interval for OR using log method
    # log(OR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d)
    a, b, c, d = own_flagged, own_unflagged, other_flagged, other_unflagged
    if all(v > 0 for v in [a, b, c, d]):
        log_or = np.log(or_val)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo = np.exp(log_or - 1.96 * se_log_or)
        ci_hi = np.exp(log_or + 1.96 * se_log_or)
    else:
        ci_lo, ci_hi = np.nan, np.nan

    fisher_rows.append({
        "judge": display,
        "or_value": round(or_val, 4),
        "p_value": round(p_val, 6),
        "ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
        "ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
    })

save_csv(pd.DataFrame(fisher_rows), "self_eval_fisher.csv")


# ══════════════════════════════════════════════════════════
# 8b. self_eval_bootstrap_or.csv
# ══════════════════════════════════════════════════════════
print("\n[8b] Self-evaluation bootstrap CIs on OR...")

rng_se = np.random.RandomState(99)
N_BOOT_SE = 10_000

def _or_ci_ha(a_, b_, c_, d_):
    """OR + log(OR) with Haldane-Anscombe correction for zero cells."""
    aa, bb, cc, dd = a_, b_, c_, d_
    if any(v == 0 for v in [aa, bb, cc, dd]):
        aa, bb, cc, dd = aa + 0.5, bb + 0.5, cc + 0.5, dd + 0.5
    or_v = (aa * dd) / (bb * cc)
    return or_v, np.log(or_v)

boot_or_rows = []
for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    bl_col = f"bl_{display}"
    if bl_col not in base.columns:
        continue
    own_responder = JUDGE_OWN_MODEL[judge]

    sub = base[[bl_col, "model"]].dropna(subset=[bl_col]).copy()
    sub["is_own"] = (sub["model"] == own_responder).astype(int)
    sub["bias_label"] = sub[bl_col].astype(int)

    own_labels = sub.loc[sub["is_own"] == 1, "bias_label"].values
    other_labels = sub.loc[sub["is_own"] == 0, "bias_label"].values
    n_own, n_other = len(own_labels), len(other_labels)

    if n_own == 0 or n_other == 0:
        continue

    # Point estimate
    a = int(own_labels.sum())
    b_val = n_own - a
    c = int(other_labels.sum())
    d_val = n_other - c
    or_point, _ = _or_ci_ha(a, b_val, c, d_val)

    # Bootstrap (stratified by is_own)
    boot_log_ors = []
    for _ in range(N_BOOT_SE):
        idx_own = rng_se.randint(0, n_own, size=n_own)
        idx_other = rng_se.randint(0, n_other, size=n_other)
        b_own = own_labels[idx_own]
        b_other = other_labels[idx_other]
        ba = int(b_own.sum())
        bb = n_own - ba
        bc = int(b_other.sum())
        bd = n_other - bc
        _, blog = _or_ci_ha(ba, bb, bc, bd)
        boot_log_ors.append(blog)

    boot_log_ors = np.array(boot_log_ors)
    ci_lo = np.exp(np.percentile(boot_log_ors, 2.5))
    ci_hi = np.exp(np.percentile(boot_log_ors, 97.5))

    # Get Fisher p from already-computed rows
    fisher_p = np.nan
    for fr in fisher_rows:
        if fr["judge"] == display:
            fisher_p = fr["p_value"]
            break

    boot_or_rows.append({
        "judge": display,
        "or_value": round(or_point, 4),
        "ci_lower_boot": round(ci_lo, 4),
        "ci_upper_boot": round(ci_hi, 4),
        "p_fisher": fisher_p,
    })
    print(f"  {display}: OR={or_point:.3f}, Boot 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

save_csv(pd.DataFrame(boot_or_rows), "self_eval_bootstrap_or.csv")


# ══════════════════════════════════════════════════════════
# 9. self_eval_logistic.csv
# ══════════════════════════════════════════════════════════
print("\n[9] Self-evaluation logistic regression...")

# Build stacked frame: one row per (response, judge) with bias_label
logistic_rows = []
for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    bl_col = f"bl_{display}"
    if bl_col not in base.columns:
        continue
    sub = base[["prompt_id", "model", "scale_direction", "response", bl_col]].copy()
    sub = sub.rename(columns={bl_col: "bias_label"})
    sub = sub.dropna(subset=["bias_label"])
    sub["bias_label"] = sub["bias_label"].astype(int)
    sub["judge_model"] = display
    sub["responder_model"] = sub["model"]
    sub["own_judge"] = (sub["model"] == JUDGE_OWN_MODEL[judge]).astype(int)
    # Version column: all v1 for this base frame
    sub["version"] = "v1"
    sub["scale"] = sub["scale_direction"]
    # Response length
    sub["response_length"] = sub["response"].astype(str).str.len()
    sub["log_length"] = np.log1p(sub["response_length"])
    logistic_rows.append(sub)

# Also add v2 data if available
for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    if (judge, "v2") not in judge_all:
        continue
    df_v2 = judge_all[(judge, "v2")].copy()
    if non_refusals is not None:
        df_v2 = df_v2.merge(
            non_refusals[MERGE_KEYS].drop_duplicates(),
            on=MERGE_KEYS, how="inner"
        )
    df_v2 = df_v2[df_v2["bias_label"].isin([0, 1])].copy()
    merged_v2 = df_v2.merge(
        base[["prompt_id", "model", "scale_direction", "response"]].drop_duplicates(subset=MERGE_KEYS),
        on=MERGE_KEYS, how="left"
    )
    merged_v2["judge_model"] = display
    merged_v2["responder_model"] = merged_v2["model"]
    merged_v2["own_judge"] = (merged_v2["model"] == JUDGE_OWN_MODEL[judge]).astype(int)
    merged_v2["version"] = "v2"
    merged_v2["scale"] = merged_v2["scale_direction"]
    merged_v2["response_length"] = merged_v2["response"].astype(str).str.len()
    merged_v2["log_length"] = np.log1p(merged_v2["response_length"])
    merged_v2["bias_label"] = merged_v2["bias_label"].astype(int)
    logistic_rows.append(merged_v2)

logistic_df = pd.DataFrame()
if logistic_rows:
    logistic_df = pd.concat(logistic_rows, ignore_index=True)
    logistic_df = logistic_df.dropna(subset=["bias_label", "log_length"])
    print(f"  Logistic regression frame: {len(logistic_df)} rows")

logistic_result_rows = []
if len(logistic_df) > 10:
    formula = "bias_label ~ C(judge_model) * C(responder_model) + C(version) + C(scale) + log_length"
    try:
        model_fit = logit(formula, data=logistic_df).fit(disp=0)
        for term in model_fit.params.index:
            logistic_result_rows.append({
                "term": term,
                "beta": round(model_fit.params[term], 4),
                "se": round(model_fit.bse[term], 4),
                "p": round(model_fit.pvalues[term], 6),
                "or_value": round(np.exp(model_fit.params[term]), 4),
            })
        print(f"  Logistic model fitted: {len(logistic_result_rows)} terms")
    except Exception as e:
        print(f"  Logistic regression failed: {e}")

save_csv(pd.DataFrame(logistic_result_rows), "self_eval_logistic.csv")


# ══════════════════════════════════════════════════════════
# 10. mcnemar_v1v2.csv
# ══════════════════════════════════════════════════════════
print("\n[10] McNemar v1 vs v2...")
mcn_rows = []

for judge in JUDGES:
    display = JUDGE_DISPLAY[judge]
    if (judge, "v1") not in judge_all or (judge, "v2") not in judge_all:
        continue
    v1 = judge_all[(judge, "v1")].copy()
    v2 = judge_all[(judge, "v2")].copy()
    # Filter to non-refusals
    if non_refusals is not None:
        nr_keys = non_refusals[MERGE_KEYS].drop_duplicates()
        v1 = v1.merge(nr_keys, on=MERGE_KEYS, how="inner")
        v2 = v2.merge(nr_keys, on=MERGE_KEYS, how="inner")

    v1 = v1[v1["bias_label"].isin([0, 1])].drop_duplicates(subset=MERGE_KEYS)
    v2 = v2[v2["bias_label"].isin([0, 1])].drop_duplicates(subset=MERGE_KEYS)

    paired = v1.merge(v2, on=MERGE_KEYS, suffixes=("_v1", "_v2"))
    if len(paired) == 0:
        continue
    # Count flips
    n_01 = int(((paired["bias_label_v1"] == 0) & (paired["bias_label_v2"] == 1)).sum())
    n_10 = int(((paired["bias_label_v1"] == 1) & (paired["bias_label_v2"] == 0)).sum())
    # McNemar test
    n_disc = n_01 + n_10
    if n_disc > 0:
        if n_disc < 25:
            # Exact binomial test
            from scipy.stats import binomtest
            try:
                p_val = binomtest(n_01, n_disc, 0.5).pvalue
            except Exception:
                p_val = np.nan
            chi2 = np.nan
        else:
            chi2 = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
            from scipy.stats import chi2 as chi2_dist
            p_val = chi2_dist.sf(chi2, df=1)
    else:
        chi2, p_val = np.nan, np.nan

    mcn_rows.append({
        "judge": display,
        "n_flips_01": n_01,
        "n_flips_10": n_10,
        "mcnemar_chi2": round(chi2, 4) if not np.isnan(chi2) else np.nan,
        "p_value": round(p_val, 6) if not np.isnan(p_val) else np.nan,
    })

save_csv(pd.DataFrame(mcn_rows), "mcnemar_v1v2.csv")


# ══════════════════════════════════════════════════════════
# 11. demographic_flagging.csv
# ══════════════════════════════════════════════════════════
print("\n[11] Demographic flagging rates...")
dem_rows = []

for method in ALL_METHODS:
    bl_col = f"bl_{method}"
    if bl_col not in base.columns:
        continue
    if "nationality" not in base.columns:
        continue
    for nat in sorted(base["nationality"].unique()):
        sub = base[base["nationality"] == nat]
        valid = sub[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        dem_rows.append({
            "method": method,
            "nationality": nat,
            "bias_rate": round(flagged / n, 4) if n > 0 else 0,
            "n": n,
        })

save_csv(pd.DataFrame(dem_rows), "demographic_flagging.csv")


print("\nAll operational statistics generated.")
