"""
generate_tables.py — Produce table_*.csv summaries from raw score files.

Run before generate_figures.py:
    python3 generate_tables.py
    python3 generate_figures.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Load responses to identify non-refusal explanations ──────────
resp = pd.read_csv("../../data/all_model_responses_clean.csv")
expl = resp[resp["step"] == "explanation"].copy()
non_ref = expl[~expl["is_refusal"]].copy()
MERGE_KEYS = ["prompt_id", "model", "scale"]
print(f"Non-refusal explanations: {len(non_ref)}")

# ── Load all score files ─────────────────────────────────────────
dict_all = pd.read_csv("../../scores/dict_scores.csv")
dict_expl = dict_all[dict_all["step"] == "explanation"].copy()
dict_df = dict_expl.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
dict_df = dict_df.rename(columns={"bias_label": "bl_biaslex", "severity": "sev_biaslex"})

hurtlex_df = pd.read_csv("../../scores/hurtlex_scores.csv")
hurtlex_df = hurtlex_df.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
hurtlex_df = hurtlex_df.rename(columns={"bias_label": "bl_hurtlex", "severity": "sev_hurtlex"})

judge_gemma = pd.read_csv("../../scores/llm_judge_scores.csv")
judge_gemma = judge_gemma.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
judge_gemma = judge_gemma.rename(columns={"bias_label": "bl_gemma4b", "severity": "sev_gemma4b"})

judge_1b = pd.read_csv("../../scores/llm_judge_scores_1b.csv")
judge_1b = judge_1b.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
judge_1b = judge_1b.rename(columns={"bias_label": "bl_gemma1b", "severity": "sev_gemma1b"})

judge_llama = pd.read_csv("../../scores/llm_judge_scores_llama.csv")
judge_llama = judge_llama.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
judge_llama = judge_llama.rename(columns={"bias_label": "bl_llama", "severity": "sev_llama"})

judge_mistral = pd.read_csv("../../scores/llm_judge_scores_mistral.csv")
judge_mistral = judge_mistral.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
judge_mistral = judge_mistral.rename(columns={"bias_label": "bl_mistral", "severity": "sev_mistral"})

tox = pd.read_csv("../../scores/toxicity_scores.csv")
tox = tox.merge(non_ref[MERGE_KEYS], on=MERGE_KEYS, how="inner")
tox = tox.rename(columns={"toxicity_label": "bl_toxicity", "toxicity_score": "sev_toxicity"})

# ── Build unified dataframe ──────────────────────────────────────
base = non_ref[["prompt_id", "model", "scale", "nationality", "disability_type", "scenario"]].copy()

methods = {
    "Bias-Lexicon": ("bl_biaslex", "sev_biaslex", dict_df),
    "HurtLex": ("bl_hurtlex", "sev_hurtlex", hurtlex_df),
    "Gemma 4.3B": ("bl_gemma4b", "sev_gemma4b", judge_gemma),
    "Gemma 1B": ("bl_gemma1b", "sev_gemma1b", judge_1b),
    "Llama 3.1": ("bl_llama", "sev_llama", judge_llama),
    "Mistral": ("bl_mistral", "sev_mistral", judge_mistral),
    "Toxicity": ("bl_toxicity", "sev_toxicity", tox),
}

for name, (bl_col, sev_col, df) in methods.items():
    cols_to_merge = MERGE_KEYS + [bl_col, sev_col]
    available = [c for c in cols_to_merge if c in df.columns]
    base = base.merge(df[available].drop_duplicates(subset=MERGE_KEYS),
                      on=MERGE_KEYS, how="left")

for col in ["bl_gemma1b", "bl_gemma4b", "bl_llama", "bl_mistral"]:
    if col in base.columns:
        base.loc[base[col] == -1, col] = np.nan

print(f"Unified dataframe: {len(base)} rows, {len(base.columns)} columns")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Bias rates by method
# ══════════════════════════════════════════════════════════════════
rows = []
for name, (bl_col, sev_col, _) in methods.items():
    if bl_col in base.columns:
        valid = base[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        rate = flagged / n if n > 0 else 0
        rows.append({"method": name, "n": n, "flagged": flagged, "bias_rate": round(rate, 4)})

table_bias_rates = pd.DataFrame(rows).sort_values("bias_rate")
table_bias_rates.to_csv("../../results/tables/table_bias_rates.csv", index=False)
print(f"table_bias_rates.csv: {len(table_bias_rates)} rows")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Kappa and Spearman matrices + pairwise detail
# ══════════════════════════════════════════════════════════════════
method_names = list(methods.keys())
bl_cols = {name: methods[name][0] for name in method_names}
sev_cols = {name: methods[name][1] for name in method_names}

kappa_matrix = pd.DataFrame(index=method_names, columns=method_names, dtype=float)
spearman_matrix = pd.DataFrame(index=method_names, columns=method_names, dtype=float)
pairwise_rows = []

for i, m1 in enumerate(method_names):
    for j, m2 in enumerate(method_names):
        if i == j:
            kappa_matrix.loc[m1, m2] = 1.0
            spearman_matrix.loc[m1, m2] = 1.0
            continue
        bl1, bl2 = bl_cols[m1], bl_cols[m2]
        sev1, sev2 = sev_cols[m1], sev_cols[m2]
        mask = base[bl1].notna() & base[bl2].notna()
        sub = base[mask]
        if len(sub) < 5:
            continue

        k = cohen_kappa_score(sub[bl1].astype(int), sub[bl2].astype(int))
        kappa_matrix.loc[m1, m2] = round(k, 4)

        rho_val = None
        if sev1 in sub.columns and sev2 in sub.columns:
            s_mask = sub[sev1].notna() & sub[sev2].notna()
            if s_mask.sum() >= 5:
                rho, p = spearmanr(sub.loc[s_mask, sev1], sub.loc[s_mask, sev2])
                spearman_matrix.loc[m1, m2] = round(rho, 4)
                rho_val = round(rho, 4)

        if i < j:
            cm = confusion_matrix(sub[bl1].astype(int), sub[bl2].astype(int), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            pairwise_rows.append({
                "method_a": m1, "method_b": m2,
                "n": len(sub), "cohens_kappa": round(k, 4),
                "spearman_rho": rho_val,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "agreement_rate": round((tp + tn) / len(sub), 4),
            })

kappa_matrix.to_csv("../../results/tables/table_kappa_matrix.csv")
spearman_matrix.to_csv("../../results/tables/table_spearman_matrix.csv")
print("table_kappa_matrix.csv and table_spearman_matrix.csv saved")

table_pairwise = pd.DataFrame(pairwise_rows)
table_pairwise.to_csv("../../results/tables/table_pairwise_detail.csv", index=False)
print(f"table_pairwise_detail.csv: {len(table_pairwise)} rows")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Confusion matrices (long format)
# ══════════════════════════════════════════════════════════════════
cm_rows = []
for _, row in table_pairwise.iterrows():
    for cell in ["TP", "FP", "FN", "TN"]:
        cm_rows.append({
            "method_a": row["method_a"], "method_b": row["method_b"],
            "cell": cell, "count": row[cell]
        })
table_cm = pd.DataFrame(cm_rows)
table_cm.to_csv("../../results/tables/table_confusion_matrices.csv", index=False)
print(f"table_confusion_matrices.csv: {len(table_cm)} rows")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Bias rates by model
# ══════════════════════════════════════════════════════════════════
bm_rows = []
for name, (bl_col, _, _) in methods.items():
    if bl_col not in base.columns:
        continue
    for model in sorted(base["model"].unique()):
        sub = base[base["model"] == model]
        valid = sub[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        bm_rows.append({"method": name, "model": model, "n": n,
                         "flagged": flagged, "bias_rate": round(flagged / n, 4) if n > 0 else 0})
table_bias_by_model = pd.DataFrame(bm_rows)
table_bias_by_model.to_csv("../../results/tables/table_bias_by_model.csv", index=False)
print(f"table_bias_by_model.csv: {len(table_bias_by_model)} rows")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Self-evaluation
# ══════════════════════════════════════════════════════════════════
judge_map = {"Gemma 4.3B": "gemma3", "Gemma 1B": "gemma3", "Llama 3.1": "llama3.1", "Mistral": "mistral"}
se_rows = []
for judge_name, (bl_col, _, _) in methods.items():
    if judge_name not in judge_map:
        continue
    for model in sorted(base["model"].unique()):
        sub = base[base["model"] == model]
        valid = sub[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        se_rows.append({
            "judge": judge_name, "evaluated_model": model,
            "own_model": model == judge_map[judge_name],
            "n": n, "flagged": flagged,
            "bias_rate": round(flagged / n, 4) if n > 0 else 0
        })
table_self = pd.DataFrame(se_rows)
table_self.to_csv("../../results/tables/table_self_evaluation.csv", index=False)
print(f"table_self_evaluation.csv: {len(table_self)} rows")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Bias rates by nationality
# ══════════════════════════════════════════════════════════════════
bn_rows = []
for name, (bl_col, _, _) in methods.items():
    if bl_col not in base.columns:
        continue
    for nat in sorted(base["nationality"].unique()):
        sub = base[base["nationality"] == nat]
        valid = sub[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        bn_rows.append({"method": name, "nationality": nat, "n": n,
                         "flagged": flagged, "bias_rate": round(flagged / n, 4) if n > 0 else 0})
pd.DataFrame(bn_rows).to_csv("../../results/tables/table_bias_by_nationality.csv", index=False)
print(f"table_bias_by_nationality.csv saved")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Bias rates by disability type
# ══════════════════════════════════════════════════════════════════
bd_rows = []
for name, (bl_col, _, _) in methods.items():
    if bl_col not in base.columns:
        continue
    for dt in sorted(base["disability_type"].unique()):
        sub = base[base["disability_type"] == dt]
        valid = sub[bl_col].dropna()
        n = len(valid)
        flagged = int((valid == 1).sum())
        bd_rows.append({"method": name, "disability_type": dt, "n": n,
                         "flagged": flagged, "bias_rate": round(flagged / n, 4) if n > 0 else 0})
pd.DataFrame(bd_rows).to_csv("../../results/tables/table_bias_by_disability.csv", index=False)
print(f"table_bias_by_disability.csv saved")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Refusals
# ══════════════════════════════════════════════════════════════════
ref_rows = []
for breakdown, col in [("model", "model"), ("nationality", "nationality"),
                        ("disability_type", "disability_type"), ("scenario", "scenario")]:
    for grp in sorted(expl[col].unique()):
        sub = expl[expl[col] == grp]
        n = len(sub)
        refs = int(sub["is_refusal"].sum())
        ref_rows.append({"breakdown": breakdown, "group": grp, "n": n,
                          "refusals": refs, "refusal_rate": round(refs / n, 4) if n > 0 else 0})
pd.DataFrame(ref_rows).to_csv("../../results/tables/table_refusals.csv", index=False)
print(f"table_refusals.csv saved")

# ══════════════════════════════════════════════════════════════════
#  TABLE: MultiWOZ
# ══════════════════════════════════════════════════════════════════
mw = pd.read_csv("../../multiwoz/multiwoz_dict_scores.csv")
n_dlg = mw["dialogue_id"].nunique()
n_turns = len(mw)
mw_rows = [
    {"method": "Bias-Lexicon (turn)", "n_dialogues": n_dlg, "n_turns": n_turns,
     "flagged": int(mw["bias_label"].sum()),
     "bias_rate": round(mw["bias_label"].mean(), 6)},
    {"method": "HurtLex (turn)", "n_dialogues": n_dlg, "n_turns": n_turns,
     "flagged": int(mw["hurtlex_label"].sum()),
     "bias_rate": round(mw["hurtlex_label"].mean(), 6)},
]
dlg_bl = mw.groupby("dialogue_id")["bias_label"].max()
dlg_hl = mw.groupby("dialogue_id")["hurtlex_label"].max()
mw_rows.append({"method": "Bias-Lexicon (dialogue)", "n_dialogues": n_dlg, "n_turns": n_turns,
                 "flagged": int(dlg_bl.sum()), "bias_rate": round(dlg_bl.mean(), 6)})
mw_rows.append({"method": "HurtLex (dialogue)", "n_dialogues": n_dlg, "n_turns": n_turns,
                 "flagged": int(dlg_hl.sum()), "bias_rate": round(dlg_hl.mean(), 6)})
pd.DataFrame(mw_rows).to_csv("../../results/tables/table_multiwoz.csv", index=False)
print(f"table_multiwoz.csv saved")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Cross-dataset comparison
# ══════════════════════════════════════════════════════════════════
cross_rows = []
for name, (bl_col, _, _) in [("Bias-Lexicon", methods["Bias-Lexicon"]),
                               ("HurtLex", methods["HurtLex"])]:
    valid = base[bl_col].dropna()
    cross_rows.append({
        "dataset": "Intersectional", "method": name,
        "n": len(valid), "flagged": int((valid == 1).sum()),
        "bias_rate": round((valid == 1).mean(), 4),
    })
cross_rows.append({"dataset": "MultiWOZ", "method": "Bias-Lexicon",
                    "n": n_turns, "flagged": int(mw["bias_label"].sum()),
                    "bias_rate": round(mw["bias_label"].mean(), 6)})
cross_rows.append({"dataset": "MultiWOZ", "method": "HurtLex",
                    "n": n_turns, "flagged": int(mw["hurtlex_label"].sum()),
                    "bias_rate": round(mw["hurtlex_label"].mean(), 6)})
pd.DataFrame(cross_rows).to_csv("../../results/tables/table_cross_dataset.csv", index=False)
print(f"table_cross_dataset.csv saved")

# ══════════════════════════════════════════════════════════════════
#  TABLE: Divergence cases
# ══════════════════════════════════════════════════════════════════
judge_bl_cols = ["bl_gemma4b", "bl_gemma1b", "bl_llama", "bl_mistral"]
judge_names_list = ["Gemma 4.3B", "Gemma 1B", "Llama 3.1", "Mistral"]
div_rows = []

for idx, row in base.iterrows():
    dict_label = row["bl_biaslex"]
    if pd.isna(dict_label):
        continue
    for jbl, jname in zip(judge_bl_cols, judge_names_list):
        judge_label = row[jbl]
        if pd.isna(judge_label):
            continue
        if int(dict_label) != int(judge_label):
            div_rows.append({
                "prompt_id": row["prompt_id"], "model": row["model"],
                "scale": row["scale"], "nationality": row["nationality"],
                "disability_type": row["disability_type"], "scenario": row["scenario"],
                "dict_label": int(dict_label), "judge": jname,
                "judge_label": int(judge_label),
                "divergence_type": "dict_only" if dict_label == 1 else "judge_only",
            })

table_div = pd.DataFrame(div_rows)
table_div.to_csv("../../results/tables/table_divergence_cases.csv", index=False)
print(f"table_divergence_cases.csv: {len(table_div)} rows")

print("\nAll tables generated.")
