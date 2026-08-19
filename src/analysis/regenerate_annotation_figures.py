#!/usr/bin/env python3
"""
Regenerate fig11–fig15 (annotation-based figures) with corrected labels.

Changes from originals:
- "MultiWOZ" → "Naturalistic dataset" in titles/legends
- "Intersectional" → "Declarative" in titles (fig11)
- "Gemma 27B" → "Gemma 4.3B" in fig14 (model size fix)
- Consistent "Coder 1 (lead)" labelling

Data sources:
- results/tables/table_inter_annotator_pairwise_intersectional.csv
- results/tables/table_inter_annotator_pairwise_multiwoz.csv
- hpc/operational/statistics/operational/pairwise_kappa.csv
- data/annotations/qualitative_analysis_merged.csv (per-coder rates)
- results/tables/table_method_accuracy.csv
- results/tables/table_method_accuracy_multiwoz.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "..", "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
TABLES = os.path.join(RESULTS, "tables")
DATA = os.path.join(ROOT, "..", "..", "data", "annotations")

CODERS = ["coder1", "coder2", "coder3", "coder4", "coder5", "coder6"]
CODER_LABELS = ["Coder 1\n(lead)", "Coder 2", "Coder 3", "Coder 4", "Coder 5", "Coder 6"]


def build_kappa_matrix(df):
    """Build 6x6 kappa matrix from pairwise CSV."""
    n = len(CODERS)
    mat = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i = CODERS.index(row["coder_a"])
        j = CODERS.index(row["coder_b"])
        mat[i, j] = row["cohens_kappa"]
        mat[j, i] = row["cohens_kappa"]
    return mat


def plot_heatmap(mat, title, outpath, vmin=-0.1, vmax=0.7, cmap="Reds"):
    """Plot a 6x6 kappa heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.eye(len(CODERS), dtype=bool)
    sns.heatmap(mat, annot=True, fmt=".3f", cmap=cmap, center=0,
                vmin=vmin, vmax=vmax, mask=mask,
                xticklabels=CODER_LABELS, yticklabels=CODER_LABELS,
                linewidths=1, linecolor="white",
                cbar_kws={"label": "Cohen's κ"}, ax=ax)
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved {os.path.basename(outpath)}")


# ═══════════════════════════════════════════════════════════
# FIG 11: Declarative inter-rater heatmap
# ═══════════════════════════════════════════════════════════
print("Fig 11: Declarative inter-rater heatmap")
pw_decl = pd.read_csv(os.path.join(TABLES, "table_inter_annotator_pairwise_intersectional.csv"))
mat_decl = build_kappa_matrix(pw_decl)
n_decl = 38  # from table_inter_annotator.csv
plot_heatmap(mat_decl,
             f"Pairwise Cohen's κ — Declarative Dataset (n={n_decl})",
             os.path.join(FIGURES, "fig11_inter_rater_heatmap_declarative.png"))

# Verify against CSV
print(f"  Coder1×Coder2 = {mat_decl[0,1]:.3f} (CSV: {pw_decl[(pw_decl.coder_a=='coder1')&(pw_decl.coder_b=='coder2')]['cohens_kappa'].iloc[0]:.4f})")
print(f"  Coder4×Coder6 = {mat_decl[3,5]:.3f} (CSV: {pw_decl[(pw_decl.coder_a=='coder4')&(pw_decl.coder_b=='coder6')]['cohens_kappa'].iloc[0]:.4f})")


# ═══════════════════════════════════════════════════════════
# FIG 12: Naturalistic dataset inter-rater heatmap
# ═══════════════════════════════════════════════════════════
print("Fig 12: Naturalistic dataset inter-rater heatmap")
pw_mwoz = pd.read_csv(os.path.join(TABLES, "table_inter_annotator_pairwise_multiwoz.csv"))
mat_mwoz = build_kappa_matrix(pw_mwoz)
n_mwoz = 46
plot_heatmap(mat_mwoz,
             f"Pairwise Cohen's κ — Naturalistic Dataset (n={n_mwoz})",
             os.path.join(FIGURES, "fig12_inter_rater_heatmap_multiwoz.png"))

print(f"  Coder1×Coder3 = {mat_mwoz[0,2]:.3f} (CSV: {pw_mwoz[(pw_mwoz.coder_a=='coder1')&(pw_mwoz.coder_b=='coder3')]['cohens_kappa'].iloc[0]:.4f})")
print(f"  Coder4×Coder6 = {mat_mwoz[3,5]:.3f} (CSV: {pw_mwoz[(pw_mwoz.coder_a=='coder4')&(pw_mwoz.coder_b=='coder6')]['cohens_kappa'].iloc[0]:.4f})")


# ═══════════════════════════════════════════════════════════
# FIG 13: Per-coder bias rates across three datasets
# ═══════════════════════════════════════════════════════════
print("Fig 13: Per-coder bias rates")

# Compute per-coder rates from merged annotations
merged = pd.read_csv(os.path.join(DATA, "qualitative_analysis_merged.csv"))
decl_labels = {c: merged[f"{c}_label"].dropna() for c in CODERS}
decl_rates = {c: labels.mean() * 100 for c, labels in decl_labels.items()}

# MultiWOZ annotations — load from each coder's file
LABEL_COL = "human_bias_label"
mwoz_rates = {}
for c in CODERS:
    coder_dir = os.path.join(DATA, c)
    mwoz_files = [f for f in os.listdir(coder_dir) if "multiwoz" in f.lower() and f.endswith(".csv")]
    if mwoz_files:
        try:
            df = pd.read_csv(os.path.join(coder_dir, mwoz_files[0]))
        except UnicodeDecodeError:
            df = pd.read_csv(os.path.join(coder_dir, mwoz_files[0]), encoding="latin-1")
        vals = pd.to_numeric(df[LABEL_COL], errors="coerce").dropna()
        mwoz_rates[c] = vals.mean() * 100 if len(vals) > 0 else 0.0
    else:
        mwoz_rates[c] = 0.0

# Operational annotations — exclude items where ANY coder marked "unclear"
oper_all = {}
for c in CODERS:
    coder_dir = os.path.join(DATA, c)
    oper_files = [f for f in os.listdir(coder_dir) if "operational" in f.lower() and (f.endswith(".csv") or f.endswith(".xlsx"))]
    if oper_files:
        fpath = os.path.join(coder_dir, oper_files[0])
        if fpath.endswith(".xlsx"):
            df = pd.read_excel(fpath)
        else:
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding="latin-1")
        if LABEL_COL in df.columns:
            raw = df[LABEL_COL].astype(str).str.strip().str.lower()
            oper_all[c] = pd.to_numeric(raw.replace("unclear", np.nan), errors="coerce").values

# Build unclear mask: exclude items where any coder has NaN
n_items = len(oper_all[CODERS[0]])
unclear_mask = np.zeros(n_items, dtype=bool)
for c in CODERS:
    if c in oper_all:
        unclear_mask |= np.isnan(oper_all[c])
print(f"  Operational: {n_items} items, {unclear_mask.sum()} with unclear, {(~unclear_mask).sum()} clean")

oper_rates = {}
for c in CODERS:
    if c in oper_all:
        vals = oper_all[c][~unclear_mask]
        oper_rates[c] = np.nanmean(vals) * 100 if len(vals) > 0 else 0.0
    else:
        oper_rates[c] = 0.0

coder_short = ["Coder 1", "Coder 2", "Coder 3", "Coder 4", "Coder 5", "Coder 6"]
x = np.arange(len(CODERS))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
d_vals = [decl_rates.get(c, 0) for c in CODERS]
o_vals = [oper_rates.get(c, 0) for c in CODERS]
m_vals = [mwoz_rates.get(c, 0) for c in CODERS]

ax.bar(x - width, d_vals, width, label="Declarative", color="#6aafe6")
ax.bar(x, o_vals, width, label="Operational", color="#f4a261")
ax.bar(x + width, m_vals, width, label="Naturalistic", color="#72c472")

ax.set_ylabel("Bias flagging rate (%)", fontsize=11)
ax.set_xlabel("Coder", fontsize=11)
ax.set_title("Per-coder bias-flagging rates across three datasets", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(coder_short)
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig13_per_coder_bias_rates.png"))
plt.close()
print(f"  Saved fig13_per_coder_bias_rates.png")
for c, d, o, m in zip(coder_short, d_vals, o_vals, m_vals):
    print(f"    {c}: decl={d:.1f}%, oper={o:.1f}%, nat={m:.1f}%")


# ═══════════════════════════════════════════════════════════
# FIG 14: Method accuracy vs 6-coder majority vote
# ═══════════════════════════════════════════════════════════
print("Fig 14: Method accuracy vs consensus")

acc_decl = pd.read_csv(os.path.join(TABLES, "table_method_accuracy.csv"))
acc_mwoz = pd.read_csv(os.path.join(TABLES, "table_method_accuracy_multiwoz.csv"))

# Clean method names — fix "Gemma 27B" → "Gemma 4.3B"
def clean_method(name):
    return name.replace("27B", "4.3B").replace("Judge (", "").replace(")", "")

# Declarative: average accuracy per method (some have multiple rows for different divergence types)
decl_avg = acc_decl.groupby("method")["accuracy"].mean().reset_index()
decl_avg["method"] = decl_avg["method"].apply(clean_method)
decl_avg = decl_avg.sort_values("accuracy", ascending=True)

# MultiWOZ
mwoz_avg = acc_mwoz.copy()
mwoz_avg["method"] = mwoz_avg["method"].apply(clean_method)
mwoz_avg = mwoz_avg.sort_values("accuracy", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: declarative divergence cases
colors_d = ["#e9967a" if "Lexicon" in m or "HurtLex" in m else "#3bb3a0" for m in decl_avg["method"]]
bars1 = ax1.barh(decl_avg["method"], decl_avg["accuracy"], color=colors_d, edgecolor="black")
for bar, acc in zip(bars1, decl_avg["accuracy"]):
    ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f"{acc:.0%}", va="center", fontsize=10)
ax1.set_xlim(0, 1.15)
ax1.set_xlabel("Accuracy vs. 6-coder majority vote", fontsize=10)
ax1.set_title("Declarative Divergence Cases", fontsize=12)

# Right panel: naturalistic dataset
colors_m = ["#e9967a" if "Lexicon" in m or "HurtLex" in m else "#3bb3a0" for m in mwoz_avg["method"]]
bars2 = ax2.barh(mwoz_avg["method"], mwoz_avg["accuracy"], color=colors_m, edgecolor="black")
for bar, acc in zip(bars2, mwoz_avg["accuracy"]):
    ax2.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f"{acc:.0%}", va="center", fontsize=10)
ax2.set_xlim(0, 1.15)
ax2.set_xlabel("Accuracy vs. 6-coder majority vote", fontsize=10)
ax2.set_title(f"Naturalistic Dataset (n={len(acc_mwoz)})", fontsize=12)

fig.suptitle("Method Accuracy Against 6-Coder Majority Vote", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig14_method_accuracy_vs_consensus.png"))
plt.close()
print(f"  Saved fig14_method_accuracy_vs_consensus.png")
for _, row in mwoz_avg.iterrows():
    print(f"    {row['method']}: {row['accuracy']:.0%}")


# ═══════════════════════════════════════════════════════════
# FIG 15: Operational inter-rater heatmap
# ═══════════════════════════════════════════════════════════
print("Fig 15: Operational inter-rater heatmap")
from sklearn.metrics import cohen_kappa_score

# Load all operational labels (handling xlsx for coder5)
oper_all_labels = {}
for c in CODERS:
    coder_dir = os.path.join(DATA, c)
    oper_files = [f for f in os.listdir(coder_dir) if "operational" in f.lower() and (f.endswith(".csv") or f.endswith(".xlsx"))]
    if oper_files:
        fpath = os.path.join(coder_dir, oper_files[0])
        if fpath.endswith(".xlsx"):
            df = pd.read_excel(fpath)
        else:
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding="latin-1")
        if LABEL_COL in df.columns:
            raw = df[LABEL_COL].astype(str).str.strip().str.lower()
            oper_all_labels[c] = pd.to_numeric(raw.replace("unclear", np.nan), errors="coerce").values

# Exclude items where ANY coder marked unclear (matches text: 50→35)
n_items = len(oper_all_labels[CODERS[0]])
oper_unclear = np.zeros(n_items, dtype=bool)
for c in CODERS:
    if c in oper_all_labels:
        oper_unclear |= np.isnan(oper_all_labels[c])
print(f"  {n_items} items, {oper_unclear.sum()} unclear, {(~oper_unclear).sum()} clean")

if len(oper_all_labels) == 6:
    n = len(CODERS)
    mat_oper = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i+1, n):
            a = oper_all_labels[CODERS[i]][~oper_unclear]
            b = oper_all_labels[CODERS[j]][~oper_unclear]
            k = cohen_kappa_score(a, b)
            mat_oper[i, j] = round(k, 3)
            mat_oper[j, i] = round(k, 3)

    # Use same style as fig11/12: Reds cmap, vmin=-0.1, vmax=0.7, with (lead) label
    plot_heatmap(mat_oper,
                 f"Pairwise Cohen's κ — Operational Dataset (n={(~oper_unclear).sum()})",
                 os.path.join(FIGURES, "fig15_inter_rater_heatmap_operational.png"),
                 vmin=-0.2, vmax=0.7, cmap="Reds")
    # Verify against text
    print(f"  Coder1×Coder6 = {mat_oper[0,5]:.3f} (text says 0.553)")
    print(f"  Coder4×Coder6 = {mat_oper[3,5]:.3f} (text says 0.462)")
    print(f"  Coder5×Coder6 = {mat_oper[4,5]:.3f} (text says 0.424)")
else:
    print(f"  ERROR: Only found {len(oper_all_labels)} coders with operational data")

print("\nDone. All figures regenerated.")
