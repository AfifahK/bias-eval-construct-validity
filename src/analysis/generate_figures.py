"""
generate_figures.py — Reusable figure generator for bias evaluation pipeline.

Reads table_*.csv files produced by generate_tables.py and produces PNG figures.
Run after generate_tables.py:
    python3 generate_tables.py
    python3 generate_figures.py

Outputs fig1_*.png through fig10_*.png in the current directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
sns.set_style("whitegrid")


def check_file(path):
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return False
    return True


# ══════════════════════════════════════════════════════════
# FIGURE 1: Bias rates by method
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_bias_rates.csv"):
    br = pd.read_csv("../../results/tables/table_bias_rates.csv").sort_values("bias_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(br)))
    bars = ax.barh(br["method"], br["bias_rate"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Bias Detection Rate", fontsize=11)
    ax.set_title(f"Bias Rates Across {len(br)} Evaluation Methods\n(n=293 non-refusal explanations)", fontsize=12)
    ax.set_xlim(0, 1.1)
    for bar, rate, n, flagged in zip(bars, br["bias_rate"], br["n"], br["flagged"]):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2,
                f"{rate:.1%} ({flagged}/{n})", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig1_bias_rates_by_method.png")
    plt.close()
    print("fig1 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 2: Cohen's κ matrix heatmap
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_kappa_matrix.csv"):
    k = pd.read_csv("../../results/tables/table_kappa_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(k, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(k, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-0.25, vmax=0.25,
                square=True, cbar_kws={"label": "Cohen's κ"}, mask=mask,
                linewidths=0.5, linecolor="gray", ax=ax)
    ax.set_title("Pairwise Cohen's κ Between Bias Evaluation Methods\n(diagonal masked; closer to 0 = chance agreement)", fontsize=12)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig2_kappa_heatmap.png")
    plt.close()
    print("fig2 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 3: Spearman ρ matrix heatmap
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_spearman_matrix.csv"):
    r = pd.read_csv("../../results/tables/table_spearman_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(r, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(r, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                square=True, cbar_kws={"label": "Spearman's ρ"}, mask=mask,
                linewidths=0.5, linecolor="gray", ax=ax)
    ax.set_title("Pairwise Spearman's ρ on Severity Scores", fontsize=12)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig3_spearman_heatmap.png")
    plt.close()
    print("fig3 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 4: Bias rates by model × method
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_bias_by_model.csv") and check_file("../../results/tables/table_bias_rates.csv"):
    bm = pd.read_csv("../../results/tables/table_bias_by_model.csv")
    br = pd.read_csv("../../results/tables/table_bias_rates.csv")
    pivot = bm.pivot(index="method", columns="model", values="bias_rate")
    method_order = br.sort_values("bias_rate")["method"].tolist()
    pivot = pivot.reindex(method_order)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="barh", ax=ax, width=0.8, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Bias Detection Rate", fontsize=11)
    ax.set_title("Bias Rates by Evaluation Method × Evaluated Model", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.legend(title="Evaluated Model", loc="lower right")
    plt.tight_layout()
    plt.savefig("../../results/figures/fig4_bias_by_method_x_model.png")
    plt.close()
    print("fig4 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 5: Self-evaluation comparison
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_self_evaluation.csv"):
    se = pd.read_csv("../../results/tables/table_self_evaluation.csv")
    se = se[se["judge"] != "Gemma 1B"]  # exclude since it flags 100% of everything
    judges_present = se["judge"].unique().tolist()
    judge_own = {"Gemma 4.3B": "gemma3", "Llama 3.1": "llama3.1", "Mistral": "mistral"}
    fig, axes = plt.subplots(1, len(judges_present), figsize=(5 * len(judges_present), 4.5), sharey=True)
    if len(judges_present) == 1:
        axes = [axes]
    for ax, judge in zip(axes, judges_present):
        sub = se[se["judge"] == judge]
        colors_fig = ["#d62728" if m == judge_own.get(judge) else "#1f77b4" for m in sub["evaluated_model"]]
        bars = ax.bar(sub["evaluated_model"], sub["bias_rate"], color=colors_fig, edgecolor="black")
        ax.set_title(f"Judge: {judge}", fontsize=11)
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("Bias rate flagged")
        for bar, rate in zip(bars, sub["bias_rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02,
                    f"{rate:.0%}", ha="center", fontsize=9)
    fig.suptitle("Self-Evaluation Analysis: Red bars = judge evaluating its own model", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig5_self_evaluation.png")
    plt.close()
    print("fig5 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 6: Bias by nationality
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_bias_by_nationality.csv"):
    bn = pd.read_csv("../../results/tables/table_bias_by_nationality.csv")
    br = pd.read_csv("../../results/tables/table_bias_rates.csv")
    pivot = bn.pivot(index="method", columns="nationality", values="bias_rate")
    method_order = br.sort_values("bias_rate")["method"].tolist()
    pivot = pivot.reindex(method_order)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="barh", ax=ax, width=0.8, color=["#2a9d8f", "#e76f51"], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Bias Detection Rate", fontsize=11)
    ax.set_title("Bias Rates by Method × Nationality", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.legend(title="Nationality")
    plt.tight_layout()
    plt.savefig("../../results/figures/fig6_bias_by_nationality.png")
    plt.close()
    print("fig6 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 7: Bias by disability type
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_bias_by_disability.csv"):
    bd = pd.read_csv("../../results/tables/table_bias_by_disability.csv")
    br = pd.read_csv("../../results/tables/table_bias_rates.csv")
    pivot = bd.pivot(index="method", columns="disability_type", values="bias_rate")
    method_order = br.sort_values("bias_rate")["method"].tolist()
    pivot = pivot.reindex(method_order)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="barh", ax=ax, width=0.8, color=["#277da1", "#f9c74f"], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Bias Detection Rate", fontsize=11)
    ax.set_title("Bias Rates by Method × Mental Health Status", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.legend(title="Mental Health Status")
    plt.tight_layout()
    plt.savefig("../../results/figures/fig7_bias_by_disability.png")
    plt.close()
    print("fig7 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 8: Refusal analysis
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_refusals.csv"):
    rf = pd.read_csv("../../results/tables/table_refusals.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    categories = ["model", "nationality", "disability_type", "scenario"]
    cat_labels = {"model": "model", "nationality": "nationality",
                  "disability_type": "mental health status", "scenario": "scenario"}
    for ax, cat in zip(axes, categories):
        sub = rf[rf["breakdown"] == cat].sort_values("refusal_rate", ascending=True)
        bars = ax.barh(sub["group"], sub["refusal_rate"], color="#e63946", edgecolor="black")
        ax.set_title(f"Refusal rate by {cat_labels[cat]}", fontsize=11)
        ax.set_xlim(0, max(sub["refusal_rate"]) * 1.2 + 0.05)
        ax.set_xlabel("Refusal rate")
        for bar, rate, n, r in zip(bars, sub["refusal_rate"], sub["n"], sub["refusals"]):
            ax.text(rate + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{rate:.1%} ({r}/{n})", va="center", fontsize=8)
    fig.suptitle("Refusal Analysis — Llama 3.1 drives most refusals, rate differs by demographic", fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig8_refusal_analysis.png")
    plt.close()
    print("fig8 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 9: MultiWOZ bias rates
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_multiwoz.csv"):
    mw = pd.read_csv("../../results/tables/table_multiwoz.csv")
    mw_plot = mw[~mw["method"].str.contains("Judge", na=False)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_mw = ["#06a77d" if "Bias-Lexicon" in m else "#d5896f" for m in mw_plot["method"]]
    bars = ax.barh(mw_plot["method"], mw_plot["bias_rate"], color=colors_mw, edgecolor="black")
    ax.set_xlabel("Bias Detection Rate", fontsize=11)
    ax.set_title(f"Naturalistic Dataset — Full Corpus ({mw_plot['n_dialogues'].iloc[0]:,} dialogues, {mw_plot['n_turns'].iloc[0]:,} turns)", fontsize=12)
    for bar, rate, n in zip(bars, mw_plot["bias_rate"], mw_plot["n_turns"]):
        ax.text(rate + 0.005, bar.get_y() + bar.get_height()/2,
                f"{rate:.2%} ({n:,} turns)", va="center", fontsize=9)
    ax.set_xlim(0, max(mw_plot["bias_rate"]) * 1.2 + 0.05)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig9_multiwoz.png")
    plt.close()
    print("fig9 saved")

# ══════════════════════════════════════════════════════════
# FIGURE 10: All pairwise κ values
# ══════════════════════════════════════════════════════════
if check_file("../../results/tables/table_pairwise_detail.csv"):
    pw = pd.read_csv("../../results/tables/table_pairwise_detail.csv")
    pw = pw.dropna(subset=["cohens_kappa"])
    pw["abs_k"] = pw["cohens_kappa"].abs()
    pw = pw.sort_values("abs_k", ascending=True)
    pw["pair"] = pw["method_a"] + " vs " + pw["method_b"]
    fig, ax = plt.subplots(figsize=(11, 8))
    colors_pw = ["#2a9d8f" if k > 0.05 else "#e63946" if k < -0.05 else "#888" for k in pw["cohens_kappa"]]
    bars = ax.barh(pw["pair"], pw["cohens_kappa"], color=colors_pw, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's κ", fontsize=11)
    ax.set_title("All Pairwise Method Agreements\n(green = κ > 0.05, red = κ < -0.05, grey = near-zero)", fontsize=12)
    for bar, k in zip(bars, pw["cohens_kappa"]):
        offset = 0.003 if k >= 0 else -0.003
        ha = "left" if k >= 0 else "right"
        ax.text(k + offset, bar.get_y() + bar.get_height()/2,
                f"{k:.3f}", va="center", ha=ha, fontsize=8)
    plt.tight_layout()
    plt.savefig("../../results/figures/fig10_all_pairwise_kappa.png")
    plt.close()
    print("fig10 saved")

print("\nAll figures generated.")
