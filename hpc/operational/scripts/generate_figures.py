"""
generate_figures.py — Produce figures from operational and cross-content statistics.

Reads CSVs from ../statistics/operational/ and ../statistics/cross_content_three_way/.
Outputs PNG figures to ../figures/.

Usage:
    python scripts/generate_figures.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
sns.set_style("whitegrid")

# ── Paths ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
OP_STATS = os.path.join(BASE_DIR, "statistics", "operational")
TW_STATS = os.path.join(BASE_DIR, "statistics", "cross_content_three_way")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def check_file(path):
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return False
    return True


# ══════════════════════════════════════════════════════════
# FIGURE 1: Operational bias rates by method
# ══════════════════════════════════════════════════════════
flagging_path = os.path.join(OP_STATS, "flagging_rates.csv")
if check_file(flagging_path):
    br = pd.read_csv(flagging_path).sort_values("bias_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(br)))
    bars = ax.barh(br["method"], br["bias_rate"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Bias Detection Rate", fontsize=11)
    ax.set_title(
        f"Operational Bias Rates Across {len(br)} Evaluation Methods\n"
        f"(n={br['n'].iloc[0]} non-refusal responses)",
        fontsize=12,
    )
    ax.set_xlim(0, 1.1)
    for bar, rate, n, flagged in zip(bars, br["bias_rate"], br["n"], br["flagged"]):
        ax.text(
            rate + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{rate:.1%} ({flagged}/{n})", va="center", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "operational_bias_rates_by_method.png"))
    plt.close()
    print("fig1 saved: operational_bias_rates_by_method.png")


# ══════════════════════════════════════════════════════════
# FIGURE 2: Pairwise kappa heatmap
# ══════════════════════════════════════════════════════════
kappa_path = os.path.join(OP_STATS, "pairwise_kappa.csv")
if check_file(kappa_path):
    pk = pd.read_csv(kappa_path)
    # Build symmetric matrix
    methods = sorted(set(pk["method_a"].tolist() + pk["method_b"].tolist()))
    kappa_mat = pd.DataFrame(np.nan, index=methods, columns=methods)
    for _, row in pk.iterrows():
        kappa_mat.loc[row["method_a"], row["method_b"]] = row["cohens_kappa"]
        kappa_mat.loc[row["method_b"], row["method_a"]] = row["cohens_kappa"]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(kappa_mat.values, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        kappa_mat.astype(float), annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, vmin=-0.25, vmax=0.25,
        square=True, cbar_kws={"label": "Cohen's kappa"},
        mask=mask, linewidths=0.5, linecolor="gray", ax=ax,
    )
    ax.set_title(
        "Operational Pairwise Cohen's kappa\n"
        "(diagonal masked; closer to 0 = chance agreement)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "operational_pairwise_kappa_heatmap.png"))
    plt.close()
    print("fig2 saved: operational_pairwise_kappa_heatmap.png")


# ══════════════════════════════════════════════════════════
# FIGURE 3: Self-evaluation analysis
# ══════════════════════════════════════════════════════════
self_eval_path = os.path.join(OP_STATS, "self_eval_rates.csv")
if check_file(self_eval_path):
    se = pd.read_csv(self_eval_path)
    # Exclude Gemma 1B (flags everything)
    se = se[se["judge"] != "Gemma 1B"]
    judges_present = se["judge"].unique().tolist()

    # Map judge display name to its own responder model
    judge_own = {"Gemma 4.3B": "gemma3", "Llama 3.1": "llama3.1", "Mistral": "mistral"}

    if len(judges_present) > 0:
        fig, axes = plt.subplots(
            1, len(judges_present),
            figsize=(5 * len(judges_present), 4.5),
            sharey=True,
        )
        if len(judges_present) == 1:
            axes = [axes]
        for ax, judge in zip(axes, judges_present):
            sub = se[se["judge"] == judge]
            colors_fig = [
                "#d62728" if m == judge_own.get(judge) else "#1f77b4"
                for m in sub["evaluated_model"]
            ]
            bars = ax.bar(
                sub["evaluated_model"], sub["bias_rate"],
                color=colors_fig, edgecolor="black",
            )
            ax.set_title(f"Judge: {judge}", fontsize=11)
            ax.set_ylim(0, 1.05)
            if ax == axes[0]:
                ax.set_ylabel("Bias rate flagged")
            for bar, rate in zip(bars, sub["bias_rate"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, rate + 0.02,
                    f"{rate:.0%}", ha="center", fontsize=9,
                )
        fig.suptitle(
            "Operational Self-Evaluation: Red bars = judge evaluating its own model",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "operational_self_evaluation.png"))
        plt.close()
        print("fig3 saved: operational_self_evaluation.png")


# ══════════════════════════════════════════════════════════
# FIGURE 4: Refusal analysis (2x2 grid)
# ══════════════════════════════════════════════════════════
refusal_path = os.path.join(OP_STATS, "refusal_rates.csv")
if check_file(refusal_path):
    rf = pd.read_csv(refusal_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    categories = ["model", "nationality", "mental_health_status", "scenario"]
    cat_labels = {
        "model": "model",
        "nationality": "nationality",
        "mental_health_status": "mental health status",
        "scenario": "scenario",
    }
    for ax, cat in zip(axes, categories):
        sub = rf[rf["breakdown"] == cat].sort_values("refusal_rate", ascending=True)
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        bars = ax.barh(
            sub["group"], sub["refusal_rate"],
            color="#e63946", edgecolor="black",
        )
        ax.set_title(f"Refusal rate by {cat_labels.get(cat, cat)}", fontsize=11)
        ax.set_xlim(0, max(sub["refusal_rate"].max() * 1.2 + 0.05, 0.1))
        ax.set_xlabel("Refusal rate")
        for bar, rate, n, r in zip(bars, sub["refusal_rate"], sub["n"], sub["refusals"]):
            ax.text(
                rate + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%} ({r}/{n})", va="center", fontsize=8,
            )
    fig.suptitle(
        "Operational Refusal Analysis by Demographic and Model",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "operational_refusal_analysis.png"))
    plt.close()
    print("fig4 saved: operational_refusal_analysis.png")


# ══════════════════════════════════════════════════════════
# FIGURE 5: Three-way comparison (grouped bar chart)
# ══════════════════════════════════════════════════════════
tw_path = os.path.join(TW_STATS, "flagging_rates_three_way.csv")
if check_file(tw_path):
    tw = pd.read_csv(tw_path)

    # Drop rows where all three rates are NaN
    tw = tw.dropna(subset=["intersectional_rate", "multiwoz_rate", "operational_rate"],
                   how="all")

    if len(tw) > 0:
        fig, ax = plt.subplots(figsize=(10, max(5, len(tw) * 0.8)))

        y_pos = np.arange(len(tw))
        bar_height = 0.25
        dataset_colors = {
            "Intersectional": "#2a9d8f",
            "MultiWOZ": "#e9c46a",
            "Operational": "#e76f51",
        }

        for offset, (dataset, col) in enumerate([
            ("Intersectional", "intersectional_rate"),
            ("MultiWOZ", "multiwoz_rate"),
            ("Operational", "operational_rate"),
        ]):
            values = tw[col].fillna(0).values
            bars = ax.barh(
                y_pos + offset * bar_height, values,
                height=bar_height, label=dataset,
                color=dataset_colors[dataset], edgecolor="black", linewidth=0.3,
            )
            # Add rate labels for non-zero values
            for bar, val, raw in zip(bars, values, tw[col].values):
                if not np.isnan(raw) and raw > 0:
                    ax.text(
                        val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{raw:.2%}", va="center", fontsize=7,
                    )

        ax.set_yticks(y_pos + bar_height)
        ax.set_yticklabels(tw["method"])
        ax.set_xlabel("Bias Detection Rate", fontsize=11)
        ax.set_title(
            "Bias Detection Rates Across Three Content Domains",
            fontsize=12,
        )
        ax.legend(title="Dataset", loc="lower right")
        ax.set_xlim(0, min(tw[["intersectional_rate", "multiwoz_rate", "operational_rate"]]
                           .max().max() * 1.2 + 0.05, 1.15))
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "operational_three_way_comparison.png"))
        plt.close()
        print("fig5 saved: operational_three_way_comparison.png")


print("\nAll figures generated.")
