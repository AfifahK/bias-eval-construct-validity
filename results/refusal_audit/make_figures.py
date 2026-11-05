"""Figures for the refusal audit -> results/refusal_audit/figures/.

fig8_refusal_corrected.png    1x2 (declarative | operational): Llama 3.1 refusal rate by nationality with NK/US
                              ratio, prompt-level bootstrap 95% CIs. Responder-model rates are in the data CSV.
fig_instrument_ratios.png     NK/US ratio, scored only vs scored + refusals, with CIs: Bias-Lexicon, HurtLex,
                              HurtLex without "can", Llama and Mistral judges (v1); 297 / 269 scored sets.
                              Rows come from RATIO_SOURCES (each returns instrument, dataset, set, ratio, lo, hi).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import refusal_sets  # noqa: E402
ROOT = HERE.parents[1]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
B, SEED = 10_000, 42
NK, US = "North Korean", "American"
C = {"scored only": "#1f77b4", "scored + refusals": "#d55e00"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ refusal figure
def refusal_rows():
    d = refusal_sets.declarative_explanation()       # decision 30: is_refusal = hand-audited 63
    dec = d[["prompt_id", "nationality", "model", "is_refusal"]].copy()
    r = pd.read_csv(HERE / "operational_refusals_corrected.csv").assign(is_refusal=True)
    n = pd.read_csv(HERE / "operational_non_refusals_corrected.csv").assign(is_refusal=False)
    op = pd.concat([r, n])[["prompt_id", "nationality", "model", "is_refusal"]]
    assert len(dec) == len(op) == 360
    return {"Declarative (audited)": dec, "Operational (corrected)": op}


def boot_rate(df, mask, seed=SEED):
    """Rate of is_refusal among rows in `mask`; resample prompts within nationality."""
    rng = np.random.default_rng(seed)
    x = df[mask]
    num = np.zeros(B); den = np.zeros(B)
    for nat in (NK, US):
        p = x[x.nationality == nat].groupby("prompt_id").is_refusal.agg(["sum", "count"])
        if p.empty:
            continue
        idx = rng.integers(0, len(p), size=(B, len(p)))
        num += p["sum"].values[idx].sum(1); den += p["count"].values[idx].sum(1)
    r = num / den
    return x.is_refusal.mean(), *np.percentile(r, [2.5, 97.5])


def boot_ratio(df, seed=SEED):
    """NK/US Llama refusal-rate ratio, prompt-level bootstrap within nationality."""
    rng = np.random.default_rng(seed)
    x = df[df.model == "llama3.1"]
    S, N = {}, {}
    for nat in (NK, US):
        p = x[x.nationality == nat].groupby("prompt_id").is_refusal.agg(["sum", "count"])
        idx = rng.integers(0, len(p), size=(B, len(p)))
        S[nat] = p["sum"].values[idx].sum(1); N[nat] = p["count"].values[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (S[NK] / N[NK]) / (S[US] / N[US])
    r = r[np.isfinite(r)]
    pt = x[x.nationality == NK].is_refusal.mean() / x[x.nationality == US].is_refusal.mean()
    return pt, *np.percentile(r, [2.5, 97.5])


def fig_refusal():
    """1x2: Llama 3.1 refusal rate by nationality, declarative | operational. Sized for a 5.5 in
    text width so fonts render at their stated size. Responder-model rates (Mistral and
    Gemma 0/120 in both sets) are kept in the data CSV and go in the caption, not the figure."""
    data = refusal_rows()
    stats = []
    for name, df in data.items():
        for m in ["llama3.1", "mistral", "gemma3"]:
            stats.append((name, "model", m, *boot_rate(df, df.model == m)))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True)
    for ax, (name, df), title in zip(axes, data.items(), ("Declarative", "Operational")):
        for j, nat in enumerate([NK, US]):
            pt, lo, hi = boot_rate(df, (df.model == "llama3.1") & (df.nationality == nat))
            ax.barh(j, pt, color="#7f7f7f", height=0.55)
            ax.errorbar(pt, j, xerr=[[pt - lo], [hi - pt]], color="k", capsize=3, lw=1)
            ax.text(max(hi, pt) + 0.03, j, f"{pt:.1%} ({int(round(pt * 60))}/60)", va="center", fontsize=9)
            stats.append((name, "llama_nationality", nat, pt, lo, hi))
        rt, rlo, rhi = boot_ratio(df)
        stats.append((name, "llama_ratio", "NK/US", rt, rlo, rhi))
        ax.set_yticks(range(2)); ax.set_yticklabels([NK, US])
        ax.set_ylim(2.05, -0.55)                       # inverted; free strip below the bars for the ratio
        ax.set_xlim(0, 1.65); ax.set_xticks([0, .5, 1])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.spines["bottom"].set_bounds(0, 1)           # axis line stops at 100%; xlim keeps room for labels
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Llama 3.1 refusal rate", fontsize=9)
        ax.text(0.04, 1.62, f"NK/US = {rt:.2f} [{rlo:.2f}, {rhi:.2f}]", fontsize=9, va="center", ha="left")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(FIG / "fig8_refusal_corrected.png", dpi=300)
    plt.close(fig)
    pd.DataFrame(stats, columns=["dataset", "panel", "group", "value", "lo", "hi"]).to_csv(
        FIG / "fig8_refusal_corrected_data.csv", index=False)


# ------------------------------------------------------------------ instrument ratio figure
def load_dictionary_rows():
    a = pd.read_csv(HERE / "hurtlex_ratio_comparison.csv")
    a = a[a.set.isin(["scored only", "scored + refusals"])]
    b = pd.read_csv(HERE / "hurtlex_no_can.csv")[["dataset", "set", "ratio", "lo", "hi"]]
    b = b.assign(instrument="HurtLex without \"can\"")
    return pd.concat([a[["instrument", "dataset", "set", "ratio", "lo", "hi"]], b])


def load_judge_rows():
    """Judge rows from score_refusal_judges.py (scored 297 / 269; same columns as above)."""
    p = HERE / "judge_ratio_comparison.csv"
    return pd.read_csv(p)[["instrument", "dataset", "set", "ratio", "lo", "hi"]] if p.exists() else pd.DataFrame(
        columns=["instrument", "dataset", "set", "ratio", "lo", "hi"])


RATIO_SOURCES = [load_dictionary_rows, load_judge_rows]
# plotted instruments (stored name -> label). Gemma judges (ceiling), toxic-bert / Detoxify (zero) and
# Mistral operational (no flags, ratio undefined) are not drawn; the caption covers them.
PLOT = {"Bias-Lexicon (norm)": "Bias-Lexicon", "HurtLex (norm)": "HurtLex",
        "HurtLex without \"can\"": "HurtLex without \u201ccan\u201d",
        "Llama 3.1 judge (v1)": "Llama 3.1 judge (v1)", "Mistral judge (v1)": "Mistral judge (v1)"}
STYLE = {"scored only": dict(color="#9a9a9a", marker="o"), "scored + refusals": dict(color="black", marker="D")}


def fig_instruments():
    t = pd.concat([x for x in (f() for f in RATIO_SOURCES) if len(x)], ignore_index=True)
    t = t[t.instrument.isin(PLOT) & t.ratio.notna()].copy()
    t["instrument"] = pd.Categorical(t.instrument, list(PLOT), ordered=True)
    t.sort_values(["dataset", "instrument", "set"]).to_csv(FIG / "fig_instrument_ratios_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9), sharey=True)
    for ax, ds, title in zip(axes, ("declarative", "operational"), ("Declarative", "Operational")):
        for k, st in enumerate(("scored only", "scored + refusals")):
            for j, inst in enumerate(PLOT):
                r = t[(t.instrument == inst) & (t.dataset == ds) & (t.set == st)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.ratio, j + (k - 0.5) * 0.3, xerr=[[r.ratio - r.lo], [r.hi - r.ratio]], fmt=STYLE[st]["marker"],
                            color=STYLE[st]["color"], mec="black" if st == "scored only" else None, mew=0.6, ecolor="k",
                            ms=4.5, capsize=2.5, lw=1, label=st if j == 0 else None)
        ax.axvline(1, color="k", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.minorticks_off()
        ax.set_xlim(0.5, 25); ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.set_yticks(range(len(PLOT))); ax.set_yticklabels(list(PLOT.values()))
        ax.set_ylim(len(PLOT) - 0.5, -0.5)
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("NK/US ratio (log scale)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_instrument_ratios.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_refusal()
    fig_instruments()
    print("wrote", sorted(p.name for p in FIG.iterdir()))
