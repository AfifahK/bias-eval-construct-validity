"""HurtLex diagnostics on refusals vs scored responses. Output: hurtlex_diagnostics.md + csvs.
1 matched terms (counts) + 5 example refusals; 2 raw counts and length; 3 NK/US mean-score ratios
(scored only; scored + refusals) for HurtLex and Bias-Lexicon, prompt-clustered bootstrap."""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_refusals as sr  # noqa: E402  (loads HurtLex + lexicon)

ROOT = sr.ROOT
hl, lex = sr.hl, sr.lex
NK, US, B, SEED = "North Korean", "American", 10_000, 42

# ---------- build row table: text, group, dataset, nationality, prompt_id ----------
import refusal_sets  # noqa: E402

# Decision 30: refusals = hand-audited 63 + 91; the 4 audit false positives are scored responses.
ref = pd.read_csv(HERE / "refusal_dictionary_scores.csv")
fp = (ref.dataset == "declarative") & (ref.prompt_id.astype(str) + "_" + ref.responder + "_" + ref.scale
                                       ).isin(refusal_sets.audit_fp_ids())
ref = ref.rename(columns={"scored_text": "text"})[["dataset", "prompt_id", "nationality", "text"]]
moved = ref[fp].assign(grp="scored")
ref = ref[~fp].assign(grp="refusal")
assert len(moved) == 4 and len(ref) == 154
d = refusal_sets.declarative_explanation()
d = d[~d.is_refusal_regex]
sd = d[["prompt_id", "nationality", "response"]].rename(columns={"response": "text"})
sd["dataset"] = "declarative"
so = pd.read_csv(HERE / "operational_non_refusals_corrected.csv")[
    ["prompt_id", "nationality", "response_clean"]].rename(columns={"response_clean": "text"})
so["dataset"] = "operational"
sc = pd.concat([sd, so]); sc["grp"] = "scored"
df = pd.concat([ref, moved, sc], ignore_index=True)

H, L = [], []
for t in df.text:
    h = hl.score_text(t); l = lex.score_text(t)
    H.append(h); L.append(l)
df["hl_count"] = [h["bias_count"] for h in H]
df["hl_norm"] = [h["bias_score_norm"] for h in H]
df["hl_terms"] = [[w for w, _ in h["matched_terms"]] for h in H]
df["hl_tokens"] = [len(hl.tokenize(hl.clean_response(t))) for t in df.text]
df["bl_count"] = [l["bias_count"] for l in L]
df["bl_norm"] = [l["bias_score_norm"] for l in L]
df["bl_terms"] = [l["matched_terms"] for l in L]
df["chars"] = df.text.str.len()
df["words"] = df.text.str.split().str.len()
assert len(df) == 154 + 4 + 293 + 269

out = ["# HurtLex diagnostics (refusals vs scored responses)", "",
       "Refusals: 63 declarative (hand-audited) / 91 operational. Scored: 297 / 269 (the 4 audit false positives are scored). Scores recomputed with the "
       "repo functions (`src/dictionary/*`); HurtLex tokens/terms are those of `hurtlex_analysis.score_text`.", ""]

# ---------- 1. terms ----------
out += ["## 1. Matched HurtLex terms (token matches, counted with repetition)", ""]
rows = []
for ds in ("declarative", "operational"):
    for g in ("refusal", "scored"):
        s = df[(df.dataset == ds) & (df.grp == g)]
        c = Counter(w for ts in s.hl_terms for w in ts)
        n_rows_with = Counter(w for ts in s.hl_terms for w in set(ts))
        tot = sum(c.values())
        out.append(f"**{ds} {g}** (n={len(s)} responses; {tot} matches; {len(c)} distinct terms; "
                   f"{int((s.hl_count>0).sum())} responses with >=1 match)")
        out.append("")
        out.append("| term | matches | responses containing |")
        out.append("|---|---|---|")
        for w, n in c.most_common(25):
            out.append(f"| {w} | {n} | {n_rows_with[w]} |")
        out.append("")
        for w, n in c.items():
            rows.append({"dataset": ds, "group": g, "term": w, "matches": n,
                         "responses_containing": n_rows_with[w]})
pd.DataFrame(rows).sort_values(["dataset", "group", "matches"], ascending=[1, 1, 0]).to_csv(
    HERE / "hurtlex_terms.csv", index=False)
out.append("Top 25 per group shown; full list in `hurtlex_terms.csv`.\n")

# ---------- 1b. examples ----------
out += ["### Five example refusals (matched terms in **bold**; every token match highlighted)", ""]
ex = df[df.grp == "refusal"].sample(5, random_state=42)
for _, r in ex.iterrows():
    txt = r.text
    for w in sorted(set(r.hl_terms), key=len, reverse=True):
        txt = re.sub(rf"(?<![A-Za-z])({re.escape(w)})(?![A-Za-z])", r"**\1**", txt, flags=re.I)
    out.append(f"- {r.dataset}, prompt {r.prompt_id}, {r.nationality}; {r.hl_count} matches / "
               f"{r.hl_tokens} tokens = {r.hl_norm}; terms: {', '.join(r.hl_terms)}")
    out.append(f"  > {txt.strip()[:900]}")
    out.append("")

# ---------- 2. raw counts and length ----------
out += ["## 2. Raw HurtLex counts and response length", "",
        "| dataset | group | n | HurtLex matches mean | median | max | HurtLex norm mean | tokens mean | tokens median | chars mean | chars median | Bias-Lexicon matches mean |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for ds in ("declarative", "operational"):
    for g in ("refusal", "scored"):
        s = df[(df.dataset == ds) & (df.grp == g)]
        out.append(f"| {ds} | {g} | {len(s)} | {s.hl_count.mean():.2f} | {s.hl_count.median():.1f} | "
                   f"{s.hl_count.max()} | {s.hl_norm.mean():.4f} | {s.hl_tokens.mean():.1f} | "
                   f"{s.hl_tokens.median():.0f} | {s.chars.mean():.0f} | {s.chars.median():.0f} | "
                   f"{s.bl_count.mean():.3f} |")
out.append("")
df.drop(columns=["text"]).assign(hl_terms=df.hl_terms.str.join("; "),
                                 bl_terms=df.bl_terms.str.join("; ")).to_csv(
    HERE / "hurtlex_row_level.csv", index=False)


# ---------- 3. NK/US mean-score ratios ----------
def boot_ratio_mean(x, col, seed=SEED):
    rng = np.random.default_rng(seed)
    S, N = {}, {}
    for nat in (NK, US):
        p = x[x.nationality == nat].groupby("prompt_id")[col].agg(["sum", "count"])
        idx = rng.integers(0, len(p), size=(B, len(p)))
        S[nat] = p["sum"].values[idx].sum(1); N[nat] = p["count"].values[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (S[NK] / N[NK]) / (S[US] / N[US])
    ok = np.isfinite(r)
    pt = x[x.nationality == NK][col].mean() / x[x.nationality == US][col].mean() \
        if x[x.nationality == US][col].mean() else np.nan
    lo, hi = np.percentile(r[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    return pt, lo, hi, int((~ok).sum())


out += ["## 3. NK/US ratio of mean score", "",
        "Ratio = mean score of North Korean rows / mean score of American rows. CI: 10,000 bootstrap "
        "resamples of prompts within nationality, percentile 95%, seed 42. 'scored + refusals' pools the "
        "scored rows with the refusals of the same dataset.", "",
        "| instrument | dataset | set | n US | n NK | mean US | mean NK | NK/US [95% CI] | dropped |",
        "|---|---|---|---|---|---|---|---|---|"]
res = []
for name, col in (("HurtLex (norm)", "hl_norm"), ("Bias-Lexicon (norm)", "bl_norm")):
    for ds in ("declarative", "operational"):
        base = df[df.dataset == ds]
        for label, x in (("scored only", base[base.grp == "scored"]), ("scored + refusals", base),
                         ("refusals only", base[base.grp == "refusal"])):
            pt, lo, hi, dr = boot_ratio_mean(x, col)
            mu, mn = x[x.nationality == US][col].mean(), x[x.nationality == NK][col].mean()
            out.append(f"| {name} | {ds} | {label} | {(x.nationality==US).sum()} | {(x.nationality==NK).sum()} | "
                       f"{mu:.4f} | {mn:.4f} | {pt:.2f} [{lo:.2f}, {hi:.2f}] | {dr} |")
            res.append({"instrument": name, "dataset": ds, "set": label, "ratio": pt, "lo": lo, "hi": hi})
pd.DataFrame(res).to_csv(HERE / "hurtlex_ratio_comparison.csv", index=False)
(HERE / "hurtlex_diagnostics.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
