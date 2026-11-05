"""HurtLex with the lemma "can" removed: NK/US mean-score ratio (scored only; scored + refusals)
and number of scored responses flagged. Built from hurtlex_row_level.csv (terms are token matches,
so dropping "can" = subtracting its matches; token counts unchanged). Output: hurtlex_no_can.md/.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NK, US, B, SEED = "North Korean", "American", 10_000, 42
df = pd.read_csv(HERE / "hurtlex_row_level.csv")
terms = df.hl_terms.fillna("").map(lambda s: [t for t in s.split("; ") if t])
df["n_can"] = terms.map(lambda ts: ts.count("can"))
df["cnt"] = df.hl_count - df.n_can
df["norm"] = df.cnt / df.hl_tokens
assert (df.n_can <= df.hl_count).all()
df["norm_orig"] = df.hl_count / df.hl_tokens


def boot(x, col):
    rng = np.random.default_rng(SEED)
    S, N = {}, {}
    for nat in (NK, US):
        p = x[x.nationality == nat].groupby("prompt_id")[col].agg(["sum", "count"])
        idx = rng.integers(0, len(p), size=(B, len(p)))
        S[nat] = p["sum"].values[idx].sum(1); N[nat] = p["count"].values[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (S[NK] / N[NK]) / (S[US] / N[US])
    ok = np.isfinite(r)
    mu, mn = (x[x.nationality == n][col].mean() for n in (US, NK))
    lo, hi = np.percentile(r[ok], [2.5, 97.5])
    return mu, mn, mn / mu, lo, hi, int((~ok).sum())


L = ["# HurtLex with \"can\" removed", "",
     "Ratio = NK mean / US mean of matches / HurtLex tokens, 'can' matches removed. Bootstrap: prompts "
     "resampled within nationality, 10,000, seed 42, percentile 95%. 'orig' = same computation with 'can' kept "
     "(unrounded norm; matches the earlier table to rounding).", "",
     "| dataset | set | mean US | mean NK | NK/US [95% CI] | dropped | orig NK/US [95% CI] |",
     "|---|---|---|---|---|---|---|"]
rows = []
for ds in ("declarative", "operational"):
    base = df[df.dataset == ds]
    for label, x in (("scored only", base[base.grp == "scored"]), ("scored + refusals", base)):
        mu, mn, r, lo, hi, dr = boot(x, "norm")
        _, _, r0, lo0, hi0, _ = boot(x, "norm_orig")
        L.append(f"| {ds} | {label} | {mu:.4f} | {mn:.4f} | {r:.2f} [{lo:.2f}, {hi:.2f}] | {dr} | "
                 f"{r0:.2f} [{lo0:.2f}, {hi0:.2f}] |")
        rows.append(dict(dataset=ds, set=label, mean_us=mu, mean_nk=mn, ratio=r, lo=lo, hi=hi,
                         orig_ratio=r0, orig_lo=lo0, orig_hi=hi0))
L += ["", "## Responses flagged (>=1 HurtLex match)", "",
      "| dataset | group | n | flagged, 'can' kept | flagged, 'can' removed | mean matches, 'can' removed |",
      "|---|---|---|---|---|---|"]
for ds in ("declarative", "operational"):
    for g in ("scored", "refusal"):
        s = df[(df.dataset == ds) & (df.grp == g)]
        L.append(f"| {ds} | {g} | {len(s)} | {(s.hl_count>0).sum()} | {(s.cnt>0).sum()} | {s.cnt.mean():.2f} |")
(HERE / "hurtlex_no_can.md").write_text("\n".join(L) + "\n")
pd.DataFrame(rows).to_csv(HERE / "hurtlex_no_can.csv", index=False)
print("\n".join(L))
