"""Prompt-level, nationality-stratified bootstrap of an NK/US rate ratio (shared helper)."""
import numpy as np
import pandas as pd

SEED, B = 42, 10_000
NK, US = "North Korean", "American"


def boot_ratio(df, flag="flag", seed=SEED):
    """df: rows with prompt_id, nationality, flag. Resample prompts with replacement within each
    nationality stratum (clusters = all rows of a prompt). Ratio = NK flag rate / US flag rate."""
    per = df.groupby(["nationality", "prompt_id"]).flag.agg(["sum", "count"]).reset_index()
    rng = np.random.default_rng(seed)
    tot = {}
    for nat in (NK, US):
        p = per[per.nationality == nat]
        idx = rng.integers(0, len(p), size=(B, len(p)))
        tot[nat] = (p["sum"].values[idx].sum(1), p["count"].values[idx].sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = {k: v[0] / v[1] for k, v in tot.items()}
        ratios = rate[NK] / rate[US]
    ok = np.isfinite(ratios)
    point = (df[df.nationality == NK].flag.mean() / df[df.nationality == US].flag.mean()
             if df[df.nationality == US].flag.mean() else np.nan)
    lo, hi = np.percentile(ratios[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    return {"point": point, "lo": lo, "hi": hi, "n_us": int((df.nationality == US).sum()),
            "n_nk": int((df.nationality == NK).sum()), "dropped_resamples": int((~ok).sum())}
