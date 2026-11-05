"""Audited precision of the 67 declarative refusals from the hand-coded blind sheet.

Usage: python3 results/refusal_audit/audit_precision.py [coded.csv]
Default input: results/refusal_audit/refusal_audit_all67.csv (filled `coder_label`: 1 = refusal, 0 = not a
refusal, ? = unclear). Joins refusal_audit_all67_key.csv. Writes audit_precision.md.
Precision = refusals (1) / all 67 coded rows; '?' rows are counted separately and also given as a
range (lower bound: '?' = not refusal; upper bound: '?' = refusal). Exact (Clopper-Pearson) 95% CI.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "refusal_audit_all67.csv"
c = pd.read_csv(src, dtype={"coder_label": str})
k = pd.read_csv(HERE / "refusal_audit_all67_key.csv")
df = k.merge(c[["audit_id", "coder_label"]], on="audit_id", validate="1:1")
df["label"] = df.coder_label.fillna("").str.strip()
bad = df[~df.label.isin(["0", "1", "?"])]
if len(bad):
    sys.exit(f"{len(bad)} rows have no valid label (need 1, 0 or ?): {bad.audit_id.tolist()[:10]}")


def cp(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return lo, hi


def row(name, s):
    n, ref, no, q = len(s), int((s.label == "1").sum()), int((s.label == "0").sum()), int((s.label == "?").sum())
    lo, hi = cp(ref, n)
    return (f"| {name} | {n} | {ref} | {no} | {q} | {ref/n:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"{ref/n:.3f} to {(ref+q)/n:.3f} |")


L = ["# Audited precision of the 67 declarative refusals", "",
     f"Source: `{src.name}` joined to `refusal_audit_all67_key.csv`. Precision = label 1 / rows. "
     "CI: exact Clopper-Pearson 95%. Last column: range if '?' rows are all non-refusals (lower) or all refusals (upper).", "",
     "| group | n | 1 (refusal) | 0 (not refusal) | ? | precision [95% CI] | range with '?' |", "|---|---|---|---|---|---|---|",
     row("all", df), row("American", df[df.nationality == "American"]),
     row("North Korean", df[df.nationality == "North Korean"]),
     row("phrase at start", df[df.phrase_at_start == 1]), row("phrase mid-response", df[df.phrase_at_start == 0])]
L += ["", f"'?' rows: {int((df.label == '?').sum())}. Rows coded 0: " +
      (", ".join(df[df.label == "0"].audit_id) or "none") + "."]
if "notes" in c.columns:
    nn = c[c.notes.fillna("").str.strip() != ""][["audit_id", "notes"]]
    if len(nn):
        L += ["", "## Coder notes", ""] + [f"- {r.audit_id} (label {df.set_index('audit_id').label[r.audit_id]}): {r.notes}" for r in nn.itertuples()]

# ---- operational audit: the mid-response regex matches among the 91 operational refusals ----
osheet, okey = HERE / "refusal_audit_operational_mid.csv", HERE / "refusal_audit_operational_mid_key.csv"
L += ["", "## Operational audit (mid-response regex matches)", ""]
if osheet.exists() and okey.exists():
    oc = pd.read_csv(osheet, dtype={"coder_label": str})
    ok = pd.read_csv(okey).merge(oc[["audit_id", "coder_label"] + (["notes"] if "notes" in oc else [])], on="audit_id")
    ok["label"] = ok.coder_label.fillna("").str.strip()
    n_all = len(pd.read_csv(HERE / "operational_refusals_corrected.csv"))
    L.append(f"Population: {n_all} corrected operational refusals; {n_all - len(ok)} have the refusal phrase at the "
             f"start of the response (not audited), {len(ok)} match mid-response (`{osheet.name}`).")
    if (~ok.label.isin(["0", "1", "?"])).any():
        L.append("Not coded yet.")
    else:
        L += ["", "| group | n | 1 (refusal) | 0 (not refusal) | ? | precision [95% CI] |", "|---|---|---|---|---|---|"]
        for nm, sub in (("mid-response matches", ok),):
            x, n = int((sub.label == "1").sum()), len(sub)
            lo, hi = cp(x, n)
            L.append(f"| {nm} | {n} | {x} | {int((sub.label == '0').sum())} | {int((sub.label == '?').sum())} | "
                     f"{x/n:.3f} [{lo:.3f}, {hi:.3f}] |")
        L += ["", "Rows coded 0: " + (", ".join(ok[ok.label == "0"].audit_id) or "none") + ".",
              "Precision is for the mid-response subset only; the start-of-response matches were not hand-coded."]
(HERE / "audit_precision.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
