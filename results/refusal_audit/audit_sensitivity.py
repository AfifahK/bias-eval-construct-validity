"""Declarative refusal counts and Llama NK/US ratio if the rows coded 0 in the 67-row audit are
treated as non-refusals. Needs a coded refusal_audit_all67.csv. Output: audit_sensitivity.md"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from boot import boot_ratio  # noqa: E402

c = pd.read_csv(HERE / "refusal_audit_all67.csv", dtype=str)
k = pd.read_csv(HERE / "refusal_audit_all67_key.csv").merge(c[["audit_id", "coder_label"]], on="audit_id")
drop = set(k[k.coder_label == "0"].sample_id)
d = pd.read_csv(HERE.parents[1] / "data/all_model_responses_clean.csv")
e = d[d.step == "explanation"].copy()
e["sid"] = e.prompt_id.astype(str) + "_" + e.model + "_" + e.scale
e["flag_orig"] = e.is_refusal.astype(int)
e["flag_aud"] = (e.is_refusal & ~e.sid.isin(drop)).astype(int)
L = ["# Declarative refusals if audit-coded 0 rows are treated as non-refusals", "",
     f"Rows coded 0 in `refusal_audit_all67.csv`: {', '.join(sorted(drop)) or 'none'} "
     f"({len(drop)} rows; sample ids {', '.join(sorted(drop))}). Bootstrap as in `robustness_summary.md` "
     "(prompts resampled within nationality, 10,000, seed 42).", "",
     "| refusal set | total | Gemma | Llama | Mistral | Llama NK/US | NK/US [95% CI] |", "|---|---|---|---|---|---|---|"]
for name, col in (("regex rule (paper)", "flag_orig"), ("minus audit-coded 0", "flag_aud")):
    ll = e[e.model == "llama3.1"].assign(flag=lambda x: x[col])
    r = boot_ratio(ll)
    nk = ll[(ll.nationality == "North Korean")][col].sum(); us = ll[ll.nationality == "American"][col].sum()
    m = e.groupby("model")[col].sum()
    L.append(f"| {name} | {int(e[col].sum())} | {m['gemma3']} | {m['llama3.1']} | {m['mistral']} | "
             f"{nk}/60 vs {us}/60 | {r['point']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] |")

# operational: same exercise for the mid-response audit sheet (when coded)
osheet, okey = HERE / "refusal_audit_operational_mid.csv", HERE / "refusal_audit_operational_mid_key.csv"
if osheet.exists() and okey.exists():
    oc = pd.read_csv(osheet, dtype=str)
    if oc.coder_label.notna().all():
        ok = pd.read_csv(okey).merge(oc[["audit_id", "coder_label"]], on="audit_id")
        odrop = set(ok[ok.coder_label == "0"].sample_id)
        ref = pd.read_csv(HERE / "operational_refusals_corrected.csv")
        non = pd.read_csv(HERE / "operational_non_refusals_corrected.csv")
        allr = pd.concat([ref.assign(f0=1), non.assign(f0=0)])
        allr["sid"] = allr.prompt_id.astype(str) + "_" + allr.model + "_" + allr.scale_direction
        allr["f1"] = ((allr.f0 == 1) & ~allr.sid.isin(odrop)).astype(int)
        L += ["", "# Operational refusals if audit-coded 0 rows are treated as non-refusals", "",
              f"Rows coded 0 in `refusal_audit_operational_mid.csv`: {', '.join(sorted(odrop)) or 'none'}. "
              "Only the mid-response matches were audited.", "",
              "| refusal set | total | Llama | Llama NK/US | NK/US [95% CI] |", "|---|---|---|---|---|"]
        for nm, col in (("regex rule (corrected)", "f0"), ("minus audit-coded 0", "f1")):
            ll = allr[allr.model == "llama3.1"].assign(flag=lambda x: x[col])
            r = boot_ratio(ll)
            nk = ll[ll.nationality == "North Korean"][col].sum(); us = ll[ll.nationality == "American"][col].sum()
            L.append(f"| {nm} | {int(allr[col].sum())} | {int(ll[col].sum())} | {nk}/60 vs {us}/60 | "
                     f"{r['point']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] |")
(HERE / "audit_sensitivity.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
