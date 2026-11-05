"""Run locally after copying results/ back. Compares sanity-slice judge labels with stored labels
for v1 and v2 (v2: no stored declarative Gemma 1B labels, those rows are skipped)."""
import sys
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve().parent
res = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"
exp = pd.read_csv(here / "sanity_expected.csv")
print(f"{'ver':3s} {'judge':18s} {'dataset':12s} n   label_match  severity_match  parse_err")
for ver in ("v1", "v2"):
    for tag in ["gemma3:latest", "gemma3:1b", "llama3.1:latest", "mistral:latest"]:
        safe = tag.replace(":", "_")
        for ds, sc in (("declarative", "scale"), ("operational", "scale_direction")):
            p = res / f"sanity_{ds}_{safe}_{ver}.csv"
            if not p.exists():
                print(f"{ver:3s} {tag:18s} {ds:12s} MISSING {p.name}"); continue
            n = pd.read_csv(p).rename(columns={sc: "scale"})
            e = exp[exp.dataset == ds]
            suf = "" if ver == "v1" else "_v2"
            L, S = f"expected_label{suf}__{safe}", f"expected_severity{suf}__{safe}"
            e = e[e[L].notna()]
            if e.empty:
                print(f"{ver:3s} {tag:18s} {ds:12s} no stored {ver} labels"); continue
            m = e.merge(n[["prompt_id", "model", "scale", "bias_label", "severity", "parse_error"]],
                        on=["prompt_id", "model", "scale"], how="left")
            print(f"{ver:3s} {tag:18s} {ds:12s} {len(m):<3d} {(m.bias_label == m[L]).sum():>3}/{len(m)}"
                  f"       {(m.severity == m[S]).sum():>3}/{len(m)}          {m.parse_error.notna().sum()}")
