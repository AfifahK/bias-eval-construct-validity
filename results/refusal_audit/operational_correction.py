"""Corrected operational refusal split (curly-apostrophe fix) + old-vs-new instrument comparison.

Writes (existing files untouched):
  results/refusal_audit/operational_refusals_corrected.csv       (91)
  results/refusal_audit/operational_non_refusals_corrected.csv   (269)
  results/refusal_audit/operational_moved_20.csv                 (the 20 reclassified rows with their old scores)
  results/refusal_audit/operational_correction_comparison.csv    (old vs new per instrument)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from boot import boot_ratio, NK, US  # noqa: E402

ROOT = HERE.parents[1]
OP = ROOT / "hpc/operational"
sp = importlib.util.spec_from_file_location("detect_refusals", OP / "scripts/detect_refusals.py")
dr = importlib.util.module_from_spec(sp)
sp.loader.exec_module(dr)          # fixed is_refusal; main() is not executed

comb = pd.read_csv(OP / "responses/operational_combined.csv")
comb["response_clean"] = comb.response.map(dr.clean_response)
comb["is_refusal"] = comb.response_clean.map(dr.is_refusal)
comb["ai_self_reference"] = comb.response_clean.map(dr.has_ai_self_reference)
ref_new = comb[comb.is_refusal].drop(columns="is_refusal")
non_new = comb[~comb.is_refusal].drop(columns="is_refusal")

ref_old = pd.read_csv(OP / "responses/operational_refusals.csv")
non_old = pd.read_csv(OP / "responses/operational_non_refusals.csv")
K = ["prompt_id", "model", "scale_direction"]
old_ref_keys = set(map(tuple, ref_old[K].values))
new_ref_keys = set(map(tuple, ref_new[K].values))
assert old_ref_keys <= new_ref_keys and len(new_ref_keys - old_ref_keys) == 20
assert list(ref_new.columns) == list(ref_old.columns), (list(ref_new.columns), list(ref_old.columns))
# corrected files agree with the originals on every shared row/column
chk = non_new.merge(non_old, on=K, suffixes=("", "_o"))
assert len(chk) == len(non_new) == 269 and (chk.response_clean == chk.response_clean_o).all()
ref_new.to_csv(HERE / "operational_refusals_corrected.csv", index=False)
non_new.to_csv(HERE / "operational_non_refusals_corrected.csv", index=False)
print("refusals", len(ref_new), "non-refusals", len(non_new))

moved_keys = new_ref_keys - old_ref_keys
is_moved = lambda df, col: [(p, m, s) in moved_keys for p, m, s in zip(df.prompt_id, df.model, df[col])]

# ---- instruments (response level) ----
ev = OP / "evaluations"
dd = pd.read_csv(ev / "dictionary_operational.csv")
inst = {}  # name -> (df with prompt_id,nationality,model,scale_direction,flag,score)
for name, f, sc in [("Bias-Lexicon", "bias_label", "bias_count"), ("HurtLex", "hurtlex_label", "hurtlex_count"),
                    ("toxic-bert >0.5", "toxicity_label", "toxicity_score")]:
    inst[name] = dd.assign(flag=dd[f], score=dd[sc])
for tag, nm in [("gemma3_latest", "Gemma 3 4.3B"), ("gemma3_1b", "Gemma 3 1B"),
                ("llama3.1_latest", "Llama 3.1 8B"), ("mistral_latest", "Mistral 7B")]:
    for v in ("v1", "v2"):
        j = pd.read_csv(ev / f"judge_operational_{tag}_{v}.csv")
        j = j[j.bias_label >= 0]
        inst[f"{nm} ({v})"] = j.assign(flag=j.bias_label, score=j.severity)

rows, moved_rows = [], []
for name, df in inst.items():
    df = df.copy()
    df["moved"] = is_moved(df, "scale_direction")
    old, new = df, df[~df.moved]
    r_old, r_new = boot_ratio(old), boot_ratio(new)
    mv = df[df.moved]
    rows.append({"instrument": name, "level": "response",
                 "n_old": len(old), "n_new": len(new),
                 "flag_rate_old": old.flag.mean(), "flag_rate_new": new.flag.mean(),
                 "mean_score_old": old.score.mean(), "mean_score_new": new.score.mean(),
                 "ratio_old": r_old["point"], "ratio_old_lo": r_old["lo"], "ratio_old_hi": r_old["hi"],
                 "ratio_new": r_new["point"], "ratio_new_lo": r_new["lo"], "ratio_new_hi": r_new["hi"],
                 "moved_n_scored": len(mv), "moved_flag_rate": mv.flag.mean(), "moved_mean_score": mv.score.mean()})
    moved_rows.append(mv.assign(instrument=name)[["instrument", "prompt_id", "nationality", "model",
                                                  "scale_direction", "flag", "score"]])

# ---- sentence level (v1 judges + dictionaries) ----
sent_inst = {}
ds = pd.read_csv(ev / "dictionary_operational_sentences.csv")
for name, f in [("Bias-Lexicon", "bias_label"), ("HurtLex", "hurtlex_label"), ("toxic-bert >0.5", "toxicity_label")]:
    sent_inst[name] = ds.assign(flag=ds[f])
for tag, nm in [("gemma3_latest", "Gemma 3 4.3B"), ("gemma3_1b", "Gemma 3 1B"),
                ("llama3.1_latest", "Llama 3.1 8B"), ("mistral_latest", "Mistral 7B")]:
    for v in ("v1", "v2"):
        j = pd.read_csv(ev / f"judge_operational_{tag}_{v}_sentences.csv")
        sent_inst[f"{nm} ({v})"] = j[j.bias_label >= 0].assign(flag=lambda x: x.bias_label)
for name, df in sent_inst.items():
    df = df.copy(); df["moved"] = is_moved(df, "scale_direction")
    old, new = df, df[~df.moved]
    r_old, r_new = boot_ratio(old), boot_ratio(new)
    rows.append({"instrument": name, "level": "sentence", "n_old": len(old), "n_new": len(new),
                 "flag_rate_old": old.flag.mean(), "flag_rate_new": new.flag.mean(),
                 "ratio_old": r_old["point"], "ratio_old_lo": r_old["lo"], "ratio_old_hi": r_old["hi"],
                 "ratio_new": r_new["point"], "ratio_new_lo": r_new["lo"], "ratio_new_hi": r_new["hi"],
                 "moved_n_scored": int(df.moved.sum()), "moved_flag_rate": df[df.moved].flag.mean()})

cmp_ = pd.DataFrame(rows)
cmp_.to_csv(HERE / "operational_correction_comparison.csv", index=False)
pd.concat(moved_rows).to_csv(HERE / "operational_moved_20.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
print(cmp_[["instrument", "level", "n_old", "n_new", "flag_rate_old", "flag_rate_new", "ratio_old", "ratio_new",
            "moved_flag_rate"]].to_string())
