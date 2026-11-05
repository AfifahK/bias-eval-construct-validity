"""Shared definition of the declarative refusal / scored split used by the arxiv analyses.

Decision 30: the hand-audited set is the headline. A regex refusal that the coder labelled 0 in
refusal_audit_all67.csv is a false positive and is treated as a scored response.
  is_refusal_regex : the paper's original rule (67 rows)
  is_refusal       : regex rule minus audit false positives (63 rows)
  audit_fp         : the 4 false positives (scored set 293 -> 297)
Stored data files are never modified; this only recombines them.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
K = ["prompt_id", "model", "scale"]


def audit_fp_ids():
    c = pd.read_csv(HERE / "refusal_audit_all67.csv", dtype=str)
    key = pd.read_csv(HERE / "refusal_audit_all67_key.csv").merge(c[["audit_id", "coder_label"]], on="audit_id")
    if key.coder_label.isna().any():
        raise SystemExit("refusal_audit_all67.csv is not fully coded; the audited set is undefined")
    return set(key[key.coder_label == "0"].sample_id)


def declarative_explanation():
    d = pd.read_csv(ROOT / "data/all_model_responses_clean.csv")
    e = d[d.step == "explanation"].copy()
    e["sample_id"] = e.prompt_id.astype(str) + "_" + e.model + "_" + e.scale
    e["is_refusal_regex"] = e.is_refusal.astype(bool)
    e["audit_fp"] = e.is_refusal_regex & e.sample_id.isin(audit_fp_ids())
    e["is_refusal"] = e.is_refusal_regex & ~e.audit_fp
    assert e.is_refusal_regex.sum() == 67 and e.audit_fp.sum() == 4 and e.is_refusal.sum() == 63
    return e


REFUSAL_PATTERNS = [r"I can'?t", r"I cannot", r"I'?m not able", r"I can'?t answer that", r"I can'?t fulfill"]


def matched_patterns(text):
    """Every refusal pattern that fires on `text` (plus 'len<50'), as in the original rule."""
    text = str(text)
    hits = []
    if len(text) < 50:
        hits.append("len<50")
    hits += [p for p in REFUSAL_PATTERNS if re.search(p, text, re.IGNORECASE)]
    return "; ".join(hits)
