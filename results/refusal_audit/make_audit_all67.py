"""Blind coding sheet for all 67 declarative refusals (explanation rows, is_refusal True).
Writes refusal_audit_all67.csv (audit_id, response, coder_label[empty], notes[empty]; shuffled, seed 42) and
refusal_audit_all67_key.csv (audit_id -> sample id, model, nationality, regex patterns, length,
phrase position, phrase_at_start). 'phrase at start' = same rule as make_robustness_summary.py:
first refusal-phrase match at character offset <= 5.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refusal_sets import matched_patterns  # noqa: E402

ROOT = HERE.parents[1]
sheet = HERE / "refusal_audit_all67.csv"
labelled = sheet.exists() and pd.read_csv(sheet, dtype=str).coder_label.notna().any()
d = pd.read_csv(ROOT / "data/all_model_responses_clean.csv")
ref = d[(d.step == "explanation") & d.is_refusal].copy()
assert len(ref) == 67
ref["sample_id"] = ref.prompt_id.astype(str) + "_" + ref.model + "_" + ref.scale
ref["pos"] = ref.response_clean.map(
    lambda t: (re.search(r"I can'?t|I cannot|I'?m not able", t, re.I) or re.search("$", t)).start())
ref["phrase_at_start"] = (ref.pos <= 5).astype(int)
assert ref.phrase_at_start.sum() == 54
ref["matched_regex_pattern"] = ref.response_clean.map(matched_patterns)
ref["char_length"] = ref.response_clean.str.len()
ref = ref.sample(frac=1, random_state=42).reset_index(drop=True)
ref["audit_id"] = [f"R{i:02d}" for i in range(1, 68)]

if not labelled:       # never overwrite hand-coded labels; the key is deterministic and always rewritten
    ref[["audit_id", "response"]].assign(coder_label="", notes="").to_csv(sheet, index=False)
ref[["audit_id", "sample_id", "model", "nationality", "matched_regex_pattern", "char_length", "pos",
     "phrase_at_start"]].to_csv(HERE / "refusal_audit_all67_key.csv", index=False)
print(ref.nationality.value_counts().to_dict(), ref.phrase_at_start.value_counts().to_dict())
