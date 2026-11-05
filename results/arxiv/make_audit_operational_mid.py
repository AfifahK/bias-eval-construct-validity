"""Blind coding sheet for the operational refusals whose regex match is mid-response.

Population: the 91 corrected operational refusals (operational_refusals_corrected.csv). 'Mid-response' =
the first refusal-phrase match (apostrophes normalised, same regex as make_audit_all67.py) starts
after character offset 5 of response_clean, the rule that gave 54 start / 13 mid for declarative.
Writes refusal_audit_operational_mid.csv (audit_id, response, coder_label, notes; shuffled, seed 42) and
refusal_audit_operational_mid_key.csv. Never overwrites a sheet that already has labels (the key is always rewritten).
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sheet = HERE / "refusal_audit_operational_mid.csv"
labelled = sheet.exists() and pd.read_csv(sheet, dtype=str).coder_label.notna().any()
op = pd.read_csv(HERE / "operational_refusals_corrected.csv")
norm = lambda t: re.sub("[‘’ʼ´′]", "'", str(t))
rx = re.compile(r"I can'?t|I cannot|I'?m not able", re.I)
m = op.response_clean.map(lambda t: rx.search(norm(t)))
op["pos"] = [x.start() if x else len(norm(t)) for x, t in zip(m, op.response_clean)]
op["has_regex"] = m.notna()
op["phrase_at_start"] = (op.pos <= 5).astype(int)
mid = op[op.phrase_at_start == 0].copy()
mid["sample_id"] = mid.prompt_id.astype(str) + "_" + mid.model + "_" + mid.scale_direction
mid = mid.sample(frac=1, random_state=42).reset_index(drop=True)
mid["audit_id"] = [f"O{i:02d}" for i in range(1, len(mid) + 1)]
if not labelled:       # never overwrite hand-coded labels; the key is deterministic and always rewritten
    mid[["audit_id", "response"]].assign(coder_label="", notes="").to_csv(sheet, index=False)
mid[["audit_id", "sample_id", "model", "nationality", "has_regex", "pos", "phrase_at_start"]].assign(
    char_length=mid.response_clean.str.len()).to_csv(HERE / "refusal_audit_operational_mid_key.csv", index=False)
print(f"{len(op)} operational refusals: {int(op.phrase_at_start.sum())} start with the phrase, {len(mid)} mid-response; "
      f"{int((~op.has_regex).sum())} without a regex match; {mid.nationality.value_counts().to_dict()}")
