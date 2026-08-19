"""
compute_cross_content.py — Three-way cross-content comparisons.

Reads operational statistics, intersectional results, and MultiWOZ data
to produce three-way comparison tables.

Outputs to ../statistics/cross_content_three_way/

Usage:
    python scripts/compute_cross_content.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
REPO_ROOT = os.path.join(BASE_DIR, "..", "..")

# Existing intersectional + MultiWOZ tables
INTERSECTIONAL_RATES = os.path.join(REPO_ROOT, "results", "tables", "table_bias_rates.csv")
MULTIWOZ_RATES = os.path.join(REPO_ROOT, "results", "tables", "table_multiwoz.csv")
CROSS_AGREEMENT = os.path.join(
    REPO_ROOT, "results", "tables", "table_multiwoz_vs_intersectional_agreement.csv"
)

# Operational statistics
OP_FLAGGING = os.path.join(BASE_DIR, "statistics", "operational", "flagging_rates.csv")

# Operational evaluation data (for computing AC1 on operational)
EVAL_DIR = os.path.join(BASE_DIR, "evaluations")
RESP_DIR = os.path.join(BASE_DIR, "responses")

# Output directory
OUT_DIR = os.path.join(BASE_DIR, "statistics", "cross_content_three_way")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Judge configuration ──────────────────────────────────
JUDGES = ["gemma3_1b", "gemma3_latest", "llama3.1_latest", "mistral_latest"]
JUDGE_DISPLAY = {
    "gemma3_1b": "Gemma 1B",
    "gemma3_latest": "Gemma 4.3B",
    "llama3.1_latest": "Llama 3.1",
    "mistral_latest": "Mistral",
}

# Merge keys for operational data
MERGE_KEYS = ["prompt_id", "model", "scale_direction"]


# ── Helpers ──────────────────────────────────────────────

def load_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  SKIP: {path} not found")
    return None


def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  Saved {name} ({len(df)} rows)")


def compute_ac1(labels_a, labels_b):
    """Compute Gwet's AC1 from two binary label arrays."""
    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)
    pa = np.mean(a == b)
    pbar = (np.mean(a) + np.mean(b)) / 2
    pe_ac1 = 2 * pbar * (1 - pbar)
    if pe_ac1 >= 1.0:
        return np.nan
    return (pa - pe_ac1) / (1 - pe_ac1)


# ── Load existing data ───────────────────────────────────
print("Loading existing tables...")

intersectional = load_if_exists(INTERSECTIONAL_RATES)
multiwoz = load_if_exists(MULTIWOZ_RATES)
cross_agreement = load_if_exists(CROSS_AGREEMENT)
op_flagging = load_if_exists(OP_FLAGGING)


# ══════════════════════════════════════════════════════════
# 1. flagging_rates_three_way.csv
# ══════════════════════════════════════════════════════════
print("\n[1] Three-way flagging rates...")
tw_rows = []

# Build lookup dictionaries
inter_lookup = {}
if intersectional is not None:
    for _, row in intersectional.iterrows():
        inter_lookup[row["method"]] = row["bias_rate"]

mw_lookup = {}
if multiwoz is not None:
    for _, row in multiwoz.iterrows():
        # Use turn-level rates for MultiWOZ (dialogue-level is a different metric)
        if "(turn)" in row["method"]:
            clean_name = row["method"].replace(" (turn)", "")
            mw_lookup[clean_name] = row["bias_rate"]

op_lookup = {}
if op_flagging is not None:
    for _, row in op_flagging.iterrows():
        op_lookup[row["method"]] = row["bias_rate"]

# Collect all method names across datasets
all_methods = sorted(set(
    list(inter_lookup.keys()) +
    list(mw_lookup.keys()) +
    list(op_lookup.keys())
))

for method in all_methods:
    tw_rows.append({
        "method": method,
        "intersectional_rate": inter_lookup.get(method, np.nan),
        "multiwoz_rate": mw_lookup.get(method, np.nan),
        "operational_rate": op_lookup.get(method, np.nan),
    })

save_csv(pd.DataFrame(tw_rows), "flagging_rates_three_way.csv")


# ══════════════════════════════════════════════════════════
# 2. ac1_three_way.csv
# ══════════════════════════════════════════════════════════
print("\n[2] Three-way AC1 agreement...")

# Extract existing AC1 from cross_agreement table
inter_ac1_lookup = {}
mw_ac1_lookup = {}
if cross_agreement is not None:
    for _, row in cross_agreement.iterrows():
        pair = row["pair"]
        if row["dataset"] == "intersectional":
            inter_ac1_lookup[pair] = row.get("ac1", np.nan)
        elif row["dataset"] == "multiwoz_500":
            mw_ac1_lookup[pair] = row.get("ac1", np.nan)

# Compute AC1 on operational data
print("  Computing operational AC1...")
op_ac1_lookup = {}

# Load operational evaluation data to compute pairwise AC1
non_refusals = load_if_exists(os.path.join(RESP_DIR, "operational_non_refusals.csv"))
dict_resp = load_if_exists(os.path.join(EVAL_DIR, "dictionary_operational.csv"))

# Build operational base frame
op_base = None
if non_refusals is not None:
    op_base = non_refusals[MERGE_KEYS].drop_duplicates().copy()

    # Dictionary labels
    DICT_METHODS = {
        "Bias-Lexicon": "bias_label",
        "HurtLex": "hurtlex_label",
        "Toxicity": "toxicity_label",
    }
    if dict_resp is not None:
        for method_name, col in DICT_METHODS.items():
            if col in dict_resp.columns:
                sub = dict_resp[MERGE_KEYS + [col]].drop_duplicates(subset=MERGE_KEYS)
                sub = sub.rename(columns={col: f"bl_{method_name}"})
                op_base = op_base.merge(sub, on=MERGE_KEYS, how="left")

    # Judge v1 labels
    for judge in JUDGES:
        display = JUDGE_DISPLAY[judge]
        path = os.path.join(EVAL_DIR, f"judge_operational_{judge}_v1.csv")
        df = load_if_exists(path)
        if df is not None:
            df = df[df["bias_label"] != -1].copy()
            sub = df[MERGE_KEYS + ["bias_label"]].drop_duplicates(subset=MERGE_KEYS)
            sub = sub.rename(columns={"bias_label": f"bl_{display}"})
            op_base = op_base.merge(sub, on=MERGE_KEYS, how="left")

# Compute all pairwise AC1 on operational
if op_base is not None:
    op_methods = [c.replace("bl_", "") for c in op_base.columns if c.startswith("bl_")]
    for i, m1 in enumerate(op_methods):
        for j, m2 in enumerate(op_methods):
            if j <= i:
                continue
            c1, c2 = f"bl_{m1}", f"bl_{m2}"
            mask = op_base[c1].notna() & op_base[c2].notna()
            sub = op_base[mask]
            if len(sub) < 5:
                continue
            ac1_val = compute_ac1(sub[c1].astype(int).values, sub[c2].astype(int).values)
            pair_key = f"{m1} vs {m2}"
            op_ac1_lookup[pair_key] = round(ac1_val, 4) if not np.isnan(ac1_val) else np.nan

# Build three-way AC1 table
# Collect all pair names
all_pairs = sorted(set(
    list(inter_ac1_lookup.keys()) +
    list(mw_ac1_lookup.keys()) +
    list(op_ac1_lookup.keys())
))

ac1_tw_rows = []
for pair in all_pairs:
    # Parse method names from pair
    parts = pair.split(" vs ")
    if len(parts) != 2:
        continue
    ac1_tw_rows.append({
        "method_a": parts[0],
        "method_b": parts[1],
        "ac1_intersectional": inter_ac1_lookup.get(pair, np.nan),
        "ac1_multiwoz": mw_ac1_lookup.get(pair, np.nan),
        "ac1_operational": op_ac1_lookup.get(pair, np.nan),
    })

save_csv(pd.DataFrame(ac1_tw_rows), "ac1_three_way.csv")


# ══════════════════════════════════════════════════════════
# 3. granularity_three_way.csv
# ══════════════════════════════════════════════════════════
print("\n[3] Three-way granularity comparison...")

gran_rows = []

# Intersectional: sentence vs response rates
# Load from existing score files if available
inter_sent_scores = os.path.join(REPO_ROOT, "scores", "dict_sentence_scores.csv")
inter_resp_scores = os.path.join(REPO_ROOT, "scores", "dict_scores.csv")

inter_sent = load_if_exists(inter_sent_scores)
inter_resp = load_if_exists(inter_resp_scores)

if inter_sent is not None and inter_resp is not None:
    inter_resp_expl = inter_resp[inter_resp["step"] == "explanation"] if "step" in inter_resp.columns else inter_resp
    for col, name in [("bias_label", "Bias-Lexicon")]:
        if col in inter_sent.columns and col in inter_resp_expl.columns:
            sent_rate = inter_sent[col].dropna().mean()
            resp_rate = inter_resp_expl[col].dropna().mean()
            inflation = resp_rate / sent_rate if sent_rate > 0 else np.nan
            gran_rows.append({
                "method": name, "dataset": "intersectional",
                "sentence_rate": round(sent_rate, 4),
                "response_rate": round(resp_rate, 4),
                "inflation": round(inflation, 4) if not np.isnan(inflation) else np.nan,
            })

# MultiWOZ: turn-level vs dialogue-level (closest analog)
if multiwoz is not None:
    for base_method in ["Bias-Lexicon", "HurtLex"]:
        turn_row = multiwoz[multiwoz["method"] == f"{base_method} (turn)"]
        dlg_row = multiwoz[multiwoz["method"] == f"{base_method} (dialogue)"]
        if len(turn_row) > 0 and len(dlg_row) > 0:
            turn_rate = turn_row.iloc[0]["bias_rate"]
            dlg_rate = dlg_row.iloc[0]["bias_rate"]
            inflation = dlg_rate / turn_rate if turn_rate > 0 else np.nan
            gran_rows.append({
                "method": base_method, "dataset": "multiwoz",
                "sentence_rate": round(turn_rate, 6),
                "response_rate": round(dlg_rate, 6),
                "inflation": round(inflation, 4) if not np.isnan(inflation) else np.nan,
            })

# Operational: sentence vs response rates
op_gran = load_if_exists(
    os.path.join(BASE_DIR, "statistics", "operational", "granularity_inter.csv")
)
if op_gran is not None:
    for _, row in op_gran.iterrows():
        gran_rows.append({
            "method": row["method"], "dataset": "operational",
            "sentence_rate": row.get("sentence_rate", np.nan),
            "response_rate": row.get("response_rate", np.nan),
            "inflation": row.get("inflation", np.nan),
        })

save_csv(pd.DataFrame(gran_rows), "granularity_three_way.csv")


print("\nAll cross-content three-way comparisons generated.")
