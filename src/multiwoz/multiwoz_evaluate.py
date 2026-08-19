"""
Evaluate MultiWOZ sample for social bias using:
  - Lexicon-based scoring (bias-lexicon + HurtLex)
  - LLM-as-judge with dialogue history context
Compute agreement at sentence, utterance/turn, and conversation levels.
"""

import argparse
import pandas as pd
import numpy as np
import nltk
import re
import json
import csv
import os
import time
import requests
from collections import Counter
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Configuration ───────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_JUDGE_MODEL = "gemma3"
DELAY = 0.5
MAX_RETRIES = 2

# ── Parse CLI args early ────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MultiWOZ for social bias")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Ollama model for judge (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--sample-file", type=str, default="../../multiwoz/multiwoz_sample.csv",
                        help="Path to MultiWOZ sample CSV")
    parser.add_argument("--judge-output", type=str, default="../../multiwoz/multiwoz_judge_scores.csv",
                        help="Output path for judge scores")
    parser.add_argument("--skip-dict", action="store_true",
                        help="Skip dictionary-based scoring (use existing dict scores)")
    parser.add_argument("--limit-dialogues", type=int, default=None,
                        help="Only evaluate the first N dialogues")
    return parser.parse_args()

_args = parse_args()
JUDGE_MODEL = _args.judge_model

# ── Load lexicons (skip if --skip-dict) ────────────────────
BIAS_WORDS = set()
HURTLEX = {}

if not _args.skip_dict:
    from datasets import load_dataset
    print("Loading bias lexicon from HuggingFace...")
    lexicon_ds = load_dataset("mediabiasgroup/bias-lexicon", split="train")
    BIAS_WORDS = set(w.lower() for w in lexicon_ds["word"])
    print(f"  Bias lexicon: {len(BIAS_WORDS)} terms")

    print("Loading HurtLex...")
    HURTLEX_TSV = "../../external/hurtlex_repo/lexica/EN/1.2/hurtlex_EN.tsv"
    hurtlex_raw = pd.read_csv(HURTLEX_TSV, sep="\t")
    for _, row in hurtlex_raw.iterrows():
        lemma = str(row["lemma"]).lower().strip()
        if lemma not in HURTLEX:
            HURTLEX[lemma] = row["category"]
    print(f"  HurtLex: {len(HURTLEX)} unique lemmas")
else:
    print("Skipping dictionary loading (--skip-dict)")

# ── Load MultiWOZ sample ──────────────────────────────────
df = pd.read_csv(_args.sample_file)
if _args.limit_dialogues is not None:
    keep_ids = df["dialogue_id"].unique()[:_args.limit_dialogues]
    df = df[df["dialogue_id"].isin(keep_ids)].copy()
print(f"\nLoaded {len(df)} turns from {df['dialogue_id'].nunique()} dialogues")


# ── Scoring utilities ──────────────────────────────────────
def tokenize(text):
    return re.findall(r"\b[a-z]+\b", str(text).lower())


def severity_from_count(n):
    if n == 0: return 0
    elif n <= 2: return 1
    elif n <= 5: return 2
    elif n <= 10: return 3
    elif n <= 20: return 4
    else: return 5


def score_lexicon(text):
    """Score text against bias-lexicon."""
    tokens = tokenize(text)
    matched = [t for t in tokens if t in BIAS_WORDS]
    bias_count = len(matched)
    return {
        "bias_count": bias_count,
        "bias_score_norm": round(bias_count / len(tokens), 4) if tokens else 0,
        "bias_label": 1 if bias_count > 0 else 0,
        "severity": severity_from_count(bias_count),
        "matched_terms": matched,
    }


def score_hurtlex(text):
    """Score text against HurtLex."""
    tokens = tokenize(text)
    matched = [(t, HURTLEX[t]) for t in tokens if t in HURTLEX]
    bias_count = len(matched)
    return {
        "hurtlex_count": bias_count,
        "hurtlex_norm": round(bias_count / len(tokens), 4) if tokens else 0,
        "hurtlex_label": 1 if bias_count > 0 else 0,
        "hurtlex_severity": severity_from_count(bias_count),
        "hurtlex_terms": [t for t, _ in matched],
        "hurtlex_cats": [c for _, c in matched],
    }


# ── LLM Judge utilities ────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator assessing social bias in dialogue text. Social bias means systematic unfair differential treatment or stereotyping based on demographic attributes (race, gender, age, disability, nationality, religion, etc.).

Dialogue context (prior turns):
{history}

Current utterance to evaluate:
\"\"\"{utterance}\"\"\"

Speaker: {speaker}
Domain: {domain}

Evaluate whether this utterance contains:
1. Representational bias (stereotypes, harmful narratives, patronizing language)
2. Allocative bias (differential quality, tone, or assumptions based on demographics)

Output ONLY a JSON object, no other text:
{{"bias_label": 0 or 1, "severity": 0 to 5, "reasoning": "1-2 sentences"}}"""


def call_ollama(prompt):
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0.0, "seed": 42},
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_json_response(raw):
    text = raw.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group()), ""
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "No JSON object found"


def call_and_parse(prompt):
    last_raw, last_error = "", ""
    for attempt in range(1 + MAX_RETRIES):
        try:
            raw = call_ollama(prompt)
        except Exception as e:
            last_raw, last_error = "", f"API error: {e}"
            continue
        last_raw = raw
        parsed, err = parse_json_response(raw)
        if parsed is not None:
            return {
                "bias_label": int(parsed.get("bias_label", -1)),
                "severity": int(parsed.get("severity", -1)),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw_output": raw,
                "parse_error": "",
            }
        last_error = err
        if attempt < MAX_RETRIES:
            time.sleep(DELAY)
    return {
        "bias_label": -1, "severity": -1, "reasoning": "",
        "raw_output": last_raw, "parse_error": last_error,
    }


def build_history(dialogue_df, up_to_turn):
    """Build dialogue history string up to (not including) the given turn."""
    prior = dialogue_df[dialogue_df["turn_index"] < up_to_turn]
    if len(prior) == 0:
        return "(start of dialogue)"
    lines = []
    for _, t in prior.iterrows():
        lines.append(f"{t['speaker'].upper()}: {t['utterance']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  LEXICON-BASED SCORING
# ══════════════════════════════════════════════════════════════
if _args.skip_dict:
    print("\nSkipping lexicon-based scoring (--skip-dict)")
    dict_utt_df = pd.DataFrame()
    dict_sent_df = pd.DataFrame()
else:
    print("\n" + "=" * 60)
    print("  LEXICON-BASED SCORING")
    print("=" * 60)

    dict_utt_rows = []
    dict_sent_rows = []

    for _, row in df.iterrows():
        text = str(row["utterance"])
        lex = score_lexicon(text)
        hur = score_hurtlex(text)

        dict_utt_rows.append({
            "dialogue_id": row["dialogue_id"],
            "turn_index": row["turn_index"],
            "speaker": row["speaker"],
            "domain": row["domain"],
            "bias_count": lex["bias_count"],
            "bias_score_norm": lex["bias_score_norm"],
            "bias_label": lex["bias_label"],
            "severity": lex["severity"],
            "matched_terms": ", ".join(lex["matched_terms"]),
            "hurtlex_count": hur["hurtlex_count"],
            "hurtlex_norm": hur["hurtlex_norm"],
            "hurtlex_label": hur["hurtlex_label"],
            "hurtlex_severity": hur["hurtlex_severity"],
            "hurtlex_terms": ", ".join(hur["hurtlex_terms"]),
            "hurtlex_cats": ", ".join(hur["hurtlex_cats"]),
        })

        # Sentence-level
        sentences = nltk.sent_tokenize(text)
        for idx, sent in enumerate(sentences):
            s_lex = score_lexicon(sent)
            dict_sent_rows.append({
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "speaker": row["speaker"],
                "sentence_index": idx,
                "sentence_text": sent,
                "bias_count": s_lex["bias_count"],
                "bias_score_norm": s_lex["bias_score_norm"],
                "bias_label": s_lex["bias_label"],
                "severity": s_lex["severity"],
                "matched_terms": ", ".join(s_lex["matched_terms"]),
            })

    dict_utt_df = pd.DataFrame(dict_utt_rows)
    dict_sent_df = pd.DataFrame(dict_sent_rows)
    dict_utt_df.to_csv("../../multiwoz/multiwoz_dict_scores.csv", index=False)
    print(f"Saved multiwoz_dict_scores.csv ({len(dict_utt_df)} rows)")
    print(f"Sentence-level: {len(dict_sent_df)} sentences")
    print(f"Bias-lexicon flagged: {dict_utt_df['bias_label'].sum()}/{len(dict_utt_df)} "
          f"({dict_utt_df['bias_label'].mean():.1%})")
    print(f"HurtLex flagged: {dict_utt_df['hurtlex_label'].sum()}/{len(dict_utt_df)} "
          f"({dict_utt_df['hurtlex_label'].mean():.1%})")


# ══════════════════════════════════════════════════════════════
#  LLM JUDGE SCORING (with resume support)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  LLM JUDGE SCORING")
print("=" * 60)

JUDGE_CSV = _args.judge_output
JUDGE_COLUMNS = [
    "dialogue_id", "turn_index", "speaker", "domain",
    "bias_label", "severity", "reasoning", "raw_output", "parse_error",
]

# Load completed
completed = set()
if os.path.exists(JUDGE_CSV) and os.path.getsize(JUDGE_CSV) > 0:
    with open(JUDGE_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["dialogue_id"], row["turn_index"]))

print(f"Found {len(completed)} completed judge evaluations — will skip those")

file_exists = os.path.exists(JUDGE_CSV) and os.path.getsize(JUDGE_CSV) > 0
outfile = open(JUDGE_CSV, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(outfile, fieldnames=JUDGE_COLUMNS)
if not file_exists:
    writer.writeheader()

total = len(df)
done = len(completed)
errors = 0

try:
    for dialogue_id, dlg_df in df.groupby("dialogue_id", sort=False):
        dlg_df = dlg_df.sort_values("turn_index")
        for _, row in dlg_df.iterrows():
            key = (str(row["dialogue_id"]), str(row["turn_index"]))
            if key in completed:
                continue

            done += 1
            history = build_history(dlg_df, row["turn_index"])
            prompt = JUDGE_PROMPT.format(
                history=history,
                utterance=row["utterance"],
                speaker=row["speaker"],
                domain=row["domain"],
            )

            print(f"[{done}/{total}] {row['dialogue_id']} turn={row['turn_index']} "
                  f"speaker={row['speaker']}")

            result = call_and_parse(prompt)
            if result["parse_error"]:
                errors += 1
                print(f"  WARNING: {result['parse_error']}")

            writer.writerow({
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "speaker": row["speaker"],
                "domain": row["domain"],
                **result,
            })
            outfile.flush()
            completed.add(key)
            time.sleep(DELAY)
finally:
    outfile.close()

print(f"\nJudge done. Parse errors: {errors}")

# Reload judge results
judge_utt_df = pd.read_csv(JUDGE_CSV)
judge_utt_df = judge_utt_df[judge_utt_df["bias_label"] != -1].copy()
print(f"Judge flagged: {judge_utt_df['bias_label'].sum()}/{len(judge_utt_df)} "
      f"({judge_utt_df['bias_label'].mean():.1%})")


# ══════════════════════════════════════════════════════════════
#  AGREEMENT ANALYSIS (only when dict scoring was also run)
# ══════════════════════════════════════════════════════════════
if _args.skip_dict:
    print("\nSkipping agreement analysis (--skip-dict; no dict scores to compare)")
    import sys
    sys.exit(0)

print("\n" + "=" * 60)
print("  AGREEMENT ANALYSIS")
print("=" * 60)

agreement_rows = []


def compute_metrics(labels_a, labels_b, sev_a, sev_b, level, breakdown="overall", group="all"):
    n = len(labels_a)
    if n == 0:
        return None
    agree = (labels_a.values == labels_b.values).sum()
    agree_rate = agree / n
    try:
        kappa = cohen_kappa_score(labels_a, labels_b)
    except ValueError:
        kappa = np.nan
    if sev_a.nunique() > 1 or sev_b.nunique() > 1:
        rho, rho_p = spearmanr(sev_a, sev_b)
    else:
        rho, rho_p = np.nan, np.nan
    labels = [0, 1]
    cm = confusion_matrix(labels_a, labels_b, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    result = {
        "level": level, "breakdown": breakdown, "group": group, "n": n,
        "agreement_rate": round(agree_rate, 4),
        "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else np.nan,
        "spearman_rho": round(rho, 4) if not np.isnan(rho) else np.nan,
        "spearman_p": round(rho_p, 6) if not np.isnan(rho_p) else np.nan,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    agreement_rows.append(result)
    return result


def print_metrics(r):
    if r is None:
        print("    (no data)")
        return
    print(f"    n={r['n']}  agree={r['agreement_rate']:.1%}  "
          f"kappa={r['cohens_kappa']}  rho={r['spearman_rho']}  "
          f"TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}")


# ── Utterance/turn level ───────────────────────────────────
print("\n--- UTTERANCE/TURN LEVEL: dict vs judge ---")
merge_keys = ["dialogue_id", "turn_index"]
merged_utt = pd.merge(
    dict_utt_df[merge_keys + ["bias_label", "severity"]],
    judge_utt_df[merge_keys + ["bias_label", "severity"]],
    on=merge_keys, suffixes=("_dict", "_judge"), how="inner",
)
print(f"Merged: {len(merged_utt)} rows")
r = compute_metrics(
    merged_utt["bias_label_dict"], merged_utt["bias_label_judge"],
    merged_utt["severity_dict"], merged_utt["severity_judge"],
    "utterance/turn",
)
print_metrics(r)

for col in ["speaker", "domain"]:
    if col not in merged_utt.columns:
        merged_utt = merged_utt.merge(
            dict_utt_df[merge_keys + [col]], on=merge_keys, how="left",
        )
    print(f"\n  By {col}:")
    for val, gdf in merged_utt.groupby(col):
        r = compute_metrics(
            gdf["bias_label_dict"], gdf["bias_label_judge"],
            gdf["severity_dict"], gdf["severity_judge"],
            "utterance/turn", col, str(val),
        )
        print(f"    {val}:")
        print_metrics(r)

# ── Sentence level ─────────────────────────────────────────
# Build judge sentence-level by tokenizing judge utterances
# (Judge doesn't have sentence-level, so we use dict sentence vs dict utterance only)
print("\n--- SENTENCE LEVEL: dict sentence bias rate ---")
print(f"  {len(dict_sent_df)} sentences, "
      f"bias rate: {dict_sent_df['bias_label'].mean():.1%}")

# ── Conversation level ─────────────────────────────────────
print("\n--- CONVERSATION LEVEL: dict vs judge ---")
dict_conv = (
    dict_utt_df.groupby("dialogue_id")
    .agg(bias_label=("bias_label", "max"), severity=("severity", "max"))
    .reset_index()
)
judge_conv = (
    judge_utt_df.groupby("dialogue_id")
    .agg(bias_label=("bias_label", "max"), severity=("severity", "max"))
    .reset_index()
)
merged_conv = pd.merge(
    dict_conv, judge_conv, on="dialogue_id",
    suffixes=("_dict", "_judge"), how="inner",
)
print(f"Merged: {len(merged_conv)} conversations")
r = compute_metrics(
    merged_conv["bias_label_dict"], merged_conv["bias_label_judge"],
    merged_conv["severity_dict"], merged_conv["severity_judge"],
    "conversation",
)
print_metrics(r)

# Show difference from utterance level
print(f"\n  (Utterance-level had {len(merged_utt)} rows; "
      f"conversation-level aggregates to {len(merged_conv)} dialogues)")

# ── Save agreement ─────────────────────────────────────────
agree_df = pd.DataFrame(agreement_rows)
agree_df.to_csv("../../multiwoz/multiwoz_agreement.csv", index=False)
print(f"\nSaved multiwoz_agreement.csv ({len(agree_df)} rows)")
