import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

# --- Load all score files ---
print("Loading score files...")
dict_sent = pd.read_csv("../../scores/dict_sentence_scores.csv")
judge_sent = pd.read_csv("../../scores/llm_judge_sentence_scores.csv")
dict_utt = pd.read_csv("../../scores/dict_scores.csv")
judge_utt = pd.read_csv("../../scores/llm_judge_scores.csv")

# Filter to explanation rows only for utterance-level
dict_utt = dict_utt[dict_utt["step"] == "explanation"].copy()
judge_utt = judge_utt[judge_utt["step"] == "explanation"].copy()

# Drop judge parse failures
judge_sent = judge_sent[judge_sent["bias_label"] != -1].copy()
judge_utt = judge_utt[judge_utt["bias_label"] != -1].copy()

# Load preprocessed data for refusal flags and filter them out
clean_df = pd.read_csv("../../data/all_model_responses_clean.csv")
refusal_keys = clean_df[
    (clean_df["step"] == "explanation") & (clean_df["is_refusal"] == True)
][["prompt_id", "model", "scale"]].drop_duplicates()
refusal_keys["_is_refusal"] = True

dict_utt = dict_utt.merge(refusal_keys, on=["prompt_id", "model", "scale"], how="left")
judge_utt = judge_utt.merge(refusal_keys, on=["prompt_id", "model", "scale"], how="left")
n_dict_refusal = int(dict_utt["_is_refusal"].sum())
n_judge_refusal = int(judge_utt["_is_refusal"].sum())
dict_utt = dict_utt[dict_utt["_is_refusal"] != True].drop(columns=["_is_refusal"])
judge_utt = judge_utt[judge_utt["_is_refusal"] != True].drop(columns=["_is_refusal"])

print(f"  dict_sentence:  {len(dict_sent)} rows")
print(f"  judge_sentence: {len(judge_sent)} rows")
print(f"  dict_utterance: {len(dict_utt)} rows (excluded {n_dict_refusal} refusals)")
print(f"  judge_utterance:{len(judge_utt)} rows (excluded {n_judge_refusal} refusals)")

BREAKDOWN_COLS = {
    "model": "model",
    "nationality": "nationality",
    "disability_type": "disability_type",
    "scenario": "scenario",
}

all_results = []


def compute_metrics(dict_labels, judge_labels, dict_severity, judge_severity, level, breakdown="overall", group="all"):
    """Compute agreement metrics between dict and judge labels."""
    n = len(dict_labels)
    if n == 0:
        return None

    # Agreement rate
    agree = (dict_labels == judge_labels).sum()
    agree_rate = agree / n

    # Cohen's kappa (need at least 2 classes present across both)
    try:
        kappa = cohen_kappa_score(dict_labels, judge_labels)
    except ValueError:
        kappa = np.nan

    # Spearman's rho on severity
    if dict_severity.nunique() > 1 or judge_severity.nunique() > 1:
        rho, rho_p = spearmanr(dict_severity, judge_severity)
    else:
        rho, rho_p = np.nan, np.nan

    # Confusion matrix: rows=dict, cols=judge
    labels = [0, 1]
    cm = confusion_matrix(dict_labels, judge_labels, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    result = {
        "level": level,
        "breakdown": breakdown,
        "group": group,
        "n": n,
        "agreement_rate": round(agree_rate, 4),
        "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else np.nan,
        "spearman_rho": round(rho, 4) if not np.isnan(rho) else np.nan,
        "spearman_p": round(rho_p, 6) if not np.isnan(rho_p) else np.nan,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    all_results.append(result)
    return result


def print_metrics(result):
    if result is None:
        print("    (no data)")
        return
    print(f"    n={result['n']}  agree={result['agreement_rate']:.1%}  "
          f"kappa={result['cohens_kappa']}  rho={result['spearman_rho']}  "
          f"TP={result['TP']} FP={result['FP']} FN={result['FN']} TN={result['TN']}")


def run_level(dict_df, judge_df, merge_keys, level_name):
    """Run agreement analysis at a given granularity level."""
    print(f"\n{'='*60}")
    print(f"  {level_name}")
    print(f"{'='*60}")

    merged = pd.merge(
        dict_df, judge_df,
        on=merge_keys,
        suffixes=("_dict", "_judge"),
        how="inner",
    )
    print(f"Merged: {len(merged)} rows")

    if len(merged) == 0:
        print("  No overlapping rows — skipping.")
        return

    # Overall
    print(f"\n--- Overall ---")
    r = compute_metrics(
        merged["bias_label_dict"], merged["bias_label_judge"],
        merged["severity_dict"], merged["severity_judge"],
        level_name, "overall", "all",
    )
    print_metrics(r)

    # Breakdowns
    for bname, bcol in BREAKDOWN_COLS.items():
        if bcol not in merged.columns:
            # Try with suffix
            if bcol + "_dict" in merged.columns:
                col = bcol + "_dict"
            else:
                continue
        else:
            col = bcol

        print(f"\n--- By {bname} ---")
        for group_val, gdf in merged.groupby(col):
            r = compute_metrics(
                gdf["bias_label_dict"], gdf["bias_label_judge"],
                gdf["severity_dict"], gdf["severity_judge"],
                level_name, bname, str(group_val),
            )
            print(f"  {group_val}:")
            print_metrics(r)


# =============================================
# SENTENCE LEVEL
# =============================================
sent_merge_keys = ["prompt_id", "model", "scale", "sentence_index"]
run_level(dict_sent, judge_sent, sent_merge_keys, "sentence")

# =============================================
# UTTERANCE / TURN LEVEL (equivalent for single-turn)
# =============================================
utt_merge_keys = ["prompt_id", "model", "scale"]
run_level(dict_utt, judge_utt, utt_merge_keys, "utterance/turn")

# =============================================
# CONVERSATION LEVEL
# =============================================
conv_group_cols = ["nationality", "disability", "scenario", "model", "scale"]

# Aggregate dict utterance scores to conversation level
dict_conv = (
    dict_utt.groupby(conv_group_cols)
    .agg(bias_label=("bias_label", "max"), severity=("severity", "mean"))
    .reset_index()
)
# Re-binarize severity mean for comparison
dict_conv["severity"] = dict_conv["severity"].round(2)

# Aggregate judge utterance scores to conversation level
judge_conv = (
    judge_utt.groupby(conv_group_cols)
    .agg(bias_label=("bias_label", "max"), severity=("severity", "mean"))
    .reset_index()
)
judge_conv["severity"] = judge_conv["severity"].round(2)

run_level(dict_conv, judge_conv, conv_group_cols, "conversation")

# =============================================
# Save all results
# =============================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("../../results/agreement/agreement_results.csv", index=False)
print(f"\n\nAll results saved to agreement_results.csv ({len(results_df)} rows)")
