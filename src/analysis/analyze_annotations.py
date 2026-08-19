#!/usr/bin/env python3
"""
Analyze completed ground truth annotations for BOTH datasets.
Supports N annotators (auto-discovered from data/coder*/ directories).
Uses Fleiss's kappa for 3+ coders, Cohen's kappa for 2.

Run AFTER annotators fill in annotation columns in their respective files.

Produces:
  - results/tables/table_ground_truth_summary.csv
  - results/tables/table_method_accuracy.csv
  - results/tables/table_ground_truth_multiwoz.csv
  - results/tables/table_method_accuracy_multiwoz.csv
  - results/tables/table_inter_annotator.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import glob as globmod

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "tables"


# ── Auto-discover coders ──────────────────────────────────────────────────────

def discover_annotators():
    """Auto-discover coder directories and their file mappings."""
    annotators = {}
    coder_dirs = sorted((ROOT / "data" / "annotations").glob("coder*"))
    for coder_dir in coder_dirs:
        coder_name = coder_dir.name  # e.g. "coder1"
        csvs = list(coder_dir.glob("*.csv"))
        if not csvs:
            continue

        # Identify intersectional and multiwoz files
        inter_file = None
        mwoz_file = None
        for f in csvs:
            fname = f.name.lower()
            if "multiwoz" in fname:
                mwoz_file = f
            else:
                inter_file = f

        # Try to extract human-readable name from filename
        # e.g. "annotation_sample_coder2.csv" -> "coder2"
        #      "annotation_sample.csv" -> use directory name
        display_name = coder_name
        if inter_file:
            stem = inter_file.stem.lower()
            stem = stem.replace("annotation_sample", "").strip("_ -")
            # Handle Google Sheets export names like "file - file.csv"
            if " - " in inter_file.name:
                stem = inter_file.stem.split(" - ")[0]
                stem = stem.replace("annotation_sample", "").strip("_ -")
            if stem:
                display_name = stem

        annotators[display_name] = {
            "dir": coder_dir,
            "intersectional": inter_file,
            "multiwoz": mwoz_file,
        }

    return annotators


# ── Agreement metrics ─────────────────────────────────────────────────────────

def cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two binary arrays."""
    y1, y2 = np.array(y1), np.array(y2)
    n = len(y1)
    if n == 0:
        return np.nan
    po = (y1 == y2).mean()
    p1 = y1.mean()
    p2 = y2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix):
    """
    Compute Fleiss's kappa for N raters on binary categories.
    matrix: (n_subjects, n_categories) array where each row sums to n_raters.
    """
    N, k = matrix.shape
    n = matrix[0].sum()  # number of raters per subject
    if N == 0 or n <= 1:
        return np.nan

    # Proportion in each category
    p_j = matrix.sum(axis=0) / (N * n)

    # Per-subject agreement
    P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()

    # Expected agreement
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def compute_agreement(labels_dict, dataset_label):
    """
    Compute inter-annotator agreement for N coders.
    labels_dict: {coder_name: array of labels} — all same length, aligned by index.
    Returns dict with metrics.
    """
    names = list(labels_dict.keys())
    n_coders = len(names)

    if n_coders < 2:
        return None

    arrays = [np.array(labels_dict[n]) for n in names]
    n_items = len(arrays[0])

    if n_coders == 2:
        # Cohen's kappa
        pct = (arrays[0] == arrays[1]).mean()
        kappa = cohens_kappa(arrays[0], arrays[1])
        return {
            "dataset": dataset_label,
            "n_coders": n_coders,
            "n_items": n_items,
            "pct_agreement": round(pct, 4),
            "kappa": round(kappa, 4),
            "kappa_type": "Cohen",
        }
    else:
        # Fleiss's kappa — build rating matrix
        # Each row: [count_of_0, count_of_1] across all raters
        stacked = np.column_stack(arrays)
        matrix = np.zeros((n_items, 2), dtype=int)
        matrix[:, 0] = (stacked == 0).sum(axis=1)
        matrix[:, 1] = (stacked == 1).sum(axis=1)

        fk = fleiss_kappa(matrix)

        # Also compute pairwise Cohen's kappas for detail
        pairwise = []
        for i in range(n_coders):
            for j in range(i + 1, n_coders):
                pk = cohens_kappa(arrays[i], arrays[j])
                pct = (arrays[i] == arrays[j]).mean()
                pairwise.append({
                    "coder_a": names[i], "coder_b": names[j],
                    "pct_agreement": round(pct, 4),
                    "cohens_kappa": round(pk, 4),
                })

        # Overall % agreement (all N agree)
        unanimous = np.all(stacked == stacked[:, 0:1], axis=1).mean()

        return {
            "dataset": dataset_label,
            "n_coders": n_coders,
            "n_items": n_items,
            "pct_unanimous": round(unanimous, 4),
            "kappa": round(fk, 4),
            "kappa_type": "Fleiss",
            "pairwise": pairwise,
        }


# ── Loading and cleaning ─────────────────────────────────────────────────────

def load_and_clean(path, annotator_name="annotator"):
    """Load annotation CSV, normalize labels, return (all_rows, clear_rows)."""
    if path is None or not path.exists():
        print(f"  [{annotator_name}] File not found: {path}")
        return None, None

    # Try utf-8 first, fallback to latin-1
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    if "human_bias_label" not in df.columns:
        print(f"  [{annotator_name}] No human_bias_label column. Skipping.")
        return None, None

    filled = df["human_bias_label"].notna() & (df["human_bias_label"].astype(str).str.strip() != "")
    n_filled = filled.sum()
    print(f"  [{annotator_name}] Annotation progress: {n_filled}/{len(df)} rows")

    if n_filled == 0:
        print(f"  [{annotator_name}] No annotations found yet. Skipping.")
        return None, None

    ann = df[filled].copy()
    ann["human_bias_label"] = ann["human_bias_label"].astype(str).str.strip().str.lower()
    label_map = {"0": 0, "1": 1, "unclear": -1, "uncertain": -1, "?": -1}
    ann["human_label"] = ann["human_bias_label"].map(label_map)
    unclear = ann["human_label"].isna() | (ann["human_label"] == -1)
    clear = ann[~unclear].copy()
    clear["human_label"] = clear["human_label"].astype(int)

    print(f"  [{annotator_name}] Clear: {len(clear)}, Unclear: {unclear.sum()}")
    return ann, clear


def load_all_annotators(annotators, file_key, merge_keys, dataset_label):
    """
    Load annotations from all discovered annotators.
    Returns: (results_dict, majority_df)
      results_dict: {name: (ann_df, clear_df)}
      majority_df: merged df with majority-vote consensus label (None if <2 coders)
    """
    results = {}
    for name, info in annotators.items():
        path = info[file_key]
        ann, clear = load_and_clean(path, annotator_name=name)
        if clear is not None and len(clear) > 0:
            results[name] = (ann, clear)

    if len(results) == 0:
        print(f"  No annotators have completed annotations for {dataset_label}.")
        return results, None, None

    if len(results) < 2:
        return results, None, None

    # Merge all coders on shared keys
    names = list(results.keys())
    merged = results[names[0]][1][merge_keys + ["human_label"]].rename(
        columns={"human_label": f"label_{names[0]}"}
    )
    for name in names[1:]:
        other = results[name][1][merge_keys + ["human_label"]].rename(
            columns={"human_label": f"label_{name}"}
        )
        merged = merged.merge(other, on=merge_keys, how="inner")

    if len(merged) == 0:
        print(f"\n  No overlapping annotations between annotators for {dataset_label}.")
        return results, None, None

    label_cols = [f"label_{n}" for n in names]

    # Agreement stats
    labels_dict = {n: merged[f"label_{n}"].values for n in names}
    agreement = compute_agreement(labels_dict, dataset_label)

    print(f"\n  INTER-ANNOTATOR AGREEMENT ({dataset_label})")
    print(f"    Coders: {len(names)} ({', '.join(names)})")
    print(f"    Overlapping items: {len(merged)}")
    print(f"    {agreement['kappa_type']}'s kappa: {agreement['kappa']:.3f}")
    if "pct_unanimous" in agreement:
        print(f"    % Unanimous: {agreement['pct_unanimous']:.1%}")
    if "pairwise" in agreement:
        for pw in agreement["pairwise"]:
            print(f"    {pw['coder_a']} vs {pw['coder_b']}: "
                  f"agree={pw['pct_agreement']:.1%}, Cohen's κ={pw['cohens_kappa']:.3f}")

    # Majority vote consensus
    merged["n_bias"] = merged[label_cols].sum(axis=1)
    threshold = len(names) / 2
    merged["majority_label"] = (merged["n_bias"] > threshold).astype(int)
    # For even number of coders, ties go to 0 (not biased) — conservative
    merged["unanimous"] = merged[label_cols].nunique(axis=1) == 1

    n_unanimous = merged["unanimous"].sum()
    n_majority_bias = (merged["majority_label"] == 1).sum()
    n_majority_nobias = (merged["majority_label"] == 0).sum()
    print(f"    Unanimous: {n_unanimous}/{len(merged)}")
    print(f"    Majority=bias: {n_majority_bias}, Majority=no-bias: {n_majority_nobias}")

    return results, merged, agreement


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_accuracy_stats(labels_true, labels_pred):
    """Return dict with accuracy, sensitivity, specificity, false_positive_rate."""
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    acc = (labels_pred == labels_true).mean()
    has_bias = labels_true == 1
    has_nobias = labels_true == 0
    sens = labels_pred[has_bias].mean() if has_bias.any() else np.nan
    spec = (1 - labels_pred[has_nobias].mean()) if has_nobias.any() else np.nan
    fp_rate = labels_pred[has_nobias].mean() if has_nobias.any() else np.nan
    return {"accuracy": acc, "sensitivity": sens, "specificity": spec, "false_positive_rate": fp_rate}


# ── Intersectional analysis ───────────────────────────────────────────────────

def analyze_intersectional(annotators):
    """Analyze intersectional annotations — rejoin with method scores from key file."""
    key_path = ROOT / "data" / "annotations" / "annotation_sample_key.csv"
    print(f"\n{'='*60}")
    print(f"  INTERSECTIONAL GROUND TRUTH")
    print(f"{'='*60}")

    if not key_path.exists():
        print(f"  Key file not found: {key_path}")
        return None

    key = pd.read_csv(key_path)
    merge_keys = ["prompt_id", "model", "scale"]

    annotator_results, majority_df, agreement = load_all_annotators(
        annotators, "intersectional", merge_keys, "intersectional"
    )
    if len(annotator_results) == 0:
        return None

    # Build key lookup from primary file (coder1)
    first_coder = list(annotators.values())[0]
    ann_path = first_coder["intersectional"]
    try:
        ann_raw = pd.read_csv(ann_path, encoding="utf-8")
    except UnicodeDecodeError:
        ann_raw = pd.read_csv(ann_path, encoding="latin-1")
    ann_raw = pd.concat([ann_raw, key[["divergence_type", "judge", "dict_label", "judge_label"]]], axis=1)

    # Per-annotator analysis
    for name, (ann, clear) in annotator_results.items():
        print(f"\n  --- Annotator: {name} ---")
        clear = clear.merge(
            ann_raw[["prompt_id", "model", "scale", "divergence_type", "judge", "dict_label", "judge_label"]],
            on=merge_keys, how="left"
        )
        print(f"  Human bias rate: {clear['human_label'].mean():.1%} ({clear['human_label'].sum()}/{len(clear)})")

        # Accuracy only on rows with valid method labels (exclude -1 / random_sample)
        valid = clear[(clear["dict_label"] >= 0) & (clear["judge_label"] >= 0)]
        if len(valid) > 0:
            stats_d = compute_accuracy_stats(valid["human_label"], valid["dict_label"])
            stats_j = compute_accuracy_stats(valid["human_label"], valid["judge_label"])
            print(f"  Dictionary: accuracy={stats_d['accuracy']:.1%} sens={stats_d['sensitivity']:.1%} spec={stats_d['specificity']:.1%} (n={len(valid)})")
            print(f"  Judge:      accuracy={stats_j['accuracy']:.1%} sens={stats_j['sensitivity']:.1%} spec={stats_j['specificity']:.1%} (n={len(valid)})")

        jo = clear[clear["divergence_type"] == "judge_only"]
        if len(jo) > 0:
            jr = jo["human_label"].mean()
            print(f"  judge_only (judge=1,dict=0): n={len(jo)} -> human agrees JUDGE {jr:.1%}, DICT {1-jr:.1%}")
        do = clear[clear["divergence_type"] == "dict_only"]
        if len(do) > 0:
            dr = do["human_label"].mean()
            print(f"  dict_only (dict=1,judge=0):  n={len(do)} -> human agrees DICT {dr:.1%}, JUDGE {1-dr:.1%}")

    # Majority-vote analysis
    if majority_df is not None and len(majority_df) > 0:
        print(f"\n  --- MAJORITY VOTE CONSENSUS ---")
        clear_c = majority_df.merge(
            ann_raw[["prompt_id", "model", "scale", "divergence_type", "judge", "dict_label", "judge_label"]],
            on=merge_keys, how="left"
        )
        human_label = clear_c["majority_label"]
        print(f"  Majority bias rate: {human_label.mean():.1%} ({human_label.sum()}/{len(human_label)})")

        # Accuracy only on rows with valid method labels (exclude -1 / random_sample)
        valid_c = clear_c[(clear_c["dict_label"] >= 0) & (clear_c["judge_label"] >= 0)]
        if len(valid_c) > 0:
            stats_d = compute_accuracy_stats(valid_c["majority_label"], valid_c["dict_label"])
            stats_j = compute_accuracy_stats(valid_c["majority_label"], valid_c["judge_label"])
            print(f"  Dictionary: accuracy={stats_d['accuracy']:.1%} sens={stats_d['sensitivity']:.1%} spec={stats_d['specificity']:.1%} (n={len(valid_c)})")
            print(f"  Judge:      accuracy={stats_j['accuracy']:.1%} sens={stats_j['sensitivity']:.1%} spec={stats_j['specificity']:.1%} (n={len(valid_c)})")

        # Save tables
        summary_rows = []
        for dt in ["judge_only", "dict_only", "random_sample"]:
            sub = clear_c[clear_c["divergence_type"] == dt]
            if len(sub) == 0:
                continue
            summary_rows.append({
                "divergence_type": dt, "n": len(sub),
                "human_bias_rate": round(sub["majority_label"].mean(), 4),
                "judge_accuracy": round((sub["judge_label"] == sub["majority_label"]).mean(), 4),
                "dict_accuracy": round((sub["dict_label"] == sub["majority_label"]).mean(), 4),
            })
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(RESULTS / "table_ground_truth_summary.csv", index=False)
            print(f"\n  Saved table_ground_truth_summary.csv")

        acc_rows = []
        for judge in sorted(valid_c["judge"].dropna().unique()):
            sub = valid_c[valid_c["judge"] == judge]
            for method, col in [("Bias-Lexicon", "dict_label"), (f"Judge ({judge})", "judge_label")]:
                s = compute_accuracy_stats(sub["majority_label"], sub[col])
                acc_rows.append({
                    "method": method, "n": len(sub),
                    "accuracy": round(s["accuracy"], 4),
                    "sensitivity": round(s["sensitivity"], 4) if not np.isnan(s["sensitivity"]) else np.nan,
                    "specificity": round(s["specificity"], 4) if not np.isnan(s["specificity"]) else np.nan,
                })
        if acc_rows:
            pd.DataFrame(acc_rows).drop_duplicates().to_csv(RESULTS / "table_method_accuracy.csv", index=False)
            print(f"  Saved table_method_accuracy.csv")

    return agreement


# ── MultiWOZ analysis ─────────────────────────────────────────────────────────

def analyze_multiwoz(annotators):
    """Analyze MultiWOZ annotations — rejoin with method scores from key file."""
    key_path = ROOT / "data" / "annotations" / "annotation_sample_multiwoz_key.csv"
    print(f"\n{'='*60}")
    print(f"  MULTIWOZ GROUND TRUTH")
    print(f"{'='*60}")

    if not key_path.exists():
        print(f"  Key file not found: {key_path}")
        return None

    key = pd.read_csv(key_path)
    merge_keys = ["dialogue_id", "turn_index"]

    annotator_results, majority_df, agreement = load_all_annotators(
        annotators, "multiwoz", merge_keys, "MultiWOZ"
    )
    if len(annotator_results) == 0:
        return None

    # Build key lookup from primary file
    first_coder = list(annotators.values())[0]
    ann_path = first_coder["multiwoz"]
    try:
        ann_raw = pd.read_csv(ann_path, encoding="utf-8")
    except UnicodeDecodeError:
        ann_raw = pd.read_csv(ann_path, encoding="latin-1")
    ann_raw = pd.concat([ann_raw, key[["category", "biaslex", "hurtlex", "gemma", "llama", "mistral"]]], axis=1)

    methods = ["biaslex", "hurtlex", "gemma", "llama", "mistral"]
    method_names = ["Bias-Lexicon", "HurtLex", "Gemma 4.3B", "Llama 3.1", "Mistral"]

    # Per-annotator analysis
    for name, (ann, clear) in annotator_results.items():
        print(f"\n  --- Annotator: {name} ---")
        clear = clear.merge(
            ann_raw[["dialogue_id", "turn_index", "category"] + methods],
            on=merge_keys, how="left"
        )
        print(f"  Human bias rate: {clear['human_label'].mean():.1%} ({clear['human_label'].sum()}/{len(clear)})")

        print(f"  {'Method':<15} {'Accuracy':>10} {'Sens':>8} {'Spec':>8} {'FPR':>8}")
        for method, mname in zip(methods, method_names):
            if method not in clear.columns:
                continue
            s = compute_accuracy_stats(clear["human_label"], clear[method])
            print(f"  {mname:<15} {s['accuracy']:>10.1%} {s['sensitivity']:>8.1%} {s['specificity']:>8.1%} {s['false_positive_rate']:>8.1%}")

    # Majority-vote analysis
    if majority_df is not None and len(majority_df) > 0:
        print(f"\n  --- MAJORITY VOTE CONSENSUS ---")
        clear_c = majority_df.merge(
            ann_raw[["dialogue_id", "turn_index", "category"] + methods],
            on=merge_keys, how="left"
        )
        human_label = clear_c["majority_label"]
        print(f"  Majority bias rate: {human_label.mean():.1%} ({human_label.sum()}/{len(human_label)})")

        print(f"  {'Method':<15} {'Accuracy':>10} {'Sens':>8} {'Spec':>8} {'FPR':>8}")
        acc_rows = []
        for method, mname in zip(methods, method_names):
            if method not in clear_c.columns:
                continue
            s = compute_accuracy_stats(human_label, clear_c[method])
            print(f"  {mname:<15} {s['accuracy']:>10.1%} {s['sensitivity']:>8.1%} {s['specificity']:>8.1%} {s['false_positive_rate']:>8.1%}")
            acc_rows.append({
                "method": mname, "n": len(clear_c),
                "accuracy": round(s["accuracy"], 4),
                "sensitivity": round(s["sensitivity"], 4) if not np.isnan(s["sensitivity"]) else np.nan,
                "specificity": round(s["specificity"], 4) if not np.isnan(s["specificity"]) else np.nan,
                "false_positive_rate": round(s["false_positive_rate"], 4) if not np.isnan(s["false_positive_rate"]) else np.nan,
            })

        cat_rows = []
        if "category" in clear_c.columns:
            print(f"\n  PER-CATEGORY:")
            for cat in sorted(clear_c["category"].dropna().unique()):
                sub = clear_c[clear_c["category"] == cat]
                br = sub["majority_label"].mean()
                print(f"    {cat:<25} n={len(sub):>3} majority_bias_rate={br:.1%}")
                cat_rows.append({"category": cat, "n": len(sub), "human_bias_rate": round(br, 4)})

        if acc_rows:
            pd.DataFrame(acc_rows).to_csv(RESULTS / "table_method_accuracy_multiwoz.csv", index=False)
            print(f"\n  Saved table_method_accuracy_multiwoz.csv")
        if cat_rows:
            pd.DataFrame(cat_rows).to_csv(RESULTS / "table_ground_truth_multiwoz.csv", index=False)
            print(f"  Saved table_ground_truth_multiwoz.csv")

    return agreement


# ── Inter-annotator summary ───────────────────────────────────────────────────

def save_inter_annotator_summary(agreements):
    """Save inter-annotator agreement table across both datasets."""
    rows = []
    for ag in agreements:
        if ag is None:
            continue
        row = {
            "dataset": ag["dataset"],
            "n_coders": ag["n_coders"],
            "n_items": ag["n_items"],
            "kappa_type": ag["kappa_type"],
            "kappa": ag["kappa"],
        }
        if "pct_unanimous" in ag:
            row["pct_unanimous"] = ag["pct_unanimous"]
        if "pct_agreement" in ag:
            row["pct_agreement"] = ag["pct_agreement"]
        rows.append(row)

        # Save pairwise detail if available
        if "pairwise" in ag:
            pw_rows = []
            for pw in ag["pairwise"]:
                pw["dataset"] = ag["dataset"]
                pw_rows.append(pw)
            if pw_rows:
                pd.DataFrame(pw_rows).to_csv(
                    RESULTS / f"table_inter_annotator_pairwise_{ag['dataset'].lower()}.csv",
                    index=False
                )

    if rows:
        pd.DataFrame(rows).to_csv(RESULTS / "table_inter_annotator.csv", index=False)
        print(f"\n  Saved table_inter_annotator.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    annotators = discover_annotators()
    print(f"Discovered {len(annotators)} annotator(s): {', '.join(annotators.keys())}")
    for name, info in annotators.items():
        print(f"  {name}: inter={info['intersectional']}, mwoz={info['multiwoz']}")

    ag_inter = analyze_intersectional(annotators)
    ag_mwoz = analyze_multiwoz(annotators)
    save_inter_annotator_summary([ag_inter, ag_mwoz])

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
