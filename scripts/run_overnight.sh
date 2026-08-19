#!/bin/bash
# Overnight pipeline: MultiWOZ subsample judges + v2 ablation on intersectional dataset
# Launched via: nohup bash run_overnight.sh &
# Expected runtime: ~18 hours (3hr MultiWOZ + 15hr v2 ablation)

PROJECT="$(cd "$(dirname "$0")" && pwd)"
LOG="$PROJECT/pipeline_v2_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOG" 2>&1
OVERALL_START=$(date +%s)

echo "=================================================================="
echo "  PIPELINE V2 — $(date)"
echo "  Log: $LOG"
echo "=================================================================="

# ── Pre-registration ──────────────────────────────────────
echo ""
echo "[Pre-registration] Timestamped: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "H1 (model identity ablation): If self-evaluation gap (Llama 35% on own"
echo "outputs vs 64% on others) persists in v2 with similar magnitude, it is an"
echo "intrinsic property of the model. If the gap disappears or attenuates"
echo "substantially (>15pp reduction), it was prompt-induced."
echo ""
echo "H2 (reproducibility): Running the modified script with --prompt-version v1"
echo "on existing prompts should produce bias_label and severity matching existing"
echo "v1 outputs."
echo ""
echo "Note: MultiWOZ prompt has NO model identity disclosure. MultiWOZ runs are"
echo "a SUBSAMPLE evaluation with the existing prompt, NOT a v2 ablation."
echo ""

# ── Phase 0: Record v1 file hashes ───────────────────────
echo "=================================================================="
echo "  PHASE 0: v1 FILE INTEGRITY BASELINE"
echo "=================================================================="
V1_HASH_FILE="/tmp/v1_hashes_$(date +%s).txt"

V1_FILES=(
    "$PROJECT/scores/llm_judge_scores.csv"
    "$PROJECT/scores/llm_judge_sentence_scores.csv"
    "$PROJECT/scores/llm_judge_scores_1b.csv"
    "$PROJECT/scores/llm_judge_sentence_scores_1b.csv"
    "$PROJECT/scores/llm_judge_scores_llama.csv"
    "$PROJECT/scores/llm_judge_sentence_scores_llama.csv"
    "$PROJECT/scores/llm_judge_scores_mistral.csv"
    "$PROJECT/scores/llm_judge_sentence_scores_mistral.csv"
    "$PROJECT/multiwoz/multiwoz_judge_scores.csv"
    "$PROJECT/multiwoz/multiwoz_dict_scores.csv"
    "$PROJECT/multiwoz/multiwoz_sample.csv"
    "$PROJECT/data/all_model_responses_clean.csv"
    "$PROJECT/data/all_model_responses.csv"
    "$PROJECT/data/prompts_dataset (1).csv"
)

> "$V1_HASH_FILE"
for f in "${V1_FILES[@]}"; do
    if [ -f "$f" ]; then
        hash=$(md5 -q "$f")
        echo "$hash  $f" >> "$V1_HASH_FILE"
        echo "  $hash  $(basename "$f")"
    else
        echo "  MISSING: $f"
    fi
done
echo "Hashes saved to $V1_HASH_FILE"

# ── Phase 2: MultiWOZ subsample (3 judges) ───────────────
echo ""
echo "=================================================================="
echo "  PHASE 2: MULTIWOZ SUBSAMPLE (500 dialogues, 3 judges)"
echo "=================================================================="

PHASE2_STATUS="PASS"
MULTIWOZ_SAMPLE="$PROJECT/multiwoz/multiwoz_sample_500.csv"
TURNS=$(wc -l < "$MULTIWOZ_SAMPLE")
TURNS=$((TURNS - 1))  # subtract header
echo "Sample: $MULTIWOZ_SAMPLE ($TURNS turns)"

for JUDGE in gemma3 llama3.1 mistral; do
    JUDGE_SAFE=$(echo "$JUDGE" | tr '.' '_')
    OUT="$PROJECT/multiwoz/multiwoz_judge_scores_500_${JUDGE_SAFE}.csv"
    echo ""
    echo "--- Judge: $JUDGE ---"
    echo "  Output: $OUT"

    START_T=$(date +%s)

    cd "$PROJECT/src/multiwoz"
    python3 multiwoz_evaluate.py \
        --judge-model "$JUDGE" \
        --sample-file "$MULTIWOZ_SAMPLE" \
        --judge-output "$OUT" \
        --skip-dict
    EXIT_CODE=$?
    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))
    ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  FAILED (exit code $EXIT_CODE) after $ELAPSED_FMT"
        PHASE2_STATUS="PARTIAL_FAIL"
    else
        ROWS=$(wc -l < "$OUT" 2>/dev/null || echo "0")
        ROWS=$((ROWS - 1))
        if [ $TURNS -gt 0 ] && [ $ELAPSED -gt 0 ]; then
            RATE=$(python3 -c "print(f'{$TURNS/$ELAPSED:.2f}')")
            # Extrapolate to 113552 turns
            FULL_SEC=$(python3 -c "print(int(113552 / ($TURNS/$ELAPSED)))")
            FULL_FMT=$(printf '%02d:%02d:%02d' $((FULL_SEC/3600)) $(((FULL_SEC%3600)/60)) $((FULL_SEC%60)))
        else
            RATE="N/A"
            FULL_FMT="N/A"
        fi
        echo "  DONE in $ELAPSED_FMT ($ROWS rows scored, $RATE turns/sec)"
        echo "  Extrapolated full-dataset: $FULL_FMT"

        # Write timing to summary
        echo "$JUDGE: $ELAPSED_FMT ($RATE turns/sec) [extrapolated full: $FULL_FMT]" >> "$PROJECT/multiwoz/judge_timing_summary.txt"
    fi
done

echo ""
echo "Phase 2 status: $PHASE2_STATUS"

# ── Phase 3: v2 ablation on intersectional dataset ───────
echo ""
echo "=================================================================="
echo "  PHASE 3: v2 ABLATION (intersectional dataset, 3 judges)"
echo "=================================================================="

PHASE3_STATUS="PASS"

for JUDGE in gemma3 llama3.1 mistral; do
    JUDGE_SAFE=$(echo "$JUDGE" | tr '.' '_')
    OUT="$PROJECT/scores/llm_judge_scores_${JUDGE_SAFE}_v2.csv"
    SENT_OUT="$PROJECT/scores/llm_judge_sentence_scores_${JUDGE_SAFE}_v2.csv"

    # Map to existing v1 filenames for the default gemma3 case
    if [ "$JUDGE" = "gemma3" ]; then
        OUT="$PROJECT/scores/llm_judge_scores_v2.csv"
        SENT_OUT="$PROJECT/scores/llm_judge_sentence_scores_v2.csv"
    fi

    echo ""
    echo "--- Judge: $JUDGE (v2 prompt) ---"
    echo "  Output: $OUT"
    echo "  Sentence: $SENT_OUT"

    START_T=$(date +%s)

    cd "$PROJECT/src/llm_judge"
    python3 llm_judge.py \
        --prompt-version v2 \
        --judge-model "$JUDGE" \
        --output "$OUT" \
        --sentence-output "$SENT_OUT"
    EXIT_CODE=$?
    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))
    ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  FAILED (exit code $EXIT_CODE) after $ELAPSED_FMT"
        PHASE3_STATUS="PARTIAL_FAIL"
    else
        ROWS=$(wc -l < "$OUT" 2>/dev/null || echo "0")
        ROWS=$((ROWS - 1))
        echo "  DONE in $ELAPSED_FMT ($ROWS utterance rows)"
    fi
done

echo ""
echo "Phase 3 status: $PHASE3_STATUS"

# ── Phase 3.2: Quick v1 vs v2 comparison ─────────────────
echo ""
echo "=================================================================="
echo "  PHASE 3.2: v1 vs v2 HEADLINE COMPARISON"
echo "=================================================================="

cd "$PROJECT"
python3 << 'PYEOF'
import pandas as pd
import os

mk = ["prompt_id", "model", "scale"]
clean = pd.read_csv("data/all_model_responses_clean.csv")
non_ref = clean[(clean["step"]=="explanation") & (~clean["is_refusal"])][mk]

def bias_rate(path):
    if not os.path.exists(path):
        return None, 0
    df = pd.read_csv(path)
    if "step" in df.columns:
        df = df[df["step"]=="explanation"]
    df = df.merge(non_ref, on=mk, how="inner")
    df = df[df["bias_label"] != -1]
    return df["bias_label"].mean(), len(df)

def self_eval(path, judge_ollama_name):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "step" in df.columns:
        df = df[df["step"]=="explanation"]
    df = df.merge(non_ref, on=mk, how="inner")
    df = df[df["bias_label"] != -1]
    result = {}
    for model in sorted(df["model"].unique()):
        sub = df[df["model"]==model]
        result[model] = sub["bias_label"].mean()
    return result

print(f"{'Judge':<12} {'v1 rate':>8} {'v1 n':>5} {'v2 rate':>8} {'v2 n':>5} {'delta':>7}")
print("-" * 50)

pairs = [
    ("Gemma",   "scores/llm_judge_scores.csv",         "scores/llm_judge_scores_v2.csv",         "gemma3"),
    ("Llama",   "scores/llm_judge_scores_llama.csv",    "scores/llm_judge_scores_llama3_1_v2.csv","llama3.1"),
    ("Mistral", "scores/llm_judge_scores_mistral.csv",  "scores/llm_judge_scores_mistral_v2.csv", "mistral"),
]

for name, v1_path, v2_path, ollama_name in pairs:
    r1, n1 = bias_rate(v1_path)
    r2, n2 = bias_rate(v2_path)
    if r1 is not None and r2 is not None:
        delta = r2 - r1
        print(f"{name:<12} {r1:>8.1%} {n1:>5} {r2:>8.1%} {n2:>5} {delta:>+7.1%}")
    elif r1 is not None:
        print(f"{name:<12} {r1:>8.1%} {n1:>5} {'N/A':>8} {'':>5}")
    else:
        print(f"{name:<12} {'N/A':>8} {'':>5} {'N/A':>8} {'':>5}")

# Self-evaluation analysis for Llama (key hypothesis)
print("\n--- Llama 3.1 Self-Evaluation (H1 test) ---")
for version, path in [("v1", "scores/llm_judge_scores_llama.csv"),
                       ("v2", "scores/llm_judge_scores_llama3_1_v2.csv")]:
    rates = self_eval(path, "llama3.1")
    if rates:
        own = rates.get("llama3.1", None)
        others = [v for k,v in rates.items() if k != "llama3.1"]
        avg_others = sum(others)/len(others) if others else None
        if own is not None and avg_others is not None:
            gap = own - avg_others
            print(f"  {version}: own={own:.1%}, others_avg={avg_others:.1%}, gap={gap:+.1%}")
        else:
            print(f"  {version}: {rates}")
    else:
        print(f"  {version}: no data")

PYEOF

# ── Phase 4: Final verification ──────────────────────────
echo ""
echo "=================================================================="
echo "  PHASE 4: FINAL VERIFICATION"
echo "=================================================================="

# 4.1 v1 file integrity check
echo ""
echo "--- 4.1: v1 file integrity ---"
INTEGRITY_OK=true
while IFS= read -r line; do
    expected_hash=$(echo "$line" | awk '{print $1}')
    filepath=$(echo "$line" | cut -d' ' -f3-)
    if [ -f "$filepath" ]; then
        actual_hash=$(md5 -q "$filepath")
        if [ "$expected_hash" != "$actual_hash" ]; then
            echo "  CRITICAL: HASH MISMATCH for $(basename "$filepath")"
            echo "    Expected: $expected_hash"
            echo "    Actual:   $actual_hash"
            INTEGRITY_OK=false
        fi
    fi
done < "$V1_HASH_FILE"

if $INTEGRITY_OK; then
    echo "  All v1 files intact. PASS."
else
    echo "  CRITICAL ERROR: v1 files were modified!"
fi

# 4.2 New file inventory
echo ""
echo "--- 4.2: New file inventory ---"
NEW_FILES=(
    "$PROJECT/scores/llm_judge_scores_v2.csv"
    "$PROJECT/scores/llm_judge_sentence_scores_v2.csv"
    "$PROJECT/scores/llm_judge_scores_llama3_1_v2.csv"
    "$PROJECT/scores/llm_judge_sentence_scores_llama3_1_v2.csv"
    "$PROJECT/scores/llm_judge_scores_mistral_v2.csv"
    "$PROJECT/scores/llm_judge_sentence_scores_mistral_v2.csv"
    "$PROJECT/multiwoz/multiwoz_sample_500.csv"
    "$PROJECT/multiwoz/multiwoz_judge_scores_500_gemma3.csv"
    "$PROJECT/multiwoz/multiwoz_judge_scores_500_llama3_1.csv"
    "$PROJECT/multiwoz/multiwoz_judge_scores_500_mistral.csv"
    "$PROJECT/multiwoz/judge_timing_summary.txt"
)

for f in "${NEW_FILES[@]}"; do
    if [ -f "$f" ]; then
        ROWS=$(wc -l < "$f")
        echo "  $(basename "$f"): $ROWS lines"
    else
        echo "  $(basename "$f"): NOT CREATED"
    fi
done

# 4.3 Summary
OVERALL_END=$(date +%s)
TOTAL_SEC=$((OVERALL_END - OVERALL_START))
TOTAL_FMT=$(printf '%02d:%02d:%02d' $((TOTAL_SEC/3600)) $(((TOTAL_SEC%3600)/60)) $((TOTAL_SEC%60)))

echo ""
echo "=================================================================="
echo "  SUMMARY"
echo "=================================================================="
echo "  Total wall-clock time: $TOTAL_FMT"
echo "  Phase 2 (MultiWOZ subsample): $PHASE2_STATUS"
echo "  Phase 3 (v2 ablation):        $PHASE3_STATUS"
echo "  v1 file integrity:            $(if $INTEGRITY_OK; then echo PASS; else echo FAIL; fi)"
echo "  Log file: $LOG"
echo ""
echo "=== DONE at $(date) ==="
