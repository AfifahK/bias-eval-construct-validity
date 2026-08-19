#!/bin/bash
# Run sentence-level MultiWOZ judges in parallel pairs.
# Night 1: Mistral + Gemma 1B
# Night 2: Llama 3.1 + Gemma 4.3B
#
# Usage:
#   bash run_sentence_judges.sh night1    # Mistral + Gemma 1B
#   bash run_sentence_judges.sh night2    # Llama 3.1 + Gemma 4.3B
#   bash run_sentence_judges.sh all       # All 4 sequential
#   bash run_sentence_judges.sh smoke     # 100-sentence smoke test with gemma3

set -e
cd "$(dirname "$0")"

case "${1:-smoke}" in
  smoke)
    echo "=== SMOKE TEST: 100 sentences with gemma3:latest ==="
    python3 multiwoz_sentence_judge.py --judge-model gemma3:latest --limit 100
    echo "=== Smoke test complete ==="
    ;;
  night1)
    echo "=== NIGHT 1: Mistral + Gemma 1B ==="
    python3 multiwoz_sentence_judge.py --judge-model mistral:latest &
    PID1=$!
    python3 multiwoz_sentence_judge.py --judge-model gemma3:1b &
    PID2=$!
    echo "Mistral PID=$PID1, Gemma1B PID=$PID2"
    wait $PID1; echo "Mistral done (exit $?)"
    wait $PID2; echo "Gemma 1B done (exit $?)"
    echo "=== Night 1 complete ==="
    ;;
  night2)
    echo "=== NIGHT 2: Llama 3.1 + Gemma 4.3B ==="
    python3 multiwoz_sentence_judge.py --judge-model llama3.1:latest &
    PID1=$!
    python3 multiwoz_sentence_judge.py --judge-model gemma3:latest &
    PID2=$!
    echo "Llama PID=$PID1, Gemma4.3B PID=$PID2"
    wait $PID1; echo "Llama done (exit $?)"
    wait $PID2; echo "Gemma 4.3B done (exit $?)"
    echo "=== Night 2 complete ==="
    ;;
  all)
    echo "=== ALL 4 JUDGES SEQUENTIAL ==="
    for model in mistral:latest gemma3:1b llama3.1:latest gemma3:latest; do
      echo "--- Starting $model ---"
      python3 multiwoz_sentence_judge.py --judge-model "$model"
    done
    echo "=== All complete ==="
    ;;
  *)
    echo "Usage: $0 {smoke|night1|night2|all}"
    exit 1
    ;;
esac
