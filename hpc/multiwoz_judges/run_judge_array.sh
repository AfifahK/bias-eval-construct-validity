#!/bin/bash
#SBATCH -A ACCOUNT-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --job-name=mwoz_judge
#SBATCH --output=logs/judge_%A_%a.out
#SBATCH --error=logs/judge_%A_%a.err
#SBATCH --array=0-47%8
# 48 tasks: 4 judges x 12 chunks (full corpus turn-level).
# %8 limits concurrent tasks.

set -e
module load python/3.8

source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate
cd /home/$USER/rds/hpc-work/hpc_run

mkdir -p logs results

# Read this task's config row
CONFIG_LINE=$(awk -F',' -v task="$SLURM_ARRAY_TASK_ID" 'NR == task+2 {print}' task_config.csv)
JUDGE=$(echo "$CONFIG_LINE"   | cut -d',' -f2)
CHUNK=$(echo "$CONFIG_LINE"   | cut -d',' -f3)
LEVEL=$(echo "$CONFIG_LINE"   | cut -d',' -f4)

CHUNK_BASE="${CHUNK%.csv}"
JUDGE_SAFE=$(echo "$JUDGE" | tr ':/' '__')
OUTPUT_FILE="results/${JUDGE_SAFE}_${CHUNK_BASE}_${LEVEL}.csv"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: judge=$JUDGE chunk=$CHUNK level=$LEVEL"

# Start Ollama on a per-task port
OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID % 100))
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODELS=/home/$USER/rds/hpc-work/ollama_models
export PATH="$HOME/bin:$PATH"

ollama serve > /tmp/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 15

# Verify model is available
ollama list | grep -q "${JUDGE%%:*}" || { echo "Model $JUDGE not found"; kill $OLLAMA_PID; exit 1; }

# Run the appropriate script based on level
if [ "$LEVEL" = "turn" ]; then
    python src/multiwoz_evaluate_hpc.py \
      --input "chunks/$CHUNK" \
      --output "$OUTPUT_FILE" \
      --judge-model "$JUDGE" \
      --prompt-version v1 \
      --ollama-host "http://${OLLAMA_HOST}"
elif [ "$LEVEL" = "sentence" ]; then
    python src/multiwoz_sentence_judge_hpc.py \
      --input "sentence_chunks/$CHUNK" \
      --output "$OUTPUT_FILE" \
      --judge-model "$JUDGE" \
      --ollama-host "http://${OLLAMA_HOST}"
else
    echo "Unknown level: $LEVEL"
    kill $OLLAMA_PID
    exit 1
fi

kill $OLLAMA_PID 2>/dev/null || true
echo "[$(date)] Task $SLURM_ARRAY_TASK_ID complete"
