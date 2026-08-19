#!/bin/bash
#SBATCH -A ACCOUNT-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH --job-name=op_judge
#SBATCH --output=logs/judge_%A_%a.out
#SBATCH --error=logs/judge_%A_%a.err
#SBATCH --array=0-7%4
# Step 5: LLM judges. 4 judges × 2 versions = 8 tasks.
# %4 limits concurrent GPU tasks.

set -e
module load python/3.8

source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate
cd /home/$USER/rds/hpc-work/hpc_run_extension

mkdir -p logs evaluations

# Judge/version config
JUDGES=("gemma3:latest" "gemma3:1b" "llama3.1:latest" "mistral:latest")
VERSIONS=("v1" "v2")

JUDGE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
VERSION_IDX=$((SLURM_ARRAY_TASK_ID % 2))

JUDGE="${JUDGES[$JUDGE_IDX]}"
VERSION="${VERSIONS[$VERSION_IDX]}"

JUDGE_SAFE=$(echo "$JUDGE" | tr ':/' '__')
OUTPUT_FILE="evaluations/judge_operational_${JUDGE_SAFE}_${VERSION}.csv"
SENT_OUTPUT_FILE="evaluations/judge_operational_${JUDGE_SAFE}_${VERSION}_sentences.csv"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: judge=$JUDGE version=$VERSION"

# Start Ollama on a per-task port
OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODELS=/home/$USER/rds/hpc-work/ollama_models
export PATH="$HOME/bin:$PATH"

ollama serve > /tmp/ollama_judge_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 15

# Verify model is available
ollama list | grep -q "${JUDGE%%:*}" || { echo "Model $JUDGE not found"; kill $OLLAMA_PID; exit 1; }

python scripts/run_judges.py \
  --judge-model "$JUDGE" \
  --prompt-version "$VERSION" \
  --ollama-host "http://${OLLAMA_HOST}/api/generate" \
  --output "$OUTPUT_FILE" \
  --sentence-output "$SENT_OUTPUT_FILE"

kill $OLLAMA_PID 2>/dev/null || true
echo "[$(date)] Task $SLURM_ARRAY_TASK_ID complete"
