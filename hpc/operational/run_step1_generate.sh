#!/bin/bash
#SBATCH -A ACCOUNT-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --job-name=op_generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
# Steps 1-3: Generate prompts, responses, detect refusals.

set -e
module load python/3.8

source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate
cd /home/$USER/rds/hpc-work/hpc_run_extension

mkdir -p logs prompts responses evaluations statistics/operational statistics/cross_content_three_way figures

# Start Ollama
export OLLAMA_MODELS=/home/$USER/rds/hpc-work/ollama_models
export PATH="$HOME/bin:$PATH"
ollama serve > /tmp/ollama_generate_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 15

echo "[$(date)] Step 1: Generating prompts..."
python scripts/generate_prompts.py

echo "[$(date)] Step 2: Generating responses..."
python scripts/generate_responses.py --ollama-host "http://127.0.0.1:11434/api/generate"

echo "[$(date)] Step 3: Detecting refusals..."
python scripts/detect_refusals.py

kill $OLLAMA_PID 2>/dev/null || true
echo "[$(date)] Steps 1-3 complete"
