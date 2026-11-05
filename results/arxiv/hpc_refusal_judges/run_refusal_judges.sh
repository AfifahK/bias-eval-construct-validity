#!/bin/bash
#SBATCH -A ACCOUNT-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH --job-name=refusal_judge
#SBATCH --output=logs/refusal_judge_%j.out
#SBATCH --error=logs/refusal_judge_%j.err
# 4 judges x 2 prompt versions (v1, then v2) x (158 refusals + 40 sanity rows) = 1,584 calls, sequential,
# one Ollama server. Expected ~3-5 h on a GPU (the v1-only job of 792 calls was budgeted at 1-2 h); 8 h
# leaves margin and the SL3 maximum is 12 h. v1 finishes before v2 starts.
# Resumable: just `sbatch` again if it is killed. Submit FROM this folder (logs/ must exist).

set -e
module load python/3.8
source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate

RUN_DIR=/home/$USER/rds/hpc-work/refusal_judge
cd "$RUN_DIR"
mkdir -p logs results

export OLLAMA_DIR=/home/$USER/rds/hpc-work/ollama_v0305
export OLLAMA_MODELS=/home/$USER/rds/hpc-work/ollama_models
export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 1000))"   # avoid clashes on shared nodes

# Same CUDA setup + GPU-verified serve as the original judge jobs
source "$RUN_DIR/setup_ollama_cuda.sh"
source "$RUN_DIR/ollama_serve_resilient.sh"
ollama_serve_resilient || { echo "no working Ollama/GPU config on $(hostname)"; exit 1; }
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# Verify all four exact tags are present
for T in gemma3:latest gemma3:1b llama3.1:latest mistral:latest; do
    "$OLLAMA_BIN" list | awk 'NR>1{print $1}' | grep -qx "$T" || { echo "MISSING MODEL TAG: $T"; exit 1; }
done

echo "[$(date)] start on $(hostname), host=$OLLAMA_HOST"
python run_refusal_judges.py --ollama-host "http://${OLLAMA_HOST}" --outdir "$RUN_DIR/results"
echo "[$(date)] complete"
