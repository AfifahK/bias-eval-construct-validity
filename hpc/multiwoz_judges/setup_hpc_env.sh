#!/bin/bash
# Run this ONCE on the HPC login node before submitting any array jobs.
# Assumes:
#   - ~/bin/ollama binary already installed
#   - ~/rds/hpc-work/ollama_models/ already populated with model files
#   - Python venv may or may not exist yet

set -e
cd /home/$USER/rds/hpc-work

# 1. Create/update the venv
module load python/3.8
if [ ! -d "multiwoz_env" ]; then
    python -m venv multiwoz_env
    echo "Created new venv"
fi
source multiwoz_env/bin/activate
pip install --upgrade pip -q
pip install -r hpc_run/requirements.txt -q
pip install nltk -q
echo "Venv ready"

# 2. Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
echo "NLTK data ready"

# 3. Verify ollama binary
export PATH="$HOME/bin:$PATH"
if command -v ollama &>/dev/null; then
    echo "Ollama binary found: $(which ollama)"
else
    echo "WARNING: ollama not found in PATH. Install to ~/bin/ollama"
fi

# 4. Verify models
export OLLAMA_MODELS=/home/$USER/rds/hpc-work/ollama_models
if [ -d "$OLLAMA_MODELS/manifests" ]; then
    echo "Models directory found:"
    ls $OLLAMA_MODELS/manifests/registry.ollama.ai/library/ 2>/dev/null || echo "  (no manifests)"
else
    echo "WARNING: Models directory not found at $OLLAMA_MODELS"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Test ollama on a GPU node (srun)"
echo "  2. Generate chunks: python hpc_run/chunk_multiwoz.py && python hpc_run/chunk_multiwoz_sentences.py"
echo "  3. Generate task config: python hpc_run/generate_task_config.py"
echo "  4. Update PLACEHOLDER in run_judge_array.sh"
echo "  5. Submit: sbatch hpc_run/run_judge_array.sh"
