#!/bin/bash
#SBATCH -A ACCOUNT-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH --job-name=op_dict
#SBATCH --output=logs/dictionary_%j.out
#SBATCH --error=logs/dictionary_%j.err
# Step 4: Dictionary methods (fast, single job).

set -e
module load python/3.8

source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate
cd /home/$USER/rds/hpc-work/hpc_run_extension

echo "[$(date)] Step 4: Running dictionary methods..."
python scripts/run_dictionary.py

echo "[$(date)] Step 4 complete"
