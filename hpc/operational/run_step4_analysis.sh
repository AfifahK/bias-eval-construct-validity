#!/bin/bash
#SBATCH -A ACCOUNT-SL3-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 02:00:00
#SBATCH --job-name=op_stats
#SBATCH --output=logs/stats_%j.out
#SBATCH --error=logs/stats_%j.err
# Steps 6-8: Statistics, cross-content comparison, figures.

set -e
module load python/3.8

source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate
cd /home/$USER/rds/hpc-work/hpc_run_extension

echo "[$(date)] Step 6: Computing statistics..."
python scripts/compute_statistics.py

echo "[$(date)] Step 7: Cross-content three-way comparison..."
python scripts/compute_cross_content.py

echo "[$(date)] Step 8: Generating figures..."
python scripts/generate_figures.py

echo "[$(date)] Steps 6-8 complete. All outputs in statistics/ and figures/"
