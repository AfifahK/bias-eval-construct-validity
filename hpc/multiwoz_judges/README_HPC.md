# HPC Setup: Full MultiWOZ Judge Run

Runs all 113,552 MultiWOZ turns through 4 LLM judges (Gemma 3 4.3B, Gemma 3 1B, Llama 3.1 8B, Mistral 7B) via SLURM array jobs on Cambridge HPC.

## Account

- **SLURM account**: ACCOUNT-SL3-GPU
- **Partition**: ampere (GPU nodes)

## Login

```bash
ssh <crsid>@login-cpu.hpc.cam.ac.uk
# Raven password + 2FA via authenticator app
```

## One-time setup

```bash
cd rds/hpc-work
bash hpc_run/setup_hpc_env.sh   # ~20 min for model downloads
```

This creates the Python venv, installs deps, and pulls all 4 judge models into `ollama_models/`.

## Before submitting

1. Generate chunks locally: `python chunk_multiwoz.py`
2. Generate task config: `python generate_task_config.py`
3. Edit `run_judge_array.sh`: replace PLACEHOLDER with (total_tasks - 1)
4. rsync everything to HPC (see below)

## Smoke test

```bash
sbatch --array=0-1%2 hpc_run/run_judge_array.sh
# Wait ~10 min, then check:
cat hpc_run/logs/judge_*.out
cat hpc_run/logs/judge_*.err
```

## Full submission

```bash
sbatch hpc_run/run_judge_array.sh
```

## Monitor

```bash
squeue -u $USER                    # running jobs
squeue -u $USER -t all -r          # individual array tasks
sacct -j <jobid>                   # post-hoc accounting
```

## Cancel

```bash
scancel <jobid>                    # whole array
scancel <jobid>_[10-20]            # specific task range
```

## Find failures

```bash
grep -l "Error\|Traceback" hpc_run/logs/judge_*.err
# Re-run only failed tasks:
sbatch --array=<failed_ids> hpc_run/run_judge_array.sh
```

## Download results

From local machine:
```bash
rsync -avz --progress \
  <crsid>@login-cpu.hpc.cam.ac.uk:/home/<crsid>/rds/hpc-work/hpc_run/results/ \
  ./hpc_results/
```

## File structure

```
hpc_run/
├── src/multiwoz_evaluate_hpc.py   # Chunk-aware, resumable judge script
├── chunks/                         # 10k-turn CSV chunks (generated locally)
├── chunk_multiwoz.py               # Splits full MultiWOZ into chunks
├── generate_task_config.py         # Creates task_config.csv for SLURM array
├── task_config.csv                 # Maps task_id → (judge, chunk)
├── run_judge_array.sh              # SLURM array job script
├── setup_hpc_env.sh                # One-time env + model setup
├── requirements.txt
├── results/                        # Output CSVs (one per judge×chunk)
└── logs/                           # SLURM stdout/stderr logs
```

## Notes

- Each array task starts its own Ollama instance on a unique port to avoid collisions
- Script is resumable: if a task is restarted, it skips already-completed turn_ids
- Results are flushed every 100 turns so partial progress is preserved on timeout
- Models are pre-pulled in setup; array jobs do NOT pull (avoids network issues on compute nodes)
- v1 prompt only (no v2 ablation for MultiWOZ since there is no model-identity field)
