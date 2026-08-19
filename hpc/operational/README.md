# Operational Bias Extension — HPC Run

## Quick start on CSD3

```bash
# 1. Upload from local machine
scp -r hpc_run_extension/ <crsid>@login.hpc.cam.ac.uk:/rds/user/<crsid>/hpc-work/

# 2. SSH in and install deps (if not already in multiwoz_env)
ssh <crsid>@login.hpc.cam.ac.uk
source /home/$USER/rds/hpc-work/multiwoz_env/bin/activate
pip install -r /home/$USER/rds/hpc-work/hpc_run_extension/requirements.txt

# 3. Run pipeline in order
cd /home/$USER/rds/hpc-work/hpc_run_extension

# Step 1-3: Generate prompts, responses, refusals (needs GPU for Ollama)
JOB1=$(sbatch --parsable run_step1_generate.sh)

# Step 4: Dictionary methods (after step 3 completes)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 run_step2_dictionary.sh)

# Step 5: LLM judges (after step 3 completes, overnight)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 run_step3_judges.sh)

# Step 6-8: Statistics + figures (after steps 4 and 5 complete)
sbatch --dependency=afterok:$JOB2:$JOB3 run_step4_analysis.sh
```

## Pulling results back
```bash
rsync -avzP <crsid>@login.hpc.cam.ac.uk:/rds/user/<crsid>/hpc-work/hpc_run_extension/{statistics,figures,evaluations,responses} ./
```

## Execution order
1. `run_step1_generate.sh` — prompts + 360 responses + refusal detection (~3-4 hrs)
2. `run_step2_dictionary.sh` — Bias-Lexicon, HurtLex, toxic-bert (~30 min)
3. `run_step3_judges.sh` — 4 judges × 2 versions, array job (~6-8 hrs)
4. `run_step4_analysis.sh` — all statistics, cross-content, figures (~1 hr)

## Sanity checks
After step 3 completes, verify:
- 120 prompts generated (60 base × 2 directions)
- ~360 responses (120 × 3 models, minus any generation failures)
- Refusal rate in sensible range (intersectional was 18.6%)
- Llama-on-Llama cell has ≥30 non-refusal responses
- All 8 judge output files present (4 judges × v1/v2)
