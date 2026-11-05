# Refusal judging on CSD3 (SLURM)

4 judges (gemma3:latest, gemma3:1b, llama3.1:latest, mistral:latest) x [158 refusals + 40 sanity rows] x
prompt v1 (with the model-identity line) and v2 (identity line removed), response level only, T=0, seed 42.
1,584 calls (792 per version), one job, resumable; v1 runs first, then v2. `--versions v1` runs v1 only.

## Run (replace <crsid>; login needs MFA so run these yourself)

```bash
# 1. copy folder up (from the machine holding this repo)
scp -r results/arxiv/hpc_refusal_judges <crsid>@login.hpc.cam.ac.uk:/home/<crsid>/rds/hpc-work/refusal_judge

# 2. on the login node
ssh <crsid>@login.hpc.cam.ac.uk
cd /home/<crsid>/rds/hpc-work/refusal_judge
sed -i 's/ACCOUNT-SL3-GPU/<YOUR-ACCOUNT>/' run_refusal_judges.sh    # your real SLURM account
sbatch run_refusal_judges.sh                                        # prints "Submitted batch job <id>"

# 3. monitor
squeue -u <crsid>
tail -f logs/refusal_judge_<id>.out
ls results/        # 32 CSVs when done (16 per prompt version); the log ends with "ALL DONE" then "complete"

# 4. if killed/timed out: just resubmit, finished rows are skipped
sbatch run_refusal_judges.sh

# 5. copy results back (from the local machine)
scp -r <crsid>@login.hpc.cam.ac.uk:/home/<crsid>/rds/hpc-work/refusal_judge/results results/arxiv/hpc_refusal_judges/results_hpc
python3 results/arxiv/hpc_refusal_judges/compare_sanity.py results/arxiv/hpc_refusal_judges/results_hpc
```

## Outputs (`results/`, existing judge-score CSV formats)
`{refusal,sanity}_{declarative,operational}_<tag>_{v1,v2}.csv`. Declarative files use the
`llm_judge_scores*.csv` columns, operational files the `judge_operational_*.csv` columns.

## Contents
- `judge_input.csv`: slice (refusal/sanity), dataset, metadata, `judge_text` (declarative: raw `response`;
  operational: `response_clean`, as in the original runs).
- `sanity_expected.csv`: stored v1 and v2 labels/severities for the 40 sanity rows (20 declarative + 20 operational,
  `random_state=42`; no stored declarative Gemma 1B v2, those rows are skipped); `compare_sanity.py` reports agreement per version.
  `build_sanity_v2.py` added the v2 columns; `judge_input.csv` is unchanged (158 refusal rows, 4 of the declarative
  ones are the audit false positives and are used as scored responses downstream).
- `src/decl_llm_judge.py`, `src/op_run_judges.py`: unmodified copies of the repo's judge code.
- `setup_ollama_cuda.sh`, `ollama_serve_resilient.sh`: unmodified copies, sourced as in the earlier jobs.
- `build_inputs.py`: regenerates the two CSVs from the repo.

## Notes
- `--ollama-host` is now the base URL; `/api/generate` is appended once (the old step-3 script doubled it).
- Wall time requested is 8 h (SL3 maximum 12 h); expect ~3-5 h on a GPU for both versions (~1-2 h for v1 alone). Resubmit if it times out.
- Python 3.8 compatible; runner needs only `requests` and `nltk` (already in `multiwoz_env`).
