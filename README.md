# Bias by Whose Definition?

Comparing seven automated methods for detecting social bias in LLM-generated conversational outputs — three dictionary-based, four LLM-as-judge — evaluated against six-coder human ground truth across three datasets. MPhil dissertation, University of Cambridge, 2026.

The methods are not interchangeable, and the disagreement is not noise.

## What this found

Bias-flagging rates on 293 non-refusal responses to intersectional prompts (nationality × mental health status × scenario) span 0% to 100%. toxic-bert flags nothing. Gemma 3 (1B) flags everything. The full range:

| Method | Paradigm | Flagging rate |
|--------|----------|--------------|
| toxic-bert (classifier) | Dictionary | 0.0% |
| Mistral 7B | LLM-judge | 14.3% |
| Bias-Lexicon | Dictionary | 23.2% |
| Llama 3.1 | LLM-judge | 58.4% |
| HurtLex | Dictionary | 91.1% |
| Gemma 3 (4.3B) | LLM-judge | 95.2% |
| Gemma 3 (1B) | LLM-judge | 100.0% |

Cohen's kappa across all 21 pairwise combinations, with bootstrap 95% CIs (10,000 resamples) and Holm–Bonferroni correction: two pairs survive at α = 0.05, both LLM-judge × LLM-judge — Llama 3.1 × Mistral (κ = 0.213) and Gemma 4.3B × Llama 3.1 (κ = 0.131). No cross-paradigm pair achieves significant agreement after correction. Dictionary methods and LLM judges are not measuring the same thing.

**Self-evaluation is model-specific, not universal.** Llama 3.1 flags 63.9% of Gemma-generated and Mistral-generated outputs but only 34.6% of its own — a 29-point gap (Fisher's exact OR = 0.31, 95% bootstrap CI [0.20, 0.47], p < 0.001). On operational content the effect strengthens: 6.1% on own outputs versus ~38% on others (OR = 0.12, CI [0.03, 0.24]). Mistral does the opposite, flagging its own outputs *more* (OR = 2.69, CI [1.70, 4.46], p < 0.001). Gemma shows no effect (OR = 0.72, CI [0.34, 1.54]) — near-ceiling flagging leaves no room for one. The Llama result holds under a v2 prompt with the model-identity line removed, and a log-length covariate does not attenuate the interaction (β = −1.39 without length, −1.43 with).

**On naturalistic dialogue (MultiWOZ 2.2),** the methods converge on what bias *isn't* and diverge on what it *is*. Turn-level rates on a 500-dialogue subsample (6,648 turns): toxic-bert 0.03%, Mistral 0.03%, Bias-Lexicon 0.15%, Llama 3.1 6.0%, Gemma 3 (4.3B) 15.7%, HurtLex 41.95%, Gemma 3 (1B) 99.7%. On the full corpus (8,437 dialogues, 113,552 turns), rates span 0.08% to 99.4% — a factor of over 1,200. All judges land within 1 pp of their subsample estimates. A tool validated only on naturalistic corpora looks reliable because it agrees with other methods on the easy cases.

**Differential refusal is the finding no method catches.** Of 360 candidate declarative responses, 67 are refusals (18.6%). Llama 3.1 accounts for 97% of them. Within Llama's refusals, North Korean prompts are refused at 2.35× the American rate. This is allocative bias enacted through refusal — not through response content — and every method misses it, because every method scores response text, and a refused prompt produces none. The operational dataset replicates: 71 of 360 refusals (19.7%), all Llama.

The disagreement replicates on a third operational dataset (deployment-framed, action-oriented prompts) — no cross-paradigm pair survives there either, and the one surviving pair weakens to κ = 0.044. But the flagging rates and method ordering shift: Mistral drops from 14.3% to 0.0%, Bias-Lexicon doubles to 49.1%, and dictionary-vs-judge rankings invert. Method behavior is content-dependent.

## What's in here

- **`src/`** — Response generation, dictionary scoring (Bias-Lexicon, HurtLex), LLM-as-judge evaluation, toxicity/sentiment scoring, thematic analysis, agreement computation, figures and tables.
- **`data/`** — Prompt templates, model responses, annotation keys. Per-coder annotations not released (participant privacy); aggregate agreement statistics live in `results/`.
- **`scores/`** — Per-method outputs on the declarative dataset.
- **`multiwoz/`** — Judge and dictionary scores on MultiWOZ. Raw dialogues not included; obtain from the [upstream repository](https://github.com/budzianowski/multiwoz) (Apache 2.0).
- **`hpc/`** — HPC pipeline for full-corpus MultiWOZ evaluation and operational dataset (SLURM scripts, judge results, statistics, figures).
- **`results/`** — Aggregate tables and figures as reported in the dissertation.
- **`data/example/`** — Synthetic fixture data for testing.
- **`tests/`** — Schema validation, agreement metric tests, smoke test.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -v           # 11 tests, <3 seconds
bash scripts/regenerate_figure.sh     # regenerates fig1 from committed results
```

Analysis scripts run from their own subdirectory (paths are relative to `../../`):

```bash
cd src/analysis && python3 generate_tables.py    # reads scores/ → writes results/tables/
cd src/analysis && python3 generate_figures.py   # reads results/tables/ → writes results/figures/
```

## Reproducing the results

Full reproduction requires:

1. **Ollama** serving Gemma 3 (4.3B), Gemma 3 (1B), Llama 3.1 (8B), and Mistral (7B).
2. **MultiWOZ 2.2** dialogues ([upstream](https://github.com/budzianowski/multiwoz), Apache 2.0).
3. **HurtLex** lexicon ([Bassignana, Basile & Patti, CLiC-it 2018](https://github.com/valeriobasile/hurtlex)).
4. **HPC access** for full-corpus evaluation (~48 GPU-hours on NVIDIA A100).

The committed `results/` and `scores/` directories contain everything needed to regenerate every figure and table without re-running the pipeline.

### Pipeline run order

```bash
# 1. Generate responses (requires Ollama)
cd src/pipeline && python3 generate_responses.py
cd src/pipeline && python3 preprocess_responses.py

# 2. Evaluation methods (any order)
cd src/dictionary && python3 lexicon_analysis.py
cd src/dictionary && python3 hurtlex_analysis.py
cd src/llm_judge  && python3 llm_judge.py
cd src/llm_judge  && python3 llm_judge.py --judge-model llama3.1 \
    --output ../../scores/llm_judge_scores_llama.csv \
    --sentence-output ../../scores/llm_judge_sentence_scores_llama.csv
cd src/llm_judge  && python3 llm_judge.py --judge-model mistral \
    --output ../../scores/llm_judge_scores_mistral.csv \
    --sentence-output ../../scores/llm_judge_sentence_scores_mistral.csv
cd src/other_methods && python3 toxicity_analysis.py
cd src/other_methods && python3 likert_analysis.py
cd src/other_methods && python3 topic_analysis.py

# 3. MultiWOZ
cd src/multiwoz && python3 multiwoz_prep.py --limit 10000
cd src/multiwoz && python3 build_subsample.py
cd src/multiwoz && python3 multiwoz_evaluate.py

# 4. Agreement + tables + figures
cd src/analysis && python3 compute_agreement.py
cd src/analysis && python3 generate_tables.py
cd src/analysis && python3 generate_figures.py

# 5. Supplementary
cd src/analysis && python3 v1v2_ablation.py
cd src/analysis && python3 dissertation_followups.py
cd src/analysis && python3 analyze_annotations.py
```

## Data availability

Per-coder annotations and rationales are not released to protect participant privacy. Included derived artifacts:

- Aggregate inter-rater agreement statistics (`results/tables/table_inter_annotator*.csv`)
- Method accuracy against human consensus (`results/tables/table_method_accuracy*.csv`)
- Ground truth summary (`results/tables/table_ground_truth_summary.csv`)
- Annotation codebook (`data/annotations/ANNOTATION_GUIDE.md`)

## Models evaluated

| Role | Model | Parameters | Serving |
|------|-------|-----------|---------|
| Responder + Judge | Gemma 3 | 4.3B | Ollama |
| Responder + Judge | Gemma 3 | 1B | Ollama |
| Responder + Judge | Llama 3.1 | 8B | Ollama |
| Responder + Judge | Mistral | 7B | Ollama |

## Citation

```
@mastersthesis{kashif2026bias,
  title  = {Bias by Whose Definition? The Construct Validity of Dictionary-Based and LLM-as-Judge Bias Evaluation in Conversational AI},
  author = {Kashif, Afifah},
  school = {University of Cambridge},
  year   = {2026}
}
```
