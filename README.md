# Bias by Whose Definition?

> **Paper:** *Refusal Is Not Scored: A Coverage Limit in Content-Based Bias Evaluation* (Kashif, Sadek, Huang, Gunes, 2026). The paper's analyses are in [`results/refusal_audit/`](results/refusal_audit/); `bash results/refusal_audit/reproduce.sh` regenerates its figures and numbers from the stored model outputs.
> `results/tables`, `results/figures` and `hpc/operational` hold the as-submitted dissertation outputs (293/289 scored, 67/71 refusals); `results/refusal_audit/` holds the audited set (297/269, 63/91) and is what this README reports.

This repository contains the evaluation harness for my Cambridge MPhil dissertation, *Bias by Whose Definition? The Construct Validity of Dictionary-Based and LLM-as-Judge Bias Evaluation in Conversational AI* (2026). The study tests whether different bias evaluation methods measure the same thing when they are given identical inputs. It uses three datasets that differ in content type. The declarative set has 60 base prompts crossing 2 nationalities (American, North Korean), 6 mental health descriptors (3 conditions, each symptomatic or remitted), and 5 social-distance scenarios drawn from the mental health stigma literature, and each prompt is issued in regular and flipped Likert-scale versions to control for acquiescence bias. The operational set keeps the same demographic crossing but asks what action to take in 5 deployment settings (housing assistance, workplace task assignment, relationship counseling, community resource allocation, caregiving), with each prompt phrased as both "help" and "decline helping." The naturalistic set is MultiWOZ 2.2, which has 113k task-oriented dialogue turns that were not written to elicit bias. Gemma 3, Llama 3.1 8B, and Mistral 7B generated 360 responses for each synthetic set. Refusals are pulled out and analyzed separately rather than dropped, and the remaining responses are scored by seven methods (three lexicon or classifier approaches and four LLM judges) against labels from six human coders on a stratified annotation sample.

## What this found

Numbers are on the hand-audited refusal set (297 declarative and 269 operational scored responses); tables are in `results/refusal_audit/corrected_stats/` and `results/refusal_audit/`.

Bias-flagging rates on 297 non-refusal responses to intersectional prompts (nationality × mental health status × scenario) span 0% to 100%. toxic-bert flags nothing. Gemma 3 (1B) flags everything it labelled (288 of 297 have a label).

| Method | Paradigm | Flagging rate |
|--------|----------|--------------|
| toxic-bert (classifier) | Dictionary | 0.0% |
| Mistral 7B | LLM-judge | 14.1% |
| Bias-Lexicon | Dictionary | 23.2% |
| Llama 3.1 | LLM-judge | 58.2% |
| HurtLex | Dictionary | 91.2% |
| Gemma 3 (4.3B) | LLM-judge | 95.3% |
| Gemma 3 (1B) | LLM-judge | 100.0% |

Cohen's kappa across all 21 pairs, with bootstrap 95% CIs (10,000 resamples) and permutation p-values under Holm–Bonferroni: two pairs survive at α = 0.05, both LLM-judge × LLM-judge: Llama 3.1 × Mistral (κ = 0.211) and Gemma 4.3B × Llama 3.1 (κ = 0.129). No cross-paradigm pair is significant after correction. Dictionary methods and LLM judges are not measuring the same thing.

**Self-evaluation is model-specific, not universal.** The declarative rates in this paragraph are v1-prompt rates; the declarative odds ratios are pooled over v1 and v2 unless stated. Llama 3.1 flags 63.8% of Gemma- and Mistral-generated outputs but only 35.1% of its own, a 29-point gap (pooled OR = 0.32, 95% bootstrap CI [0.20, 0.48], p < 0.001). With the model-identity line removed (v2 only) the gap holds: 35.1% on own outputs versus 62.1% on others (OR = 0.33, CI [0.17, 0.59]). On operational content the effect strengthens: 3.4% on own outputs versus 37.9% on others (v1 prompt only; OR = 0.059, bootstrap CI [0.023, 0.218]). Mistral does the opposite on declarative content, flagging its own outputs more (pooled OR = 2.72, CI [1.71, 4.49]). Gemma shows no effect (pooled OR = 0.76, CI [0.37, 1.64]); near-ceiling flagging leaves no room for one. A log-length covariate does not attenuate the Llama interaction in the stacked logistic regression (β = −1.245 without length, −1.275 with; OR = 0.28, 95% CI [0.10, 0.80] with length).

**On naturalistic dialogue (MultiWOZ 2.2),** the methods converge on what bias *isn't* and diverge on what it *is*. Turn-level rates on a 500-dialogue subsample (6,648 turns): toxic-bert 0.03%, Mistral 0.03%, Bias-Lexicon 0.15%, Llama 3.1 6.0%, Gemma 3 (4.3B) 15.7%, HurtLex 41.95%, Gemma 3 (1B) 99.7%. On the full corpus (8,437 dialogues, 113,552 turns), judge rates span 0.08% to 99.4%, a factor of over 1,200. All judges land within 1 pp of their subsample estimates. A tool validated only on naturalistic corpora looks reliable because it agrees with other methods on the easy cases.

**Differential refusal is the finding no method catches.** Of 360 declarative responses, 63 are refusals (17.5%), all from Llama 3.1. North Korean prompts are refused at 2.15× the American rate (43 vs 20 of 60 each; 95% CI 1.52–3.31). This is allocative bias enacted through refusal, not through response content, and every method misses it, because every method scores response text and a refused prompt produces none. The operational dataset replicates: 91 of 360 are refusals (25.3%), all Llama, with a North Korean/American ratio of 1.76 (58 vs 33 of 60; CI 1.51–2.07).

The disagreement replicates on the operational dataset (deployment-framed, action-oriented prompts): no pair survives Holm correction there (smallest adjusted p = 0.16, Gemma 4.3B × Llama 3.1, κ = 0.020; the toxic-bert × Mistral pair is constant and has no kappa). Flagging rates and method ordering shift: Mistral drops from 14.1% to 0.0%, Bias-Lexicon more than doubles to 52.4%, and it now flags more than Llama 3.1 (34.2%). Method behavior is content-dependent.

## What's in here

- **`src/`** — Response generation, dictionary scoring (Bias-Lexicon, HurtLex), LLM-as-judge evaluation, toxicity/sentiment scoring, thematic analysis, agreement computation, figures and tables.
- **`data/`** — Prompt templates, model responses, annotation keys. Per-coder annotations not released (participant privacy); aggregate agreement statistics live in `results/`.
- **`scores/`** — Per-method outputs on the declarative dataset.
- **`multiwoz/`** — Judge and dictionary scores on MultiWOZ. Raw dialogues not included; obtain from the [upstream repository](https://github.com/budzianowski/multiwoz) (Apache 2.0).
- **`hpc/`** — HPC pipeline for full-corpus MultiWOZ evaluation and operational dataset (SLURM scripts, judge results, statistics, figures).
- **`results/`** — Aggregate tables and figures as submitted, plus `refusal_audit/` (audited set).
- **`data/example/`** — Synthetic fixture data for testing.
- **`tests/`** — Unit tests for preprocessing (refusal detection, response cleaning), dictionary scoring (Bias-Lexicon, HurtLex tokenization and matching), inter-rater agreement metrics, and fixture schema validation. 85 tests, all importing from `src/`.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -v           # 85 tests
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

## Refusal audit (`results/refusal_audit/`)

Hand audit of the refusal rule and the corrected-set analyses; see [`results/refusal_audit/README.md`](results/refusal_audit/README.md).

```bash
bash results/refusal_audit/reproduce.sh    # regenerate numbers, tables and figures; add --clean to rebuild from scratch
git status --short results/refusal_audit   # empty = outputs match the committed ones
```

Needs python3 (3.9+) with the packages in `requirements.txt`. The HurtLex lexicon is not redistributed: obtain `hurtlex_EN.tsv` (English, 1.2) from the [HurtLex repository](https://github.com/valeriobasile/hurtlex) and `export HURTLEX_TSV=/path/to/hurtlex_EN.tsv`.

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
