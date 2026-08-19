# Synthetic Example Data

These files contain **obviously synthetic fixture data** for testing the bias evaluation pipeline. They are not real experimental data.

## Files

- **prompts.csv** — 4 synthetic prompts (IDs 901--904) using fake nationalities (Testlandian, Examplestani) and fake conditions (test condition alpha/beta).
- **responses.csv** — 8 synthetic model responses (4 prompts x 2 models: testmodel_a, testmodel_b). Two responses (prompt 901/testmodel_b and prompt 903/testmodel_b) contain deliberately biased language for testing bias-detection scoring.
- **dict_scores.csv** — Dictionary-based bias scores for all 8 responses. The two biased responses have `bias_label=1`.
- **judge_scores.csv** — LLM-judge bias scores for all 8 responses. The two biased responses have `bias_label=1, severity=3`.

## Purpose

Use these files to verify that analysis scripts, scoring pipelines, and statistical tests run correctly on well-understood data with known ground truth before applying them to real experimental outputs.
