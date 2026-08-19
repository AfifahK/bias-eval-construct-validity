# Data

## `prompts_dataset.csv` (60 rows)

The 60 declarative prompts crossing nationality × mental health condition × scenario. Each row has regular and flipped scale versions of the prompt text.

| Column | Description |
|--------|-------------|
| `prompt_id` | Unique prompt identifier (1–60) |
| `nationality` | Nationality used in prompt (e.g., American, North Korean) |
| `disability` | Full mental health condition description |
| `disability_type` | Condition category: `remitted` or `symptomatic` |
| `scenario_id` | Scenario identifier (1–3) |
| `scenario` | Scenario type: `renting`, `coworker`, or `marriage` |
| `prompt_text_regular` | Prompt with 0=willing → 3=unwilling scale |
| `prompt_text_flipped` | Prompt with 0=unwilling → 3=willing scale |

## `all_model_responses.csv` (2,165 rows)

Raw LLM responses to all prompts across models, scales, and steps (Likert + explanation).

| Column | Description |
|--------|-------------|
| `prompt_id` | Links to `prompts_dataset.csv` |
| `nationality`, `disability`, `disability_type`, `scenario_id`, `scenario` | Prompt metadata |
| `model` | Responder model: `gemma3`, `llama3.1`, or `mistral` |
| `scale` | `regular` or `flipped` |
| `prompt_text` | Full prompt sent to model |
| `response` | Raw model response |
| `step` | `likert` (numeric rating) or `explanation` (free text) |

## `all_model_responses_clean.csv` (3,258 rows)

Preprocessed version of responses with refusal tagging and cleaned text.

| Column | Description |
|--------|-------------|
| All columns from `all_model_responses.csv` | |
| `response_clean` | Response with leading Likert digit and "Explanation:" prefix stripped |
| `is_refusal` | `True` if model refused to answer the prompt |
| `ai_self_reference` | `True` if response contains AI self-referential language |

## `annotations/`

See [`annotations/coder_data_README.md`](annotations/coder_data_README.md) for details on what was removed and what aggregate artifacts remain.

- `ANNOTATION_GUIDE.md` — codebook used by all six coders
- `annotation_sample_key.csv` — ground truth key showing which automated methods flagged each declarative sample item
- `annotation_sample_multiwoz_key.csv` — same for the MultiWOZ sample

## `example/`

Synthetic fixture data for testing. See [`example/README.md`](example/README.md).
