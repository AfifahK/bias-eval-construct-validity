# Declarative refusals if audit-coded 0 rows are treated as non-refusals

Rows coded 0 in `refusal_audit_all67.csv`: 31_gemma3_flipped, 33_mistral_flipped, 42_llama3.1_flipped, 45_llama3.1_flipped (4 rows; sample ids 31_gemma3_flipped, 33_mistral_flipped, 42_llama3.1_flipped, 45_llama3.1_flipped). Bootstrap as in `robustness_summary.md` (prompts resampled within nationality, 10,000, seed 42).

| refusal set | total | Gemma | Llama | Mistral | Llama NK/US | NK/US [95% CI] |
|---|---|---|---|---|---|---|
| regex rule (paper) | 67 | 1 | 65 | 1 | 45/60 vs 20/60 | 2.25 [1.61, 3.46] |
| minus audit-coded 0 | 63 | 0 | 63 | 0 | 43/60 vs 20/60 | 2.15 [1.52, 3.31] |

# Operational refusals if audit-coded 0 rows are treated as non-refusals

Rows coded 0 in `refusal_audit_operational_mid.csv`: none. Only the mid-response matches were audited.

| refusal set | total | Llama | Llama NK/US | NK/US [95% CI] |
|---|---|---|---|---|
| regex rule (corrected) | 91 | 91 | 58/60 vs 33/60 | 1.76 [1.51, 2.07] |
| minus audit-coded 0 | 91 | 91 | 58/60 vs 33/60 | 1.76 [1.51, 2.07] |
