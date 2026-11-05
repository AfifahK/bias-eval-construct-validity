# Judge outputs on refusals (HPC run, response level, T=0, seed 42)

Descriptive numbers only. v1 = prompt with the model-identity line, v2 = without. Refusals: declarative 63 (hand-audited), operational 91. Scored: declarative 297, operational 269. Parse failures (bias_label -1) are dropped per judge. Bootstrap: prompts resampled within nationality, 10,000 resamples, seed 42.

## 1. Flag rate on refusals

| Judge | Ver | Dataset | n refusals | parse fail | flagged | flag rate [95% CI] | US | NK |
|---|---|---|---|---|---|---|---|---|
| Gemma 3 (4.3B) judge | v1 | declarative | 63 | 0 | 54 | 0.86 [0.77, 0.94] | 0.70 (n=20) | 0.93 (n=43) |
| Gemma 3 (4.3B) judge | v1 | operational | 91 | 0 | 63 | 0.69 [0.59, 0.79] | 0.61 (n=33) | 0.74 (n=58) |
| Gemma 3 (4.3B) judge | v2 | declarative | 63 | 0 | 55 | 0.87 [0.77, 0.95] | 0.75 (n=20) | 0.93 (n=43) |
| Gemma 3 (4.3B) judge | v2 | operational | 91 | 0 | 71 | 0.78 [0.69, 0.87] | 0.73 (n=33) | 0.81 (n=58) |
| Gemma 3 (1B) judge | v1 | declarative | 63 | 1 | 62 | 1.00 [1.00, 1.00] | 1.00 (n=20) | 1.00 (n=42) |
| Gemma 3 (1B) judge | v1 | operational | 91 | 1 | 90 | 1.00 [1.00, 1.00] | 1.00 (n=33) | 1.00 (n=57) |
| Gemma 3 (1B) judge | v2 | declarative | 63 | 1 | 62 | 1.00 [1.00, 1.00] | 1.00 (n=19) | 1.00 (n=43) |
| Gemma 3 (1B) judge | v2 | operational | 91 | 1 | 90 | 1.00 [1.00, 1.00] | 1.00 (n=33) | 1.00 (n=57) |
| Llama 3.1 judge | v1 | declarative | 63 | 0 | 9 | 0.14 [0.06, 0.23] | 0.15 (n=20) | 0.14 (n=43) |
| Llama 3.1 judge | v1 | operational | 91 | 0 | 8 | 0.09 [0.03, 0.16] | 0.06 (n=33) | 0.10 (n=58) |
| Llama 3.1 judge | v2 | declarative | 63 | 0 | 7 | 0.11 [0.04, 0.19] | 0.15 (n=20) | 0.09 (n=43) |
| Llama 3.1 judge | v2 | operational | 91 | 0 | 10 | 0.11 [0.04, 0.18] | 0.09 (n=33) | 0.12 (n=58) |
| Mistral judge | v1 | declarative | 63 | 0 | 1 | 0.02 [0.00, 0.05] | 0.05 (n=20) | 0.00 (n=43) |
| Mistral judge | v1 | operational | 91 | 0 | 0 | 0.00 [0.00, 0.00] | 0.00 (n=33) | 0.00 (n=58) |
| Mistral judge | v2 | declarative | 63 | 0 | 1 | 0.02 [0.00, 0.05] | 0.05 (n=20) | 0.00 (n=43) |
| Mistral judge | v2 | operational | 91 | 0 | 0 | 0.00 [0.00, 0.00] | 0.00 (n=33) | 0.00 (n=58) |

## 2. NK/US flag ratio, scored only vs scored + refusals

| Judge | Dataset | Scored only [95% CI] | n US / NK | Scored + refusals [95% CI] | n US / NK |
|---|---|---|---|---|---|
| Gemma 3 (4.3B) judge (v1) | declarative | 1.08 [1.02, 1.15] | 160 / 137 | 1.09 [1.03, 1.16] | 180 / 180 |
| Gemma 3 (4.3B) judge (v1) | operational | 1.04 [1.01, 1.07] | 147 / 122 | 1.02 [0.96, 1.08] | 180 / 180 |
| Gemma 3 (4.3B) judge (v2) | declarative | 1.08 [1.01, 1.16] | 160 / 137 | 1.09 [1.01, 1.17] | 180 / 180 |
| Gemma 3 (4.3B) judge (v2) | operational | 1.05 [1.01, 1.09] | 147 / 122 | 1.03 [0.97, 1.10] | 180 / 180 |
| Gemma 3 (1B) judge (v1) | declarative | 1.00 [1.00, 1.00] | 158 / 130 | 1.00 [1.00, 1.00] | 178 / 172 |
| Gemma 3 (1B) judge (v1) | operational | 1.00 [1.00, 1.00] | 145 / 122 | 1.00 [1.00, 1.00] | 178 / 179 |
| Gemma 3 (1B) judge (v2) | operational | 1.00 [1.00, 1.00] | 147 / 122 | 1.00 [1.00, 1.00] | 180 / 179 |
| Llama 3.1 judge (v1) | declarative | 2.38 [1.81, 3.31] | 160 / 137 | 2.03 [1.55, 2.84] | 180 / 180 |
| Llama 3.1 judge (v1) | operational | 6.19 [3.77, 13.21] | 147 / 122 | 4.88 [3.07, 9.67] | 180 / 180 |
| Llama 3.1 judge (v2) | declarative | 2.63 [1.98, 3.69] | 160 / 137 | 2.20 [1.65, 3.12] | 180 / 180 |
| Llama 3.1 judge (v2) | operational | 8.55 [5.00, 20.84] | 147 / 122 | 6.00 [3.68, 12.57] | 180 / 180 |
| Mistral judge (v1) | declarative | 4.96 [2.35, 16.17] | 160 / 137 | 3.78 [1.79, 12.33] | 180 / 180 |
| Mistral judge (v1) | operational | -- | 147 / 122 | -- | 180 / 180 |
| Mistral judge (v2) | declarative | 4.15 [2.03, 11.74] | 160 / 137 | 3.20 [1.57, 9.00] | 180 / 180 |
| Mistral judge (v2) | operational | -- | 147 / 122 | -- | 180 / 180 |

Gemma 3 (1B) has no stored declarative v2 labels, so no declarative v2 ratio is computed for it (its v2 refusal flag rate is in section 1). Dropped bootstrap resamples (zero US rate) are in `judge_ratio_comparison.csv`.

Verbatim rationales for every judged row: `judge_refusal_rationales.csv` (`audited_refusal` = 1 for the 154 analysed refusals, 0 for the 4 declarative audit false positives).
