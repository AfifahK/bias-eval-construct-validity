# HurtLex with "can" removed

Ratio = NK mean / US mean of matches / HurtLex tokens, 'can' matches removed. Bootstrap: prompts resampled within nationality, 10,000, seed 42, percentile 95%. 'orig' = same computation with 'can' kept (unrounded norm; matches the earlier table to rounding).

| dataset | set | mean US | mean NK | NK/US [95% CI] | dropped | orig NK/US [95% CI] |
|---|---|---|---|---|---|---|
| declarative | scored only | 0.0205 | 0.0174 | 0.85 [0.68, 1.06] | 0 | 0.79 [0.66, 0.94] |
| declarative | scored + refusals | 0.0195 | 0.0149 | 0.77 [0.60, 0.97] | 0 | 1.18 [1.02, 1.37] |
| operational | scored only | 0.0141 | 0.0152 | 1.08 [0.89, 1.30] | 0 | 0.92 [0.80, 1.07] |
| operational | scored + refusals | 0.0130 | 0.0126 | 0.97 [0.80, 1.18] | 0 | 1.50 [1.28, 1.77] |

## Responses flagged (>=1 HurtLex match)

| dataset | group | n | flagged, 'can' kept | flagged, 'can' removed | mean matches, 'can' removed |
|---|---|---|---|---|---|
| declarative | scored | 297 | 271 | 254 | 2.10 |
| declarative | refusal | 63 | 63 | 13 | 0.27 |
| operational | scored | 269 | 265 | 243 | 3.07 |
| operational | refusal | 91 | 91 | 23 | 0.40 |
