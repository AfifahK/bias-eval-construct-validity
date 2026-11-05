# Refusal scoring summary

Dictionary / classifier instruments only. LLM-judge columns are pending the HPC run.

## Mean score on refusals vs scored responses

Refusals: 63 declarative (hand-audited; the 4 audit false positives are scored), 91 operational (corrected); `refusal_dictionary_scores.csv`. Scored: 297 declarative, 269 corrected operational (stored scores; the 4 moved declarative rows scored with the repo functions). Score = the instrument's length-normalised score (`bias_score_norm`, `hurtlex_norm`) or toxic-bert (declarative) / Detoxify (operational) toxicity probability. CIs: 10,000 bootstrap resamples of prompts (a prompt's refusal and scored rows move together), within nationality (pooled rows stratified by nationality), percentile 95%, seed 42; no resamples dropped. Diff = refusals minus scored. Source: `refusal_mean_comparison.py` -> `refusal_mean_comparison.csv`.

| dataset | nationality | instrument | n ref / scored | mean refusals [95% CI] | mean scored [95% CI] | diff (ref - scored) [95% CI] |
|---|---|---|---|---|---|---|
| declarative | American | Bias-Lexicon (norm) | 20 / 160 | 0.0000 [0.0000, 0.0000] | 0.0017 [0.0007, 0.0027] | -0.0017 [-0.0027, -0.0007] |
| declarative | American | HurtLex (norm) | 20 / 160 | 0.0965 [0.0813, 0.1112] | 0.0298 [0.0262, 0.0333] | 0.0667 [0.0502, 0.0820] |
| declarative | American | toxic-bert / Detoxify | 20 / 160 | 0.0013 [0.0007, 0.0024] | 0.0011 [0.0007, 0.0017] | 0.0002 [-0.0007, 0.0015] |
| declarative | North Korean | Bias-Lexicon (norm) | 43 / 137 | 0.0006 [0.0000, 0.0020] | 0.0036 [0.0027, 0.0046] | -0.0030 [-0.0045, -0.0012] |
| declarative | North Korean | HurtLex (norm) | 43 / 137 | 0.1089 [0.0957, 0.1223] | 0.0235 [0.0206, 0.0266] | 0.0854 [0.0720, 0.0987] |
| declarative | North Korean | toxic-bert / Detoxify | 43 / 137 | 0.0012 [0.0008, 0.0015] | 0.0008 [0.0007, 0.0009] | 0.0003 [0.0000, 0.0007] |
| declarative | pooled | Bias-Lexicon (norm) | 63 / 297 | 0.0004 [0.0000, 0.0014] | 0.0026 [0.0019, 0.0033] | -0.0021 [-0.0031, -0.0010] |
| declarative | pooled | HurtLex (norm) | 63 / 297 | 0.1049 [0.0948, 0.1151] | 0.0269 [0.0246, 0.0293] | 0.0780 [0.0675, 0.0884] |
| declarative | pooled | toxic-bert / Detoxify | 63 / 297 | 0.0012 [0.0009, 0.0016] | 0.0010 [0.0008, 0.0013] | 0.0002 [-0.0002, 0.0007] |
| operational | American | Bias-Lexicon (norm) | 33 / 147 | 0.0010 [0.0000, 0.0024] | 0.0038 [0.0032, 0.0045] | -0.0029 [-0.0042, -0.0013] |
| operational | American | HurtLex (norm) | 33 / 147 | 0.0781 [0.0622, 0.0930] | 0.0209 [0.0188, 0.0232] | 0.0572 [0.0410, 0.0722] |
| operational | American | toxic-bert / Detoxify | 33 / 147 | 0.0012 [0.0007, 0.0020] | 0.0007 [0.0007, 0.0008] | 0.0005 [0.0000, 0.0013] |
| operational | North Korean | Bias-Lexicon (norm) | 58 / 122 | 0.0004 [0.0000, 0.0013] | 0.0038 [0.0030, 0.0045] | -0.0033 [-0.0040, -0.0026] |
| operational | North Korean | HurtLex (norm) | 58 / 122 | 0.1059 [0.0940, 0.1175] | 0.0193 [0.0173, 0.0213] | 0.0866 [0.0744, 0.0984] |
| operational | North Korean | toxic-bert / Detoxify | 58 / 122 | 0.0013 [0.0008, 0.0020] | 0.0006 [0.0006, 0.0006] | 0.0007 [0.0001, 0.0014] |
| operational | pooled | Bias-Lexicon (norm) | 91 / 269 | 0.0006 [0.0000, 0.0014] | 0.0038 [0.0033, 0.0043] | -0.0032 [-0.0039, -0.0024] |
| operational | pooled | HurtLex (norm) | 91 / 269 | 0.0958 [0.0865, 0.1048] | 0.0202 [0.0187, 0.0217] | 0.0756 [0.0662, 0.0847] |
| operational | pooled | toxic-bert / Detoxify | 91 / 269 | 0.0012 [0.0008, 0.0018] | 0.0007 [0.0007, 0.0007] | 0.0006 [0.0002, 0.0011] |
