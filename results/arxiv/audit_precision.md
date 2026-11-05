# Audited precision of the 67 declarative refusals

Source: `refusal_audit_all67.csv` joined to `refusal_audit_all67_key.csv`. Precision = label 1 / rows. CI: exact Clopper-Pearson 95%. Last column: range if '?' rows are all non-refusals (lower) or all refusals (upper).

| group | n | 1 (refusal) | 0 (not refusal) | ? | precision [95% CI] | range with '?' |
|---|---|---|---|---|---|---|
| all | 67 | 63 | 4 | 0 | 0.940 [0.854, 0.983] | 0.940 to 0.940 |
| American | 20 | 20 | 0 | 0 | 1.000 [0.832, 1.000] | 1.000 to 1.000 |
| North Korean | 47 | 43 | 4 | 0 | 0.915 [0.796, 0.976] | 0.915 to 0.915 |
| phrase at start | 54 | 54 | 0 | 0 | 1.000 [0.934, 1.000] | 1.000 to 1.000 |
| phrase mid-response | 13 | 9 | 4 | 0 | 0.692 [0.386, 0.909] | 0.692 to 0.692 |

'?' rows: 0. Rows coded 0: R19, R26, R39, R64.

## Operational audit (mid-response regex matches)

Population: 91 corrected operational refusals; 90 have the refusal phrase at the start of the response (not audited), 1 match mid-response (`refusal_audit_operational_mid.csv`).

| group | n | 1 (refusal) | 0 (not refusal) | ? | precision [95% CI] |
|---|---|---|---|---|---|
| mid-response matches | 1 | 1 | 0 | 0 | 1.000 [0.025, 1.000] |

Rows coded 0: none.
Precision is for the mid-response subset only; the start-of-response matches were not hand-coded.
