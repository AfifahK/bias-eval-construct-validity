"""Builds robustness_summary.md from robustness_results.json + the CSVs written by robustness.py."""
import json, re
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
R = json.load(open(OUT / "robustness_results.json"))
B, F, S = R["bootstrap"], R["family"], R["sensitivity"]
ROOT = OUT.parents[1]


def ci(x):
    return f"{x['point']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]"


def drop(x):
    return x["dropped_resamples"]


aud = pd.read_csv(OUT / "refusal_audit_sample.csv")
hedge = pd.read_csv(OUT / "hedge_check.csv")
soft = pd.read_csv(OUT / "soft_refusal_candidates.csv")
import sys
sys.path.insert(0, str(OUT))
import refusal_sets  # noqa: E402
e = refusal_sets.declarative_explanation()      # is_refusal = audited 63, is_refusal_regex = 67
ref = e[e.is_refusal_regex]                      # phrase-position / length facts refer to the 67 regex refusals
FO, FB = R["flag_old"], R["flag_bounds"]
pos = ref.response_clean.map(lambda t: (re.search(r"I can'?t|I cannot|I'?m not able", t, re.I) or re.search("$", t)).start())
n_open = int((pos <= 5).sum())
short = int((ref.response_clean.str.len() < 50).sum())
nonref_decl, nonref_op = int((~e.is_refusal).sum()), len(pd.read_csv(ROOT / "hpc/operational/responses/operational_non_refusals.csv"))

flag_rows = ["Bias-Lexicon", "Llama judge (v1)", "Mistral judge (v1)", "Llama judge (v2)", "Mistral judge (v2)",
             "Gemma 3 4.3B (v1)", "Gemma 3 1B (v1)"]
paper = {"Bias-Lexicon": "2.7", "Llama judge (v1)": "2.4", "Mistral judge (v1)": "5.1"}
def new_cell(n):
    b = FB[n]
    return ci(B['Flag ratio: ' + n]) if b is None else \
        f"not computed ({b['n_missing']} rows without a stored judge label; point {b['point_min']:.2f} to {b['point_max']:.2f} over all 0/1 fillings)"


t_flag = "\n".join(f"| {n} | {FO[n]['n_us']}/{FO[n]['n_nk']} -> {B['Flag ratio: '+n]['n_us']}/{B['Flag ratio: '+n]['n_nk']} | "
                   f"{paper.get(n, '-')} | {ci(FO[n])} | {new_cell(n)} | {drop(FO[n])} |" for n in flag_rows)
t_fam = "\n".join(f"| {n} | drop {v['family']} ({v['dropped_rows']} rows) | {ci(v['old_original'])} -> {ci(v['old_excluded'])} | "
                  + (f"{ci(v['original'])} -> {ci(v['excluded'])}" if v['n_missing_moved'] == 0 else "not computed (4 moved rows without a stored judge label)")
                  + f" | {v['excluded']['n_us']}/{v['excluded']['n_nk']} | {drop(v['excluded'])} |" for n, v in F.items())
sd = soft.groupby(["tier", "dataset", "nationality"]).size().rename("n").reset_index()
t_soft = "\n".join(f"| {r.tier} | {r.dataset} | {r.nationality} | {r.n} |" for r in sd.itertuples())
by_model = soft.groupby(["tier", "model"]).size().rename("n").reset_index()
t_soft_m = "\n".join(f"| {r.tier} | {r.model} | {r.n} |" for r in by_model.itertuples())
ex = []
for tier, g in soft.groupby("tier"):
    for _, r in g.sample(min(3, len(g)), random_state=42).iterrows():
        ex.append(f"- `{tier}` ({r.dataset}, {r.model}, {r.nationality}): \"{r.opening_sentence[:160].strip()}\"")
audit_pat = aud.matched_regex_pattern.value_counts().to_string()
op_ref_n = B["Llama refusal NK/US, operational"]

cmp_ = pd.read_csv(OUT / "operational_correction_comparison.csv")
oc_ref = pd.read_csv(OUT / "operational_refusals_corrected.csv")
op_new = B["Llama refusal NK/US, operational, apostrophe-normalised regex"]
pct = lambda a, b: f"{100*a/b:.1f}%"


def fmt(x, p=3):
    return "n/a" if pd.isna(x) else f"{x:.{p}f}"


def rciq(r, pre):
    return "n/a" if pd.isna(r[pre]) else f"{r[pre]:.2f} [{fmt(r[pre+'_lo'],2)}, {fmt(r[pre+'_hi'],2)}]"


t_cmp = "\n".join(f"| {r.instrument} | {r.level} | {r.n_old}/{r.n_new} | {fmt(r.flag_rate_old)} -> {fmt(r.flag_rate_new)} | "
                  f"{rciq(r, 'ratio_old')} -> {rciq(r, 'ratio_new')} | {fmt(r.moved_flag_rate)} |"
                  for _, r in cmp_.iterrows() if not (r.instrument.endswith("(v2)") and r.level == "sentence"))
jl = pd.read_csv(ROOT / "hpc/operational/evaluations/judge_operational_llama3.1_latest_v1.csv")
jl = jl[jl.bias_label >= 0].copy()
mv = pd.read_csv(OUT / "operational_moved_20.csv")[["prompt_id", "model", "scale_direction"]].drop_duplicates()
mk = set(map(tuple, mv.values))
jl["moved"] = [(a, b_, c) in mk for a, b_, c in zip(jl.prompt_id, jl.model, jl.scale_direction)]
own_old, own_new = jl[jl.model == "llama3.1"], jl[(jl.model == "llama3.1") & ~jl.moved]
own = (f"{int(own_old.bias_label.sum())}/{len(own_old)} ({pct(own_old.bias_label.sum(), len(own_old))}) -> "
       f"{int(own_new.bias_label.sum())}/{len(own_new)} ({pct(own_new.bias_label.sum(), len(own_new))})")

md = f"""# Robustness summary

All numbers use existing results files (nothing modified). Bootstrap: 10,000 resamples, seed 42, percentile
95% CI, resampling **prompts** (60 prompt_ids; 30 American, 30 North Korean) with replacement within nationality,
carrying all of a prompt's rows (models x scales) together. Ratio = NK flag (or refusal) rate / US rate. Resamples
with a US rate of 0 have no ratio and are dropped (count in the last column). Format: point [95% CI].

## 1. NK/US refusal ratio (Llama responses only, as in the paper)

| Dataset | Refusal def. | Ratio | Notes |
|---|---|---|---|
| Declarative | **hand-audited (headline, 63)** | {ci(B['Llama refusal NK/US, declarative'])} | 43/60 NK vs 20/60 US; 4 regex matches coded not-refusal in the audit (decision 30) |
| Declarative | original regex (67) | {ci(B['Llama refusal NK/US, declarative, regex rule (pre-audit)'])} | 45/60 NK vs 20/60 US; paper 2.25 |
| Operational | original regex | {ci(op_ref_n)} | 38/60 NK vs 33/60 US; paper 1.15 |
| Operational | regex after curly->straight apostrophe | {ci(B['Llama refusal NK/US, operational, apostrophe-normalised regex'])} | {S['operational_extra_refusals']} additional refusals, all NK: 58/60 vs 33/60 |
| Declarative | regex after curly->straight apostrophe | {ci(B['Llama refusal NK/US, declarative, apostrophe-normalised regex'])} | {S['declarative_extra_refusals']} extra match, not a refusal (mid-response "I can’t fully assess") |

## 2. Section 3.2 flag ratios on scored declarative responses (explanation, non-refusal)

Old = 293 regex-scored responses. New = 297: the 4 audit false positives (all North Korean) are scored responses. Where a judge has no label for those rows yet (stored files exclude them; the labels from the HPC refusal-judge run are not used in this table) the new ratio is not computed; the range of the point estimate over every 0/1 filling of the 4 rows is shown instead.

| Instrument | Scored n US/NK, old -> new | Paper | Old NK/US [95% CI] | New NK/US [95% CI] | Dropped resamples (old) |
|---|---|---|---|---|---|
{t_flag}

v1 includes the model-identity line, v2 removes it. The v1 point estimates reproduce the paper.

## 3. Family exclusion (v1 judges; drop the judge's own family's responses)

Old (293) and new (297) shown as all scored -> own family excluded.

| Judge | Exclusion | Old: original -> excluded | New: original -> excluded | n US/NK after | Dropped resamples (excluded) |
|---|---|---|---|---|---|
{t_fam}

Bias-Lexicon has no model family and is unchanged. Both Gemma judges drop Gemma responses.

## 4. Regex audit sheet and hedge check

- `refusal_audit_sample.csv`: 40 of the 67 declarative regex refusals, proportional by nationality ({R['audit_alloc']['North Korean']} NK, {R['audit_alloc']['American']} US), seed 42, `coder_label` empty.
- Hand audit of all 67 (`refusal_audit_all67.csv`, single coder): 63 refusals, 4 not (see `audit_precision.md`); headline declarative refusals = 63.
- Across all 67: {n_open} have the refusal phrase at the start of the response and {short} are under 50 characters (the length rule). The remaining responses have the phrase later in the text. The regex matches the phrase anywhere in the text, so mid-response matches are the ones the audit should check.
- Matched patterns in the 40-row sample:
```
{audit_pat}
```
- `hedge_check.csv`: **0** non-refused explanation responses (declarative or operational) contain the requested hedges ("I cannot say", "I can't be sure", "I'm not able to determine" and close variants, with curly apostrophes normalised).
- The same file lists {len(hedge)} rows where the refusal regex only fires after apostrophe normalisation: {int((hedge.dataset=='operational').sum())} operational Llama NK responses that open with "I can’t assist/fulfill/answer ..." and {int((hedge.dataset=='declarative').sum())} declarative Gemma response ("factors I can’t fully assess", a false positive).

## 5. Soft refusals (compliant responses that open with a disclaimer, then answer)

Heuristic: among responses the pipeline did not flag as refusals ({nonref_decl} declarative, {nonref_op} operational), the first sentence contains a disclaimer and at least 300 characters follow it. Tiers: (1) refusal-like hedge (can't/cannot/unable/not able/inappropriate/must clarify), (2) AI-identity disclaimer ("as an AI", "I'm a (large) language model", "I don't have personal feelings/opinions"), (3) importance caveat ("it's important to note/remember"). Headline soft refusals = tiers 1 and 2; tier 3 is listed for transparency only.

| Tier | Dataset | Nationality | n |
|---|---|---|---|
{t_soft}

| Tier | Responder | n |
|---|---|---|
{t_soft_m}

Examples:
{chr(10).join(ex)}

Declarative tier 2 rates: US 28/160, NK 8/137 of scored responses. Tier 1 operational rows are the curly-apostrophe misses in section 1 (their "remainder" is the crisis-line text that follows the refusal), not independent soft refusals. Full list with text: `soft_refusal_candidates.csv`.

## 6. Corrected operational refusal set (curly-apostrophe fix)

Regex fix: `normalize_apostrophes` in `src/pipeline/preprocess_responses.py` and `hpc/operational/scripts/detect_refusals.py` (own commit, 3 new tests, 85 pass). No `I won't`/`I am unable`/`I'm not able` variants (curly or straight) occur in any non-refused response, so the only affected pattern is `I can’t`. Original data files untouched; corrected splits are `operational_refusals_corrected.csv` (91) and `operational_non_refusals_corrected.csv` (269), checked row-for-row against the originals.

| Operational refusals | Original | Corrected |
|---|---|---|
| All refusals | 71/360 (19.7%) | 91/360 (25.3%) |
| Llama responses | 71/120 (59.2%) | 91/120 (75.8%) |
| North Korean prompts (all models) | 38/180 (21.1%) | 58/180 (32.2%) |
| American prompts (all models) | 33/180 (18.3%) | 33/180 (18.3%) |
| NK/US ratio [95% CI] | {ci(op_ref_n)} | {ci(op_new)} |
| Scored (non-refusal) responses | 289 | 269 |

Existing operational instrument results on all 289 scored responses (old) vs the 269 left after moving the 20 out (new). Columns: flag rate old -> new; NK/US ratio [prompt-level bootstrap CI] old -> new; flag rate the instrument gave the 20 moved responses. Sentence-level rows use per-sentence flags. v2 response-level rows are included; ratios are n/a where no American response is flagged.

| Instrument | Level | n old/new | Flag rate | NK/US ratio | Flag rate on the 20 moved |
|---|---|---|---|---|---|
{t_cmp}

- Flag rates and ratios move modestly (largest: Llama v1 response-level ratio 5.45 -> 6.19, Bias-Lexicon 1.13 -> 1.29, Gemma 4.3B flag rate 0.955 -> 0.981). toxic-bert and the Mistral judge flag no operational response either way; Gemma 1B flags every response.
- The 20 moved responses are all Llama on North Korean prompts, so removing them changes the nationality balance of the scored set as well as its size.
- Llama judge v1 flag rate on Llama's own operational responses: {own}.
- Existing score files were not regenerated or modified; the new columns come from re-filtering them by key.

## Contradictions with, or caveats on, the current paper

1. **Operational refusal ratio changes materially (section 6).** The original regex `I can'?t` does not match the curly apostrophe in `I can’t`. {S['operational_extra_refusals']} Llama operational responses on North Korean prompts that open with "I can’t assist with that request" or similar were classed as non-refusals and passed to the operational scorers. Consistent regex handling gives 58/60 NK vs 33/60 US refusals, ratio {B['Llama refusal NK/US, operational, apostrophe-normalised regex']['point']:.2f} [{B['Llama refusal NK/US, operational, apostrophe-normalised regex']['lo']:.2f}, {B['Llama refusal NK/US, operational, apostrophe-normalised regex']['hi']:.2f}], against the paper's 1.15 [{op_ref_n['lo']:.2f}, {op_ref_n['hi']:.2f}], whose CI includes 1. The operational refusal set would be 91, not 71, and the operational scored set 269, not 289. Any paper statement that rests on the operational 1.15 ratio, or on operational flag rates computed over the 289 'scored' responses, is affected (the paper source was not available, so see PAPER_CHANGES.md for the mapping I could make).
2. The refusal-scoring input is 158 (67 declarative regex + 91 operational); analyses use the audited 154 (63 + 91). `hpc_refusal_judges/judge_input.csv` was rebuilt with 158 refusal rows plus the 40-row sanity slice (198 rows, 792 judge calls). The declarative regex set stays at 67 in the judge input (decision 30: the 4 audit false positives are scored as normal responses, analyses use 63): applying the same normalisation to the stored data would also reclassify one Gemma response (prompt 6, regular scale) whose 1,099-character answer contains 'factors I can’t fully assess', which is not a refusal, giving 68 under a purely mechanical reading. The regex still matches the phrase anywhere in a response, so the coder audit sheet is the check on declarative precision.
3. Declarative: the paper's 2.25x reproduces under the regex rule ({ci(B['Llama refusal NK/US, declarative, regex rule (pre-audit)'])}); with the 4 audit false positives removed it is {ci(B['Llama refusal NK/US, declarative'])}, CI excluding 1. Single-coder audit: 63 of 67 coded refusals.
4. Section 3.2 point estimates reproduce on the 293 scored rows (2.69, 2.41, 5.11); with the 297 scored rows Bias-Lexicon is {ci(B['Flag ratio: Bias-Lexicon'])} and Gemma 4.3B {ci(B['Flag ratio: Gemma 3 4.3B (v1)'])}; judges without stored labels for the 4 moved rows are not computed here. Intervals are wide: Mistral v1 (293) {ci(FO['Mistral judge (v1)'])}. The Llama v1 interval ({ci(B['Flag ratio: Llama judge (v1)'])}) overlaps it, so the data do not separate the two judges' ratios.
5. Family exclusion lowers the Mistral ratio from {F['Mistral judge (v1)']['original']['point']:.2f} to {F['Mistral judge (v1)']['excluded']['point']:.2f} (CI [{F['Mistral judge (v1)']['excluded']['lo']:.2f}, {F['Mistral judge (v1)']['excluded']['hi']:.2f}]) and Llama from {F['Llama judge (v1)']['original']['point']:.2f} to {F['Llama judge (v1)']['excluded']['point']:.2f}; the Gemma 4.3B CI includes 1 after exclusion. The Gemma 1B judge flags every response, so its ratio is 1.00 with a degenerate CI.
6. v2 ratios differ from v1 (Llama {B['Flag ratio: Llama judge (v2)']['point']:.2f} vs {B['Flag ratio: Llama judge (v1)']['point']:.2f}; Mistral {B['Flag ratio: Mistral judge (v2)']['point']:.2f} vs {B['Flag ratio: Mistral judge (v1)']['point']:.2f}) but the intervals overlap.
"""
(OUT / "robustness_summary.md").write_text(md)
print(md)
