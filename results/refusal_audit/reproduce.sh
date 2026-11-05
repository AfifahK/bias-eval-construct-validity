#!/usr/bin/env bash
# Regenerate every refusal-audit number, table and figure from the committed data.
# Usage (from anywhere):  bash results/refusal_audit/reproduce.sh [--clean]
#   --clean   delete the generated outputs first, so the run proves they are rebuilt from data
# Needs: python3 (3.9+, or set PYTHON=...) with pandas, numpy, scipy, matplotlib, nltk, datasets, transformers, torch,
# detoxify (cached HF models: mediabiasgroup/bias-lexicon, unitary/toxic-bert). HurtLex EN 1.2 TSV is
# not in this repo: set HURTLEX_TSV=/path/to/hurtlex_EN.tsv. Without it the three steps that rescore
# text (dictionary scoring, HurtLex diagnostics, "can" removed) are skipped and the committed
# outputs of those steps are used by the later steps.
# Not run here (need Ollama / HPC): LLM judge calls. The 16 HPC judge CSVs in
# hpc_refusal_judges/results_hpc/ are aggregated by score_refusal_judges.py (step 8b) when present.
# Never overwritten: hand-coded audit sheets (refusal_audit_all67.csv, refusal_audit_operational_mid.csv);
# only their deterministic key files are rebuilt.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"   # Python 3.9+
A=results/refusal_audit
export HURTLEX_TSV="${HURTLEX_TSV:-}"   # path to hurtlex_EN.tsv; if unset the rescoring steps are skipped

if [[ "${1:-}" == "--clean" ]]; then
  rm -f $A/robustness_results.json $A/robustness_summary.md $A/refusal_audit_sample.csv $A/hedge_check.csv \
        $A/soft_refusal_candidates.csv $A/operational_refusals_corrected.csv \
        $A/operational_non_refusals_corrected.csv $A/operational_moved_20.csv \
        $A/operational_correction_comparison.csv $A/refusal_dictionary_scores.csv \
        $A/refusal_mean_comparison.csv $A/refusal_mean_comparison.md $A/refusal_scoring_summary.md $A/hurtlex_diagnostics.md \
        $A/hurtlex_terms.csv $A/hurtlex_row_level.csv $A/hurtlex_ratio_comparison.csv \
        $A/hurtlex_no_can.csv $A/hurtlex_no_can.md $A/refusal_audit_all67_key.csv $A/audit_precision.md $A/audit_sensitivity.md $A/refusal_audit_operational_mid_key.csv \
        $A/judge_refusal_flags.csv $A/judge_ratio_comparison.csv $A/judge_refusal_rationales.csv $A/judge_refusal_summary.md
  rm -rf $A/figures $A/tables
fi

step() { echo "== $*"; }
HAVE_HL=1; [[ -f "$HURTLEX_TSV" ]] || { HAVE_HL=0; echo "!! HurtLex TSV not found ($HURTLEX_TSV): skipping rescoring steps"; }

step "0 blind audit sheets + keys (the audited refusal set depends on them)"; $PY $A/make_audit_all67.py >/dev/null
step "1 refusal robustness (audit sheet, hedges, bootstrap CIs, family exclusion)"; $PY $A/robustness.py >/dev/null
step "2 corrected operational split + operational mid-response audit sheet"; $PY $A/operational_correction.py >/dev/null; $PY $A/make_audit_operational_mid.py >/dev/null
step "3 robustness summary"; $PY $A/make_robustness_summary.py >/dev/null
if [[ $HAVE_HL == 1 ]]; then
  step "4 dictionary / toxicity scoring of the 158 refusals"; $PY $A/score_refusal_dictionaries.py >/dev/null
fi
step "5 refusal vs scored mean comparison"; $PY $A/refusal_mean_comparison.py >/dev/null
if [[ $HAVE_HL == 1 ]]; then
  step "6 HurtLex diagnostics"; $PY $A/hurtlex_diagnostics.py >/dev/null
fi
step "7 HurtLex without 'can'"; $PY $A/hurtlex_no_can.py >/dev/null
if ls $A/hpc_refusal_judges/results_hpc/refusal_*_v1.csv >/dev/null 2>&1; then
  step "8b judge outputs on refusals ($(ls $A/hpc_refusal_judges/results_hpc/*.csv | wc -l) HPC CSVs)"; $PY $A/score_refusal_judges.py >/dev/null
else echo "== no HPC judge CSVs present; judge rows stay pending"; fi
step "9 figures and tables"; $PY $A/make_figures.py >/dev/null; $PY $A/make_tables.py >/dev/null
if [[ -f $A/refusal_audit_all67.csv ]] && $PY - <<'PYEOF'
import pandas as pd, sys
sys.exit(0 if pd.read_csv("results/refusal_audit/refusal_audit_all67.csv", dtype=str).coder_label.notna().any() else 1)
PYEOF
then step "10 audit precision + sensitivity (coder labels found)"; $PY $A/audit_precision.py >/dev/null; $PY $A/audit_sensitivity.py >/dev/null
else step "10 audit precision skipped (refusal_audit_all67.csv not coded yet)"; fi
echo "done. Compare with committed outputs: git status --short $A"
