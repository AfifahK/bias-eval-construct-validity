"""Tests for src/analysis/compute_agreement.py: compute_metrics."""

import numpy as np
import pandas as pd
import pytest
from src.analysis.compute_agreement import compute_metrics


class TestComputeMetrics:
    def test_perfect_agreement(self):
        labels = pd.Series([0, 1, 1, 0, 1, 0, 0, 1])
        severity = pd.Series([0, 3, 2, 0, 1, 0, 0, 4])
        result = compute_metrics(labels, labels, severity, severity, "test")
        assert result["agreement_rate"] == 1.0
        assert result["cohens_kappa"] == 1.0
        assert result["n"] == 8

    def test_total_disagreement(self):
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        b = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
        sev_a = pd.Series([0, 0, 0, 0, 3, 3, 3, 3])
        sev_b = pd.Series([3, 3, 3, 3, 0, 0, 0, 0])
        result = compute_metrics(a, b, sev_a, sev_b, "test")
        assert result["agreement_rate"] == 0.0
        assert result["cohens_kappa"] < 0
        assert result["TP"] == 0
        assert result["TN"] == 0

    def test_both_raters_flag_almost_nothing(self):
        """When both raters say 0 on almost everything, kappa should be
        low or undefined — high agreement rate masks low kappa due to
        base-rate problem (prevalence paradox)."""
        # 100 items, both say 0 on 98, rater A says 1 on item 99, rater B says 1 on item 100
        a = pd.Series([0] * 98 + [1, 0])
        b = pd.Series([0] * 98 + [0, 1])
        sev_a = pd.Series([0] * 98 + [1, 0])
        sev_b = pd.Series([0] * 98 + [0, 1])
        result = compute_metrics(a, b, sev_a, sev_b, "test")
        # Agreement rate is high (98%)
        assert result["agreement_rate"] == 0.98
        # But kappa should be very low or negative (no agreement on positives)
        assert result["cohens_kappa"] < 0.1

    def test_empty_input(self):
        result = compute_metrics(
            pd.Series([], dtype=int), pd.Series([], dtype=int),
            pd.Series([], dtype=int), pd.Series([], dtype=int),
            "test",
        )
        assert result is None

    def test_constant_labels_kappa_nan(self):
        """When both raters give the same constant label, kappa is undefined."""
        a = pd.Series([0, 0, 0, 0])
        b = pd.Series([0, 0, 0, 0])
        sev = pd.Series([0, 0, 0, 0])
        result = compute_metrics(a, b, sev, sev, "test")
        assert result["agreement_rate"] == 1.0
        # kappa undefined when only one class present — sklearn returns 0 or nan
        # Our function handles this via try/except
        assert result["cohens_kappa"] == 0.0 or np.isnan(result["cohens_kappa"])

    def test_spearman_nan_when_constant(self):
        """Spearman is nan when both severity series have no variance."""
        a = pd.Series([1, 1, 0])
        b = pd.Series([1, 0, 0])
        sev = pd.Series([0, 0, 0])
        result = compute_metrics(a, b, sev, sev, "test")
        assert np.isnan(result["spearman_rho"])

    def test_confusion_matrix_cells(self):
        a = pd.Series([0, 0, 1, 1])
        b = pd.Series([0, 1, 0, 1])
        sev = pd.Series([0, 0, 1, 1])
        result = compute_metrics(a, b, sev, sev, "test")
        assert result["TP"] == 1
        assert result["TN"] == 1
        assert result["FP"] == 1
        assert result["FN"] == 1

    def test_breakdown_and_group_passed_through(self):
        a = pd.Series([0, 1])
        b = pd.Series([0, 1])
        sev = pd.Series([0, 1])
        result = compute_metrics(a, b, sev, sev, "sentence", "model", "gpt4")
        assert result["level"] == "sentence"
        assert result["breakdown"] == "model"
        assert result["group"] == "gpt4"
