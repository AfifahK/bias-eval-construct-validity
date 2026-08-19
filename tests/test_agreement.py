"""Test inter-rater agreement utilities."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def test_perfect_agreement():
    labels = [0, 1, 1, 0, 1, 0, 0, 1]
    kappa = cohen_kappa_score(labels, labels)
    assert kappa == 1.0


def test_imperfect_agreement():
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=200).tolist()
    b = rng.integers(0, 2, size=200).tolist()
    kappa = cohen_kappa_score(a, b)
    assert kappa < 1.0
