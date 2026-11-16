"""Property, mathematical invariant, and statistical tests for kept metrics."""

from __future__ import annotations

import random

import pytest

from apart_epistemics.metrics import (
    bootstrap_ci,
    compute_page_decomposition,
    holm_bonferroni,
    paired_wilcoxon_p,
)


def test_pages_diversity_prediction_theorem_holds_universally() -> None:
    rng = random.Random(123)
    for _ in range(50):
        truth = rng.uniform(-2.0, 6.0)
        estimates = [truth + rng.gauss(0.2, 0.8) for _ in range(30)]
        decomp = compute_page_decomposition(estimates, truth)
        # Core identity: crowd_error == avg_individual_error - diversity
        assert decomp["identity_residual"] < 1e-12
        assert decomp["crowd_error"] == pytest.approx(
            decomp["avg_individual_error"] - decomp["diversity"], abs=1e-12
        )
        assert decomp["iqr"] >= 0.0


def test_statistical_tests_wilcoxon_bootstrap_and_holm() -> None:
    # Clear positive difference across 20 cells
    x = [0.20 + 0.01 * i for i in range(20)]
    y = [0.55 + 0.01 * i for i in range(20)]
    p_val = paired_wilcoxon_p(x, y)
    assert p_val < 0.001
    assert paired_wilcoxon_p(x, x) == pytest.approx(1.0)

    lo, hi = bootstrap_ci([yi - xi for xi, yi in zip(x, y, strict=True)], seed=42)
    assert lo > 0.30
    assert hi < 0.40

    adj = holm_bonferroni([0.005, 0.02, 0.04, 0.50])
    assert adj[0] == pytest.approx(0.02)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.08)
    assert adj[3] == pytest.approx(0.50)
