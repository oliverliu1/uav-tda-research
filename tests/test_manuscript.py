import numpy as np
import pytest

from uav_tda import config, manuscript


def test_manuscript_seeds_constant():
    assert config.MANUSCRIPT_SEEDS == (42, 7, 123, 0, 1, 2, 3, 4, 5, 6)
    assert config.MANUSCRIPT_SEEDS[:3] == config.PROBE_SEEDS


def test_stratified_resample_preserves_class_counts():
    labels = np.array(["A"] * 10 + ["B"] * 5)
    rng = np.random.default_rng(0)
    idx = manuscript.stratified_resample_indices(labels, rng)
    assert len(idx) == 15
    resampled = labels[idx]
    assert (resampled == "A").sum() == 10
    assert (resampled == "B").sum() == 5


def test_bootstrap_ci_contains_truth_and_is_deterministic():
    rng = np.random.default_rng(1)
    per_seed = []
    for _ in range(3):
        y = np.array([0] * 200 + [1] * 200)
        scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.2, 1, 200)])
        per_seed.append((y, scores))
    lo, hi = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=0)
    assert 0.5 < lo < hi < 1.0
    # true AUC for N(0,1) vs N(1.2,1) is Phi(1.2/sqrt(2)) ~= 0.802
    assert lo < 0.802 < hi
    lo2, hi2 = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=0)
    assert (lo, hi) == (lo2, hi2)
    lo3, hi3 = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=1)
    assert (lo, hi) != (lo3, hi3)


def test_bootstrap_never_drops_a_class():
    # tiny minority class: unstratified resampling would frequently lose it
    y = np.array([0] * 98 + [1] * 2)
    scores = np.arange(100, dtype=float)
    lo, hi = manuscript.bootstrap_mean_auc_ci([(y, scores)], B=200, bootstrap_seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)  # no ValueError from single-class replicate
