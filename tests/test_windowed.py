import numpy as np
import pytest

from uav_tda import config, windowed


def test_config_constants():
    assert config.WINDOW_SIZES == (25, 50, 100, 200)
    assert config.WINDOWED_REPEATS == 10
    assert config.WINDOWED_SHUFFLE_SEEDS == tuple(range(10))


def test_windowed_sparse_covers_all_manifolds():
    # Fixed by the Task-1 benchmark gate (see comment in config.py).
    assert set(config.WINDOWED_SPARSE) == set(config.MANIFOLDS)
    for eps in config.WINDOWED_SPARSE.values():
        assert eps is None or isinstance(eps, float)


def test_make_windows_drops_trailing_partial():
    wins = windowed.make_windows(105, 25)
    assert len(wins) == 4
    assert all(len(w) == 25 for w in wins)
    assert wins[0].tolist() == list(range(25))
    assert wins[3].tolist() == list(range(75, 100))  # rows 100-104 dropped


def test_make_windows_respects_permutation():
    order = np.arange(50)[::-1]  # reversed
    wins = windowed.make_windows(50, 10, order=order)
    assert wins[0].tolist() == list(range(49, 39, -1))
    # every row appears at most once across windows
    flat = np.concatenate(wins)
    assert len(np.unique(flat)) == len(flat)


def test_window_diagram_shape_and_determinism_exact():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 5))
    d1 = windowed.window_diagram(pts, max_edge=0.8, max_hom_dim=1, sparse=None)
    d2 = windowed.window_diagram(pts, max_edge=0.8, max_hom_dim=1, sparse=None)
    assert d1.ndim == 2 and d1.shape[1] == 3
    assert np.array_equal(d1, d2)  # exact Rips: bit-identical, no sort needed
    assert set(np.unique(d1[:, 0])) <= {0.0, 1.0}


# --- Task 2: w2_distance, baseline_medoid_diagram, labeling helpers --------

def _diag(rows):
    """Build an (n, 3) [dim, birth, death] diagram from a list of (dim, b, d)."""
    return np.array(rows, dtype=float) if rows else np.empty((0, 3), dtype=float)


def test_w2_distance_zero_on_identical():
    d = _diag([(0, 0.0, 0.5), (0, 0.1, 0.3), (1, 0.2, 0.6)])
    assert windowed.w2_distance(d, d, max_edge=1.0, max_hom_dim=1) == pytest.approx(0.0, abs=1e-9)


def test_w2_distance_symmetry():
    d1 = _diag([(0, 0.0, 0.5), (0, 0.1, 0.3), (1, 0.2, 0.6)])
    d2 = _diag([(0, 0.0, 0.4), (0, 0.15, 0.35), (1, 0.25, 0.55)])
    fwd = windowed.w2_distance(d1, d2, max_edge=1.0, max_hom_dim=1)
    rev = windowed.w2_distance(d2, d1, max_edge=1.0, max_hom_dim=1)
    assert fwd == pytest.approx(rev, abs=1e-9)
    assert fwd > 0.0


def test_baseline_medoid_diagram_picks_similar_pair():
    similar_a = _diag([(0, 0.0, 0.5), (0, 0.1, 0.3)])
    similar_b = _diag([(0, 0.0, 0.52), (0, 0.11, 0.31)])
    outlier = _diag([(0, 0.0, 5.0), (0, 3.0, 9.0)])
    idx, medoid = windowed.baseline_medoid_diagram(
        [similar_a, similar_b, outlier], max_edge=10.0, max_hom_dim=0,
    )
    assert idx in (0, 1)
    assert np.array_equal(medoid, [similar_a, similar_b][idx])


def test_window_majority_label_normal_majority():
    labels = ["Normal Traffic"] * 6 + ["Sybil Attack"] * 4
    assert windowed.window_majority_label(labels) == "Normal Traffic"
    assert windowed.window_attack_frac(labels) == pytest.approx(0.4)


def test_window_majority_label_attack_majority():
    labels = ["Normal Traffic"] * 4 + ["Sybil Attack"] * 6
    assert windowed.window_majority_label(labels) == "Sybil Attack"
    assert windowed.window_attack_frac(labels) == pytest.approx(0.6)


@pytest.mark.slow
def test_run_windowed_smoke_w100():
    from uav_tda.workspace import Workspace

    ws = Workspace.default()
    window_df, stats, timing = windowed.run_windowed(ws, w=100)

    assert len(window_df) == 18326 // 100 == 183

    score_cols = [c for c in window_df.columns if c.startswith(("W2_", "Z2_"))]
    assert len(score_cols) == 20  # 3 manifolds + 7 subsets, raw + Z2
    for c in score_cols:
        assert np.isfinite(window_df[c].to_numpy()).all()

    assert set(stats.keys()) == {"c2", "network", "physical"}
    for m, (mean, std) in stats.items():
        assert std > 0

    assert timing["total_s"] > 0
    assert timing["n_windows"] == len(window_df)

    attack_heavy = window_df[window_df["attack_frac"] > 0.9]
    pure_normal = window_df[window_df["attack_frac"] == 0.0]
    assert len(attack_heavy) > 0 and len(pure_normal) > 0
    assert attack_heavy["Z2_all_three"].mean() > pure_normal["Z2_all_three"].mean()
