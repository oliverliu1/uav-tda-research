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
