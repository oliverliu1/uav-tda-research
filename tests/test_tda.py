import numpy as np
import pytest

from uav_tda import config


def test_persistence_for_point_deterministic_in_process():
    """Two identical calls must agree — sparse Rips determinism gate.

    If this FAILS, sparse Rips is nondeterministic: STOP, report to controller;
    diagram comparisons must switch to W2-tolerance comparisons plan-wide.
    """
    from uav_tda.tda import _persistence_for_point
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(50, 10))
    q = rng.normal(size=10)
    a = _persistence_for_point(q, ref, max_edge=0.5, max_simplex_dim=3, sparse=0.5)
    b = _persistence_for_point(q, ref, max_edge=0.5, max_simplex_dim=3, sparse=0.5)
    assert a.shape == b.shape
    assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


from tests import monolith_harness as mh
from uav_tda.workspace import Workspace


@pytest.mark.slow
def test_tda_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "tda")
    from uav_tda.data import run_prep
    from uav_tda.tda import run_tda
    ws = Workspace.at(tmp_path)
    (tmp_path / "data").symlink_to(repo_root / "data")
    ws.ensure()
    run_prep(ws, debug=True)
    run_tda(ws, debug=True)
    mh.assert_json_equal(oracle / "outputs" / "max_edge_lengths.json",
                         ws.outputs_dir / "max_edge_lengths.json")
    assert np.array_equal(np.load(oracle / "outputs" / "reference_indices.npy"),
                          np.load(ws.outputs_dir / "reference_indices.npy"))
    for m in config.MANIFOLDS:
        for s in ("train", "val", "test"):
            mh.assert_diagram_pkls_equal(
                oracle / "outputs" / "persistence_diagrams" / f"{m}_{s}.pkl",
                ws.persistence_dir / f"{m}_{s}.pkl")


@pytest.mark.slow
def test_tda_spotcheck_against_production_artifacts(repo_root):
    """Recompute 3 real test flows per manifold against repo reference cloud;
    must match the production {m}_test.pkl entries (clean-rebuild lineage)."""
    import json
    import pickle

    import pandas as pd
    from uav_tda.tda import _persistence_for_point
    out = repo_root / "outputs"
    ref_idx = np.load(out / "reference_indices.npy")
    max_edge = json.loads((out / "max_edge_lengths.json").read_text())
    for m in config.MANIFOLDS:
        ref = pd.read_csv(out / f"{m}_train.csv").to_numpy()[ref_idx]
        test_pts = pd.read_csv(out / f"{m}_test.csv").to_numpy()
        stored = pickle.loads((out / "persistence_diagrams" / f"{m}_test.pkl").read_bytes())
        for i in (0, 100, 5000):
            fresh = _persistence_for_point(
                test_pts[i], ref, max_edge[m],
                config.MAX_HOM_DIM[m] + 1, config.SPARSE_RIPS_EPSILON[m])
            assert np.allclose(np.sort(fresh, axis=0),
                               np.sort(np.asarray(stored[i]), axis=0)), (m, i)
