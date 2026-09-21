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

# Calibrated per the amended Task-4 ruling (see task-4-report.md, fix round 1):
# gates derived from the max deltas observed between the monolith oracle and a
# package tda run on debug data, over 25 sampled diagrams x all H-dims x all
# three splits per manifold. Derivation: count_delta = max(3, 2x observed);
# pers_rel = max(0.05, 2x observed rounded up to nearest 0.05); w2 = 3x
# observed rounded up.
# OBSERVED (calibrated 2026-09-20):
#   c2:      max_count_delta=6, max_pers_rel=0.1247, max_w2=0.04719
#   network: max_count_delta=5, max_pers_rel=0.1724, max_w2=0.000163
GATES = {
    "c2": {"count_delta": 12, "pers_rel": 0.25, "w2": 0.15},
    "network": {"count_delta": 10, "pers_rel": 0.35, "w2": 0.00049},
}


@pytest.mark.slow
def test_tda_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    import json
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
    max_edge = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    for s in ("train", "val", "test"):
        # physical (exact Rips) is cross-process reproducible: exact comparison.
        mh.assert_diagram_pkls_equal(
            oracle / "outputs" / "persistence_diagrams" / f"physical_{s}.pkl",
            ws.persistence_dir / f"physical_{s}.pkl")
        # sparse manifolds: statistical comparator (see Task-4 ruling).
        for m in ("c2", "network"):
            mh.assert_diagram_pkls_statistically_equal(
                oracle / "outputs" / "persistence_diagrams" / f"{m}_{s}.pkl",
                ws.persistence_dir / f"{m}_{s}.pkl",
                max_hom_dim=config.MAX_HOM_DIM[m], max_edge=max_edge[m],
                sample=25, gates=GATES[m])


@pytest.mark.slow
def test_tda_spotcheck_against_production_artifacts(repo_root):
    """Recompute real test flows against the repo reference cloud. physical:
    exact match vs production pkl. c2/network: statistical gates only (sparse
    Rips is not bit-reproducible across processes — Task-4 ruling).

    The W2 gate here is SCALE-RELATIVE (5% of the larger total persistence per
    dim) with a production-calibrated absolute floor for small/near-empty dims:
    the debug-calibrated absolute gates in GATES do not transfer to
    production-scale diagrams (observed production W2 on 2026-09-20: c2
    0.0-0.0275, network 0.0-0.0123, vs network's debug-calibrated absolute gate
    of 0.00049). The floor cannot reuse GATES[m]["w2"] either: for network H1
    the total persistence is only ~0.12-0.14 at production scale while W2
    reaches ~10% of it (0.0123 > 0.05*0.118), so the floor is calibrated as 3x
    the max production W2 observed per manifold on 2026-09-20 (c2: 3x0.0275,
    network: 3x0.0123)."""
    SPOT_W2_FLOOR = {"c2": 0.085, "network": 0.037}
    import json
    import pickle

    import pandas as pd
    from gudhi.hera import wasserstein_distance
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
            if config.SPARSE_RIPS_EPSILON[m] is None:
                assert np.allclose(
                    fresh[np.lexsort(fresh.T[::-1])],
                    np.asarray(stored[i])[np.lexsort(np.asarray(stored[i]).T[::-1])]), (m, i)
            else:
                for k in range(config.MAX_HOM_DIM[m] + 1):
                    xa = mh._clamped_dim_slice(fresh, k, max_edge[m])
                    xb = mh._clamped_dim_slice(np.asarray(stored[i]), k, max_edge[m])
                    assert abs(len(xa) - len(xb)) <= GATES[m]["count_delta"], (m, i, k)
                    ta = float((xa[:, 1] - xa[:, 0]).sum()) if len(xa) else 0.0
                    tb = float((xb[:, 1] - xb[:, 0]).sum()) if len(xb) else 0.0
                    w2 = float(wasserstein_distance(xa, xb, order=2.0, internal_p=2.0))
                    w2_gate = max(0.05 * max(ta, tb), SPOT_W2_FLOOR[m])
                    assert w2 <= w2_gate, (m, i, k, w2, w2_gate)
