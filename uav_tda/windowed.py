"""Phase 5 — time-windowed multi-manifold persistence (core).

Implements the windowed variant of docs/superpowers/specs/
2026-09-21-windowed-variant-design.md §2: non-overlapping FlowID-order
windows of W consecutive flows (trailing partial dropped), one
Vietoris-Rips complex per (window, manifold) where the W standardized
feature vectors ARE the point cloud (Bruillard et al. 2016 construction),
plus the exact-Rips benchmark gate that fixes config.WINDOWED_SPARSE.

`window_diagram` mirrors the body of `tda._persistence_for_point` minus
the query-point stacking, so diagram conventions ((n, 3) [dim, birth,
death] float rows; empty -> (0, 3)) match the per-flow arm exactly.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.windowed")


def make_windows(n_rows: int, w: int, order=None) -> list:
    """Return index arrays for non-overlapping windows of w rows.

    Windows are consecutive slices of `order` (identity row order when
    None, e.g. FlowID order; a permutation for the shuffle control).
    The trailing partial window is dropped, per spec §2.1.
    """
    import numpy as np

    if order is None:
        order = np.arange(n_rows)
    else:
        order = np.asarray(order)
    n_windows = n_rows // w
    return [order[i * w:(i + 1) * w] for i in range(n_windows)]


def window_diagram(points, max_edge: float, max_hom_dim: int, sparse):
    """Compute one (sparse) Rips persistence diagram for a window's point cloud.

    Mirrors tda._persistence_for_point minus the query-point stacking:
    the window's points are the whole cloud. Returns an (n, 3) float
    array of [dim, birth, death] rows; empty diagram -> (0, 3).
    """
    import gudhi
    import numpy as np

    kwargs: dict = {"points": points, "max_edge_length": max_edge}
    if sparse is not None:
        kwargs["sparse"] = sparse
    rips = gudhi.RipsComplex(**kwargs)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_hom_dim + 1)
    raw = simplex_tree.persistence()
    if not raw:
        return np.empty((0, 3), dtype=float)
    return np.array(
        [[float(dim), float(birth), float(death)] for dim, (birth, death) in raw],
        dtype=float,
    )


def benchmark_exact_rips(ws: Workspace, w: int = 200, n_trials: int = 3) -> dict:
    """Time exact (sparse=None) Rips calls per manifold on real test windows.

    For each manifold, takes the first n_trials windows of size w from the
    test split and times one exact `window_diagram` call per window with
    the frozen per-manifold max_edge_lengths and MAX_HOM_DIM. Returns
    {manifold: {"median_s": float, "w": int, "trials_s": [float, ...]}}.
    Used by the Task-1 benchmark gate to fix config.WINDOWED_SPARSE.
    """
    import json
    import time

    import numpy as np

    from .tda import load_split_manifold

    with (ws.outputs_dir / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)

    results: dict = {}
    for manifold in config.MANIFOLDS:
        points = load_split_manifold(ws, manifold, "test")
        windows = make_windows(len(points), w)[:n_trials]
        max_edge = max_edge_lengths[manifold]
        max_hom_dim = config.MAX_HOM_DIM[manifold]
        trials = []
        for idx in windows:
            start = time.perf_counter()
            window_diagram(points[idx], max_edge, max_hom_dim, sparse=None)
            trials.append(time.perf_counter() - start)
        results[manifold] = {
            "median_s": float(np.median(trials)),
            "w": w,
            "trials_s": trials,
        }
        log.info("benchmark %s: median %.3fs over %d trials (w=%d)",
                 manifold, results[manifold]["median_s"], len(trials), w)
    return results
