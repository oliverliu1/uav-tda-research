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


def w2_distance(d1, d2, max_edge: float, max_hom_dim: int) -> float:
    """Hera Wasserstein-2 distance between two diagrams, summed over dims.

    Per-dim slices (birth, death) via probe._slice_dim (inf deaths clamped
    to max_edge, matching the per-flow arm), dims 0..max_hom_dim inclusive.
    """
    from gudhi.hera import wasserstein_distance

    from .probe import _slice_dim

    total = 0.0
    for dim in range(max_hom_dim + 1):
        a = _slice_dim(d1, dim, max_edge)
        b = _slice_dim(d2, dim, max_edge)
        total += float(wasserstein_distance(a, b, order=2, internal_p=2.0))
    return total


def baseline_medoid_diagram(diagrams: list, max_edge: float, max_hom_dim: int):
    """W2-medoid of a list of diagrams: the one minimizing summed W2 to the rest.

    Builds the full symmetric pairwise W2 distance matrix (zero diagonal)
    via w2_distance, then returns (argmin-row-sum index, that diagram).
    """
    import numpy as np

    n = len(diagrams)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = w2_distance(diagrams[i], diagrams[j], max_edge, max_hom_dim)
            dist[i, j] = d
            dist[j, i] = d
    medoid_idx = int(np.argmin(dist.sum(axis=1)))
    return medoid_idx, diagrams[medoid_idx]


_NORMAL_LABEL = "Normal Traffic"


def window_majority_label(labels) -> str:
    """Modal label of a window.

    Ties are broken: attack labels beat "Normal Traffic", then
    alphabetically among tied attack labels (arbitrary but deterministic;
    documented per the Task-2 brief).
    """
    from collections import Counter

    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [lbl for lbl, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    attack_candidates = sorted(c for c in candidates if c != _NORMAL_LABEL)
    if attack_candidates:
        return attack_candidates[0]
    return _NORMAL_LABEL


def window_attack_frac(labels) -> float:
    """Fraction of non-"Normal Traffic" flows in a window."""
    import numpy as np

    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels != _NORMAL_LABEL))


def run_windowed(ws: Workspace, w: int, order_seed=None, repeat: int = 0):
    """Coupled single-invocation windowed pass (spec §2.3).

    (a) val-Normal rows (original order) chunked into windows -> per-manifold
        diagrams -> W2-medoid baseline + val-window score distribution ->
        z-stats (mean/std, degenerate-sigma guard).
    (b) test split (permuted iff order_seed is not None) -> windows ->
        diagrams -> per-manifold W2 scores vs the SAME baselines.
    (c) window_df: window_idx, start_pos, majority_label, attack_frac,
        W2_<manifold> x3, W2_<subset> x7, and Z2_ variants of all 10
        (z-scored manifolds then subset sums recomputed, matching
        metrics.apply_znorm semantics).
    (d) timing: total_s, n_windows, per_window_s, rips_s, w2_s, baseline_s.

    ``repeat`` is metadata only; it is not consumed by any RNG here.
    """
    import json
    import time

    import numpy as np
    import pandas as pd

    from .metrics import MANIFOLD_SUBSETS
    from .tda import load_labels_for_split, load_split_manifold

    t_total0 = time.perf_counter()
    timers = {"rips_s": 0.0, "w2_s": 0.0, "baseline_s": 0.0}

    with (ws.outputs_dir / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)
    max_hom_dim = config.MAX_HOM_DIM

    # --- (a) val-Normal baseline ------------------------------------------------
    labels_val = load_labels_for_split(ws, "val").to_numpy()
    normal_mask = labels_val == _NORMAL_LABEL

    baseline_diagrams: dict = {}
    stats: dict = {}
    for m in config.MANIFOLDS:
        points_val = load_split_manifold(ws, m, "val")[normal_mask]
        max_edge = max_edge_lengths[m]
        hd = max_hom_dim[m]
        sparse = config.WINDOWED_SPARSE[m]

        val_windows = make_windows(len(points_val), w)
        val_diagrams = []
        for idx in val_windows:
            t0 = time.perf_counter()
            diag = window_diagram(points_val[idx], max_edge, hd, sparse)
            timers["rips_s"] += time.perf_counter() - t0
            val_diagrams.append(diag)

        t0 = time.perf_counter()
        _, baseline_diag = baseline_medoid_diagram(val_diagrams, max_edge, hd)
        timers["baseline_s"] += time.perf_counter() - t0
        baseline_diagrams[m] = baseline_diag

        val_scores = []
        for diag in val_diagrams:
            t0 = time.perf_counter()
            score = w2_distance(diag, baseline_diag, max_edge, hd)
            timers["w2_s"] += time.perf_counter() - t0
            val_scores.append(score)
        val_scores = np.asarray(val_scores, dtype=float)
        mean, std = float(np.mean(val_scores)), float(np.std(val_scores))
        if std < 1e-12:
            log.warning("run_windowed: manifold %s has degenerate val-window "
                        "std=%.3g; using 1.0", m, std)
            std = 1.0
        stats[m] = (mean, std)

    # --- (b) test split, scored against the same baselines ---------------------
    labels_test = load_labels_for_split(ws, "test").to_numpy()
    n_test = len(labels_test)
    if order_seed is None:
        order = None
    else:
        order = np.random.default_rng(order_seed).permutation(n_test)
    test_windows = make_windows(n_test, w, order=order)

    points_test = {m: load_split_manifold(ws, m, "test") for m in config.MANIFOLDS}

    rows = []
    for i, idx in enumerate(test_windows):
        window_labels = labels_test[idx]
        row = {
            "window_idx": i,
            "start_pos": i * w,
            "majority_label": window_majority_label(window_labels),
            "attack_frac": window_attack_frac(window_labels),
        }
        for m in config.MANIFOLDS:
            max_edge = max_edge_lengths[m]
            hd = max_hom_dim[m]
            sparse = config.WINDOWED_SPARSE[m]

            t0 = time.perf_counter()
            diag = window_diagram(points_test[m][idx], max_edge, hd, sparse)
            timers["rips_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            score = w2_distance(diag, baseline_diagrams[m], max_edge, hd)
            timers["w2_s"] += time.perf_counter() - t0
            row[f"W2_{m}"] = score
        rows.append(row)

    window_df = pd.DataFrame(rows)
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        window_df[f"W2_{subset}"] = sum(window_df[f"W2_{m}"] for m in manifolds)
    for m in config.MANIFOLDS:
        mean, std = stats[m]
        window_df[f"Z2_{m}"] = (window_df[f"W2_{m}"] - mean) / std
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        window_df[f"Z2_{subset}"] = sum(window_df[f"Z2_{m}"] for m in manifolds)

    n_windows = len(test_windows)
    total_s = time.perf_counter() - t_total0
    timing = {
        "total_s": total_s,
        "n_windows": n_windows,
        "per_window_s": (total_s / n_windows) if n_windows else float("nan"),
        "rips_s": timers["rips_s"],
        "w2_s": timers["w2_s"],
        "baseline_s": timers["baseline_s"],
    }
    log.info("run_windowed: w=%d order_seed=%s repeat=%d -> %d windows in %.1fs",
              w, order_seed, repeat, n_windows, total_s)

    return window_df, stats, timing


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
