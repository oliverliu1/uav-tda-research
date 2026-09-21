"""Reconstructed unsupervised Wasserstein-2 probe (paper §III.F).

Reproduces the lost tools/quick_unsup_probe.py: 200 flows/class from the test
split, per-flow persistence diagrams truncated to the top-K longest bars,
Hera approximate Wasserstein-2 (relative error delta) to a per-manifold
baseline barcode, summed over homology dimensions.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .config import (MANIFOLDS, MAX_HOM_DIM, PROBE_DELTA, PROBE_PER_CLASS,
                     PROBE_TOP_K, SPARSE_RIPS_EPSILON)
from .metrics import MANIFOLD_SUBSETS
from .paths import OUTPUTS_DIR, PERSISTENCE_DIR


def truncate_top_k(diagram_dim: np.ndarray, k: int) -> np.ndarray:
    """Keep the k finite bars with the largest persistence (death - birth)."""
    if diagram_dim.shape[0] <= k:
        return diagram_dim
    persistence = diagram_dim[:, 1] - diagram_dim[:, 0]
    keep = np.argsort(persistence)[::-1][:k]
    return diagram_dim[keep]


def _load_baseline_barcodes(max_edge_lengths: dict) -> dict:
    """Per-manifold per-dim baseline diagram from the 500 reference points."""
    import gudhi

    ref_idx = np.load(OUTPUTS_DIR / "reference_indices.npy")
    baselines: dict = {}
    for manifold in MANIFOLDS:
        pts = pd.read_csv(OUTPUTS_DIR / f"{manifold}_train.csv").to_numpy()[ref_idx]
        max_edge = max_edge_lengths[manifold]
        eps = SPARSE_RIPS_EPSILON.get(manifold)
        kwargs = {"points": pts, "max_edge_length": max_edge}
        if eps is not None:
            kwargs["sparse"] = eps
        st = gudhi.RipsComplex(**kwargs).create_simplex_tree(
            max_dimension=MAX_HOM_DIM[manifold] + 1)
        raw = st.persistence()
        if raw:
            diag = np.array([[float(d), float(b), float(de)] for d, (b, de) in raw],
                            dtype=float)
        else:
            diag = np.empty((0, 3), dtype=float)
        baselines[manifold] = {
            dim: _slice_dim(diag, dim, max_edge)
            for dim in range(MAX_HOM_DIM[manifold] + 1)
        }
    return baselines


def _slice_dim(diagram: np.ndarray, dim: int, max_edge: float) -> np.ndarray:
    """(birth, death) pairs of one homology dimension, inf deaths clamped to max_edge.

    Phase-3 diagrams (outputs/persistence_diagrams/<m>_<split>.pkl) are numeric
    (n, 3) float ndarrays with rows [dim, birth, death] (pipeline.py line 534).
    Inf deaths are CLAMPED to max_edge, matching pipeline.py:diagram_dim_slice.
    """
    if diagram.size == 0:
        return np.empty((0, 2))
    bd = diagram[diagram[:, 0] == dim][:, 1:3].copy()
    bd[~np.isfinite(bd[:, 1]), 1] = max_edge
    return bd


def _w2_subprocess_target(queue, d1, d2, order, internal_p, delta) -> None:  # noqa: PLR0913
    """Child-process entry point used by ``_w2_with_timeout``.

    Ported from ``tools/quick_unsup_probe.py:_hera_subprocess_target``
    (lines 125-140 of the recovered original). Renamed to
    ``_w2_subprocess_target`` and given a queue-first signature per this
    plan's interface. Imported inside the function so this module stays
    importable without gudhi where it isn't needed.
    """
    try:
        from gudhi.hera import wasserstein_distance  # noqa: PLC0415

        result = wasserstein_distance(
            d1, d2, order=order, internal_p=internal_p, delta=delta,
        )
        queue.put(float(result))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", repr(exc)))


def _w2_with_timeout(
    d1: np.ndarray,
    d2: np.ndarray,
    order: float,
    internal_p: float,
    delta: float,
    timeout_sec: float,
) -> float:
    """Hera W_2 wrapped in a fork+terminate timeout.

    Ported from ``tools/quick_unsup_probe.py:_w2_with_timeout`` (lines
    143-186 of the recovered original). Pathological persistence diagrams
    can cause hera's auction LP to loop for minutes without releasing the
    GIL, so signal.alarm-based interruption does not work (verified
    empirically against gudhi.hera 3.x). Each W_2 call runs in a forked
    subprocess; if wall time exceeds ``timeout_sec``, the child is killed
    and NaN is returned. Callers treat NaN as a zero contribution to the
    cross-dim sum and increment a timeout counter.

    The subprocess target is resolved via the module attribute
    (``probe._w2_subprocess_target``) at call time rather than captured as
    a local/global reference, so tests can ``monkeypatch.setattr(probe,
    "_w2_subprocess_target", ...)`` and have the fork observe the patched
    callable.
    """
    import multiprocessing as mp  # noqa: PLC0415

    import uav_tda.probe as _probe_module  # noqa: PLC0415

    ctx = mp.get_context("fork")
    q: "mp.Queue" = ctx.Queue()
    p = ctx.Process(
        target=_probe_module._w2_subprocess_target,
        args=(q, d1, d2, order, internal_p, delta),
        daemon=True,
    )
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(0.5)
        if p.is_alive():
            p.kill()
            p.join(0.5)
        return float("nan")
    try:
        value = q.get(timeout=0.5)
    except Exception:  # noqa: BLE001
        return float("nan")
    if isinstance(value, tuple) and value and value[0] == "error":
        return float("nan")
    return float(value)


def _load_test_style_inputs(max_edge_lengths: dict, split: str) -> tuple:
    """Load per-manifold diagrams (pickled) and labels for a split."""
    labels = pd.read_csv(OUTPUTS_DIR / f"labels_{split}.csv")["label"].to_numpy()
    per_manifold_diagrams = {
        m: pickle.loads((PERSISTENCE_DIR / f"{m}_{split}.pkl").read_bytes())
        for m in MANIFOLDS
    }
    return labels, per_manifold_diagrams


def _probe_distances(
    baselines: dict,
    max_edge_lengths: dict,
    idx: np.ndarray,
    labels: np.ndarray,
    per_manifold_diagrams: dict,
    top_k: int,
    delta: float,
    w2_timeout: float | None,
    stats_out: dict | None,
) -> pd.DataFrame:
    """Compute per-flow, per-manifold W2 distances to the baselines for ``idx``.

    Shared core used by both the test-split sampled pass and the val-split
    Normal-only pass in ``run_probe_with_znorm``, so both passes use the
    SAME in-memory baseline barcode realization.
    """
    from gudhi.hera import wasserstein_distance as wdist

    timeout_counters = {m: 0 for m in MANIFOLDS}
    records = []
    for i in idx:
        row = {"test_idx": int(i), "label": labels[i]}
        for m in MANIFOLDS:
            max_edge = max_edge_lengths[m]
            diag = per_manifold_diagrams[m][i]   # (n, 3) ndarray [dim, birth, death]
            total = 0.0
            for dim in range(MAX_HOM_DIM[m] + 1):
                flow_k = truncate_top_k(_slice_dim(diag, dim, max_edge), top_k)
                base_k = truncate_top_k(baselines[m][dim], top_k)
                if w2_timeout is None:
                    w = float(wdist(flow_k, base_k, order=2.0, internal_p=2.0, delta=delta))
                else:
                    w = _w2_with_timeout(
                        flow_k, base_k, order=2.0, internal_p=2.0,
                        delta=delta, timeout_sec=w2_timeout,
                    )
                    if not np.isfinite(w):
                        timeout_counters[m] += 1
                        w = 0.0
                total += w
            row[f"W2_{m}"] = total
        records.append(row)

    df = pd.DataFrame(records)
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)

    if stats_out is not None:
        for m in MANIFOLDS:
            stats_out[f"n_timeouts_{m}"] = (
                stats_out.get(f"n_timeouts_{m}", 0) + timeout_counters[m]
            )
    return df


def run_probe(seed: int, per_class: int | None = PROBE_PER_CLASS,
              top_k: int = PROBE_TOP_K, delta: float = PROBE_DELTA,
              w2_timeout: float | None = None, n_jobs: int = -1,
              split: str = "test", class_filter: str | None = None,
              stats_out: dict | None = None) -> pd.DataFrame:
    """Sample (or take all of) one or more classes from ``split`` and score them.

    Default arguments reproduce the original behavior exactly: sample
    ``per_class`` flows (without replacement) from EVERY class present in
    ``labels_test.csv``, per seed. When ``class_filter`` is given, only that
    class's flows are considered; ``per_class=None`` then means "take ALL of
    that class's flows" (order preserved, no RNG draw). With
    ``class_filter=None``, ``per_class`` must be an int (today's sampling
    semantics).

    ``w2_timeout=None`` bypasses the fork wrapper entirely (zero overhead) —
    this is the default and preserves prior behavior byte-for-byte.
    """
    import json

    max_edge_lengths = json.loads((OUTPUTS_DIR / "max_edge_lengths.json").read_text())
    baselines = _load_baseline_barcodes(max_edge_lengths)
    labels, per_manifold_diagrams = _load_test_style_inputs(max_edge_lengths, split)

    if class_filter is None:
        if per_class is None:
            raise ValueError("per_class must be an int when class_filter is None")
        rng = np.random.default_rng(seed)
        idx = np.concatenate([
            rng.choice(np.where(labels == cls)[0], size=per_class, replace=False)
            for cls in np.unique(labels)
        ])
    else:
        class_idx = np.where(labels == class_filter)[0]
        if per_class is None:
            idx = class_idx
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(class_idx, size=per_class, replace=False)

    return _probe_distances(
        baselines, max_edge_lengths, idx, labels, per_manifold_diagrams,
        top_k, delta, w2_timeout, stats_out,
    )


def run_probe_with_znorm(
    seed: int, per_class: int = 200, top_k: int = 50, delta: float = 0.2,
    w2_timeout: float | None = None, n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Coupled test+val probe run for §III.E Z-normalized subset scoring.

    Loads the baseline barcodes ONCE (a single sparse-Rips realization on
    the 500 reference points) and reuses that same in-memory ``baselines``
    dict for BOTH:
      - the val-split pass: ALL 3,926 validation "Normal Traffic" flows
        (``split="val"``, ``class_filter="Normal Traffic"``, ``per_class=None``);
      - the test-split pass: the seeded stratified sample (today's
        ``run_probe`` default semantics, ``split="test"``).

    This same-run coupling is the point: the Z-norm stats derived from the
    val pass (via ``metrics.znorm_stats_from_val``) must be computed against
    distances produced by the SAME baseline barcode realization as the test
    distances they are later applied to, since the sparse-Rips baseline
    complex on the 500 reference points is randomized independently per
    ``_load_baseline_barcodes`` call (no seed control over gudhi's sparse
    approximation). Calling ``run_probe`` twice independently would risk
    scoring test flows against a different baseline than the one the val
    Normal-traffic statistics were computed from.

    Returns ``(test_df, val_df, stats, timeout_counts)`` where ``stats`` is
    ``metrics.znorm_stats_from_val(val_df)`` and ``timeout_counts`` merges
    the ``stats_out`` counters from both passes (keys
    ``n_timeouts_<manifold>``).
    """
    import json

    from .metrics import znorm_stats_from_val

    max_edge_lengths = json.loads((OUTPUTS_DIR / "max_edge_lengths.json").read_text())
    baselines = _load_baseline_barcodes(max_edge_lengths)

    timeout_counts: dict = {}

    val_labels, val_diagrams = _load_test_style_inputs(max_edge_lengths, "val")
    val_idx = np.where(val_labels == "Normal Traffic")[0]
    val_df = _probe_distances(
        baselines, max_edge_lengths, val_idx, val_labels, val_diagrams,
        top_k, delta, w2_timeout, timeout_counts,
    )

    test_labels, test_diagrams = _load_test_style_inputs(max_edge_lengths, "test")
    rng = np.random.default_rng(seed)
    test_idx = np.concatenate([
        rng.choice(np.where(test_labels == cls)[0], size=per_class, replace=False)
        for cls in np.unique(test_labels)
    ])
    test_df = _probe_distances(
        baselines, max_edge_lengths, test_idx, test_labels, test_diagrams,
        top_k, delta, w2_timeout, timeout_counts,
    )

    stats = znorm_stats_from_val(val_df)
    return test_df, val_df, stats, timeout_counts
