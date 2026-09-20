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


def run_probe(seed: int, per_class: int = PROBE_PER_CLASS, top_k: int = PROBE_TOP_K,
              delta: float = PROBE_DELTA, w2_timeout: float | None = None,
              n_jobs: int = -1) -> pd.DataFrame:
    import json

    from gudhi.hera import wasserstein_distance as wdist

    max_edge_lengths = json.loads((OUTPUTS_DIR / "max_edge_lengths.json").read_text())
    baselines = _load_baseline_barcodes(max_edge_lengths)
    labels = pd.read_csv(OUTPUTS_DIR / "labels_test.csv")["label"].to_numpy()

    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(np.where(labels == cls)[0], size=per_class, replace=False)
        for cls in np.unique(labels)
    ])

    per_manifold_diagrams = {
        m: pickle.loads((PERSISTENCE_DIR / f"{m}_test.pkl").read_bytes())
        for m in MANIFOLDS
    }

    # NOTE: w2_timeout is accepted for CLI parity with the original probe but is
    # NOT enforced in this plan (it only affected the seed-123 C2 artifact). It is
    # wired in Phase 3. Passing a value here is a documented no-op for now.
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
                total += float(wdist(flow_k, base_k, order=2.0, internal_p=2.0, delta=delta))
            row[f"W2_{m}"] = total
        records.append(row)

    df = pd.DataFrame(records)
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)
    return df
