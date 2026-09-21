"""Manuscript-grade statistics: ten-seed cluster bootstrap AUC CIs.

Phase 4 adds a bootstrap-CI treatment of the across-seed mean AUC for the
manuscript's ten-seed evaluation (see config.MANUSCRIPT_SEEDS). Resampling
is stratified per class label so that a replicate never drops a minority
class and roc_auc_score never fails with a single-class ValueError.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def stratified_resample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample indices with replacement, stratified per unique label.

    For each unique label, draws that class's count of indices (with
    replacement) from that class's positions in `labels`, then concatenates
    across classes. The returned indices index into the original rows.
    """
    labels = np.asarray(labels)
    idx_parts = []
    for label in np.unique(labels):
        class_positions = np.flatnonzero(labels == label)
        draw = rng.choice(class_positions, size=len(class_positions), replace=True)
        idx_parts.append(draw)
    return np.concatenate(idx_parts)


def bootstrap_mean_auc_ci(
    per_seed: list[tuple[np.ndarray, np.ndarray]],
    B: int = 2000,
    bootstrap_seed: int = 0,
    labels_per_seed: list[np.ndarray] | None = None,
) -> tuple[float, float]:
    """Cluster bootstrap CI for the across-seed mean AUC.

    `per_seed` is a list of (y_binary, scores) pairs, one per seed.
    `labels_per_seed[i]`, if given, carries the multiclass labels used for
    stratified resampling of seed i; when None, resampling is stratified on
    the binary labels `y_binary`.

    Each of B replicates: for every seed, stratified-resample that seed's
    rows, compute roc_auc_score on the resample, then average across seeds.
    Returns the 2.5th/97.5th percentiles of the B replicate means.
    """
    rng = np.random.default_rng(bootstrap_seed)
    replicate_means = np.empty(B, dtype=float)

    for b in range(B):
        seed_aucs = np.empty(len(per_seed), dtype=float)
        for i, (y_binary, scores) in enumerate(per_seed):
            strat_labels = labels_per_seed[i] if labels_per_seed is not None else y_binary
            idx = stratified_resample_indices(strat_labels, rng)
            seed_aucs[i] = roc_auc_score(y_binary[idx], scores[idx])
        replicate_means[b] = seed_aucs.mean()

    ci_lo, ci_hi = np.percentile(replicate_means, [2.5, 97.5])
    return float(ci_lo), float(ci_hi)
