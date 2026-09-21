"""Recovered anomaly-detection metrics: probe-distance dataframe -> AUCs.

Definitions match the paper (§IV) and the surviving probe CSVs:
- combined/subset score is the SUM of per-manifold Wasserstein-2 distances;
- per-attack AUC is one-versus-REST against all other classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import EXPECTED_CLASSES, MANIFOLDS

NORMAL = "Normal Traffic"
ATTACK_CLASSES = tuple(c for c in EXPECTED_CLASSES if c != NORMAL)

MANIFOLD_SUBSETS: dict[str, tuple[str, ...]] = {
    "c2_only": ("c2",),
    "network_only": ("network",),
    "physical_only": ("physical",),
    "c2_network": ("c2", "network"),
    "c2_physical": ("c2", "physical"),
    "network_physical": ("network", "physical"),
    "all_three": ("c2", "network", "physical"),
}


def _subset_score(df: pd.DataFrame, manifolds: tuple[str, ...]) -> np.ndarray:
    """Summed Wasserstein-2 distance across the given manifolds."""
    return np.sum([df[f"W2_{m}"].to_numpy() for m in manifolds], axis=0)


def binary_auc_by_subset(df: pd.DataFrame) -> dict[str, float]:
    """Normal-vs-any-attack AUC for each of the 7 manifold subsets."""
    is_attack = (df["label"] != NORMAL).astype(int).to_numpy()
    out: dict[str, float] = {}
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        score = _subset_score(df, manifolds)
        out[subset] = float(roc_auc_score(is_attack, score))
    return out


def per_attack_auc(df: pd.DataFrame) -> pd.DataFrame:
    """One-vs-rest AUC for each (attack class, single manifold)."""
    label = df["label"].to_numpy()
    rows = []
    for attack in ATTACK_CLASSES:
        y = (label == attack).astype(int)
        for m in MANIFOLDS:
            score = df[f"W2_{m}"].to_numpy()
            rows.append({"attack_class": attack, "manifold": m,
                         "auc": float(roc_auc_score(y, score))})
    return pd.DataFrame(rows)


def aggregate_over_seeds(dfs: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Mean/std of binary-subset and per-attack AUCs across seeds."""
    bin_rows, attack_rows = [], []
    for seed, df in dfs.items():
        for subset, auc in binary_auc_by_subset(df).items():
            bin_rows.append({"seed": seed, "subset": subset, "auc": auc})
        pa = per_attack_auc(df)
        pa["seed"] = seed
        attack_rows.append(pa)
    bin_df = pd.DataFrame(bin_rows)
    binary = (bin_df.groupby("subset")["auc"]
              .agg(["mean", "std"]).reset_index())
    per = (pd.concat(attack_rows).groupby(["attack_class", "manifold"])["auc"]
           .agg(["mean", "std"]).reset_index())
    return {"binary": binary, "per_attack": per}


# --- Phase 3: Z-normalized scoring (paper §III.E / §V correction) -----------
# Canonical forward-looking scoring: per-manifold W2, Z-normalized on held-out
# VALIDATION Normal-Traffic distances, then summed across manifolds. The raw
# unweighted-sum functions above are frozen (golden-master-locked to the
# published extended-abstract numbers) and must not change.
# NOTE: stats and the distances they normalize must come from the SAME probe
# run (same sparse-Rips baseline realization) — see run_probe_with_znorm.

import logging

_log = logging.getLogger("uav_tda.metrics")


def znorm_stats_from_val(val_df: pd.DataFrame) -> dict:
    """Per-manifold (mean, std) of W2 over validation Normal-Traffic rows."""
    normal = val_df[val_df["label"] == NORMAL]
    stats: dict = {}
    for m in MANIFOLDS:
        x = normal[f"W2_{m}"].to_numpy(dtype=float)
        mean, std = float(np.mean(x)), float(np.std(x))
        if std < 1e-12:
            _log.warning("znorm: manifold %s has degenerate std=%.3g; using 1.0", m, std)
            std = 1.0
        stats[m] = (mean, std)
    return stats


def apply_znorm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Copy of df with manifold W2 columns z-scored and subset sums recomputed."""
    out = df.copy()
    for m in MANIFOLDS:
        mean, std = stats[m]
        out[f"W2_{m}"] = (out[f"W2_{m}"].astype(float) - mean) / std
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        out[f"W2_{subset}"] = sum(out[f"W2_{m}"] for m in manifolds)
    return out


def binary_auc_by_subset_znorm(df: pd.DataFrame, stats: dict) -> dict:
    """Normal-vs-attack AUC per subset on z-normalized scores."""
    return binary_auc_by_subset(apply_znorm(df, stats))
