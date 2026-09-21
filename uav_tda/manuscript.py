"""Manuscript-grade statistics: ten-seed cluster bootstrap AUC CIs.

Phase 4 adds a bootstrap-CI treatment of the across-seed mean AUC for the
manuscript's ten-seed evaluation (see config.MANUSCRIPT_SEEDS). Resampling
is stratified per class label so that a replicate never drops a minority
class and roc_auc_score never fails with a single-class ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import metrics
from .metrics import ATTACK_CLASSES, MANIFOLD_SUBSETS, NORMAL
from .paths import TABLES_DIR
from .provenance import write_provenance


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


# --- Ten-seed campaign orchestration -----------------------------------------
# Reuses the Phase-3 coupled test+val Z-normalized probe (probe.run_probe_with_znorm)
# and its artifact-writing routine, shared here so both the `uav-tda probe --znorm`
# single-seed CLI path and this multi-seed campaign path write identical artifacts.

def _rebuild_paths(seed: int, rebuild_dir: Path | None = None) -> dict[str, Path]:
    rebuild = rebuild_dir if rebuild_dir is not None else (TABLES_DIR / "rebuild")
    return {
        "raw": rebuild / f"probe_distances_seed{seed}.csv",
        "znorm": rebuild / f"probe_distances_seed{seed}_znorm.csv",
        "val": rebuild / f"val_normal_distances_seed{seed}.csv",
        "stats": rebuild / f"znorm_stats_seed{seed}.json",
    }


def run_znorm_probe_and_write(
    seed: int, w2_timeout: float | None,
    per_class: int = 200, top_k: int = 50, delta: float = 0.2,
    rebuild_dir: Path | None = None,
) -> tuple[dict[str, Path], pd.DataFrame, dict]:
    """Run the coupled test+val znorm probe for one seed and write rebuild/ artifacts.

    Shared by the `uav-tda probe --znorm` single-seed CLI path and the
    ten-seed manuscript campaign (`run_missing_seeds`) so both write
    byte-identical artifact layouts. Returns (paths, test_df, stats) where
    `paths` is the dict of written file paths, `test_df` is the raw
    (un-normalized) test probe dataframe, and `stats` is
    `{manifold: (mean, std)}` from the coupled validation pass.
    """
    from . import probe  # local import: avoids a probe<->manuscript import cycle

    paths = _rebuild_paths(seed, rebuild_dir)
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)

    test_df, val_df, stats, timeout_counts = probe.run_probe_with_znorm(
        seed=seed, per_class=per_class, top_k=top_k, delta=delta,
        w2_timeout=w2_timeout,
    )
    znorm_df = metrics.apply_znorm(test_df, stats)

    params = {
        "seed": seed, "per_class": per_class, "top_k": top_k,
        "delta": delta, "w2_timeout": w2_timeout,
    }

    test_df.to_csv(paths["raw"], index=False)
    write_provenance(paths["raw"], params)

    znorm_df.to_csv(paths["znorm"], index=False)
    write_provenance(paths["znorm"], params)

    val_df.to_csv(paths["val"], index=False)
    write_provenance(paths["val"], params)

    stats_payload = {
        "stats": {m: {"mean": mean, "std": std} for m, (mean, std) in stats.items()},
        "timeout_counts": timeout_counts,
        "params": params,
    }
    paths["stats"].write_text(json.dumps(stats_payload, indent=2, sort_keys=True))
    write_provenance(paths["stats"], params)

    return paths, test_df, stats


def missing_seeds(seeds: tuple[int, ...], rebuild_dir: Path) -> list[int]:
    """Seeds among `seeds` lacking a probe-distances CSV or znorm-stats JSON."""
    missing = []
    for seed in seeds:
        raw_csv = rebuild_dir / f"probe_distances_seed{seed}.csv"
        stats_json = rebuild_dir / f"znorm_stats_seed{seed}.json"
        if not raw_csv.exists() or not stats_json.exists():
            missing.append(seed)
    return missing


def run_missing_seeds(seeds: tuple[int, ...], w2_timeout: float = 30.0) -> None:
    """Run the Phase-3 znorm probe for each seed in `seeds` lacking artifacts.

    Writes into the default `results/tables/rebuild/` location via
    `run_znorm_probe_and_write` (the same writer the `uav-tda probe --znorm`
    CLI path uses), so already-present seeds are left untouched.
    """
    rebuild_dir = TABLES_DIR / "rebuild"
    for seed in missing_seeds(seeds, rebuild_dir):
        run_znorm_probe_and_write(seed=seed, w2_timeout=w2_timeout, rebuild_dir=rebuild_dir)


def load_seed_frames(
    seeds: tuple[int, ...], rebuild_dir: Path,
) -> dict[int, tuple[pd.DataFrame, dict]]:
    """Load per-seed raw test probe df + znorm stats ({manifold: (mean, std)})."""
    out: dict[int, tuple[pd.DataFrame, dict]] = {}
    for seed in seeds:
        raw_csv = rebuild_dir / f"probe_distances_seed{seed}.csv"
        stats_json = rebuild_dir / f"znorm_stats_seed{seed}.json"
        df = pd.read_csv(raw_csv)
        payload = json.loads(stats_json.read_text())
        stats = {m: (v["mean"], v["std"]) for m, v in payload["stats"].items()}
        out[seed] = (df, stats)
    return out


def build_binary_auc_table(
    seed_frames: dict[int, tuple[pd.DataFrame, dict]],
    B: int = 2000, bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap-CI mean binary (Normal-vs-attack) AUC per subset x scoring.

    Rows: subset (the 7 `metrics.MANIFOLD_SUBSETS` keys) x scoring in
    {raw, znorm}. Mean/std are taken across per-seed AUCs (std ddof=1, the
    pandas default, matching Phase 3's `metrics.aggregate_over_seeds`). CI
    is `bootstrap_mean_auc_ci` over the per-seed (y_binary, scores) pairs,
    stratified per-seed-resample on the seed's multiclass labels; znorm rows
    use `metrics.apply_znorm(df, stats)` (that seed's own stats) to derive
    the subset score.
    """
    rows = []
    for subset in MANIFOLD_SUBSETS:
        for scoring in ("raw", "znorm"):
            per_seed = []
            labels_per_seed = []
            aucs = []
            for seed, (df, stats) in seed_frames.items():
                score_df = metrics.apply_znorm(df, stats) if scoring == "znorm" else df
                y = (df["label"] != NORMAL).astype(int).to_numpy()
                scores = score_df[f"W2_{subset}"].to_numpy()
                per_seed.append((y, scores))
                labels_per_seed.append(df["label"].to_numpy())
                aucs.append(float(roc_auc_score(y, scores)))

            aucs_arr = np.asarray(aucs, dtype=float)
            mean = float(aucs_arr.mean())
            std = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_mean_auc_ci(
                per_seed, B=B, bootstrap_seed=bootstrap_seed, labels_per_seed=labels_per_seed)
            rows.append({
                "subset": subset, "scoring": scoring,
                "mean": mean, "std": std,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n_seeds": len(seed_frames),
            })
    return pd.DataFrame(rows)


def build_attribution_table(
    seed_frames: dict[int, tuple[pd.DataFrame, dict]],
    B: int = 2000, bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap-CI mean one-vs-rest AUC per (attack class, manifold).

    Rows: attack x manifold, raw scores only (per-manifold W2 distance is
    unmodified by Z-normalization's affine per-manifold rescaling, so
    per-manifold AUC is identical raw vs. znorm — a monotone, strictly
    increasing transform of a single column never changes ROC AUC). `mean`
    is the across-seed mean of per-seed `metrics.per_attack_auc` values
    (std ddof=1); `ci_lo`/`ci_hi` via `bootstrap_mean_auc_ci` with
    y = (label == attack), stratified on each seed's multiclass labels.
    `dominant` is True for exactly the manifold with the largest `mean`
    AUC within each attack class.
    """
    manifolds = list(metrics.MANIFOLDS)
    per_attack_per_manifold: dict[tuple[str, str], list[float]] = {}
    ci_by_key: dict[tuple[str, str], tuple[float, float]] = {}

    for attack in ATTACK_CLASSES:
        for m in manifolds:
            per_seed = []
            labels_per_seed = []
            aucs = []
            for seed, (df, _stats) in seed_frames.items():
                y = (df["label"].to_numpy() == attack).astype(int)
                scores = df[f"W2_{m}"].to_numpy()
                per_seed.append((y, scores))
                labels_per_seed.append(df["label"].to_numpy())
                aucs.append(float(roc_auc_score(y, scores)))
            per_attack_per_manifold[(attack, m)] = aucs
            ci_by_key[(attack, m)] = bootstrap_mean_auc_ci(
                per_seed, B=B, bootstrap_seed=bootstrap_seed, labels_per_seed=labels_per_seed)

    rows = []
    for attack in ATTACK_CLASSES:
        means = {}
        for m in manifolds:
            aucs_arr = np.asarray(per_attack_per_manifold[(attack, m)], dtype=float)
            means[m] = float(aucs_arr.mean())
        dominant_manifold = max(means, key=means.get)
        for m in manifolds:
            aucs_arr = np.asarray(per_attack_per_manifold[(attack, m)], dtype=float)
            std = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
            ci_lo, ci_hi = ci_by_key[(attack, m)]
            rows.append({
                "attack_class": attack, "manifold": m,
                "mean": means[m], "std": std,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "dominant": m == dominant_manifold,
            })
    return pd.DataFrame(rows)
