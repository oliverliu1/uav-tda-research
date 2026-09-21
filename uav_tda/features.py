"""Phase 4 — feature extraction, ported verbatim from pipeline.py SECTION 8.

Converts per-flow persistence diagrams into two parallel feature blocks:
  1. Summary statistics — 8 scalars per (manifold, dim).
  2. Persistence images — 20x20 flattened per (manifold, dim), using
     persim's PersistenceImager. The imager is fit on training-set
     diagrams only and reused for val/test.
Both blocks plus the original (scaled) features are concatenated into a
"combined" frame for the supervised ablation. Columns that are entirely
zero across the training set are dropped (and the same columns dropped
from val/test). The kept-column lists are saved to feature_names.json.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.features")


def load_diagrams_pkl(ws: Workspace, manifold: str, split: str) -> list:
    """Load the per-flow persistence-diagram pickle written by Phase 3."""
    import pickle

    path = ws.persistence_dir / f"{manifold}_{split}.pkl"
    with path.open("rb") as fh:
        return pickle.load(fh)


def diagram_dim_slice(diagram, dim: int, max_edge: float):
    """Return (n, 2) birth/death array for one H-dim; replace inf deaths with max_edge."""
    import numpy as np

    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    mask = diagram[:, 0].astype(int) == dim
    bd = diagram[mask, 1:3].astype(float)
    if bd.size:
        inf_rows = np.isinf(bd[:, 1])
        bd[inf_rows, 1] = max_edge
    return bd


def summary_stats_row(bd) -> dict:
    """Return the 8 summary statistics for one (birth, death) array."""
    import numpy as np

    if len(bd) == 0:
        return {name: 0.0 for name in config.SUMMARY_STAT_NAMES}
    persistence = bd[:, 1] - bd[:, 0]
    total = float(persistence.sum())
    if total > 0:
        probs = persistence / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0
    return {
        "count": float(len(bd)),
        "mean_persistence": float(persistence.mean()),
        "std_persistence": float(persistence.std()),
        "max_persistence": float(persistence.max()),
        "total_persistence": total,
        "mean_birth": float(bd[:, 0].mean()),
        "mean_death": float(bd[:, 1].mean()),
        "persistence_entropy": entropy,
    }


def configure_square_imager(imager, resolution: tuple) -> None:
    """Force a fitted PersistenceImager to a square `resolution[0]`-side grid.

    persim's PersistenceImager has a scalar pixel_size, so the natural grid
    is rectangular when birth_range and pers_range widths differ. Forcing
    both ranges to the union covers all data with a true 20x20 image whose
    empty pixels become all-zero columns and are dropped downstream.
    """
    b_min, b_max = imager.birth_range
    p_min, p_max = imager.pers_range
    lo = min(b_min, p_min)
    hi = max(b_max, p_max)
    if hi <= lo:
        hi = lo + 1.0  # degenerate range; pick something non-zero
    imager.birth_range = (lo, hi)
    imager.pers_range = (lo, hi)
    imager.pixel_size = (hi - lo) / resolution[0]


def fit_imagers(train_diagrams_by_manifold: dict) -> dict:
    """Fit one PersistenceImager per (manifold, dim) on training diagrams only."""
    from persim import PersistenceImager

    imagers: dict = {}
    for manifold, per_dim_bd in train_diagrams_by_manifold.items():
        for dim, all_bd_arrays in per_dim_bd.items():
            non_empty = [arr for arr in all_bd_arrays if len(arr) > 0]
            imager = PersistenceImager()
            if non_empty:
                imager.fit(non_empty)
            else:
                imager.birth_range = (0.0, 1.0)
                imager.pers_range = (0.0, 1.0)
            configure_square_imager(imager, config.PI_RESOLUTION)
            imagers[(manifold, dim)] = imager
    return imagers


def collect_diagrams_by_dim(ws: Workspace, manifold: str, split: str, max_edge: float, max_hom_dim: int):
    """Return {dim: list[per-flow (birth, death) arrays]} for one manifold-split."""
    diagrams = load_diagrams_pkl(ws, manifold, split)
    by_dim: dict = {k: [] for k in range(max_hom_dim + 1)}
    for diag in diagrams:
        for dim in by_dim:
            by_dim[dim].append(diagram_dim_slice(diag, dim, max_edge))
    return by_dim


def build_summary_columns(manifold: str, dim: int, bd_arrays: list) -> dict:
    """Return {column_name: list[float] (n_flows)} of summary stats for (manifold, dim)."""
    rows = [summary_stats_row(bd) for bd in bd_arrays]
    return {
        f"{manifold}_H{dim}_{stat}": [row[stat] for row in rows]
        for stat in config.SUMMARY_STAT_NAMES
    }


def build_image_columns(manifold: str, dim: int, bd_arrays: list, imager) -> dict:
    """Return {column_name: list[float]} for the flattened 20x20 persistence image."""
    import numpy as np

    target_len = config.PI_RESOLUTION[0] * config.PI_RESOLUTION[1]
    flat_images: list = []
    for bd in bd_arrays:
        if len(bd) == 0:
            flat_images.append(np.zeros(target_len, dtype=float))
            continue
        img = np.asarray(imager.transform(bd), dtype=float)
        flat = img.flatten()
        if flat.size != target_len:
            # imager may produce a slightly off-by-one grid for degenerate ranges
            padded = np.zeros(target_len, dtype=float)
            padded[: min(flat.size, target_len)] = flat[: min(flat.size, target_len)]
            flat = padded
        flat_images.append(flat)
    stacked = np.vstack(flat_images)
    return {
        f"{manifold}_H{dim}_img_{i:03d}": stacked[:, i].tolist()
        for i in range(target_len)
    }


def build_split_frames(
    ws: Workspace, split: str, max_edge_lengths: dict, imagers: dict,
) -> tuple:
    """Build (summary_df, images_df) for one split across all manifolds and dims."""
    import pandas as pd

    summary_cols: dict = {}
    image_cols: dict = {}
    for manifold in config.MANIFOLDS:
        max_edge = max_edge_lengths[manifold]
        max_hom = config.MAX_HOM_DIM[manifold]
        by_dim = collect_diagrams_by_dim(ws, manifold, split, max_edge, max_hom)
        for dim, bd_arrays in by_dim.items():
            summary_cols.update(build_summary_columns(manifold, dim, bd_arrays))
            image_cols.update(
                build_image_columns(manifold, dim, bd_arrays, imagers[(manifold, dim)])
            )
    return pd.DataFrame(summary_cols), pd.DataFrame(image_cols)


def collect_train_diagrams_for_imagers(ws: Workspace, max_edge_lengths: dict) -> dict:
    """Reload training diagrams once, organised as {manifold: {dim: [bd_arrays...]}}."""
    out: dict = {}
    for manifold in config.MANIFOLDS:
        max_edge = max_edge_lengths[manifold]
        max_hom = config.MAX_HOM_DIM[manifold]
        out[manifold] = collect_diagrams_by_dim(ws, manifold, "train", max_edge, max_hom)
    return out


def all_zero_train_columns(train_df) -> list:
    """Return columns whose training values are all zero (to be dropped from every split)."""
    return [col for col in train_df.columns if (train_df[col] == 0).all()]


def validate_no_nan(*frames, names: list) -> None:
    """Raise if any frame contains NaN; otherwise no-op."""
    for name, frame in zip(names, frames):
        if frame.isna().any().any():
            bad = frame.columns[frame.isna().any()].tolist()
            raise AssertionError(f"{name} contains NaN in: {bad[:10]}")


def validate_feature_counts(summary_df, images_df, expected_pairs: int) -> None:
    """Raise unless raw summary/image column counts match the (mfd, dim) formula."""
    summary_expected = expected_pairs * len(config.SUMMARY_STAT_NAMES)
    image_expected = expected_pairs * config.PI_RESOLUTION[0] * config.PI_RESOLUTION[1]
    if summary_df.shape[1] != summary_expected:
        raise AssertionError(
            f"summary cols {summary_df.shape[1]} != expected {summary_expected}"
        )
    if images_df.shape[1] != image_expected:
        raise AssertionError(
            f"image cols {images_df.shape[1]} != expected {image_expected}"
        )


def manifold_for_column(col: str) -> str:
    """Return the manifold prefix from a feature column name, or 'original'."""
    for manifold in config.MANIFOLDS:
        if col.startswith(f"{manifold}_H"):
            return manifold
    return "original"


def report_dropped_columns(dropped: list) -> dict:
    """Bucket dropped columns by manifold and return counts per manifold."""
    counts: dict = {m: 0 for m in config.MANIFOLDS}
    counts["original"] = 0
    for col in dropped:
        counts[manifold_for_column(col)] += 1
    return counts


def fit_and_save_imagers(ws: Workspace, max_edge_lengths: dict) -> dict:
    """Fit one imager per (manifold, dim) on training diagrams and pickle the dict."""
    import pickle

    log.info("fitting persistence imagers on training diagrams")
    train_by_md = collect_train_diagrams_for_imagers(ws, max_edge_lengths)
    imagers = fit_imagers(train_by_md)
    with (ws.outputs_dir / "persistence_imagers.pkl").open("wb") as fh:
        pickle.dump(imagers, fh)
    return imagers


def build_all_splits(ws: Workspace, max_edge_lengths: dict, imagers: dict) -> tuple:
    """Return (per_split_summary, per_split_images) dicts keyed by split name."""
    expected_pairs = sum(config.MAX_HOM_DIM[m] + 1 for m in config.MANIFOLDS)
    per_split_summary: dict = {}
    per_split_images: dict = {}
    for split in ("train", "val", "test"):
        log.info("building features for split=%s", split)
        summary_df, images_df = build_split_frames(ws, split, max_edge_lengths, imagers)
        validate_feature_counts(summary_df, images_df, expected_pairs)
        per_split_summary[split] = summary_df
        per_split_images[split] = images_df
    return per_split_summary, per_split_images


def determine_drops(per_split_summary: dict, per_split_images: dict) -> tuple:
    """Pick all-zero training columns; log dropped breakdown per manifold."""
    drop_summary = all_zero_train_columns(per_split_summary["train"])
    drop_images = all_zero_train_columns(per_split_images["train"])
    log.info(
        "dropping all-zero training columns: summary=%d images=%d",
        len(drop_summary), len(drop_images),
    )
    log.info(
        "dropped breakdown: summary=%s images=%s",
        report_dropped_columns(drop_summary),
        report_dropped_columns(drop_images),
    )
    return drop_summary, drop_images


def save_features_for_split(
    ws: Workspace, split: str, summary_df, images_df, drop_summary: list, drop_images: list,
) -> tuple:
    """Drop, validate, and write the three CSVs for one split; return per-frame shapes."""
    import pandas as pd

    summary = summary_df.drop(columns=drop_summary)
    images = images_df.drop(columns=drop_images)
    original = pd.read_csv(ws.outputs_dir / f"original_features_{split}.csv")
    validate_no_nan(summary, images, original, names=["summary", "images", "original"])
    summary.to_csv(ws.tda_features_dir / f"summary_{split}.csv", index=False)
    images.to_csv(ws.tda_features_dir / f"images_{split}.csv", index=False)
    combined = pd.concat([original, summary, images], axis=1)
    combined.to_csv(ws.tda_features_dir / f"combined_{split}.csv", index=False)
    return summary.shape[1], images.shape[1], combined.shape[1]


def run_features(ws: Workspace, debug: bool = False) -> None:
    """Phase 4: extract summary stats and persistence images from Phase 3 diagrams."""
    import json

    ws.tda_features_dir.mkdir(parents=True, exist_ok=True)

    with (ws.outputs_dir / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)

    imagers = fit_and_save_imagers(ws, max_edge_lengths)
    per_split_summary, per_split_images = build_all_splits(ws, max_edge_lengths, imagers)
    drop_summary, drop_images = determine_drops(per_split_summary, per_split_images)

    feature_names = {
        "summary": [c for c in per_split_summary["train"].columns if c not in drop_summary],
        "images": [c for c in per_split_images["train"].columns if c not in drop_images],
    }
    with (ws.tda_features_dir / "feature_names.json").open("w") as fh:
        json.dump(feature_names, fh, indent=2)

    for split in ("train", "val", "test"):
        n_summary, n_images, n_combined = save_features_for_split(
            ws, split, per_split_summary[split], per_split_images[split],
            drop_summary, drop_images,
        )
        log.info(
            "saved split=%s: summary=%d images=%d combined=%d",
            split, n_summary, n_images, n_combined,
        )

    log.info("features complete -> %s", ws.tda_features_dir)
