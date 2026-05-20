"""Multi-manifold persistent-homology pipeline for UAV intrusion detection on UAVIDS-2025.

Single-file orchestration for the AIAA SciTech 2027 paper. Each phase is exposed as an
argparse subcommand. Run ``python pipeline.py --help`` to see the available commands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable


# ==== SECTION 1: PATHS ====

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "UAVIDS-2025.csv"

OUTPUTS_DIR = REPO_ROOT / "outputs"
PERSISTENCE_DIR = OUTPUTS_DIR / "persistence_diagrams"
TDA_FEATURES_DIR = OUTPUTS_DIR / "tda_features"

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

LOGS_DIR = REPO_ROOT / "logs"


# ==== SECTION 2: DATASET SCHEMA ====

EXPECTED_COLUMNS = (
    "FlowID", "FlowDuration/s", "SrcAddr", "SrcPort", "DstAddr", "DstPort",
    "Protocol", "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps", "MeanPacketSize",
    "PacketDropRate", "AverageHopCount", "label",
)
EXPECTED_CLASSES = (
    "Normal Traffic", "Blackhole Attack", "Wormhole Attack",
    "Sybil Attack", "Flooding Attack",
)
DROP_COLUMNS = ("FlowID", "Protocol")
LABEL_COLUMN = "label"

# Known port values in the dataset; encoded as one-hot binary columns.
KNOWN_PORTS = (9, 654)


# ==== SECTION 3: MANIFOLDS ====
# Three disjoint functional categories per the UAVIDS-2025 paper.
# Categorical columns (addresses, ports) appear here in their post-encoding names.

C2_FEATURES = (
    "SrcAddr_last_octet",
    "SrcPort_9", "SrcPort_654",
    "DstAddr_last_octet",
    "DstPort_9", "DstPort_654",
    "FlowDuration/s",
)
NETWORK_FEATURES = (
    "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanPacketSize",
)
PHYSICAL_FEATURES = (
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps",
    "PacketDropRate", "AverageHopCount",
)
MANIFOLDS = {
    "c2": C2_FEATURES,
    "network": NETWORK_FEATURES,
    "physical": PHYSICAL_FEATURES,
}


# ==== SECTION 4: PIPELINE CONFIG ====

# Stratified split fractions.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# TDA reference-set settings.
REFERENCE_CLOUD_SIZE = 500          # k-medoid sample of normal training flows
QUERY_NEIGHBORHOOD_SIZE = 50        # subset drawn per query for windowing
MAX_EDGE_PERCENTILE = 25            # percentile of pairwise distances within reference cloud
# Spec item 5 prescribes 95; benchmarking found that at 95 a single Rips call
# costs 13+ seconds for C2/Network at sdim=2 and OOMs at sdim=3. p50 was still
# too slow on C2 (~8h full run for H_1 only). p25 keeps the simplex tree small
# enough that sparse Rips (below) reaches sdim=3 in 1-2s/call on C2.

# Cap homology dimensions per manifold (Physical is 5D; H2 there is noise).
MAX_HOM_DIM = {"c2": 2, "network": 2, "physical": 1}

# Sparse-Rips epsilon per manifold. None = exact Rips. Sparse=0.5 trades exact
# persistence for a controlled approximation; needed for C2/Network sdim=3
# to stay under ~2s/call. Physical at sdim=2 is already fast and stays exact.
SPARSE_RIPS_EPSILON: dict = {"c2": 0.5, "network": 0.5, "physical": None}

# Persistence images.
PI_RESOLUTION = (20, 20)

# Names of the eight summary statistics extracted per (manifold, dim).
SUMMARY_STAT_NAMES = (
    "count", "mean_persistence", "std_persistence", "max_persistence",
    "total_persistence", "mean_birth", "mean_death", "persistence_entropy",
)

# Hyperparameter grids (Phase 5).
LR_C_GRID = (0.1, 1.0, 10.0)
RF_N_ESTIMATORS_GRID = (100, 300)
RF_MAX_DEPTH_GRID = (10, 20, None)
SVM_C_GRID = (0.1, 1.0, 10.0)
SVM_GAMMA_GRID = ("scale", "auto")
GRID_CV_FOLDS = 3

# Feature sets and model names used by the supervised pipeline.
FEATURE_SETS = ("original", "summary_only", "summary_plus_images", "combined")
MODEL_NAMES = ("logreg", "rf", "svm")

# SVC with RBF kernel scales O(N^2) to O(N^3); at N=85k it is intractable.
# Train SVM on a stratified subsample of this size (set to None to disable).
SVM_MAX_TRAIN_ROWS = 5000

# Curated RF feature subset (Phase 5, item 14).
TOP_K_RF_IMPORTANCE = 30
TOP_K_MUTUAL_INFO = 30

# Reproducibility.
PRIMARY_SEED = 42
SEEDS = (42, 7, 2024)

# Unsupervised thresholding (Phase 6).
THRESHOLD_PERCENTILE = 95

# Debug mode loads only this many rows of the CSV.
DEBUG_SAMPLE_ROWS = 5000


# ==== SECTION 5: ENVIRONMENT SETUP ====

ALL_DIRS = (
    OUTPUTS_DIR, PERSISTENCE_DIR, TDA_FEATURES_DIR,
    RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR,
    LOGS_DIR,
)


def ensure_directories() -> None:
    """Create every output directory used by the pipeline."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure root logging to stdout and a timestamped logfile."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"pipeline_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    log = logging.getLogger("pipeline")
    log.info("logging to %s", log_path)
    return log


# ==== SECTION 6: PHASE 2 — DATA PREP ====
# Loads UAVIDS-2025, encodes categoricals, splits 70/15/15, scales per manifold
# on training data only, and writes the artifacts every later phase consumes.


def validate_manifolds_disjoint() -> None:
    """Raise if any feature name appears in more than one manifold."""
    counts: dict[str, int] = {}
    for cols in MANIFOLDS.values():
        for col in cols:
            counts[col] = counts.get(col, 0) + 1
    duplicates = sorted(c for c, n in counts.items() if n > 1)
    if duplicates:
        raise AssertionError(f"features in multiple manifolds: {duplicates}")


def validate_schema(df) -> None:
    """Raise unless the dataset has exactly the expected 23 columns and 5 classes."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols or extra_cols:
        raise AssertionError(
            f"schema mismatch: missing={sorted(missing_cols)} extra={sorted(extra_cols)}"
        )
    found = set(df[LABEL_COLUMN].unique())
    missing_classes = set(EXPECTED_CLASSES) - found
    unexpected_classes = found - set(EXPECTED_CLASSES)
    if missing_classes:
        raise AssertionError(f"missing classes: {sorted(missing_classes)}")
    if unexpected_classes:
        raise AssertionError(f"unexpected classes: {sorted(unexpected_classes)}")


def load_raw_dataset(debug: bool):
    """Load the UAVIDS-2025 CSV; debug mode returns a stratified head sample.

    The CSV is sorted by class, so a literal head(5000) would give one class only.
    Debug mode therefore reads the full file and keeps the first
    DEBUG_SAMPLE_ROWS // n_classes rows per class — fast, deterministic, and
    representative.
    """
    import pandas as pd

    log = logging.getLogger("pipeline.prep")
    log.info("loading %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("loaded %d rows, %d columns", len(df), df.shape[1])
    validate_schema(df)
    if debug:
        per_class = DEBUG_SAMPLE_ROWS // len(EXPECTED_CLASSES)
        df = (
            df.groupby(LABEL_COLUMN, group_keys=False)
              .apply(lambda g: g.head(per_class))
              .reset_index(drop=True)
        )
        log.info("debug stratified sample -> %d rows (%d per class)", len(df), per_class)
    return df


def encode_features(df):
    """Drop FlowID/Protocol, encode IPs as last octet, one-hot known ports."""
    encoded = df.drop(columns=list(DROP_COLUMNS))
    for col in ("SrcAddr", "DstAddr"):
        encoded[f"{col}_last_octet"] = (
            encoded[col].astype(str).str.split(".").str[-1].astype(int)
        )
    encoded = encoded.drop(columns=["SrcAddr", "DstAddr"])
    for col in ("SrcPort", "DstPort"):
        unexpected = set(encoded[col].unique()) - set(KNOWN_PORTS)
        if unexpected:
            raise AssertionError(f"{col} has unexpected values: {sorted(unexpected)}")
        for port in KNOWN_PORTS:
            encoded[f"{col}_{port}"] = (encoded[col] == port).astype(int)
    encoded = encoded.drop(columns=["SrcPort", "DstPort"])
    return encoded


def stratified_three_way_split(labels, seed: int):
    """Return sorted (train_idx, val_idx, test_idx) numpy arrays for a 70/15/15 split."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=labels.iloc[holdout_idx],
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def validate_split_indices(n_total: int, train_idx, val_idx, test_idx) -> None:
    """Raise unless the three index sets are pairwise disjoint and union to [0, n_total)."""
    train_set, val_set, test_set = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    overlaps = [
        ("train/val", train_set & val_set),
        ("train/test", train_set & test_set),
        ("val/test", val_set & test_set),
    ]
    for name, overlap in overlaps:
        if overlap:
            raise AssertionError(f"split indices overlap in {name}: {len(overlap)} rows")
    union = train_set | val_set | test_set
    if union != set(range(n_total)):
        missing = set(range(n_total)) - union
        raise AssertionError(f"split indices miss {len(missing)} rows of {n_total}")


def validate_class_balance(labels, train_idx, val_idx, test_idx) -> None:
    """Raise if any per-split class proportion drifts more than 0.5pp from overall."""
    overall = labels.value_counts(normalize=True)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        split_props = labels.iloc[idx].value_counts(normalize=True)
        for cls in EXPECTED_CLASSES:
            diff = abs(float(split_props.get(cls, 0.0)) - float(overall.get(cls, 0.0)))
            if diff > 0.005:
                raise AssertionError(
                    f"{name} proportion for {cls!r} drifts {diff:.4f} > 0.005"
                )


def scale_manifold(train_df, val_df, test_df):
    """Fit StandardScaler on train, transform all three splits."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df), columns=val_df.columns, index=val_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), columns=test_df.columns, index=test_df.index,
    )
    return train_scaled, val_scaled, test_scaled, scaler


def summarize_splits(labels, splits: dict) -> None:
    """Log a per-split row count and class-count table."""
    import pandas as pd

    rows = []
    for name, idx in splits.items():
        sub = labels.iloc[idx]
        row: dict = {"split": name, "rows": len(sub)}
        for cls in EXPECTED_CLASSES:
            row[cls] = int((sub == cls).sum())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("split")
    logging.getLogger("pipeline.prep").info("split summary:\n%s", table.to_string())


def assert_manifold_columns_present(encoded) -> None:
    """Raise if any manifold's declared columns are missing from the encoded frame."""
    for name, cols in MANIFOLDS.items():
        missing = set(cols) - set(encoded.columns)
        if missing:
            raise AssertionError(
                f"{name} manifold missing columns after encoding: {sorted(missing)}"
            )


def save_label_artifacts(labels, splits: dict) -> None:
    """Write split-index .npy files and per-split label CSVs to outputs/."""
    import numpy as np

    for name, idx in splits.items():
        np.save(OUTPUTS_DIR / f"{name}_indices.npy", idx)
        labels.iloc[idx].to_csv(OUTPUTS_DIR / f"labels_{name}.csv", index=False)


def process_manifolds(encoded, splits: dict) -> tuple[dict, dict]:
    """Scale each manifold on train, save per-split CSVs, return (scalers, combined_chunks)."""
    log = logging.getLogger("pipeline.prep")
    scalers: dict = {}
    combined_chunks: dict[str, list] = {name: [] for name in splits}
    for manifold_name, cols in MANIFOLDS.items():
        sub = encoded[list(cols)]
        per_split = {
            name: sub.iloc[idx].reset_index(drop=True) for name, idx in splits.items()
        }
        train_scaled, val_scaled, test_scaled, scaler = scale_manifold(
            per_split["train"], per_split["val"], per_split["test"]
        )
        scalers[manifold_name] = scaler
        scaled_by_split = {"train": train_scaled, "val": val_scaled, "test": test_scaled}
        for split_name, scaled in scaled_by_split.items():
            scaled.to_csv(OUTPUTS_DIR / f"{manifold_name}_{split_name}.csv", index=False)
            combined_chunks[split_name].append(scaled)
        log.info("scaled manifold=%s features=%d", manifold_name, len(cols))
    return scalers, combined_chunks


def save_combined_originals(combined_chunks: dict) -> None:
    """Concat the scaled manifolds per split and write original_features_{split}.csv."""
    import pandas as pd

    for name, chunks in combined_chunks.items():
        combined = pd.concat(chunks, axis=1)
        combined.to_csv(OUTPUTS_DIR / f"original_features_{name}.csv", index=False)


def cmd_prep(args: argparse.Namespace) -> None:
    """Phase 2: load, encode, split, scale, and save per-manifold CSVs."""
    import pickle

    log = logging.getLogger("pipeline.prep")
    validate_manifolds_disjoint()

    df = load_raw_dataset(args.debug)
    encoded = encode_features(df)
    labels = encoded[LABEL_COLUMN].reset_index(drop=True)
    encoded = encoded.drop(columns=[LABEL_COLUMN]).reset_index(drop=True)
    assert_manifold_columns_present(encoded)

    train_idx, val_idx, test_idx = stratified_three_way_split(labels, seed=PRIMARY_SEED)
    validate_split_indices(len(labels), train_idx, val_idx, test_idx)
    validate_class_balance(labels, train_idx, val_idx, test_idx)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    save_label_artifacts(labels, splits)
    scalers, combined_chunks = process_manifolds(encoded, splits)
    save_combined_originals(combined_chunks)

    with (OUTPUTS_DIR / "scalers.pkl").open("wb") as fh:
        pickle.dump(scalers, fh)

    summarize_splits(labels, splits)
    log.info("prep complete -> %s", OUTPUTS_DIR)


# ==== SECTION 7: PHASE 3 — TDA PERSISTENCE ====
# Computes a per-flow Vietoris-Rips persistence diagram against a fixed
# reference cloud of REFERENCE_CLOUD_SIZE training-Normal medoids. Each
# query flow's point cloud is {query} ∪ reference so the diagram describes
# how that flow sits relative to a canonical normal-traffic shape rather
# than a same-class neighborhood.
#
# Note on the simplex-tree dimension: the prompt caps homology at H_k for
# each manifold (k=2 for C2/Network, k=1 for Physical). Reliably computing
# H_k via Rips requires (k+1)-simplices to fill k-cycles, so the simplex
# tree's max_dimension is MAX_HOM_DIM[manifold] + 1.


def load_split_manifold(manifold: str, split: str):
    """Return the (n_flows, n_features) array for one manifold-split CSV."""
    import pandas as pd

    return pd.read_csv(OUTPUTS_DIR / f"{manifold}_{split}.csv").values


def load_labels_for_split(split: str):
    """Return the label Series for one split CSV."""
    import pandas as pd

    return pd.read_csv(OUTPUTS_DIR / f"labels_{split}.csv")[LABEL_COLUMN]


def sample_reference_indices(seed: int):
    """Return REFERENCE_CLOUD_SIZE train positions chosen by k-medoids on Normal flows.

    Runs KMedoids on c2_train restricted to Normal-Traffic rows when
    scikit-learn-extra is importable; otherwise falls back to KMeans plus
    nearest-training-point projection. Returns absolute positions into the
    training-split row order (shared across all three manifold CSVs).
    """
    import numpy as np

    log = logging.getLogger("pipeline.tda")
    labels = load_labels_for_split("train")
    normal_positions = np.where(labels.values == "Normal Traffic")[0]
    if len(normal_positions) < REFERENCE_CLOUD_SIZE:
        raise AssertionError(
            f"only {len(normal_positions)} Normal Traffic rows in train; "
            f"need >= {REFERENCE_CLOUD_SIZE}"
        )
    c2_train = load_split_manifold("c2", "train")
    normal_c2 = c2_train[normal_positions]
    try:
        from sklearn_extra.cluster import KMedoids

        log.info("k-medoids backend: sklearn_extra.KMedoids")
        km = KMedoids(
            n_clusters=REFERENCE_CLOUD_SIZE,
            random_state=seed,
            method="alternate",
        )
        km.fit(normal_c2)
        local_idx = km.medoid_indices_
    except ImportError:
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans

        log.info("k-medoids backend: KMeans + nearest-point fallback")
        km = KMeans(n_clusters=REFERENCE_CLOUD_SIZE, random_state=seed, n_init=10)
        km.fit(normal_c2)
        local_idx = cdist(km.cluster_centers_, normal_c2).argmin(axis=1)
    abs_idx = np.sort(normal_positions[np.asarray(local_idx)])
    log.info(
        "sampled %d reference indices from %d Normal train rows",
        len(abs_idx), len(normal_positions),
    )
    return abs_idx


def compute_reference_clouds(reference_idx) -> dict:
    """Return {manifold: (REFERENCE_CLOUD_SIZE, n_features) array} by indexing train."""
    return {
        manifold: load_split_manifold(manifold, "train")[reference_idx]
        for manifold in MANIFOLDS
    }


def compute_max_edge_lengths(reference_clouds: dict) -> dict:
    """Return {manifold: 95th-percentile pairwise distance within its reference cloud}."""
    import numpy as np
    from scipy.spatial.distance import pdist

    return {
        manifold: float(np.percentile(pdist(cloud), MAX_EDGE_PERCENTILE))
        for manifold, cloud in reference_clouds.items()
    }


def _persistence_for_point(query_point, reference_points, max_edge, max_simplex_dim, sparse):
    """Compute one (sparse) Rips persistence diagram for {reference} ∪ {query}."""
    import gudhi
    import numpy as np

    point_cloud = np.vstack([reference_points, query_point.reshape(1, -1)])
    kwargs: dict = {"points": point_cloud, "max_edge_length": max_edge}
    if sparse is not None:
        kwargs["sparse"] = sparse
    rips = gudhi.RipsComplex(**kwargs)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_simplex_dim)
    raw = simplex_tree.persistence()
    if not raw:
        return np.empty((0, 3), dtype=float)
    return np.array(
        [[float(dim), float(birth), float(death)] for dim, (birth, death) in raw],
        dtype=float,
    )


def compute_diagrams_for_split(
    manifold: str,
    split: str,
    reference_points,
    max_edge: float,
    max_simplex_dim: int,
    sparse,
    n_jobs: int,
) -> list:
    """Compute per-flow persistence diagrams for one (manifold, split) in parallel."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    log = logging.getLogger("pipeline.tda")
    points = load_split_manifold(manifold, split)
    log.info(
        "computing: manifold=%s split=%s flows=%d max_edge=%.4f simplex_dim=%d sparse=%s",
        manifold, split, len(points), max_edge, max_simplex_dim, sparse,
    )
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    gen = parallel(
        delayed(_persistence_for_point)(p, reference_points, max_edge, max_simplex_dim, sparse)
        for p in points
    )
    return list(tqdm(gen, total=len(points), desc=f"{manifold}/{split}"))


def validate_diagrams(diagrams, manifold: str, split: str) -> None:
    """Raise unless every feature has birth >= 0, death >= 0, death > birth."""
    for i, diag in enumerate(diagrams):
        if diag.size == 0:
            continue
        births, deaths = diag[:, 1], diag[:, 2]
        if (births < 0).any():
            raise AssertionError(f"{manifold}/{split} flow {i}: negative birth")
        if (deaths < 0).any():
            raise AssertionError(f"{manifold}/{split} flow {i}: negative death")
        if (deaths <= births).any():
            raise AssertionError(f"{manifold}/{split} flow {i}: death <= birth")


def save_diagrams(manifold: str, split: str, diagrams: list) -> None:
    """Persist diagrams as both pickle (raw list) and numpy object array."""
    import pickle

    import numpy as np

    with (PERSISTENCE_DIR / f"{manifold}_{split}.pkl").open("wb") as fh:
        pickle.dump(diagrams, fh)
    array = np.empty(len(diagrams), dtype=object)
    for i, diag in enumerate(diagrams):
        array[i] = diag
    np.save(PERSISTENCE_DIR / f"{manifold}_{split}.npy", array, allow_pickle=True)


def summarize_diagram_counts(diagrams: list, max_hom_dim: int) -> dict:
    """Return {f'mean_H{k}': mean per-flow count} for k in [0, max_hom_dim]."""
    import numpy as np

    counts: dict = {k: [] for k in range(max_hom_dim + 1)}
    for diag in diagrams:
        if diag.size == 0:
            for k in counts:
                counts[k].append(0)
            continue
        dims = diag[:, 0].astype(int)
        for k in counts:
            counts[k].append(int((dims == k).sum()))
    return {f"mean_H{k}": float(np.mean(vals)) for k, vals in counts.items()}


def cmd_tda(args: argparse.Namespace) -> None:
    """Phase 3: compute per-flow persistence diagrams against the reference cloud."""
    import json

    import numpy as np
    import pandas as pd

    log = logging.getLogger("pipeline.tda")

    reference_idx = sample_reference_indices(args.seed)
    np.save(OUTPUTS_DIR / "reference_indices.npy", reference_idx)
    reference_clouds = compute_reference_clouds(reference_idx)
    max_edge_lengths = compute_max_edge_lengths(reference_clouds)
    with (OUTPUTS_DIR / "max_edge_lengths.json").open("w") as fh:
        json.dump(max_edge_lengths, fh, indent=2)
    log.info("max edge lengths: %s", max_edge_lengths)

    manifolds = list(MANIFOLDS) if args.manifold == "all" else [args.manifold]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    summary_rows: list = []
    for manifold in manifolds:
        max_simplex_dim = MAX_HOM_DIM[manifold] + 1
        ref_points = reference_clouds[manifold]
        max_edge = max_edge_lengths[manifold]
        sparse = SPARSE_RIPS_EPSILON.get(manifold)
        for split in splits:
            diagrams = compute_diagrams_for_split(
                manifold, split, ref_points, max_edge, max_simplex_dim, sparse, n_jobs=-1,
            )
            validate_diagrams(diagrams, manifold, split)
            save_diagrams(manifold, split, diagrams)
            stats = summarize_diagram_counts(diagrams, MAX_HOM_DIM[manifold])
            summary_rows.append(
                {"manifold": manifold, "split": split, "n_flows": len(diagrams), **stats}
            )
            log.info("done %s/%s: %s", manifold, split, stats)

    table = pd.DataFrame(summary_rows)
    log.info("tda summary:\n%s", table.to_string(index=False))
    table.to_csv(OUTPUTS_DIR / "tda_summary.csv", index=False)


# ==== SECTION 8: PHASE 4 — FEATURE EXTRACTION ====
# Converts per-flow persistence diagrams into two parallel feature blocks:
#   1. Summary statistics — 8 scalars per (manifold, dim).
#   2. Persistence images — 20x20 flattened per (manifold, dim), using
#      persim's PersistenceImager. The imager is fit on training-set
#      diagrams only and reused for val/test.
# Both blocks plus the original (scaled) features are concatenated into a
# "combined" frame for the supervised ablation. Columns that are entirely
# zero across the training set are dropped (and the same columns dropped
# from val/test). The kept-column lists are saved to feature_names.json.


def load_diagrams_pkl(manifold: str, split: str) -> list:
    """Load the per-flow persistence-diagram pickle written by Phase 3."""
    import pickle

    path = PERSISTENCE_DIR / f"{manifold}_{split}.pkl"
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
        return {name: 0.0 for name in SUMMARY_STAT_NAMES}
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
            configure_square_imager(imager, PI_RESOLUTION)
            imagers[(manifold, dim)] = imager
    return imagers


def collect_diagrams_by_dim(manifold: str, split: str, max_edge: float, max_hom_dim: int):
    """Return {dim: list[per-flow (birth, death) arrays]} for one manifold-split."""
    diagrams = load_diagrams_pkl(manifold, split)
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
        for stat in SUMMARY_STAT_NAMES
    }


def build_image_columns(manifold: str, dim: int, bd_arrays: list, imager) -> dict:
    """Return {column_name: list[float]} for the flattened 20x20 persistence image."""
    import numpy as np

    target_len = PI_RESOLUTION[0] * PI_RESOLUTION[1]
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
    split: str, max_edge_lengths: dict, imagers: dict,
) -> tuple:
    """Build (summary_df, images_df) for one split across all manifolds and dims."""
    import pandas as pd

    summary_cols: dict = {}
    image_cols: dict = {}
    for manifold in MANIFOLDS:
        max_edge = max_edge_lengths[manifold]
        max_hom = MAX_HOM_DIM[manifold]
        by_dim = collect_diagrams_by_dim(manifold, split, max_edge, max_hom)
        for dim, bd_arrays in by_dim.items():
            summary_cols.update(build_summary_columns(manifold, dim, bd_arrays))
            image_cols.update(
                build_image_columns(manifold, dim, bd_arrays, imagers[(manifold, dim)])
            )
    return pd.DataFrame(summary_cols), pd.DataFrame(image_cols)


def collect_train_diagrams_for_imagers(max_edge_lengths: dict) -> dict:
    """Reload training diagrams once, organised as {manifold: {dim: [bd_arrays...]}}."""
    out: dict = {}
    for manifold in MANIFOLDS:
        max_edge = max_edge_lengths[manifold]
        max_hom = MAX_HOM_DIM[manifold]
        out[manifold] = collect_diagrams_by_dim(manifold, "train", max_edge, max_hom)
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
    summary_expected = expected_pairs * len(SUMMARY_STAT_NAMES)
    image_expected = expected_pairs * PI_RESOLUTION[0] * PI_RESOLUTION[1]
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
    for manifold in MANIFOLDS:
        if col.startswith(f"{manifold}_H"):
            return manifold
    return "original"


def report_dropped_columns(dropped: list) -> dict:
    """Bucket dropped columns by manifold and return counts per manifold."""
    counts: dict = {m: 0 for m in MANIFOLDS}
    counts["original"] = 0
    for col in dropped:
        counts[manifold_for_column(col)] += 1
    return counts


def fit_and_save_imagers(max_edge_lengths: dict) -> dict:
    """Fit one imager per (manifold, dim) on training diagrams and pickle the dict."""
    import pickle

    log = logging.getLogger("pipeline.features")
    log.info("fitting persistence imagers on training diagrams")
    train_by_md = collect_train_diagrams_for_imagers(max_edge_lengths)
    imagers = fit_imagers(train_by_md)
    with (OUTPUTS_DIR / "persistence_imagers.pkl").open("wb") as fh:
        pickle.dump(imagers, fh)
    return imagers


def build_all_splits(max_edge_lengths: dict, imagers: dict) -> tuple:
    """Return (per_split_summary, per_split_images) dicts keyed by split name."""
    log = logging.getLogger("pipeline.features")
    expected_pairs = sum(MAX_HOM_DIM[m] + 1 for m in MANIFOLDS)
    per_split_summary: dict = {}
    per_split_images: dict = {}
    for split in ("train", "val", "test"):
        log.info("building features for split=%s", split)
        summary_df, images_df = build_split_frames(split, max_edge_lengths, imagers)
        validate_feature_counts(summary_df, images_df, expected_pairs)
        per_split_summary[split] = summary_df
        per_split_images[split] = images_df
    return per_split_summary, per_split_images


def determine_drops(per_split_summary: dict, per_split_images: dict) -> tuple:
    """Pick all-zero training columns; log dropped breakdown per manifold."""
    log = logging.getLogger("pipeline.features")
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
    split: str, summary_df, images_df, drop_summary: list, drop_images: list,
) -> tuple:
    """Drop, validate, and write the three CSVs for one split; return per-frame shapes."""
    import pandas as pd

    summary = summary_df.drop(columns=drop_summary)
    images = images_df.drop(columns=drop_images)
    original = pd.read_csv(OUTPUTS_DIR / f"original_features_{split}.csv")
    validate_no_nan(summary, images, original, names=["summary", "images", "original"])
    summary.to_csv(TDA_FEATURES_DIR / f"summary_{split}.csv", index=False)
    images.to_csv(TDA_FEATURES_DIR / f"images_{split}.csv", index=False)
    combined = pd.concat([original, summary, images], axis=1)
    combined.to_csv(TDA_FEATURES_DIR / f"combined_{split}.csv", index=False)
    return summary.shape[1], images.shape[1], combined.shape[1]


def cmd_features(args: argparse.Namespace) -> None:
    """Phase 4: extract summary stats and persistence images from Phase 3 diagrams."""
    import json

    log = logging.getLogger("pipeline.features")
    with (OUTPUTS_DIR / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)

    imagers = fit_and_save_imagers(max_edge_lengths)
    per_split_summary, per_split_images = build_all_splits(max_edge_lengths, imagers)
    drop_summary, drop_images = determine_drops(per_split_summary, per_split_images)

    feature_names = {
        "summary": [c for c in per_split_summary["train"].columns if c not in drop_summary],
        "images": [c for c in per_split_images["train"].columns if c not in drop_images],
    }
    with (TDA_FEATURES_DIR / "feature_names.json").open("w") as fh:
        json.dump(feature_names, fh, indent=2)

    for split in ("train", "val", "test"):
        n_summary, n_images, n_combined = save_features_for_split(
            split, per_split_summary[split], per_split_images[split],
            drop_summary, drop_images,
        )
        log.info(
            "saved split=%s: summary=%d images=%d combined=%d",
            split, n_summary, n_images, n_combined,
        )

    log.info("features complete -> %s", TDA_FEATURES_DIR)


# ==== SECTION 9: PHASE 5 — SUPERVISED PIPELINE ====
# For each (feature_set, model, seed) cell:
#   1. Build the hyperparameter grid for the model.
#   2. For each grid combo: fit on training set, score on validation set.
#   3. Take the combo with the best val accuracy; refit on full training.
#   4. Evaluate on the test set (accuracy, weighted/macro F1, weighted AUC,
#      per-class precision/recall/F1, per-class AUC, confusion matrix).
# RandomForest also runs a "curated" variant: the feature subset is the union
# of (top-30 by RF feature_importances_, top-30 by mutual_info_classif),
# frozen at seed=42 then evaluated across all seeds.
# SVM training is capped at SVM_MAX_TRAIN_ROWS via stratified sampling because
# SVC with RBF on the full 85k training set is intractable.


def load_feature_split(feature_set: str, split: str):
    """Return the feature DataFrame for one feature_set / split pair."""
    import pandas as pd

    if feature_set == "original":
        return pd.read_csv(OUTPUTS_DIR / f"original_features_{split}.csv")
    if feature_set == "summary_only":
        return pd.read_csv(TDA_FEATURES_DIR / f"summary_{split}.csv")
    if feature_set == "summary_plus_images":
        summary = pd.read_csv(TDA_FEATURES_DIR / f"summary_{split}.csv")
        images = pd.read_csv(TDA_FEATURES_DIR / f"images_{split}.csv")
        return pd.concat([summary, images], axis=1)
    if feature_set == "combined":
        return pd.read_csv(TDA_FEATURES_DIR / f"combined_{split}.csv")
    raise ValueError(f"unknown feature set: {feature_set}")


def load_labels_for(split: str):
    """Return the label array for one split."""
    return load_labels_for_split(split).values


def build_model(name: str, seed: int, **params):
    """Construct an unfitted estimator for one of {logreg, rf, svm}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    if name == "logreg":
        return LogisticRegression(
            C=params["C"], max_iter=2000, multi_class="auto", solver="lbfgs",
            random_state=seed, n_jobs=-1,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            random_state=seed, n_jobs=-1,
        )
    if name == "svm":
        return SVC(
            C=params["C"], gamma=params["gamma"], kernel="rbf",
            probability=True, random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def grid_combos(model_name: str) -> list:
    """Return the list of hyperparameter dicts to try for one model."""
    if model_name == "logreg":
        return [{"C": c} for c in LR_C_GRID]
    if model_name == "rf":
        return [
            {"n_estimators": n, "max_depth": d}
            for n in RF_N_ESTIMATORS_GRID for d in RF_MAX_DEPTH_GRID
        ]
    if model_name == "svm":
        return [
            {"C": c, "gamma": g}
            for c in SVM_C_GRID for g in SVM_GAMMA_GRID
        ]
    raise ValueError(f"unknown model: {model_name}")


def maybe_subsample_for_svm(X_train, y_train, model_name: str, seed: int):
    """Stratified-subsample training data when training SVM; otherwise pass through."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    if model_name != "svm" or SVM_MAX_TRAIN_ROWS is None or len(X_train) <= SVM_MAX_TRAIN_ROWS:
        return X_train, y_train
    keep_frac = SVM_MAX_TRAIN_ROWS / len(X_train)
    keep_idx, _ = train_test_split(
        np.arange(len(X_train)), train_size=keep_frac,
        stratify=y_train, random_state=seed,
    )
    return X_train.iloc[keep_idx], y_train[keep_idx]


def select_best_params(model_name: str, X_train, y_train, X_val, y_val, seed: int) -> dict:
    """Try every grid combo; return the one with the highest val accuracy."""
    from sklearn.metrics import accuracy_score

    log = logging.getLogger("pipeline.supervised")
    X_fit, y_fit = maybe_subsample_for_svm(X_train, y_train, model_name, seed)
    best_acc = -1.0
    best_params: dict = {}
    for params in grid_combos(model_name):
        model = build_model(model_name, seed, **params)
        model.fit(X_fit, y_fit)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        log.debug("%s seed=%s params=%s val_acc=%.4f", model_name, seed, params, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
    log.info(
        "best %s (seed=%s): params=%s val_acc=%.4f",
        model_name, seed, best_params, best_acc,
    )
    return best_params


def fit_final_model(model_name: str, X_train, y_train, params: dict, seed: int):
    """Fit the model on full training data with the chosen hyperparameters."""
    X_fit, y_fit = maybe_subsample_for_svm(X_train, y_train, model_name, seed)
    model = build_model(model_name, seed, **params)
    model.fit(X_fit, y_fit)
    return model


def evaluate_model(model, X_test, y_test) -> tuple:
    """Compute headline metrics, per-class metrics, and the confusion matrix."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, roc_auc_score,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    # Per-class AUC (one-vs-rest)
    per_class_auc = {}
    y_test_arr = np.asarray(y_test)
    for i, cls in enumerate(classes):
        binary = (y_test_arr == cls).astype(int)
        try:
            per_class_auc[cls] = float(roc_auc_score(binary, y_proba[:, i]))
        except ValueError:
            per_class_auc[cls] = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_auc": float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        ),
    }
    for cls, val in per_class_auc.items():
        metrics[f"auc_{cls}"] = val
    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=classes), index=classes, columns=classes,
    )
    return y_pred, metrics, report, cm


def validate_supervised(y_pred, y_test, metrics: dict) -> None:
    """Raise on length mismatch, missing predicted classes, or AUC out of [0, 1]."""
    if len(y_pred) != len(y_test):
        raise AssertionError(f"len(y_pred)={len(y_pred)} != len(y_test)={len(y_test)}")
    missing = set(EXPECTED_CLASSES) - set(map(str, y_pred))
    if missing:
        raise AssertionError(f"classes predicted with zero support: {sorted(missing)}")
    for key, val in metrics.items():
        if "auc" in key and val == val and not (0.0 <= val <= 1.0):  # NaN-safe
            raise AssertionError(f"{key}={val} outside [0, 1]")


def select_curated_features(rf_model, X_train, y_train) -> list:
    """Union of top-K RF importances and top-K mutual-info features."""
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif

    importances = np.asarray(rf_model.feature_importances_)
    top_rf = set(
        X_train.columns[np.argsort(importances)[::-1][:TOP_K_RF_IMPORTANCE]]
    )
    mi = mutual_info_classif(X_train, y_train, random_state=PRIMARY_SEED)
    top_mi = set(X_train.columns[np.argsort(mi)[::-1][:TOP_K_MUTUAL_INFO]])
    return sorted(top_rf | top_mi)


def run_one_combo(
    feature_set: str, model_name: str, seed: int, curated_cols: list | None = None,
) -> tuple:
    """Train + tune + evaluate one (feature_set, model, seed) cell; return artefacts."""
    log = logging.getLogger("pipeline.supervised")
    X_train = load_feature_split(feature_set, "train")
    X_val = load_feature_split(feature_set, "val")
    X_test = load_feature_split(feature_set, "test")
    if curated_cols is not None:
        X_train = X_train[curated_cols]
        X_val = X_val[curated_cols]
        X_test = X_test[curated_cols]
    y_train = load_labels_for("train")
    y_val = load_labels_for("val")
    y_test = load_labels_for("test")
    log.info(
        "fitting %s on %s (n_train=%d, n_features=%d, seed=%s)",
        model_name, feature_set, len(X_train), X_train.shape[1], seed,
    )
    best_params = select_best_params(model_name, X_train, y_train, X_val, y_val, seed)
    model = fit_final_model(model_name, X_train, y_train, best_params, seed)
    y_pred, metrics, report, cm = evaluate_model(model, X_test, y_test)
    validate_supervised(y_pred, y_test, metrics)
    return model, metrics, report, cm, best_params


def save_seed42_artifacts(
    feature_set: str, model_name: str, model, report, cm, curated: bool,
) -> None:
    """Pickle the seed=42 model and write per-class + confusion-matrix CSVs."""
    import pickle

    suffix = "_curated" if curated else ""
    stem = f"{feature_set}_{model_name}{suffix}"
    with (MODELS_DIR / f"{stem}_seed42.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    report.to_csv(TABLES_DIR / f"per_class_metrics_{stem}.csv")
    cm.to_csv(TABLES_DIR / f"confusion_matrix_{stem}.csv")


def metrics_to_row(
    feature_set: str, model_name: str, seed: int, metrics: dict, params: dict, curated: bool,
) -> dict:
    """Flatten one cell's metrics + chosen params into a single-row dict."""
    row = {
        "feature_set": feature_set, "model": model_name, "seed": seed,
        "curated": curated, **metrics, "best_params": str(params),
    }
    return row


def run_seeds_for_combo(
    feature_set: str, model_name: str, curated_cols: list | None = None,
) -> list:
    """Run all three seeds for one (feature_set, model) cell; return list of rows."""
    rows: list = []
    for seed in SEEDS:
        model, metrics, report, cm, params = run_one_combo(
            feature_set, model_name, seed, curated_cols=curated_cols,
        )
        if seed == PRIMARY_SEED:
            save_seed42_artifacts(
                feature_set, model_name, model, report, cm,
                curated=(curated_cols is not None),
            )
        rows.append(
            metrics_to_row(
                feature_set, model_name, seed, metrics, params,
                curated=(curated_cols is not None),
            )
        )
    return rows


def summarize_metrics(rows: list):
    """Return a DataFrame of mean ± std for every numeric metric per (feature_set, model)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in
                   {"feature_set", "model", "seed", "curated", "best_params"}]
    group_cols = ["feature_set", "model", "curated"]
    means = df.groupby(group_cols)[metric_cols].mean().add_suffix("_mean")
    stds = df.groupby(group_cols)[metric_cols].std().add_suffix("_std")
    return means.join(stds).reset_index()


def plot_supervised_comparison(summary_df, out_path) -> None:
    """Grouped bar chart of accuracy by (feature_set, model) with std error bars."""
    import matplotlib.pyplot as plt
    import numpy as np

    base = summary_df[~summary_df["curated"]].copy()
    pivot_mean = base.pivot(index="feature_set", columns="model", values="accuracy_mean")
    pivot_std = base.pivot(index="feature_set", columns="model", values="accuracy_std")
    pivot_mean = pivot_mean.loc[list(FEATURE_SETS), list(MODEL_NAMES)]
    pivot_std = pivot_std.loc[list(FEATURE_SETS), list(MODEL_NAMES)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FEATURE_SETS))
    width = 0.8 / len(MODEL_NAMES)
    for i, model in enumerate(MODEL_NAMES):
        ax.bar(
            x + i * width, pivot_mean[model].values, width,
            yerr=pivot_std[model].values, label=model, capsize=3,
        )
    ax.set_xticks(x + width * (len(MODEL_NAMES) - 1) / 2)
    ax.set_xticklabels(list(FEATURE_SETS), rotation=15)
    ax.set_ylabel("test accuracy")
    ax.set_title("Supervised accuracy by feature set and model (mean ± std over 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curated_importance(rf_model, feature_names: list, out_path) -> None:
    """Top-30 features of the curated RF, color-coded by feature origin."""
    import matplotlib.pyplot as plt
    import numpy as np

    importances = np.asarray(rf_model.feature_importances_)
    top_idx = np.argsort(importances)[::-1][:30]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]
    colors = []
    for name in top_names:
        if "_img_" in name:
            colors.append("tab:blue")
        elif any(stat in name for stat in SUMMARY_STAT_NAMES):
            colors.append("tab:orange")
        else:
            colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(range(len(top_names))[::-1], top_vals, color=colors)
    ax.set_yticks(range(len(top_names))[::-1])
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("RF feature importance")
    ax.set_title("Top-30 curated-RF features (blue=image, orange=summary, gray=original)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def freeze_curated_subset(feature_set: str) -> tuple:
    """Run seed-42 RF on all features, derive curated subset, refit, return (cols, model)."""
    log = logging.getLogger("pipeline.supervised")
    X_train = load_feature_split(feature_set, "train")
    y_train = load_labels_for("train")
    base_model = build_model("rf", PRIMARY_SEED, n_estimators=300, max_depth=None)
    base_model.fit(X_train, y_train)
    curated = select_curated_features(base_model, X_train, y_train)
    log.info(
        "curated subset for %s: %d features (top-30 RF ∪ top-30 MI)",
        feature_set, len(curated),
    )
    curated_model = build_model("rf", PRIMARY_SEED, n_estimators=300, max_depth=None)
    curated_model.fit(X_train[curated], y_train)
    return curated, curated_model


def cmd_supervised(args: argparse.Namespace) -> None:
    """Phase 5: tune, train, evaluate classifiers across feature sets and seeds."""
    import pandas as pd

    log = logging.getLogger("pipeline.supervised")
    all_rows: list = []
    curated_subsets: dict = {}
    curated_models: dict = {}

    for feature_set in FEATURE_SETS:
        for model_name in MODEL_NAMES:
            log.info("=== feature_set=%s model=%s ===", feature_set, model_name)
            all_rows.extend(run_seeds_for_combo(feature_set, model_name))

        # Curated RF variant per feature set.
        curated_cols, curated_model = freeze_curated_subset(feature_set)
        curated_subsets[feature_set] = curated_cols
        curated_models[feature_set] = curated_model
        log.info("=== feature_set=%s model=rf_curated ===", feature_set)
        all_rows.extend(run_seeds_for_combo(feature_set, "rf", curated_cols=curated_cols))

    metrics_df = pd.DataFrame(all_rows)
    summary_df = summarize_metrics(all_rows)
    metrics_df.to_csv(TABLES_DIR / "supervised_metrics.csv", index=False)
    summary_df.to_csv(TABLES_DIR / "supervised_summary.csv", index=False)

    plot_supervised_comparison(summary_df, FIGURES_DIR / "supervised_comparison.png")
    # Use the combined feature-set's curated RF for the importance plot.
    plot_curated_importance(
        curated_models["combined"], curated_subsets["combined"],
        FIGURES_DIR / "feature_importance_curated.png",
    )

    log_headline_results(summary_df)
    log.info("supervised complete -> tables=%s figures=%s", TABLES_DIR, FIGURES_DIR)


def log_headline_results(summary_df) -> None:
    """Print mean ± std test accuracy for the combined+RF / summary_plus_images+RF cells."""
    log = logging.getLogger("pipeline.supervised")
    log.info("HEADLINE results (mean ± std test accuracy across 3 seeds):")
    for fs in ("combined", "summary_plus_images"):
        for curated in (False, True):
            sel = summary_df[
                (summary_df["feature_set"] == fs)
                & (summary_df["model"] == "rf")
                & (summary_df["curated"] == curated)
            ]
            if sel.empty:
                continue
            mu = float(sel["accuracy_mean"].iloc[0])
            sd = float(sel["accuracy_std"].iloc[0])
            tag = "rf_curated" if curated else "rf"
            log.info("  %s / %s: %.4f ± %.4f", fs, tag, mu, sd)


# ==== SECTION 10: PHASE 6 — UNSUPERVISED PIPELINE ====
# Single baseline barcode per manifold = Rips on just the 500 reference points
# (no query). Each flow's anomaly score per manifold = Wasserstein-2 distance
# from its Phase 3 diagram to that manifold's baseline barcode, summed over
# H-dims. Threshold per manifold = 95th percentile of val Normal distances.
# A flow is flagged if ANY manifold's distance exceeds its threshold. The
# attack-type inference rule maps each class to its most-common training
# pattern of (c2_flag, network_flag, physical_flag) and is applied to test.


def get_wasserstein_backend() -> tuple:
    """Return (callable, backend_name) preferring GUDHI Hera, falling back to gudhi.wasserstein."""
    try:
        from gudhi.hera import wasserstein_distance as wdist
        return wdist, "hera"
    except ImportError:
        from gudhi.wasserstein import wasserstein_distance as wdist
        return wdist, "gudhi.wasserstein"


def compute_baseline_barcodes(max_edge_lengths: dict, reference_indices) -> dict:
    """Run Rips on each manifold's 500 reference points; return per-dim baseline diagrams."""
    import gudhi
    import numpy as np

    log = logging.getLogger("pipeline.unsupervised")
    baselines: dict = {}
    for manifold in MANIFOLDS:
        train = load_split_manifold(manifold, "train")
        ref = train[reference_indices]
        max_edge = max_edge_lengths[manifold]
        max_simplex_dim = MAX_HOM_DIM[manifold] + 1
        sparse = SPARSE_RIPS_EPSILON.get(manifold)
        kwargs: dict = {"points": ref, "max_edge_length": max_edge}
        if sparse is not None:
            kwargs["sparse"] = sparse
        rips = gudhi.RipsComplex(**kwargs)
        simplex_tree = rips.create_simplex_tree(max_dimension=max_simplex_dim)
        raw = simplex_tree.persistence()
        if raw:
            diag = np.array(
                [[float(d), float(b), float(de)] for d, (b, de) in raw], dtype=float,
            )
        else:
            diag = np.empty((0, 3), dtype=float)
        per_dim = {
            k: diagram_dim_slice(diag, k, max_edge)
            for k in range(MAX_HOM_DIM[manifold] + 1)
        }
        baselines[manifold] = per_dim
        log.info(
            "baseline barcode %s: per_dim shapes=%s",
            manifold, {k: v.shape[0] for k, v in per_dim.items()},
        )
    return baselines


def _wasserstein_for_flow(diagram, baseline_per_dim: dict, max_edge: float, max_hom_dim: int) -> float:
    """Per-flow Wasserstein-2 distance to baseline (summed across H-dims)."""
    from gudhi.hera import wasserstein_distance as wdist

    total = 0.0
    for k in range(max_hom_dim + 1):
        flow_bd = diagram_dim_slice(diagram, k, max_edge)
        total += float(wdist(flow_bd, baseline_per_dim[k], order=2.0))
    return total


def distances_for_split(
    manifold: str, split: str, baseline_per_dim: dict, max_edge: float,
    max_hom_dim: int, n_jobs: int,
):
    """Compute per-flow Wasserstein distances for one (manifold, split) in parallel."""
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm import tqdm

    log = logging.getLogger("pipeline.unsupervised")
    diagrams = load_diagrams_pkl(manifold, split)
    log.info("computing W: manifold=%s split=%s flows=%d", manifold, split, len(diagrams))
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    gen = parallel(
        delayed(_wasserstein_for_flow)(d, baseline_per_dim, max_edge, max_hom_dim)
        for d in diagrams
    )
    return np.array(list(tqdm(gen, total=len(diagrams), desc=f"W/{manifold}/{split}")))


def validate_distances(distances: dict) -> None:
    """Raise if any computed Wasserstein distance is negative."""
    import numpy as np

    for key, arr in distances.items():
        if (np.asarray(arr) < 0).any():
            raise AssertionError(f"{key}: negative Wasserstein distance")


def compute_thresholds(distances_val: dict, val_labels) -> dict:
    """Per-manifold threshold = 95th percentile of val Normal-Traffic distances."""
    import numpy as np

    mask = (val_labels == "Normal Traffic").values
    return {
        manifold: float(np.percentile(distances_val[manifold][mask], THRESHOLD_PERCENTILE))
        for manifold in MANIFOLDS
    }


def flag_distances(distances: dict, thresholds: dict) -> dict:
    """Return {manifold: bool array} where True means distance > threshold."""
    import numpy as np

    return {
        manifold: (np.asarray(distances[manifold]) > thresholds[manifold])
        for manifold in MANIFOLDS
    }


def derive_inference_rule(train_flags: dict, train_labels) -> dict:
    """Return {class_name: (c2, network, physical) most common pattern in train}."""
    from collections import Counter

    rule: dict = {}
    for cls in EXPECTED_CLASSES:
        mask = (train_labels == cls).values
        if not mask.any():
            continue
        patterns = list(
            zip(
                train_flags["c2"][mask].astype(int).tolist(),
                train_flags["network"][mask].astype(int).tolist(),
                train_flags["physical"][mask].astype(int).tolist(),
            )
        )
        rule[cls] = tuple(int(x) for x in Counter(patterns).most_common(1)[0][0])
    return rule


def apply_inference_rule(rule: dict, flags: dict) -> list:
    """Map per-flow (c2, network, physical) flag patterns to predicted class names."""
    # Invert {class: pattern} → {pattern: class}; on collision keep first per insertion order
    # (Python dicts preserve insertion order from 3.7+).
    pattern_to_class: dict = {}
    for cls, pat in rule.items():
        pattern_to_class.setdefault(pat, cls)
    default = "Normal Traffic"
    c2 = flags["c2"].astype(int).tolist()
    net = flags["network"].astype(int).tolist()
    phy = flags["physical"].astype(int).tolist()
    return [pattern_to_class.get((c, n, p), default) for c, n, p in zip(c2, net, phy)]


def build_distances_frame(
    distances_by_split: dict, flags_by_split: dict, labels_by_split: dict,
    predicted_by_split: dict,
):
    """Combine per-split distance / flag / label arrays into one long-form DataFrame."""
    import pandas as pd

    frames = []
    for split in ("train", "val", "test"):
        d = distances_by_split[split]
        f = flags_by_split[split]
        frames.append(
            pd.DataFrame({
                "split": split,
                "label": labels_by_split[split].values,
                "c2_distance": d["c2"],
                "network_distance": d["network"],
                "physical_distance": d["physical"],
                "c2_flag": f["c2"].astype(int),
                "network_flag": f["network"].astype(int),
                "physical_flag": f["physical"].astype(int),
                "predicted_class": predicted_by_split[split],
            })
        )
    return pd.concat(frames, ignore_index=True)


def compute_per_class_auc(distances: dict, labels) -> "pandas.DataFrame":
    """Per (manifold, attack class) AUC: distance vs Normal-vs-class binary labels."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    rows = []
    for manifold in MANIFOLDS:
        d = np.asarray(distances[manifold])
        for cls in EXPECTED_CLASSES:
            if cls == "Normal Traffic":
                continue
            mask = (labels == cls) | (labels == "Normal Traffic")
            y = (labels[mask] == cls).astype(int).values
            try:
                auc = float(roc_auc_score(y, d[mask.values]))
            except ValueError:
                auc = float("nan")
            rows.append({"manifold": manifold, "attack_class": cls, "auc": auc})
    return pd.DataFrame(rows)


def warn_low_auc(per_class_auc) -> None:
    """Log a warning for every (manifold, class) cell whose AUC is below 0.5."""
    log = logging.getLogger("pipeline.unsupervised")
    suspect = per_class_auc[per_class_auc["auc"] < 0.5]
    if suspect.empty:
        return
    for _, row in suspect.iterrows():
        log.warning(
            "per-class AUC inversion: manifold=%s class=%s auc=%.4f (audit needed)",
            row["manifold"], row["attack_class"], row["auc"],
        )


def compute_overall_metrics(distances: dict, predicted, labels) -> dict:
    """Binary normal-vs-attack AUC + multi-class inference-rule accuracy."""
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score

    is_attack = (labels != "Normal Traffic").astype(int).values
    # Pool: per-flow max distance across manifolds (higher = more anomalous).
    pooled = np.maximum.reduce([np.asarray(distances[m]) for m in MANIFOLDS])
    try:
        bin_auc = float(roc_auc_score(is_attack, pooled))
    except ValueError:
        bin_auc = float("nan")
    return {
        "binary_normal_vs_attack_auc": bin_auc,
        "multiclass_inference_accuracy": float(accuracy_score(labels.values, predicted)),
        "n_flows": int(len(labels)),
    }


def plot_distance_distributions(distances_df, out_path) -> None:
    """3×5 grid of histograms: distance distribution per (manifold, label class)."""
    import matplotlib.pyplot as plt

    test_df = distances_df[distances_df["split"] == "test"]
    fig, axes = plt.subplots(
        len(MANIFOLDS), len(EXPECTED_CLASSES), figsize=(18, 9), sharex="row",
    )
    for i, manifold in enumerate(MANIFOLDS):
        col = f"{manifold}_distance"
        for j, cls in enumerate(EXPECTED_CLASSES):
            ax = axes[i, j]
            sub = test_df[test_df["label"] == cls][col]
            ax.hist(sub, bins=50, color="steelblue", alpha=0.85)
            if i == 0:
                ax.set_title(cls, fontsize=9)
            if j == 0:
                ax.set_ylabel(manifold, fontsize=10)
    fig.suptitle("Test-set Wasserstein distance distributions by manifold × class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_unsupervised_roc(distances_df, out_path) -> None:
    """Per-manifold ROC panels with one curve per attack class (test split only)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    test_df = distances_df[distances_df["split"] == "test"]
    fig, axes = plt.subplots(1, len(MANIFOLDS), figsize=(15, 5), sharey=True)
    for i, manifold in enumerate(MANIFOLDS):
        ax = axes[i]
        col = f"{manifold}_distance"
        for cls in EXPECTED_CLASSES:
            if cls == "Normal Traffic":
                continue
            mask = test_df["label"].isin(["Normal Traffic", cls])
            y = (test_df.loc[mask, "label"] == cls).astype(int).values
            score = test_df.loc[mask, col].values
            fpr, tpr, _ = roc_curve(y, score)
            ax.plot(fpr, tpr, label=cls)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(manifold)
        ax.set_xlabel("FPR")
        if i == 0:
            ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    fig.suptitle("Test-set ROC (Normal vs each attack) per manifold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pattern_heatmap(distances_df, out_path) -> None:
    """Heatmap of observed flag-pattern frequency per true class on the test split."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    test_df = distances_df[distances_df["split"] == "test"].copy()
    test_df["pattern"] = (
        test_df["c2_flag"].astype(str)
        + test_df["network_flag"].astype(str)
        + test_df["physical_flag"].astype(str)
    )
    counts = (
        test_df.groupby(["label", "pattern"]).size().unstack(fill_value=0)
        .reindex(EXPECTED_CLASSES, axis=0, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("flag pattern (c2 net phy)")
    ax.set_ylabel("true class")
    ax.set_title("Test-set flag-pattern frequency per class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_all_distances(baselines: dict, max_edge_lengths: dict) -> dict:
    """Run Wasserstein for every (manifold, split); return nested distances dict."""
    log = logging.getLogger("pipeline.unsupervised")
    distances_by_split: dict = {s: {} for s in ("train", "val", "test")}
    for manifold in MANIFOLDS:
        baseline = baselines[manifold]
        max_edge = max_edge_lengths[manifold]
        max_hom_dim = MAX_HOM_DIM[manifold]
        for split in ("train", "val", "test"):
            distances_by_split[split][manifold] = distances_for_split(
                manifold, split, baseline, max_edge, max_hom_dim, n_jobs=-1,
            )
    log.info("all Wasserstein distances computed")
    return distances_by_split


def save_unsupervised_tables(
    distances_df, per_class_auc, overall_metrics: dict, rule: dict,
) -> None:
    """Persist all four output CSVs to results/tables/."""
    import pandas as pd

    distances_df.to_csv(TABLES_DIR / "unsupervised_distances.csv", index=False)
    per_class_auc.to_csv(TABLES_DIR / "unsupervised_per_class_auc.csv", index=False)
    pd.DataFrame([overall_metrics]).to_csv(
        TABLES_DIR / "unsupervised_overall_metrics.csv", index=False,
    )
    rule_rows = [
        {"class": cls, "c2_flag": p[0], "network_flag": p[1], "physical_flag": p[2]}
        for cls, p in rule.items()
    ]
    pd.DataFrame(rule_rows).to_csv(TABLES_DIR / "inference_rule.csv", index=False)


def fit_rule_and_predict(
    distances_by_split: dict, labels_by_split: dict, thresholds: dict,
) -> tuple:
    """Compute flags per split, derive rule on train, apply rule to every split."""
    import json

    log = logging.getLogger("pipeline.unsupervised")
    flags_by_split = {
        split: flag_distances(distances_by_split[split], thresholds)
        for split in ("train", "val", "test")
    }
    rule = derive_inference_rule(flags_by_split["train"], labels_by_split["train"])
    with (OUTPUTS_DIR / "inference_rule.json").open("w") as fh:
        json.dump({cls: list(pat) for cls, pat in rule.items()}, fh, indent=2)
    log.info("inference rule: %s", rule)
    predicted_by_split = {
        split: apply_inference_rule(rule, flags_by_split[split])
        for split in ("train", "val", "test")
    }
    return rule, flags_by_split, predicted_by_split


def produce_unsupervised_plots(distances_df) -> None:
    """Render the three Phase 6 figures into results/figures/."""
    plot_distance_distributions(
        distances_df, FIGURES_DIR / "unsupervised_distance_distributions.png",
    )
    plot_unsupervised_roc(distances_df, FIGURES_DIR / "unsupervised_roc_curves.png")
    plot_pattern_heatmap(distances_df, FIGURES_DIR / "unsupervised_pattern_heatmap.png")


def cmd_unsupervised(args: argparse.Namespace) -> None:
    """Phase 6: Wasserstein-distance anomaly detection and pattern-based attack typing."""
    import json

    import numpy as np

    log = logging.getLogger("pipeline.unsupervised")
    _, backend_name = get_wasserstein_backend()
    log.info("Wasserstein backend: %s", backend_name)

    with (OUTPUTS_DIR / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)
    reference_indices = np.load(OUTPUTS_DIR / "reference_indices.npy")

    baselines = compute_baseline_barcodes(max_edge_lengths, reference_indices)
    distances_by_split = compute_all_distances(baselines, max_edge_lengths)
    for split, dists in distances_by_split.items():
        validate_distances({f"{split}_{m}": dists[m] for m in MANIFOLDS})

    labels_by_split = {s: load_labels_for_split(s) for s in ("train", "val", "test")}
    thresholds = compute_thresholds(distances_by_split["val"], labels_by_split["val"])
    with (OUTPUTS_DIR / "thresholds.json").open("w") as fh:
        json.dump(thresholds, fh, indent=2)
    log.info("thresholds: %s", thresholds)

    rule, flags_by_split, predicted_by_split = fit_rule_and_predict(
        distances_by_split, labels_by_split, thresholds,
    )
    distances_df = build_distances_frame(
        distances_by_split, flags_by_split, labels_by_split, predicted_by_split,
    )

    test_labels = labels_by_split["test"]
    per_class_auc = compute_per_class_auc(distances_by_split["test"], test_labels)
    warn_low_auc(per_class_auc)
    overall = compute_overall_metrics(
        distances_by_split["test"], predicted_by_split["test"], test_labels,
    )
    log.info("test overall: %s", overall)
    save_unsupervised_tables(distances_df, per_class_auc, overall, rule)
    produce_unsupervised_plots(distances_df)

    log.info("unsupervised complete -> tables=%s figures=%s", TABLES_DIR, FIGURES_DIR)


# ==== SECTION 11: PHASE 7 — EVALUATION + REPORTING ====
# Distills Phases 5 + 6 outputs into paper-ready tables and figures.
# Runs two ablations (drop each manifold, drop each feature group) on top of
# the combined+RF supervised setup. Renders three figures (methodology
# flowchart, persistence-barcode examples per class × {C2, Network}, and the
# composite headline-results figure). Logs a final summary block of every
# number a paper reviewer would expect.


def classify_columns(columns: list) -> dict:
    """Bucket feature columns into original / summary / images / other."""
    original_set = set(C2_FEATURES) | set(NETWORK_FEATURES) | set(PHYSICAL_FEATURES)
    summary_suffixes = tuple(f"_{stat}" for stat in SUMMARY_STAT_NAMES)
    out: dict = {"original": [], "summary": [], "images": [], "other": []}
    for col in columns:
        if col in original_set:
            out["original"].append(col)
        elif "_img_" in col:
            out["images"].append(col)
        elif col.endswith(summary_suffixes):
            out["summary"].append(col)
        else:
            out["other"].append(col)
    return out


def columns_for_manifold(columns: list, manifold: str) -> list:
    """Return columns belonging to one manifold (original, summary, or image)."""
    original_lookup = {
        "c2": set(C2_FEATURES),
        "network": set(NETWORK_FEATURES),
        "physical": set(PHYSICAL_FEATURES),
    }
    originals = original_lookup[manifold]
    prefix = f"{manifold}_H"
    return [c for c in columns if c in originals or c.startswith(prefix)]


def best_supervised_per_feature_set(summary_df) -> "pandas.DataFrame":
    """Pick the (model, curated) row with the highest accuracy_mean per feature set."""
    non_curated = summary_df[~summary_df["curated"]]
    rows = []
    for fs in FEATURE_SETS:
        sel = non_curated[non_curated["feature_set"] == fs]
        if sel.empty:
            continue
        best = sel.loc[sel["accuracy_mean"].idxmax()].to_dict()
        rows.append(best)
    return __import__("pandas").DataFrame(rows)


def build_final_supervised(summary_df) -> "pandas.DataFrame":
    """Final paper table: best model per feature set with mean ± std for headline metrics."""
    import pandas as pd

    best = best_supervised_per_feature_set(summary_df)
    keep_cols = [
        "feature_set", "model",
        "accuracy_mean", "accuracy_std",
        "weighted_f1_mean", "weighted_f1_std",
        "macro_f1_mean", "macro_f1_std",
        "weighted_auc_mean", "weighted_auc_std",
    ]
    return best[[c for c in keep_cols if c in best.columns]].reset_index(drop=True)


def build_final_unsupervised(per_class_auc, overall) -> "pandas.DataFrame":
    """Per-attack AUC by manifold + pattern-classifier metrics on test."""
    import pandas as pd

    pivot = per_class_auc.pivot(index="attack_class", columns="manifold", values="auc")
    pivot = pivot.reset_index()
    pivot.columns.name = None
    overall_row = pd.DataFrame(
        [{
            "attack_class": "<overall>",
            **{m: float("nan") for m in MANIFOLDS},
            "binary_normal_vs_attack_auc": float(overall["binary_normal_vs_attack_auc"].iloc[0]),
            "multiclass_inference_accuracy": float(overall["multiclass_inference_accuracy"].iloc[0]),
        }]
    )
    return pd.concat([pivot, overall_row], ignore_index=True)


def fit_rf_with_seed(X_train, y_train, seed: int):
    """Fit an RF with the spec's typical-best params; used by ablations."""
    return build_model("rf", seed, n_estimators=300, max_depth=None).fit(X_train, y_train)


def evaluate_rf_on_test(model, X_test, y_test) -> dict:
    """Return accuracy + weighted F1 + weighted AUC for one fitted RF."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "weighted_auc": float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        ),
    }


def run_ablation_one_setting(label: str, drop_cols: list) -> list:
    """Train + evaluate RF on combined-minus-drop_cols across all SEEDS."""
    log = logging.getLogger("pipeline.evaluate")
    rows = []
    for seed in SEEDS:
        X_train = load_feature_split("combined", "train").drop(columns=drop_cols, errors="ignore")
        X_test = load_feature_split("combined", "test").drop(columns=drop_cols, errors="ignore")
        y_train = load_labels_for("train")
        y_test = load_labels_for("test")
        model = fit_rf_with_seed(X_train, y_train, seed)
        metrics = evaluate_rf_on_test(model, X_test, y_test)
        rows.append({"ablation": label, "seed": seed, "n_features": X_train.shape[1], **metrics})
        log.info("ablation %s seed=%d acc=%.4f", label, seed, metrics["accuracy"])
    return rows


def run_manifold_ablation() -> "pandas.DataFrame":
    """Drop each manifold's contribution from `combined`; refit RF; report deltas."""
    import pandas as pd

    combined_cols = list(load_feature_split("combined", "train").columns)
    rows = []
    for manifold in MANIFOLDS:
        drop_cols = columns_for_manifold(combined_cols, manifold)
        rows.extend(run_ablation_one_setting(f"drop_{manifold}", drop_cols))
    return pd.DataFrame(rows)


def run_feature_group_ablation() -> "pandas.DataFrame":
    """Drop each feature group (original / summary / images) from `combined`; refit RF."""
    import pandas as pd

    combined_cols = list(load_feature_split("combined", "train").columns)
    classified = classify_columns(combined_cols)
    rows = []
    for group in ("original", "summary", "images"):
        rows.extend(run_ablation_one_setting(f"drop_{group}", classified[group]))
    return pd.DataFrame(rows)


def summarize_ablation(df) -> "pandas.DataFrame":
    """Collapse per-seed ablation rows to mean ± std per `ablation` label."""
    metric_cols = [c for c in df.columns if c not in {"ablation", "seed", "n_features"}]
    means = df.groupby("ablation")[metric_cols].mean().add_suffix("_mean")
    stds = df.groupby("ablation")[metric_cols].std().add_suffix("_std")
    n_feats = df.groupby("ablation")["n_features"].first()
    return means.join(stds).join(n_feats).reset_index()


def plot_methodology_flowchart(out_path) -> None:
    """Render a phase-by-phase flowchart of the pipeline, no embedded numbers."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    phases = [
        ("Phase 2", "Data prep:\nencode, split,\nscale per manifold"),
        ("Phase 3", "Persistence:\nRips on\n{query} ∪ ref_500"),
        ("Phase 4", "Features:\nsummary stats +\npersistence images"),
        ("Phase 5", "Supervised:\nLR / RF / SVM\n× feature sets"),
        ("Phase 6", "Unsupervised:\nWasserstein vs\nbaseline barcode"),
        ("Phase 7", "Evaluation:\nablations +\nheadline figures"),
    ]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    box_w, box_h = 1.6, 1.0
    for i, (title, body) in enumerate(phases):
        x = i * 2.2
        rect = patches.FancyBboxPatch(
            (x, 0), box_w, box_h, boxstyle="round,pad=0.05",
            facecolor="#e8f0fe", edgecolor="#1a73e8", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, 0.8, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + box_w / 2, 0.35, body, ha="center", va="center", fontsize=8)
        if i + 1 < len(phases):
            ax.annotate("", xy=(x + 2.2, 0.5), xytext=(x + box_w, 0.5),
                        arrowprops={"arrowstyle": "->", "color": "#1a73e8"})
    ax.set_xlim(-0.2, len(phases) * 2.2)
    ax.set_ylim(-0.1, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_example_flow_per_class(labels) -> dict:
    """Return {class_name: first test-row index with that label}."""
    out: dict = {}
    labels_arr = labels.values if hasattr(labels, "values") else labels
    for cls in EXPECTED_CLASSES:
        matches = (labels_arr == cls).nonzero()[0]
        if len(matches) > 0:
            out[cls] = int(matches[0])
    return out


def plot_barcode_examples(out_path) -> None:
    """One test-flow barcode per class × {C2, Network}; bars colored by H-dim."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels_test = load_labels_for_split("test")
    examples = pick_example_flow_per_class(labels_test)
    manifolds_to_show = ["c2", "network"]
    fig, axes = plt.subplots(
        len(manifolds_to_show), len(EXPECTED_CLASSES), figsize=(20, 6), sharex="row",
    )
    cmap = plt.get_cmap("tab10")
    for r, manifold in enumerate(manifolds_to_show):
        diagrams = load_diagrams_pkl(manifold, "test")
        for c, cls in enumerate(EXPECTED_CLASSES):
            ax = axes[r, c]
            if cls not in examples:
                ax.set_visible(False)
                continue
            diag = diagrams[examples[cls]]
            for i, (dim, birth, death) in enumerate(diag):
                if np.isinf(death):
                    death = max(birth + 0.5, birth * 1.1)
                ax.hlines(i, birth, death, color=cmap(int(dim)), lw=1.0)
            if r == 0:
                ax.set_title(cls, fontsize=9)
            if c == 0:
                ax.set_ylabel(manifold, fontsize=10)
            ax.set_yticks([])
    fig.suptitle("Persistence barcodes: one test flow per class × manifold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_headline_results(summary_df, metrics_df, out_path) -> None:
    """Composite paper Fig 4: baseline-vs-best-TDA bars per model + per-class AUC heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    non_curated = summary_df[~summary_df["curated"]]
    baseline = non_curated[non_curated["feature_set"] == "original"].set_index("model")
    tda_sets = [fs for fs in FEATURE_SETS if fs != "original"]
    best_tda = (
        non_curated[non_curated["feature_set"].isin(tda_sets)]
        .sort_values("accuracy_mean", ascending=False)
        .groupby("model").head(1).set_index("model")
    )

    auc_cols = [c for c in metrics_df.columns if c.startswith("auc_")]
    # Pick the overall best non-curated row for the heatmap.
    best_overall_idx = non_curated["accuracy_mean"].idxmax()
    best_overall = non_curated.loc[best_overall_idx]
    matching = metrics_df[
        (metrics_df["feature_set"] == best_overall["feature_set"])
        & (metrics_df["model"] == best_overall["model"])
        & (~metrics_df["curated"])
    ]
    per_class_means = matching[auc_cols].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(MODEL_NAMES))
    axes[0].bar(x - 0.2, [baseline.loc[m, "accuracy_mean"] for m in MODEL_NAMES],
                width=0.4, label="original", color="tab:gray")
    axes[0].bar(x + 0.2, [best_tda.loc[m, "accuracy_mean"] for m in MODEL_NAMES],
                width=0.4, label="best TDA", color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(MODEL_NAMES))
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Baseline vs best TDA feature set per model")
    axes[0].legend()

    heatmap_df = pd.DataFrame({
        "class": [c.replace("auc_", "") for c in auc_cols],
        "AUC": per_class_means.values,
    }).set_index("class")
    sns.heatmap(heatmap_df.T, annot=True, fmt=".3f", cmap="YlGn", ax=axes[1],
                cbar_kws={"label": "per-class AUC"}, vmin=0.5, vmax=1.0)
    axes[1].set_title(f"Per-class AUC — best cell ({best_overall['feature_set']} / {best_overall['model']})")
    fig.suptitle("Headline results")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def log_final_summary(
    final_sup, final_unsup, manifold_ablation_summary, feature_ablation_summary,
) -> None:
    """Emit a single human-readable block with every paper-abstract number."""
    log = logging.getLogger("pipeline.evaluate")
    log.info("=" * 70)
    log.info("FINAL SUMMARY — values are mean ± std over 3 seeds where applicable")
    log.info("=" * 70)
    log.info("Supervised best per feature set:")
    for _, row in final_sup.iterrows():
        log.info(
            "  %-22s %-7s  acc=%.4f ± %.4f   w-AUC=%.4f ± %.4f",
            row["feature_set"], row["model"],
            row["accuracy_mean"], row["accuracy_std"],
            row["weighted_auc_mean"], row["weighted_auc_std"],
        )
    log.info("Unsupervised:")
    for _, row in final_unsup.iterrows():
        if row["attack_class"] == "<overall>":
            log.info(
                "  overall   binary AUC=%.4f   inference acc=%.4f",
                row["binary_normal_vs_attack_auc"], row["multiclass_inference_accuracy"],
            )
        else:
            log.info(
                "  %-18s c2=%.3f net=%.3f phy=%.3f",
                row["attack_class"], row["c2"], row["network"], row["physical"],
            )
    log.info("Ablation — drop manifold (RF on combined minus that manifold):")
    for _, row in manifold_ablation_summary.iterrows():
        log.info(
            "  %-15s acc=%.4f ± %.4f  (n_features=%d)",
            row["ablation"], row["accuracy_mean"], row["accuracy_std"], row["n_features"],
        )
    log.info("Ablation — drop feature group:")
    for _, row in feature_ablation_summary.iterrows():
        log.info(
            "  %-15s acc=%.4f ± %.4f  (n_features=%d)",
            row["ablation"], row["accuracy_mean"], row["accuracy_std"], row["n_features"],
        )
    log.info("=" * 70)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Phase 7: final tables, ablations, paper figures, and a printable summary block."""
    import pandas as pd

    log = logging.getLogger("pipeline.evaluate")
    summary_df = pd.read_csv(TABLES_DIR / "supervised_summary.csv")
    metrics_df = pd.read_csv(TABLES_DIR / "supervised_metrics.csv")
    per_class_auc = pd.read_csv(TABLES_DIR / "unsupervised_per_class_auc.csv")
    overall_unsup = pd.read_csv(TABLES_DIR / "unsupervised_overall_metrics.csv")

    final_sup = build_final_supervised(summary_df)
    final_unsup = build_final_unsupervised(per_class_auc, overall_unsup)
    final_sup.to_csv(TABLES_DIR / "final_supervised.csv", index=False)
    final_unsup.to_csv(TABLES_DIR / "final_unsupervised.csv", index=False)

    log.info("running manifold ablation")
    manifold_ablation = run_manifold_ablation()
    log.info("running feature-group ablation")
    feature_ablation = run_feature_group_ablation()
    manifold_ablation.to_csv(TABLES_DIR / "ablation_manifolds.csv", index=False)
    feature_ablation.to_csv(TABLES_DIR / "ablation_features.csv", index=False)
    manifold_ablation_summary = summarize_ablation(manifold_ablation)
    feature_ablation_summary = summarize_ablation(feature_ablation)

    plot_methodology_flowchart(FIGURES_DIR / "methodology_flowchart.png")
    plot_barcode_examples(FIGURES_DIR / "persistence_barcode_examples.png")
    plot_headline_results(summary_df, metrics_df, FIGURES_DIR / "headline_results.png")

    log_final_summary(
        final_sup, final_unsup, manifold_ablation_summary, feature_ablation_summary,
    )
    log.info("evaluate complete -> tables=%s figures=%s", TABLES_DIR, FIGURES_DIR)


def cmd_all(args: argparse.Namespace) -> None:
    """Run every implemented phase in order (prep → tda → features → supervised → unsupervised → evaluate)."""
    log = logging.getLogger("pipeline.all")
    log.info("=== running full pipeline ===")
    # cmd_tda expects subcommand-specific attrs that the `all` subparser doesn't define.
    for attr, default in (("manifold", "all"), ("split", "all"), ("seed", PRIMARY_SEED)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    for phase in (cmd_prep, cmd_tda, cmd_features, cmd_supervised, cmd_unsupervised, cmd_evaluate):
        log.info("--- starting %s ---", phase.__name__)
        phase(args)
    log.info("=== pipeline complete ===")


COMMAND_DISPATCH: dict[str, Callable[[argparse.Namespace], None]] = {
    "prep": cmd_prep,
    "tda": cmd_tda,
    "features": cmd_features,
    "supervised": cmd_supervised,
    "unsupervised": cmd_unsupervised,
    "evaluate": cmd_evaluate,
    "all": cmd_all,
}


# ==== SECTION 12: CLI ====


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands and shared flags."""
    # Shared flags attached to every subparser so `pipeline.py <cmd> --debug` works.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        help=f"Sample only the first {DEBUG_SAMPLE_ROWS} rows of the dataset.",
    )
    common.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Multi-manifold persistent-homology pipeline for UAVIDS-2025.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="phase")

    sub.add_parser(
        "prep", parents=[common],
        help="Phase 2: load, split, scale, and save per-manifold CSVs.",
    )

    tda_p = sub.add_parser(
        "tda", parents=[common],
        help="Phase 3: compute per-flow persistence diagrams.",
    )
    tda_p.add_argument(
        "--manifold",
        choices=("c2", "network", "physical", "all"),
        default="all",
        help="Which manifold(s) to compute persistence for.",
    )
    tda_p.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which split(s) to compute diagrams for.",
    )
    tda_p.add_argument(
        "--seed", type=int, default=PRIMARY_SEED,
        help="Seed for the reference-cloud sampler.",
    )

    sub.add_parser(
        "features", parents=[common],
        help="Phase 4: extract summary stats and persistence images.",
    )
    sub.add_parser(
        "supervised", parents=[common],
        help="Phase 5: train and evaluate supervised classifiers.",
    )
    sub.add_parser(
        "unsupervised", parents=[common],
        help="Phase 6: Wasserstein-based detection and attack-type inference.",
    )
    sub.add_parser(
        "evaluate", parents=[common],
        help="Phase 7: ablation, final tables, and paper figures.",
    )
    sub.add_parser(
        "all", parents=[common],
        help="Run every phase in order.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, set up environment, and dispatch the requested phase."""
    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_directories()
    setup_logging(args.verbose)

    COMMAND_DISPATCH[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
