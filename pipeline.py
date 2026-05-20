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

# Hyperparameter grids (Phase 5).
LR_C_GRID = (0.1, 1.0, 10.0)
RF_N_ESTIMATORS_GRID = (100, 300)
RF_MAX_DEPTH_GRID = (10, 20, None)
SVM_C_GRID = (0.1, 1.0, 10.0)
SVM_GAMMA_GRID = ("scale", "auto")
GRID_CV_FOLDS = 3

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


# ==== SECTION 8: SUBCOMMAND STUBS ====
# Stubs for phases not yet implemented; logged so the CLI is exercisable.


def cmd_features(args: argparse.Namespace) -> None:
    """Phase 4: summary stats and persistence images from diagrams."""
    logging.getLogger("pipeline.features").info(
        "features: not implemented yet (debug=%s)", args.debug
    )


def cmd_supervised(args: argparse.Namespace) -> None:
    """Phase 5: classifier training, tuning, and evaluation."""
    logging.getLogger("pipeline.supervised").info(
        "supervised: not implemented yet (debug=%s)", args.debug
    )


def cmd_unsupervised(args: argparse.Namespace) -> None:
    """Phase 6: Wasserstein-distance anomaly detection."""
    logging.getLogger("pipeline.unsupervised").info(
        "unsupervised: not implemented yet (debug=%s)", args.debug
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Phase 7: ablations, final tables, and paper figures."""
    logging.getLogger("pipeline.evaluate").info(
        "evaluate: not implemented yet (debug=%s)", args.debug
    )


def cmd_all(args: argparse.Namespace) -> None:
    """Run every phase in sequence."""
    logging.getLogger("pipeline.all").info(
        "all: not implemented yet (debug=%s)", args.debug
    )


COMMAND_DISPATCH: dict[str, Callable[[argparse.Namespace], None]] = {
    "prep": cmd_prep,
    "tda": cmd_tda,
    "features": cmd_features,
    "supervised": cmd_supervised,
    "unsupervised": cmd_unsupervised,
    "evaluate": cmd_evaluate,
    "all": cmd_all,
}


# ==== SECTION 9: CLI ====


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
