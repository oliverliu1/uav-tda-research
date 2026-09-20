"""Phase 2 — data prep, ported verbatim from pipeline.py SECTION 6.

Loads UAVIDS-2025, encodes categoricals, splits 70/15/15, scales per manifold
on training data only, and writes the artifacts every later phase consumes.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.data")


def validate_manifolds_disjoint() -> None:
    """Raise if any feature name appears in more than one manifold."""
    counts: dict[str, int] = {}
    for cols in config.MANIFOLDS.values():
        for col in cols:
            counts[col] = counts.get(col, 0) + 1
    duplicates = sorted(c for c, n in counts.items() if n > 1)
    if duplicates:
        raise AssertionError(f"features in multiple manifolds: {duplicates}")


def validate_schema(df) -> None:
    """Raise unless the dataset has exactly the expected 23 columns and 5 classes."""
    missing_cols = set(config.EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(config.EXPECTED_COLUMNS)
    if missing_cols or extra_cols:
        raise AssertionError(
            f"schema mismatch: missing={sorted(missing_cols)} extra={sorted(extra_cols)}"
        )
    found = set(df[config.LABEL_COLUMN].unique())
    missing_classes = set(config.EXPECTED_CLASSES) - found
    unexpected_classes = found - set(config.EXPECTED_CLASSES)
    if missing_classes:
        raise AssertionError(f"missing classes: {sorted(missing_classes)}")
    if unexpected_classes:
        raise AssertionError(f"unexpected classes: {sorted(unexpected_classes)}")


def load_raw_dataset(ws: Workspace, debug: bool):
    """Load the UAVIDS-2025 CSV; debug mode returns a stratified head sample.

    The CSV is sorted by class, so a literal head(5000) would give one class only.
    Debug mode therefore reads the full file and keeps the first
    DEBUG_SAMPLE_ROWS // n_classes rows per class — fast, deterministic, and
    representative.
    """
    import pandas as pd

    log.info("loading %s", ws.data_csv)
    df = pd.read_csv(ws.data_csv)
    log.info("loaded %d rows, %d columns", len(df), df.shape[1])
    validate_schema(df)
    if debug:
        per_class = config.DEBUG_SAMPLE_ROWS // len(config.EXPECTED_CLASSES)
        df = (
            df.groupby(config.LABEL_COLUMN, group_keys=False)
              .apply(lambda g: g.head(per_class))
              .reset_index(drop=True)
        )
        log.info("debug stratified sample -> %d rows (%d per class)", len(df), per_class)
    return df


def encode_features(df):
    """Drop FlowID/Protocol, encode IPs as last octet, one-hot known ports."""
    encoded = df.drop(columns=list(config.DROP_COLUMNS))
    for col in ("SrcAddr", "DstAddr"):
        encoded[f"{col}_last_octet"] = (
            encoded[col].astype(str).str.split(".").str[-1].astype(int)
        )
    encoded = encoded.drop(columns=["SrcAddr", "DstAddr"])
    for col in ("SrcPort", "DstPort"):
        unexpected = set(encoded[col].unique()) - set(config.KNOWN_PORTS)
        if unexpected:
            raise AssertionError(f"{col} has unexpected values: {sorted(unexpected)}")
        for port in config.KNOWN_PORTS:
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
        test_size=(config.VAL_FRAC + config.TEST_FRAC),
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
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
        for cls in config.EXPECTED_CLASSES:
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
        for cls in config.EXPECTED_CLASSES:
            row[cls] = int((sub == cls).sum())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("split")
    log.info("split summary:\n%s", table.to_string())


def assert_manifold_columns_present(encoded) -> None:
    """Raise if any manifold's declared columns are missing from the encoded frame."""
    for name, cols in config.MANIFOLDS.items():
        missing = set(cols) - set(encoded.columns)
        if missing:
            raise AssertionError(
                f"{name} manifold missing columns after encoding: {sorted(missing)}"
            )


def save_label_artifacts(ws: Workspace, labels, splits: dict) -> None:
    """Write split-index .npy files and per-split label CSVs to outputs/."""
    import numpy as np

    for name, idx in splits.items():
        np.save(ws.outputs_dir / f"{name}_indices.npy", idx)
        labels.iloc[idx].to_csv(ws.outputs_dir / f"labels_{name}.csv", index=False)


def process_manifolds(ws: Workspace, encoded, splits: dict) -> tuple[dict, dict]:
    """Scale each manifold on train, save per-split CSVs, return (scalers, combined_chunks)."""
    scalers: dict = {}
    combined_chunks: dict[str, list] = {name: [] for name in splits}
    for manifold_name, cols in config.MANIFOLDS.items():
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
            scaled.to_csv(ws.outputs_dir / f"{manifold_name}_{split_name}.csv", index=False)
            combined_chunks[split_name].append(scaled)
        log.info("scaled manifold=%s features=%d", manifold_name, len(cols))
    return scalers, combined_chunks


def save_combined_originals(ws: Workspace, combined_chunks: dict) -> None:
    """Concat the scaled manifolds per split and write original_features_{split}.csv."""
    import pandas as pd

    for name, chunks in combined_chunks.items():
        combined = pd.concat(chunks, axis=1)
        combined.to_csv(ws.outputs_dir / f"original_features_{name}.csv", index=False)


def run_prep(ws: Workspace, debug: bool = False, seed: int = config.PRIMARY_SEED) -> None:
    """Phase 2: load, encode, split, scale, and save per-manifold CSVs."""
    import pickle

    validate_manifolds_disjoint()

    df = load_raw_dataset(ws, debug)
    encoded = encode_features(df)
    labels = encoded[config.LABEL_COLUMN].reset_index(drop=True)
    encoded = encoded.drop(columns=[config.LABEL_COLUMN]).reset_index(drop=True)
    assert_manifold_columns_present(encoded)

    train_idx, val_idx, test_idx = stratified_three_way_split(labels, seed=seed)
    validate_split_indices(len(labels), train_idx, val_idx, test_idx)
    validate_class_balance(labels, train_idx, val_idx, test_idx)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    save_label_artifacts(ws, labels, splits)
    scalers, combined_chunks = process_manifolds(ws, encoded, splits)
    save_combined_originals(ws, combined_chunks)

    with (ws.outputs_dir / "scalers.pkl").open("wb") as fh:
        pickle.dump(scalers, fh)

    summarize_splits(labels, splits)
    log.info("prep complete -> %s", ws.outputs_dir)
