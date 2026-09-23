"""
06_tda_features_extraction.py

TDA Feature Extraction

Extracts statistical features from persistence diagrams to use in
supervised classification. Converts topological information into
numerical features that can be fed into ML models.

Features extracted per homology dimension:
- Number of features
- Mean/std/min/max persistence
- Mean/std birth times
- Mean/std death times
- Total persistence
- Persistence entropy

DATA LEAKAGE PREVENTION:
- Loads separate train/test persistence diagrams (produced by Scripts 03-05)
- Extracts features independently for each split
- Saves separate train/test feature files consumed by Script 07
- The combined_features files preserve the original row order of the full
  dataset so that train_indices / test_indices index them correctly

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Paths
OUTPUT_DIR = "outputs"
PERSISTENCE_DIR = f"{OUTPUT_DIR}/persistence_diagrams"
FEATURES_DIR = f"{OUTPUT_DIR}/tda_features"

os.makedirs(FEATURES_DIR, exist_ok=True)

print("=" * 80)
print("MULTI-MANIFOLD TDA PIPELINE - STEP 6: FEATURE EXTRACTION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD PERSISTENCE DIAGRAMS (TRAIN AND TEST SEPARATELY)
# ==============================================================================
print("Loading persistence diagrams from all three manifolds (train + test splits)...")
print("-" * 80)

try:
    # Train diagrams
    c2_train = np.load(f"{PERSISTENCE_DIR}/c2_persistence_diagrams_train.npy")
    network_train = np.load(f"{PERSISTENCE_DIR}/network_persistence_diagrams_train.npy")
    physical_train = np.load(f"{PERSISTENCE_DIR}/physical_persistence_diagrams_train.npy")

    # Test diagrams
    c2_test = np.load(f"{PERSISTENCE_DIR}/c2_persistence_diagrams_test.npy")
    network_test = np.load(f"{PERSISTENCE_DIR}/network_persistence_diagrams_test.npy")
    physical_test = np.load(f"{PERSISTENCE_DIR}/physical_persistence_diagrams_test.npy")

    # Labels and split indices
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]
    train_indices = np.load(f"{OUTPUT_DIR}/train_indices.npy")
    test_indices = np.load(f"{OUTPUT_DIR}/test_indices.npy")

    print(f"✓ C2 train diagrams:       {c2_train.shape}")
    print(f"✓ C2 test diagrams:        {c2_test.shape}")
    print(f"✓ Network train diagrams:  {network_train.shape}")
    print(f"✓ Network test diagrams:   {network_test.shape}")
    print(f"✓ Physical train diagrams: {physical_train.shape}")
    print(f"✓ Physical test diagrams:  {physical_test.shape}")
    print(f"✓ Labels: {len(labels):,} samples total")
    print(f"✓ Train indices: {len(train_indices):,}  |  Test indices: {len(test_indices):,}\n")

    # Sanity checks
    assert len(c2_train) == len(train_indices), (
        f"C2 train diagram count ({len(c2_train)}) != train_indices ({len(train_indices)})"
    )
    assert len(c2_test) == len(test_indices), (
        f"C2 test diagram count ({len(c2_test)}) != test_indices ({len(test_indices)})"
    )
    print("✓ Shape sanity checks passed\n")

except FileNotFoundError as e:
    print(f"✗ ERROR: Missing persistence diagrams: {e}")
    print("  Please run Scripts 3-5 first.")
    exit(1)

# ==============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ==============================================================================
print("Defining feature extraction functions...")
print("-" * 80)


def extract_features_from_diagram(diagram, manifold_name):
    """
    Extract statistical features from a single persistence diagram.

    diagram: (max_features, 3) array where columns are (birth, death, dimension)

    Returns: dictionary of features
    """
    features = {}

    # Extract for each homology dimension
    for dim in [0, 1, 2]:
        dim_mask = diagram[:, 2] == dim

        if not dim_mask.any():
            # No features of this dimension
            prefix = f"{manifold_name}_H{dim}"
            features[f"{prefix}_count"] = 0
            features[f"{prefix}_mean_persistence"] = 0
            features[f"{prefix}_std_persistence"] = 0
            features[f"{prefix}_max_persistence"] = 0
            features[f"{prefix}_total_persistence"] = 0
            features[f"{prefix}_mean_birth"] = 0
            features[f"{prefix}_mean_death"] = 0
            features[f"{prefix}_entropy"] = 0
            continue

        # Get birth, death times for this dimension
        births = diagram[dim_mask, 0]
        deaths = diagram[dim_mask, 1]

        # Filter out zero features (padding)
        valid_mask = (deaths - births) > 0
        births = births[valid_mask]
        deaths = deaths[valid_mask]

        if len(births) == 0:
            # All features were padding
            prefix = f"{manifold_name}_H{dim}"
            features[f"{prefix}_count"] = 0
            features[f"{prefix}_mean_persistence"] = 0
            features[f"{prefix}_std_persistence"] = 0
            features[f"{prefix}_max_persistence"] = 0
            features[f"{prefix}_total_persistence"] = 0
            features[f"{prefix}_mean_birth"] = 0
            features[f"{prefix}_mean_death"] = 0
            features[f"{prefix}_entropy"] = 0
            continue

        # Compute persistence (lifetime)
        persistence = deaths - births

        # Feature prefix
        prefix = f"{manifold_name}_H{dim}"

        # Extract features
        features[f"{prefix}_count"] = len(persistence)
        features[f"{prefix}_mean_persistence"] = persistence.mean()
        features[f"{prefix}_std_persistence"] = persistence.std()
        features[f"{prefix}_max_persistence"] = persistence.max()
        features[f"{prefix}_total_persistence"] = persistence.sum()
        features[f"{prefix}_mean_birth"] = births.mean()
        features[f"{prefix}_mean_death"] = deaths.mean()

        # Persistence entropy (measure of complexity)
        if persistence.sum() > 0:
            probs = persistence / persistence.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            features[f"{prefix}_entropy"] = entropy
        else:
            features[f"{prefix}_entropy"] = 0

    return features


def extract_features_for_split(c2_diags, network_diags, physical_diags, split_name):
    """
    Extract TDA features for all samples in one split.
    All three diagram arrays must have the same first dimension.
    """
    n = len(c2_diags)
    assert len(network_diags) == n and len(physical_diags) == n

    all_features = []
    start_time = datetime.now()

    for idx in range(n):
        sample_features = {}
        sample_features.update(extract_features_from_diagram(c2_diags[idx], "C2"))
        sample_features.update(extract_features_from_diagram(network_diags[idx], "Network"))
        sample_features.update(extract_features_from_diagram(physical_diags[idx], "Physical"))
        all_features.append(sample_features)

        if (idx + 1) % 10000 == 0 or (idx + 1) == n:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(
                f"  [{split_name}] {idx+1:,}/{n:,} "
                f"({(idx+1)/n*100:.1f}%) | Elapsed: {elapsed:.1f}s"
            )

    return pd.DataFrame(all_features).fillna(0)


print("✓ Feature extraction functions defined\n")

# ==============================================================================
# EXTRACT FEATURES — TRAIN SPLIT
# ==============================================================================
print("=" * 80)
print("EXTRACTING TDA FEATURES — TRAIN SPLIT")
print("=" * 80)

tda_train_df = extract_features_for_split(
    c2_train, network_train, physical_train, "train"
)
print(f"\n✓ Train TDA features: {tda_train_df.shape}")

# ==============================================================================
# EXTRACT FEATURES — TEST SPLIT
# ==============================================================================
print()
print("=" * 80)
print("EXTRACTING TDA FEATURES — TEST SPLIT")
print("=" * 80)

tda_test_df = extract_features_for_split(
    c2_test, network_test, physical_test, "test"
)
print(f"\n✓ Test TDA features: {tda_test_df.shape}")

# ==============================================================================
# RECONSTRUCT FULL-DATASET ORDER
# Reassemble a (n_total, n_features) array in the original row order so that
# train_indices and test_indices index it correctly in Script 07.
# ==============================================================================
print()
print("-" * 80)
print("Reassembling full-dataset feature matrix in original row order...")
print("-" * 80)

n_total = len(labels)
n_features = tda_train_df.shape[1]
feature_names = list(tda_train_df.columns)

tda_full_array = np.zeros((n_total, n_features))
tda_full_array[train_indices] = tda_train_df.values
tda_full_array[test_indices] = tda_test_df.values

tda_features_df = pd.DataFrame(tda_full_array, columns=feature_names)

print(f"✓ Full TDA feature matrix reconstructed: {tda_features_df.shape}")
print(f"  - Train rows placed at positions: train_indices ({len(train_indices):,} rows)")
print(f"  - Test rows placed at positions:  test_indices ({len(test_indices):,} rows)")
print()

# ==============================================================================
# FEATURE SUMMARY
# ==============================================================================
print("-" * 80)
print("TDA Feature Summary:")
print("-" * 80)

manifolds = ["C2", "Network", "Physical"]
dimensions = [0, 1, 2]

for manifold in manifolds:
    manifold_cols = [col for col in tda_features_df.columns if col.startswith(manifold)]
    print(f"\n{manifold} manifold: {len(manifold_cols)} features")
    for dim in dimensions:
        dim_cols = [col for col in manifold_cols if f"_H{dim}_" in col]
        print(f"  - H{dim}: {len(dim_cols)} features")

print(f"\n✓ Total TDA features: {tda_features_df.shape[1]}")
print()

# ==============================================================================
# SAVE TDA FEATURES
# ==============================================================================
print("-" * 80)
print("Saving TDA features...")
print("-" * 80)

# --- Per-split files (primary outputs for Script 07) ---
train_labels = labels.values[train_indices]
test_labels = labels.values[test_indices]

tda_train_with_labels = tda_train_df.copy()
tda_train_with_labels["label"] = train_labels
tda_train_with_labels.to_csv(f"{FEATURES_DIR}/tda_features_train.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/tda_features_train.csv  ({tda_train_df.shape})")

tda_test_with_labels = tda_test_df.copy()
tda_test_with_labels["label"] = test_labels
tda_test_with_labels.to_csv(f"{FEATURES_DIR}/tda_features_test.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/tda_features_test.csv   ({tda_test_df.shape})")

# --- Full-dataset reconstruction (for analysis / compatibility) ---
tda_features_df.to_csv(f"{FEATURES_DIR}/tda_features.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/tda_features.csv  (full, original row order)")

tda_with_labels = tda_features_df.copy()
tda_with_labels["label"] = labels
tda_with_labels.to_csv(f"{FEATURES_DIR}/tda_features_with_labels.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/tda_features_with_labels.csv")

# Save feature names
with open(f"{FEATURES_DIR}/tda_feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print(f"✓ Saved: {FEATURES_DIR}/tda_feature_names.pkl")

# ==============================================================================
# CREATE COMBINED FEATURE SET (Original + TDA)
# ==============================================================================
print("\n" + "-" * 80)
print("Creating combined feature set (Original + TDA)...")
print("-" * 80)

# Load original features
original_features = pd.read_csv(f"{OUTPUT_DIR}/original_features.csv")
print(f"✓ Original features loaded: {original_features.shape}")

# Combine (both are in full-dataset row order)
combined_features = pd.concat([original_features, tda_features_df], axis=1)
print(f"✓ Combined features created: {combined_features.shape}")
print(f"  - Original features: {original_features.shape[1]}")
print(f"  - TDA features:      {tda_features_df.shape[1]}")
print(f"  - Total features:    {combined_features.shape[1]}")

# Save full combined
combined_features.to_csv(f"{FEATURES_DIR}/combined_features.csv", index=False)
combined_with_labels = combined_features.copy()
combined_with_labels["label"] = labels
combined_with_labels.to_csv(f"{FEATURES_DIR}/combined_features_with_labels.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/combined_features.csv")
print(f"✓ Saved: {FEATURES_DIR}/combined_features_with_labels.csv")

# Save per-split combined files for Script 07
original_arr = original_features.values

combined_train = pd.DataFrame(
    combined_features.values[train_indices],
    columns=combined_features.columns,
)
combined_train["label"] = train_labels
combined_train.to_csv(f"{FEATURES_DIR}/combined_features_train.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/combined_features_train.csv")

combined_test = pd.DataFrame(
    combined_features.values[test_indices],
    columns=combined_features.columns,
)
combined_test["label"] = test_labels
combined_test.to_csv(f"{FEATURES_DIR}/combined_features_test.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/combined_features_test.csv")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("✓ TDA FEATURE EXTRACTION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\nFeature sets created:")
print(f"  1. TDA-only features: {tda_features_df.shape[1]} features")
print(f"  2. Combined (Original + TDA): {combined_features.shape[1]} features")

print("\nFiles saved:")
print(f"  PRIMARY (for Script 07, leak-free):")
print(f"    - {FEATURES_DIR}/combined_features_train.csv  ({combined_train.shape[0]:,} rows)")
print(f"    - {FEATURES_DIR}/combined_features_test.csv   ({combined_test.shape[0]:,} rows)")
print(f"  SECONDARY (analysis / compatibility):")
print(f"    - {FEATURES_DIR}/tda_features_train.csv")
print(f"    - {FEATURES_DIR}/tda_features_test.csv")
print(f"    - {FEATURES_DIR}/tda_features.csv")
print(f"    - {FEATURES_DIR}/tda_features_with_labels.csv")
print(f"    - {FEATURES_DIR}/combined_features.csv")
print(f"    - {FEATURES_DIR}/combined_features_with_labels.csv")
print(f"    - {FEATURES_DIR}/tda_feature_names.pkl")

print("\nNext: Run 07_tda_enhanced_models.py (45-60 min)")
print("=" * 80)
