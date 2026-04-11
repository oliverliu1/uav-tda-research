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
# LOAD PERSISTENCE DIAGRAMS
# ==============================================================================
print("Loading persistence diagrams from all three manifolds...")
print("-" * 80)

try:
    c2_diagrams = np.load(f"{PERSISTENCE_DIR}/c2_persistence_diagrams.npy")
    network_diagrams = np.load(f"{PERSISTENCE_DIR}/network_persistence_diagrams.npy")
    physical_diagrams = np.load(f"{PERSISTENCE_DIR}/physical_persistence_diagrams.npy")
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]

    print(f"✓ C2 diagrams loaded: {c2_diagrams.shape}")
    print(f"✓ Network diagrams loaded: {network_diagrams.shape}")
    print(f"✓ Physical diagrams loaded: {physical_diagrams.shape}")
    print(f"✓ Labels loaded: {len(labels):,} samples\n")

except FileNotFoundError as e:
    print(f"✗ ERROR: Missing persistence diagrams")
    print(f"  {e}")
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


print("✓ Feature extraction functions defined")
print()

# ==============================================================================
# EXTRACT FEATURES FOR ALL SAMPLES
# ==============================================================================
print("=" * 80)
print("EXTRACTING TDA FEATURES FROM ALL MANIFOLDS")
print("=" * 80)
print(f"Processing {len(c2_diagrams):,} samples...")
print()

start_time = datetime.now()

all_features = []

for idx in range(len(c2_diagrams)):
    sample_features = {}

    # Extract from C2 manifold
    c2_feats = extract_features_from_diagram(c2_diagrams[idx], "C2")
    sample_features.update(c2_feats)

    # Extract from Network manifold
    network_feats = extract_features_from_diagram(network_diagrams[idx], "Network")
    sample_features.update(network_feats)

    # Extract from Physical manifold
    physical_feats = extract_features_from_diagram(physical_diagrams[idx], "Physical")
    sample_features.update(physical_feats)

    all_features.append(sample_features)

    # Progress
    if (idx + 1) % 10000 == 0 or (idx + 1) == len(c2_diagrams):
        elapsed = (datetime.now() - start_time).total_seconds()
        print(
            f"  Processed {idx+1:,}/{len(c2_diagrams):,} "
            f"({(idx+1)/len(c2_diagrams)*100:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s"
        )

total_time = (datetime.now() - start_time).total_seconds()

print()
print(f"✓ Feature extraction complete in {total_time:.1f} seconds")
print()

# Convert to DataFrame
tda_features_df = pd.DataFrame(all_features)

print(f"TDA features extracted: {tda_features_df.shape}")
print(f"  - Samples: {len(tda_features_df):,}")
print(f"  - Features per sample: {tda_features_df.shape[1]}")
print()

# ==============================================================================
# FEATURE SUMMARY
# ==============================================================================
print("-" * 80)
print("TDA Feature Summary:")
print("-" * 80)

# Count features per manifold and dimension
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

# Save features
tda_features_df.to_csv(f"{FEATURES_DIR}/tda_features.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/tda_features.csv")
print(f"  Shape: {tda_features_df.shape}")

# Save with labels for easy loading
tda_with_labels = tda_features_df.copy()
tda_with_labels["label"] = labels
tda_with_labels.to_csv(f"{FEATURES_DIR}/tda_features_with_labels.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/tda_features_with_labels.csv")

# Save feature names
feature_names = list(tda_features_df.columns)
with open(f"{FEATURES_DIR}/tda_feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print(f"✓ Saved: {FEATURES_DIR}/tda_feature_names.pkl")

# ==============================================================================
# CREATE COMBINED FEATURE SET
# ==============================================================================
print("\n" + "-" * 80)
print("Creating combined feature set (Original + TDA)...")
print("-" * 80)

# Load original features
original_features = pd.read_csv(f"{OUTPUT_DIR}/original_features.csv")
print(f"✓ Original features loaded: {original_features.shape}")

# Combine
combined_features = pd.concat([original_features, tda_features_df], axis=1)
print(f"✓ Combined features created: {combined_features.shape}")
print(f"  - Original features: {original_features.shape[1]}")
print(f"  - TDA features: {tda_features_df.shape[1]}")
print(f"  - Total features: {combined_features.shape[1]}")

# Save
combined_features.to_csv(f"{FEATURES_DIR}/combined_features.csv", index=False)
print(f"✓ Saved: {FEATURES_DIR}/combined_features.csv")

# Save with labels
combined_with_labels = combined_features.copy()
combined_with_labels["label"] = labels
combined_with_labels.to_csv(
    f"{FEATURES_DIR}/combined_features_with_labels.csv", index=False
)
print(f"✓ Saved: {FEATURES_DIR}/combined_features_with_labels.csv")

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
print(f"  - {FEATURES_DIR}/tda_features.csv")
print(f"  - {FEATURES_DIR}/tda_features_with_labels.csv")
print(f"  - {FEATURES_DIR}/combined_features.csv")
print(f"  - {FEATURES_DIR}/combined_features_with_labels.csv")
print(f"  - {FEATURES_DIR}/tda_feature_names.pkl")

print("\nNext: Run 07_tda_enhanced_models.py (45-60 min)")
print("=" * 80)
