"""
05_tda_physical_manifold.py

Persistent Homology Computation - Physical Manifold (GUDHI)

Computes Vietoris-Rips filtration and persistent homology for the
Physical manifold (2D spatial proxy).

Features: MeanDelay, AverageHopCount (proxies for spatial position)

DATA LEAKAGE PREVENTION:
- Train/test indices are loaded from outputs/ (created by Script 01)
- NearestNeighbors is FIT on training data only
- Train neighborhoods: query train points against train-only index
- Test neighborhoods: query test points against test-only index
- No test point ever appears in a train neighborhood and vice versa

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import gudhi as gd
import pickle
import os
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

# Paths
OUTPUT_DIR = "outputs"
PERSISTENCE_DIR = f"{OUTPUT_DIR}/persistence_diagrams"

os.makedirs(PERSISTENCE_DIR, exist_ok=True)

print("=" * 80)
print("MULTI-MANIFOLD TDA PIPELINE - STEP 5: PHYSICAL MANIFOLD PERSISTENT HOMOLOGY")
print("Using GUDHI Library")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD PHYSICAL MANIFOLD DATA AND TRAIN/TEST INDICES
# ==============================================================================
print("Loading Physical manifold and split indices...")

try:
    physical_data = pd.read_csv(f"{OUTPUT_DIR}/physical_manifold_scaled.csv")
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]
    train_indices = np.load(f"{OUTPUT_DIR}/train_indices.npy")
    test_indices = np.load(f"{OUTPUT_DIR}/test_indices.npy")

    print(f"✓ Physical manifold loaded: {physical_data.shape}")
    print(f"✓ Features: {list(physical_data.columns)}")
    print(f"✓ Labels loaded: {labels.shape[0]} samples")
    print(f"✓ Train indices: {len(train_indices):,} samples")
    print(f"✓ Test indices:  {len(test_indices):,} samples")
    print(f"✓ Memory usage: {physical_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

except FileNotFoundError as e:
    print(f"✗ ERROR: Required file not found: {e}")
    print("  Please run 01_data_scripts.py first.")
    exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
print("-" * 80)
print("Configuring TDA Parameters...")
print("-" * 80)

N_NEIGHBORS = 100  # Size of local neighborhood for each sample
MAX_EDGE_LENGTH = 5.0
MAX_DIMENSION = 2

print(f"Configuration:")
print(f"  - Manifold: Physical (2-dimensional spatial proxy)")
print(f"  - Approach: NearestNeighbors (local topology per sample, leak-free)")
print(f"  - Neighborhood size: {N_NEIGHBORS} nearest neighbors")
print(f"  - Maximum edge length: {MAX_EDGE_LENGTH}")
print(f"  - Maximum dimension: {MAX_DIMENSION} (H₀, H₁, H₂)")
print(f"  - Train samples: {len(train_indices):,}")
print(f"  - Test samples:  {len(test_indices):,}")
print()

print("Expected attack signatures:")
print("  - Wormhole: Physical β₀ doesn't match Network β₁")
print("  - Normal spatial clustering patterns")
print()

# ==============================================================================
# BUILD NEAREST NEIGHBOR INDICES (separate for train and test)
#
# KEY DESIGN: Each split has its own NearestNeighbors index so that
# neighborhoods never cross the train/test boundary.
# ==============================================================================
print("-" * 80)
print("Building NearestNeighbors indices...")
print("-" * 80)

all_points = physical_data.values

train_points = all_points[train_indices]
test_points = all_points[test_indices]

# Train NN index: fit on train, query train
print("  Fitting NearestNeighbors on training data...")
nn_train = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto", n_jobs=-1)
nn_train.fit(train_points)
_, train_neighbor_indices = nn_train.kneighbors(train_points)
print(f"  ✓ Train neighborhoods ready: {train_neighbor_indices.shape}")

# Test NN index: fit on test, query test
print("  Fitting NearestNeighbors on test data...")
nn_test = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto", n_jobs=-1)
nn_test.fit(test_points)
_, test_neighbor_indices = nn_test.kneighbors(test_points)
print(f"  ✓ Test neighborhoods ready: {test_neighbor_indices.shape}")
print()


# ==============================================================================
# HELPER: compute persistence diagrams for one split
# ==============================================================================
def compute_persistence_for_split(split_points, neighbor_indices, split_name):
    """
    Compute per-sample persistence diagrams for one split.

    split_points:     (n, d) array of points for this split
    neighbor_indices: (n, k) array — row i contains the k neighbor indices
                      into split_points for sample i
    split_name:       "train" or "test" (for progress messages)

    Returns list of GUDHI persistence pairs (one list per sample).
    """
    n_samples = len(split_points)
    all_diagrams = []
    PROGRESS_INTERVAL = 5000
    start_time = datetime.now()

    for idx in range(n_samples):
        # Neighborhood is entirely within split_points — no cross-split mixing
        neighbor_idx = neighbor_indices[idx]
        window_points = split_points[neighbor_idx]

        rips_complex = gd.RipsComplex(
            points=window_points, max_edge_length=MAX_EDGE_LENGTH
        )
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=MAX_DIMENSION)
        simplex_tree.compute_persistence()
        all_diagrams.append(simplex_tree.persistence())

        if (idx + 1) % PROGRESS_INTERVAL == 0 or (idx + 1) == n_samples:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx + 1) / elapsed
            remaining = (n_samples - (idx + 1)) / rate
            print(
                f"  [{split_name}] {idx+1:,}/{n_samples:,} "
                f"({(idx+1)/n_samples*100:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f} min"
            )

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"  [{split_name}] Complete in {elapsed_min:.1f} minutes")
    return all_diagrams, elapsed_min


def diagrams_to_array(diagrams, max_edge_length):
    """Convert list of GUDHI persistence pairs to (n, max_features, 3) numpy array."""
    max_features = max(len(d) for d in diagrams)
    arr = np.zeros((len(diagrams), max_features, 3))
    for i, diagram in enumerate(diagrams):
        for j, (dim, (birth, death)) in enumerate(diagram):
            if death == float("inf"):
                death = max_edge_length
            arr[i, j, 0] = birth
            arr[i, j, 1] = death
            arr[i, j, 2] = dim
    return arr


# ==============================================================================
# COMPUTE PERSISTENCE DIAGRAMS
# ==============================================================================
print("=" * 80)
print("COMPUTING PERSISTENT HOMOLOGY — TRAIN SPLIT")
print("=" * 80)
print("Physical manifold is 2D - fastest computation (~30-60 minutes per split)")
print()

train_diagrams_raw, train_time = compute_persistence_for_split(
    train_points, train_neighbor_indices, "train"
)

print()
print("=" * 80)
print("COMPUTING PERSISTENT HOMOLOGY — TEST SPLIT")
print("=" * 80)
print()

test_diagrams_raw, test_time = compute_persistence_for_split(
    test_points, test_neighbor_indices, "test"
)

total_time = train_time + test_time
print()
print(f"✓ All computation complete in {total_time:.1f} minutes ({total_time/60:.2f} hours)")
print()

# ==============================================================================
# CONVERT TO NUMPY FORMAT
# ==============================================================================
print("-" * 80)
print("Converting to numpy format...")
print("-" * 80)

persistence_diagrams_physical_train = diagrams_to_array(train_diagrams_raw, MAX_EDGE_LENGTH)
persistence_diagrams_physical_test = diagrams_to_array(test_diagrams_raw, MAX_EDGE_LENGTH)

print(f"✓ Train diagrams shape: {persistence_diagrams_physical_train.shape}")
print(f"✓ Test diagrams shape:  {persistence_diagrams_physical_test.shape}")
print(f"  Format: (n_samples, max_features, 3)  where 3 = (birth, death, dimension)")
print()

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================
print("-" * 80)
print("Saving persistence diagrams...")
print("-" * 80)

# Train diagrams
np.save(
    f"{PERSISTENCE_DIR}/physical_persistence_diagrams_train.npy",
    persistence_diagrams_physical_train,
)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_persistence_diagrams_train.npy")
file_size = os.path.getsize(
    f"{PERSISTENCE_DIR}/physical_persistence_diagrams_train.npy"
) / (1024**2)
print(f"  File size: {file_size:.2f} MB")

with open(f"{PERSISTENCE_DIR}/physical_gudhi_diagrams_train.pkl", "wb") as f:
    pickle.dump(train_diagrams_raw, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_gudhi_diagrams_train.pkl")

# Test diagrams
np.save(
    f"{PERSISTENCE_DIR}/physical_persistence_diagrams_test.npy",
    persistence_diagrams_physical_test,
)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_persistence_diagrams_test.npy")
file_size = os.path.getsize(
    f"{PERSISTENCE_DIR}/physical_persistence_diagrams_test.npy"
) / (1024**2)
print(f"  File size: {file_size:.2f} MB")

with open(f"{PERSISTENCE_DIR}/physical_gudhi_diagrams_test.pkl", "wb") as f:
    pickle.dump(test_diagrams_raw, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_gudhi_diagrams_test.pkl")

# Metadata
metadata = {
    "manifold": "Physical",
    "library": f"GUDHI {gd.__version__}",
    "n_train_samples": len(persistence_diagrams_physical_train),
    "n_test_samples": len(persistence_diagrams_physical_test),
    "n_neighbors": N_NEIGHBORS,
    "max_dimension": MAX_DIMENSION,
    "homology_dimensions": [0, 1, 2],
    "max_edge_length": MAX_EDGE_LENGTH,
    "features": list(physical_data.columns),
    "computation_time_minutes": total_time,
    "computation_date": datetime.now().isoformat(),
    "approach": "nearest_neighbors_split_aware",
    "leakage_prevention": (
        "Train NN index fit on train only; test NN index fit on test only. "
        "No cross-split neighborhood mixing."
    ),
}

with open(f"{PERSISTENCE_DIR}/physical_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_metadata.pkl")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print("\n" + "=" * 80)
print("PERSISTENCE DIAGRAM SUMMARY - PHYSICAL MANIFOLD")
print("=" * 80)

for split_name, diagrams, split_indices in [
    ("TRAIN", persistence_diagrams_physical_train, train_indices),
    ("TEST", persistence_diagrams_physical_test, test_indices),
]:
    print(f"\n[{split_name}] Shape: {diagrams.shape}  Memory: {diagrams.nbytes/(1024**2):.2f} MB")

    for dim in [0, 1, 2]:
        dim_mask = diagrams[:, :, 2] == dim
        n_features = dim_mask.sum(axis=1)
        births = diagrams[:, :, 0][dim_mask]
        deaths = diagrams[:, :, 1][dim_mask]
        lifetimes = deaths - births
        lifetimes = lifetimes[lifetimes > 0]

        print(f"  H{dim}: mean {n_features.mean():.2f} features/sample  ", end="")
        if len(lifetimes) > 0:
            print(f"mean persistence={lifetimes.mean():.4f}  max={lifetimes.max():.4f}")
        else:
            print()

    # Per-attack breakdown
    split_labels = labels.values[split_indices]
    print(f"\n  Per-attack breakdown [{split_name}]:")
    for attack in np.unique(split_labels):
        mask = split_labels == attack
        h0 = (diagrams[mask, :, 2] == 0).sum(axis=1).mean()
        h1 = (diagrams[mask, :, 2] == 1).sum(axis=1).mean()
        print(f"    {attack}: {mask.sum():,} samples  avg H₀={h0:.2f}  avg H₁={h1:.2f}")

print("\n" + "=" * 80)
print("✓ PHYSICAL MANIFOLD TDA COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nOutputs created:")
print(f"  - {PERSISTENCE_DIR}/physical_persistence_diagrams_train.npy")
print(f"  - {PERSISTENCE_DIR}/physical_persistence_diagrams_test.npy")
print(f"  - {PERSISTENCE_DIR}/physical_gudhi_diagrams_train.pkl")
print(f"  - {PERSISTENCE_DIR}/physical_gudhi_diagrams_test.pkl")
print(f"  - {PERSISTENCE_DIR}/physical_metadata.pkl")
print("\nNext: Run 06_tda_features_extraction.py (15-30 min)")
print("=" * 80)
