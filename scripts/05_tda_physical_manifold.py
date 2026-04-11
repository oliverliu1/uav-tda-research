"""
05_tda_physical_manifold.py

Persistent Homology Computation - Physical Manifold (GUDHI)

Computes Vietoris-Rips filtration and persistent homology for the
Physical manifold (2D spatial proxy).

Features: MeanDelay, AverageHopCount (proxies for spatial position)

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import gudhi as gd
import pickle
import os
from datetime import datetime
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
# LOAD PHYSICAL MANIFOLD DATA
# ==============================================================================
print("Loading Physical manifold...")

try:
    physical_data = pd.read_csv(f"{OUTPUT_DIR}/physical_manifold_scaled.csv")
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]

    print(f"✓ Physical manifold loaded: {physical_data.shape}")
    print(f"✓ Features: {list(physical_data.columns)}")
    print(f"✓ Labels loaded: {labels.shape[0]} samples")
    print(
        f"✓ Memory usage: {physical_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
    )

except FileNotFoundError:
    print("✗ ERROR: Physical manifold data not found.")
    print("  Please run 01_data_prep.py first.")
    exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
print("-" * 80)
print("Configuring TDA Parameters...")
print("-" * 80)

# Physical manifold is only 2D - fastest to compute
WINDOW_SIZE = 100
MAX_EDGE_LENGTH = 5.0
MAX_DIMENSION = 2

print(f"Configuration:")
print(f"  - Manifold: Physical (2-dimensional spatial proxy)")
print(f"  - Window size: {WINDOW_SIZE} nearest neighbors")
print(f"  - Maximum edge length: {MAX_EDGE_LENGTH}")
print(f"  - Maximum dimension: {MAX_DIMENSION} (H₀, H₁, H₂)")
print(f"  - Total samples: {len(physical_data):,}")
print()

print("Expected attack signatures:")
print("  - Wormhole: Physical β₀ doesn't match Network β₁")
print("  - Normal spatial clustering patterns")
print()

# ==============================================================================
# COMPUTE PERSISTENCE DIAGRAMS
# ==============================================================================
print("=" * 80)
print("COMPUTING PERSISTENT HOMOLOGY")
print("=" * 80)
print("Physical manifold is 2D - fastest computation (~30-60 minutes)")
print()

all_persistence_diagrams = []
all_points = physical_data.values

PROGRESS_INTERVAL = 5000
start_time = datetime.now()

for idx in range(len(physical_data)):
    # Get local window
    start_idx = max(0, idx - WINDOW_SIZE // 2)
    end_idx = min(len(physical_data), idx + WINDOW_SIZE // 2)
    window_points = all_points[start_idx:end_idx]

    # Build Vietoris-Rips complex
    rips_complex = gd.RipsComplex(points=window_points, max_edge_length=MAX_EDGE_LENGTH)

    # Create simplex tree
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=MAX_DIMENSION)

    # Compute persistence
    simplex_tree.compute_persistence()

    # Get persistence pairs
    persistence_pairs = simplex_tree.persistence()

    # Store
    all_persistence_diagrams.append(persistence_pairs)

    # Progress reporting
    if (idx + 1) % PROGRESS_INTERVAL == 0 or (idx + 1) == len(physical_data):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (len(physical_data) - (idx + 1)) / rate

        print(
            f"Processed {idx+1:,}/{len(physical_data):,} ({(idx+1)/len(physical_data)*100:.1f}%) | "
            f"Rate: {rate:.1f} samples/sec | ETA: {remaining/60:.1f} min"
        )

total_time = (datetime.now() - start_time).total_seconds() / 60
print()
print(f"✓ Computation complete in {total_time:.1f} minutes ({total_time/60:.2f} hours)")
print()

# ==============================================================================
# CONVERT TO NUMPY FORMAT
# ==============================================================================
print("-" * 80)
print("Converting to numpy format...")
print("-" * 80)

max_features = max(len(diagram) for diagram in all_persistence_diagrams)
print(f"  - Maximum features in any diagram: {max_features}")

persistence_diagrams_physical = np.zeros((len(physical_data), max_features, 3))

for sample_idx, diagram in enumerate(all_persistence_diagrams):
    for feature_idx, (dimension, (birth, death)) in enumerate(diagram):
        if death == float("inf"):
            death = MAX_EDGE_LENGTH

        persistence_diagrams_physical[sample_idx, feature_idx, 0] = birth
        persistence_diagrams_physical[sample_idx, feature_idx, 1] = death
        persistence_diagrams_physical[sample_idx, feature_idx, 2] = dimension

print(f"✓ Converted to shape: {persistence_diagrams_physical.shape}")
print()

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================
print("-" * 80)
print("Saving persistence diagrams...")
print("-" * 80)

np.save(
    f"{PERSISTENCE_DIR}/physical_persistence_diagrams.npy",
    persistence_diagrams_physical,
)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_persistence_diagrams.npy")
file_size = os.path.getsize(f"{PERSISTENCE_DIR}/physical_persistence_diagrams.npy") / (
    1024**2
)
print(f"  File size: {file_size:.2f} MB")

with open(f"{PERSISTENCE_DIR}/physical_gudhi_diagrams.pkl", "wb") as f:
    pickle.dump(all_persistence_diagrams, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_gudhi_diagrams.pkl")

metadata = {
    "manifold": "Physical",
    "library": f"GUDHI {gd.__version__}",
    "n_samples": len(persistence_diagrams_physical),
    "window_size": WINDOW_SIZE,
    "max_dimension": MAX_DIMENSION,
    "homology_dimensions": [0, 1, 2],
    "max_edge_length": MAX_EDGE_LENGTH,
    "features": list(physical_data.columns),
    "computation_time_minutes": total_time,
    "computation_date": datetime.now().isoformat(),
    "approach": "sliding_window",
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

print(f"\nComputation statistics:")
print(f"  - Total time: {total_time:.1f} min ({total_time/60:.2f} hours)")
print(f"  - Samples processed: {len(persistence_diagrams_physical):,}")
print(
    f"  - Avg time per sample: {(total_time*60)/len(persistence_diagrams_physical):.2f} seconds"
)

print(f"\nDiagram statistics:")
print(f"  - Shape: {persistence_diagrams_physical.shape}")
print(f"  - Memory: {persistence_diagrams_physical.nbytes / (1024**2):.2f} MB")

for dim in [0, 1, 2]:
    dim_mask = persistence_diagrams_physical[:, :, 2] == dim
    n_features = dim_mask.sum(axis=1)

    births = persistence_diagrams_physical[:, :, 0][dim_mask]
    deaths = persistence_diagrams_physical[:, :, 1][dim_mask]
    lifetimes = deaths - births
    lifetimes = lifetimes[lifetimes > 0]

    print(f"\n  H{dim}:")
    print(f"    - Mean features/sample: {n_features.mean():.2f}")
    print(f"    - Std features/sample: {n_features.std():.2f}")
    print(f"    - Max features: {n_features.max()}")
    if len(lifetimes) > 0:
        print(f"    - Mean persistence: {lifetimes.mean():.4f}")
        print(f"    - Max persistence: {lifetimes.max():.4f}")

print("\n" + "-" * 80)
print("Per-attack type analysis:")
print("-" * 80)

for attack in labels.unique():
    mask = labels == attack
    attack_diagrams = persistence_diagrams_physical[mask]

    h0_count = (attack_diagrams[:, :, 2] == 0).sum(axis=1).mean()
    h1_count = (attack_diagrams[:, :, 2] == 1).sum(axis=1).mean()

    print(f"\n{attack}:")
    print(f"  - Samples: {mask.sum():,}")
    print(f"  - Avg H₀ features: {h0_count:.2f}")
    print(f"  - Avg H₁ features: {h1_count:.2f}")

print("\n" + "=" * 80)
print("✓ PHYSICAL MANIFOLD TDA COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nOutputs created:")
print(f"  - {PERSISTENCE_DIR}/physical_persistence_diagrams.npy")
print(f"  - {PERSISTENCE_DIR}/physical_gudhi_diagrams.pkl")
print(f"  - {PERSISTENCE_DIR}/physical_metadata.pkl")
print("\nNext: Run 06_tda_features_extraction.py (15-30 min)")
print("=" * 80)
