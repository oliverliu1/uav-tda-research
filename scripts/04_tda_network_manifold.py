"""
04_tda_network_manifold.py

Persistent Homology Computation - Network Manifold (GUDHI)

Computes Vietoris-Rips filtration and persistent homology for the
Network manifold (15D traffic space).

Features: TxPackets, RxPackets, LostPackets, TxBytes, RxBytes, rates,
          MeanPacketSize, MeanDelay, MeanJitter, Throughput, etc.

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
print("MULTI-MANIFOLD TDA PIPELINE - STEP 4: NETWORK MANIFOLD PERSISTENT HOMOLOGY")
print("Using GUDHI Library")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD NETWORK MANIFOLD DATA
# ==============================================================================
print("Loading Network manifold...")

try:
    network_data = pd.read_csv(f"{OUTPUT_DIR}/network_manifold_scaled.csv")
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]

    print(f"✓ Network manifold loaded: {network_data.shape}")
    print(f"✓ Features: {list(network_data.columns)}")
    print(f"✓ Labels loaded: {labels.shape[0]} samples")
    print(
        f"✓ Memory usage: {network_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
    )

except FileNotFoundError:
    print("✗ ERROR: Network manifold data not found.")
    print("  Please run 01_data_prep.py first.")
    exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
print("-" * 80)
print("Configuring TDA Parameters...")
print("-" * 80)

# Network manifold is 15D - more complex than C2
# Use slightly larger window to capture more structure
WINDOW_SIZE = 100  # Keep same for consistency
MAX_EDGE_LENGTH = 5.0
MAX_DIMENSION = 2

print(f"Configuration:")
print(f"  - Manifold: Network (15-dimensional traffic space)")
print(f"  - Window size: {WINDOW_SIZE} nearest neighbors")
print(f"  - Maximum edge length: {MAX_EDGE_LENGTH}")
print(f"  - Maximum dimension: {MAX_DIMENSION} (H₀, H₁, H₂)")
print(f"  - Total samples: {len(network_data):,}")
print()

print("Expected attack signatures:")
print("  - Blackhole: Network fragmentation (β₀ increases)")
print("  - Wormhole: Impossible loops (β₁ anomalies)")
print("  - Flooding: Dense clumping (short-lived features)")
print()

# ==============================================================================
# COMPUTE PERSISTENCE DIAGRAMS
# ==============================================================================
print("=" * 80)
print("COMPUTING PERSISTENT HOMOLOGY")
print("=" * 80)
print("⚠️  WARNING: Network manifold is 15D - this will take 3-6 hours")
print("Using sliding window approach")
print()

all_persistence_diagrams = []
all_points = network_data.values

PROGRESS_INTERVAL = 5000
start_time = datetime.now()

for idx in range(len(network_data)):
    # Get local window
    start_idx = max(0, idx - WINDOW_SIZE // 2)
    end_idx = min(len(network_data), idx + WINDOW_SIZE // 2)
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
    if (idx + 1) % PROGRESS_INTERVAL == 0 or (idx + 1) == len(network_data):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (len(network_data) - (idx + 1)) / rate

        print(
            f"Processed {idx+1:,}/{len(network_data):,} ({(idx+1)/len(network_data)*100:.1f}%) | "
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

persistence_diagrams_network = np.zeros((len(network_data), max_features, 3))

for sample_idx, diagram in enumerate(all_persistence_diagrams):
    for feature_idx, (dimension, (birth, death)) in enumerate(diagram):
        if death == float("inf"):
            death = MAX_EDGE_LENGTH

        persistence_diagrams_network[sample_idx, feature_idx, 0] = birth
        persistence_diagrams_network[sample_idx, feature_idx, 1] = death
        persistence_diagrams_network[sample_idx, feature_idx, 2] = dimension

print(f"✓ Converted to shape: {persistence_diagrams_network.shape}")
print()

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================
print("-" * 80)
print("Saving persistence diagrams...")
print("-" * 80)

np.save(
    f"{PERSISTENCE_DIR}/network_persistence_diagrams.npy", persistence_diagrams_network
)
print(f"✓ Saved: {PERSISTENCE_DIR}/network_persistence_diagrams.npy")
file_size = os.path.getsize(f"{PERSISTENCE_DIR}/network_persistence_diagrams.npy") / (
    1024**2
)
print(f"  File size: {file_size:.2f} MB")

with open(f"{PERSISTENCE_DIR}/network_gudhi_diagrams.pkl", "wb") as f:
    pickle.dump(all_persistence_diagrams, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/network_gudhi_diagrams.pkl")

metadata = {
    "manifold": "Network",
    "library": f"GUDHI {gd.__version__}",
    "n_samples": len(persistence_diagrams_network),
    "window_size": WINDOW_SIZE,
    "max_dimension": MAX_DIMENSION,
    "homology_dimensions": [0, 1, 2],
    "max_edge_length": MAX_EDGE_LENGTH,
    "features": list(network_data.columns),
    "computation_time_minutes": total_time,
    "computation_date": datetime.now().isoformat(),
    "approach": "sliding_window",
}

with open(f"{PERSISTENCE_DIR}/network_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/network_metadata.pkl")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print("\n" + "=" * 80)
print("PERSISTENCE DIAGRAM SUMMARY - NETWORK MANIFOLD")
print("=" * 80)

print(f"\nComputation statistics:")
print(f"  - Total time: {total_time:.1f} min ({total_time/60:.2f} hours)")
print(f"  - Samples processed: {len(persistence_diagrams_network):,}")
print(
    f"  - Avg time per sample: {(total_time*60)/len(persistence_diagrams_network):.2f} seconds"
)

print(f"\nDiagram statistics:")
print(f"  - Shape: {persistence_diagrams_network.shape}")
print(f"  - Memory: {persistence_diagrams_network.nbytes / (1024**2):.2f} MB")

for dim in [0, 1, 2]:
    dim_mask = persistence_diagrams_network[:, :, 2] == dim
    n_features = dim_mask.sum(axis=1)

    births = persistence_diagrams_network[:, :, 0][dim_mask]
    deaths = persistence_diagrams_network[:, :, 1][dim_mask]
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
    attack_diagrams = persistence_diagrams_network[mask]

    h0_count = (attack_diagrams[:, :, 2] == 0).sum(axis=1).mean()
    h1_count = (attack_diagrams[:, :, 2] == 1).sum(axis=1).mean()

    print(f"\n{attack}:")
    print(f"  - Samples: {mask.sum():,}")
    print(f"  - Avg H₀ features: {h0_count:.2f}")
    print(f"  - Avg H₁ features: {h1_count:.2f}")

print("\n" + "=" * 80)
print("✓ NETWORK MANIFOLD TDA COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nOutputs created:")
print(f"  - {PERSISTENCE_DIR}/network_persistence_diagrams.npy")
print(f"  - {PERSISTENCE_DIR}/network_gudhi_diagrams.pkl")
print(f"  - {PERSISTENCE_DIR}/network_metadata.pkl")
print("\nNext: Run 05_tda_physical_manifold.py (this will take 30-60 min)")
print("=" * 80)
