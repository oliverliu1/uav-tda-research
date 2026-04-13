"""
2W_wasserstein_baseline.py

Wasserstein Baseline Construction

Computes "healthy baseline" persistence diagrams from Normal Traffic flows.
These baselines represent expected topological signatures of non-compromised swarm.

As per methodology: "A healthy baseline barcode is pre-computed for each manifold
using label-filtered Normal Traffic records."

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
WASSERSTEIN_DIR = f"{OUTPUT_DIR}/wasserstein"

os.makedirs(WASSERSTEIN_DIR, exist_ok=True)

print("=" * 80)
print("WASSERSTEIN APPROACH - STEP 2W: BASELINE BARCODE CONSTRUCTION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MAX_EDGE_LENGTH = 5.0
MAX_DIMENSION = 2
BASELINE_SAMPLE_SIZE = (
    1000  # Sample from normal traffic for baseline (reduced for memory)
)

print("Configuration:")
print(f"  - Baseline sample size: {BASELINE_SAMPLE_SIZE:,} Normal Traffic flows")
print(f"  - Max edge length: {MAX_EDGE_LENGTH}")
print(f"  - Max dimension: {MAX_DIMENSION} (H₀, H₁, H₂)")
print()

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading manifold data and labels...")
print("-" * 80)

try:
    c2_data = pd.read_csv(f"{OUTPUT_DIR}/c2_manifold_scaled.csv")
    network_data = pd.read_csv(f"{OUTPUT_DIR}/network_manifold_scaled.csv")
    physical_data = pd.read_csv(f"{OUTPUT_DIR}/physical_manifold_scaled.csv")
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]

    print(f"✓ C2 manifold: {c2_data.shape}")
    print(f"✓ Network manifold: {network_data.shape}")
    print(f"✓ Physical manifold: {physical_data.shape}")
    print(f"✓ Labels: {len(labels):,} samples")
    print()

except FileNotFoundError as e:
    print(f"✗ ERROR: {e}")
    exit(1)

# ==============================================================================
# FILTER NORMAL TRAFFIC
# ==============================================================================
print("Filtering Normal Traffic for baseline...")
print("-" * 80)

normal_mask = labels == "Normal Traffic"
print(f"Total Normal Traffic samples: {normal_mask.sum():,}")

# Sample for baseline construction
np.random.seed(42)
normal_indices = np.where(normal_mask)[0]
if len(normal_indices) > BASELINE_SAMPLE_SIZE:
    baseline_indices = np.random.choice(
        normal_indices, BASELINE_SAMPLE_SIZE, replace=False
    )
    print(f"Sampled {BASELINE_SAMPLE_SIZE:,} flows for baseline construction")
else:
    baseline_indices = normal_indices
    print(f"Using all {len(baseline_indices):,} Normal Traffic flows")

baseline_c2 = c2_data.iloc[baseline_indices].values
baseline_network = network_data.iloc[baseline_indices].values
baseline_physical = physical_data.iloc[baseline_indices].values

print()

# ==============================================================================
# COMPUTE BASELINE BARCODES
# ==============================================================================
print("=" * 80)
print("COMPUTING BASELINE BARCODES FOR EACH MANIFOLD")
print("=" * 80)
print()

baselines = {}

# ------------------------------------------------------------------------------
# C2 MANIFOLD BASELINE
# ------------------------------------------------------------------------------
print("-" * 80)
print("1. C2 Manifold Baseline (5D)")
print("-" * 80)

start_time = datetime.now()

print("Building Vietoris-Rips complex from baseline points...")
rips_c2 = gd.RipsComplex(points=baseline_c2, max_edge_length=MAX_EDGE_LENGTH)
simplex_tree_c2 = rips_c2.create_simplex_tree(max_dimension=MAX_DIMENSION)

print("Computing persistence...")
simplex_tree_c2.compute_persistence()
baseline_barcode_c2 = simplex_tree_c2.persistence()

elapsed = (datetime.now() - start_time).total_seconds()
print(f"✓ C2 baseline computed in {elapsed:.1f} seconds")
print(f"  - Features: {len(baseline_barcode_c2)}")
print()

baselines["C2"] = {
    "barcode": baseline_barcode_c2,
    "n_samples": len(baseline_c2),
    "dimension": baseline_c2.shape[1],
    "n_features": len(baseline_barcode_c2),
}

# ------------------------------------------------------------------------------
# NETWORK MANIFOLD BASELINE
# ------------------------------------------------------------------------------
print("-" * 80)
print("2. Network Manifold Baseline (15D)")
print("-" * 80)

start_time = datetime.now()

print("Building Vietoris-Rips complex from baseline points...")
rips_network = gd.RipsComplex(points=baseline_network, max_edge_length=MAX_EDGE_LENGTH)
simplex_tree_network = rips_network.create_simplex_tree(max_dimension=MAX_DIMENSION)

print("Computing persistence...")
simplex_tree_network.compute_persistence()
baseline_barcode_network = simplex_tree_network.persistence()

elapsed = (datetime.now() - start_time).total_seconds()
print(f"✓ Network baseline computed in {elapsed:.1f} seconds")
print(f"  - Features: {len(baseline_barcode_network)}")
print()

baselines["Network"] = {
    "barcode": baseline_barcode_network,
    "n_samples": len(baseline_network),
    "dimension": baseline_network.shape[1],
    "n_features": len(baseline_barcode_network),
}

# ------------------------------------------------------------------------------
# PHYSICAL MANIFOLD BASELINE
# ------------------------------------------------------------------------------
print("-" * 80)
print("3. Physical Manifold Baseline (2D)")
print("-" * 80)

start_time = datetime.now()

print("Building Vietoris-Rips complex from baseline points...")
rips_physical = gd.RipsComplex(
    points=baseline_physical, max_edge_length=MAX_EDGE_LENGTH
)
simplex_tree_physical = rips_physical.create_simplex_tree(max_dimension=MAX_DIMENSION)

print("Computing persistence...")
simplex_tree_physical.compute_persistence()
baseline_barcode_physical = simplex_tree_physical.persistence()

elapsed = (datetime.now() - start_time).total_seconds()
print(f"✓ Physical baseline computed in {elapsed:.1f} seconds")
print(f"  - Features: {len(baseline_barcode_physical)}")
print()

baselines["Physical"] = {
    "barcode": baseline_barcode_physical,
    "n_samples": len(baseline_physical),
    "dimension": baseline_physical.shape[1],
    "n_features": len(baseline_barcode_physical),
}

# ==============================================================================
# SAVE BASELINES
# ==============================================================================
print("=" * 80)
print("SAVING BASELINE BARCODES")
print("=" * 80)

with open(f"{WASSERSTEIN_DIR}/baseline_barcodes.pkl", "wb") as f:
    pickle.dump(baselines, f)

print(f"✓ Saved: {WASSERSTEIN_DIR}/baseline_barcodes.pkl")

# Save metadata
metadata = {
    "baseline_sample_size": BASELINE_SAMPLE_SIZE,
    "baseline_indices": baseline_indices.tolist(),
    "max_edge_length": MAX_EDGE_LENGTH,
    "max_dimension": MAX_DIMENSION,
    "computation_date": datetime.now().isoformat(),
    "manifolds": {
        "C2": baselines["C2"],
        "Network": baselines["Network"],
        "Physical": baselines["Physical"],
    },
}

with open(f"{WASSERSTEIN_DIR}/baseline_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"✓ Saved: {WASSERSTEIN_DIR}/baseline_metadata.pkl")
print()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("=" * 80)
print("BASELINE BARCODE SUMMARY")
print("=" * 80)

for manifold in ["C2", "Network", "Physical"]:
    info = baselines[manifold]
    barcode = info["barcode"]

    print(f"\n{manifold} Manifold:")
    print(f"  - Baseline samples: {info['n_samples']:,}")
    print(f"  - Dimension: {info['dimension']}D")
    print(f"  - Total features: {info['n_features']}")

    # Count by homology dimension
    for dim in [0, 1, 2]:
        dim_features = [f for f in barcode if f[0] == dim]
        print(f"  - H{dim} features: {len(dim_features)}")

print("\n" + "=" * 80)
print("✓ BASELINE BARCODES COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nOutputs created:")
print(f"  - {WASSERSTEIN_DIR}/baseline_barcodes.pkl")
print(f"  - {WASSERSTEIN_DIR}/baseline_metadata.pkl")
print("\nNext: Run 3W_wasserstein_per_flow.py")
print("(This will compute persistence for all 122K flows - takes 8-12 hours)")
print("=" * 80)
