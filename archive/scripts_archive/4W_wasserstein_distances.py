"""
4W_wasserstein_distances.py

Wasserstein Distance Computation

Computes Wasserstein distance between each flow's persistence diagram
and the healthy baseline barcode for each manifold.

Uses GUDHI's Wasserstein distance implementation.

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import gudhi
import warnings
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = "outputs"
WASSERSTEIN_DIR = f"{OUTPUT_DIR}/wasserstein"
PERSISTENCE_DIR = f"{WASSERSTEIN_DIR}/per_flow_diagrams"

print("=" * 80)
print("WASSERSTEIN APPROACH - STEP 4W: WASSERSTEIN DISTANCE COMPUTATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD BASELINES
# ==============================================================================
print("Loading baseline barcodes...")
print("-" * 80)

try:
    with open(f"{WASSERSTEIN_DIR}/baseline_barcodes.pkl", 'rb') as f:
        baselines = pickle.load(f)
    
    print("✓ Baseline barcodes loaded:")
    for manifold in ['C2', 'Network', 'Physical']:
        n_features = baselines[manifold]['n_features']
        print(f"  - {manifold}: {n_features} features")
    print()
    
except FileNotFoundError:
    print("✗ ERROR: Baseline barcodes not found")
    print("  Please run 2W_wasserstein_baseline.py first")
    exit(1)

# ==============================================================================
# LOAD PER-FLOW DIAGRAMS
# ==============================================================================
print("Loading per-flow persistence diagrams...")
print("-" * 80)

try:
    with open(f"{PERSISTENCE_DIR}/c2_all_diagrams.pkl", 'rb') as f:
        c2_diagrams = pickle.load(f)
    with open(f"{PERSISTENCE_DIR}/network_all_diagrams.pkl", 'rb') as f:
        network_diagrams = pickle.load(f)
    with open(f"{PERSISTENCE_DIR}/physical_all_diagrams.pkl", 'rb') as f:
        physical_diagrams = pickle.load(f)
    
    print(f"✓ C2 diagrams: {len(c2_diagrams):,}")
    print(f"✓ Network diagrams: {len(network_diagrams):,}")
    print(f"✓ Physical diagrams: {len(physical_diagrams):,}")
    print()
    
except FileNotFoundError:
    print("✗ ERROR: Per-flow diagrams not found")
    print("  Please run 3W_wasserstein_per_flow.py first")
    exit(1)

# Load labels
labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")['label']

# ==============================================================================
# HELPER FUNCTION: Convert GUDHI diagram to numpy format
# ==============================================================================
def gudhi_to_numpy(diagram, max_edge_length=5.0):
    """
    Convert GUDHI persistence diagram to numpy array format for Wasserstein.
    GUDHI format: list of (dimension, (birth, death))
    Output: list of (birth, death) tuples
    """
    points = []
    for dim, (birth, death) in diagram:
        if death == float('inf'):
            death = max_edge_length
        points.append((birth, death))
    return np.array(points)

# Prepare baseline diagrams in numpy format
baseline_c2_np = gudhi_to_numpy(baselines['C2']['barcode'])
baseline_network_np = gudhi_to_numpy(baselines['Network']['barcode'])
baseline_physical_np = gudhi_to_numpy(baselines['Physical']['barcode'])

# ==============================================================================
# COMPUTE WASSERSTEIN DISTANCES - C2 MANIFOLD
# ==============================================================================
print("=" * 80)
print("COMPUTING WASSERSTEIN DISTANCES: C2 MANIFOLD")
print("=" * 80)
print()

from gudhi.wasserstein import wasserstein_distance

c2_distances = []
start_time = datetime.now()

for idx, diagram in enumerate(c2_diagrams):
    # Convert to numpy
    diagram_np = gudhi_to_numpy(diagram)
    
    # Compute Wasserstein distance (order=2)
    try:
        dist = wasserstein_distance(diagram_np, baseline_c2_np, order=2)
        c2_distances.append(dist)
    except Exception as e:
        # Handle edge cases (empty diagrams, etc.)
        c2_distances.append(0.0)
    
    # Progress
    if (idx + 1) % 10000 == 0 or (idx + 1) == len(c2_diagrams):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (len(c2_diagrams) - (idx + 1)) / rate
        
        print(f"C2: {idx+1:,}/{len(c2_diagrams):,} ({(idx+1)/len(c2_diagrams)*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f} min")

c2_time = (datetime.now() - start_time).total_seconds() / 60
print(f"✓ C2 complete in {c2_time:.1f} minutes\n")

# ==============================================================================
# COMPUTE WASSERSTEIN DISTANCES - NETWORK MANIFOLD
# ==============================================================================
print("=" * 80)
print("COMPUTING WASSERSTEIN DISTANCES: NETWORK MANIFOLD")
print("=" * 80)
print()

network_distances = []
start_time = datetime.now()

for idx, diagram in enumerate(network_diagrams):
    diagram_np = gudhi_to_numpy(diagram)
    
    try:
        dist = wasserstein_distance(diagram_np, baseline_network_np, order=2)
        network_distances.append(dist)
    except Exception as e:
        network_distances.append(0.0)
    
    if (idx + 1) % 10000 == 0 or (idx + 1) == len(network_diagrams):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (len(network_diagrams) - (idx + 1)) / rate
        
        print(f"Network: {idx+1:,}/{len(network_diagrams):,} ({(idx+1)/len(network_diagrams)*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f} min")

network_time = (datetime.now() - start_time).total_seconds() / 60
print(f"✓ Network complete in {network_time:.1f} minutes\n")

# ==============================================================================
# COMPUTE WASSERSTEIN DISTANCES - PHYSICAL MANIFOLD
# ==============================================================================
print("=" * 80)
print("COMPUTING WASSERSTEIN DISTANCES: PHYSICAL MANIFOLD")
print("=" * 80)
print()

physical_distances = []
start_time = datetime.now()

for idx, diagram in enumerate(physical_diagrams):
    diagram_np = gudhi_to_numpy(diagram)
    
    try:
        dist = wasserstein_distance(diagram_np, baseline_physical_np, order=2)
        physical_distances.append(dist)
    except Exception as e:
        physical_distances.append(0.0)
    
    if (idx + 1) % 10000 == 0 or (idx + 1) == len(physical_diagrams):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (len(physical_diagrams) - (idx + 1)) / rate
        
        print(f"Physical: {idx+1:,}/{len(physical_diagrams):,} ({(idx+1)/len(physical_diagrams)*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f} min")

physical_time = (datetime.now() - start_time).total_seconds() / 60
print(f"✓ Physical complete in {physical_time:.1f} minutes\n")

# ==============================================================================
# CREATE RESULTS DATAFRAME
# ==============================================================================
print("=" * 80)
print("CREATING RESULTS DATAFRAME")
print("=" * 80)

results_df = pd.DataFrame({
    'flow_id': range(len(labels)),
    'label': labels,
    'c2_distance': c2_distances,
    'network_distance': network_distances,
    'physical_distance': physical_distances
})

print(f"\n✓ Results dataframe created: {results_df.shape}")
print(f"\nDistance statistics:")
print(results_df[['c2_distance', 'network_distance', 'physical_distance']].describe())
print()

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("-" * 80)
print("Saving Wasserstein distances...")
print("-" * 80)

results_df.to_csv(f"{WASSERSTEIN_DIR}/wasserstein_distances.csv", index=False)
print(f"✓ Saved: {WASSERSTEIN_DIR}/wasserstein_distances.csv")

# Save as pickle for faster loading
with open(f"{WASSERSTEIN_DIR}/wasserstein_distances.pkl", 'wb') as f:
    pickle.dump(results_df, f)
print(f"✓ Saved: {WASSERSTEIN_DIR}/wasserstein_distances.pkl")

# Save distance arrays separately
np.save(f"{WASSERSTEIN_DIR}/c2_distances.npy", np.array(c2_distances))
np.save(f"{WASSERSTEIN_DIR}/network_distances.npy", np.array(network_distances))
np.save(f"{WASSERSTEIN_DIR}/physical_distances.npy", np.array(physical_distances))
print(f"✓ Saved: distance arrays as .npy files")

total_time = c2_time + network_time + physical_time

print("\n" + "=" * 80)
print("✓ WASSERSTEIN DISTANCE COMPUTATION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"\nTotal time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
print(f"  - C2: {c2_time:.1f} min")
print(f"  - Network: {network_time:.1f} min")
print(f"  - Physical: {physical_time:.1f} min")
print(f"\nDistances computed: {len(labels):,} flows × 3 manifolds")
print("\nNext: Run 5W_wasserstein_detection.py (10 min)")
print("=" * 80)
