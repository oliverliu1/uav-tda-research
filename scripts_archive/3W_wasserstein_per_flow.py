"""
3W_wasserstein_per_flow.py

Per-Flow Persistence Computation

Computes persistence diagram for EACH individual flow (all 122K).
Each flow is treated as a single point cloud for anomaly detection.

This is different from supervised approach - here each flow gets its own barcode
which is then compared to the baseline using Wasserstein distance.

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
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = "outputs"
WASSERSTEIN_DIR = f"{OUTPUT_DIR}/wasserstein"
PERSISTENCE_DIR = f"{WASSERSTEIN_DIR}/per_flow_diagrams"

os.makedirs(PERSISTENCE_DIR, exist_ok=True)

print("=" * 80)
print("WASSERSTEIN APPROACH - STEP 3W: PER-FLOW PERSISTENCE COMPUTATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MAX_EDGE_LENGTH = 5.0
MAX_DIMENSION = 2
BATCH_SIZE = 10000  # Save results every 10K flows
NEIGHBORHOOD_SIZE = 50  # Points around each flow to build topology

print("Configuration:")
print(f"  - Total flows: 122,171")
print(f"  - Neighborhood size: {NEIGHBORHOOD_SIZE} points")
print(f"  - Max edge length: {MAX_EDGE_LENGTH}")
print(f"  - Max dimension: {MAX_DIMENSION}")
print(f"  - Batch save interval: {BATCH_SIZE:,} flows")
print()
print("⚠️  ESTIMATED TIME: 8-12 hours")
print("    Recommended: Run overnight Saturday → Sunday morning")
print()

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading manifold data...")
print("-" * 80)

try:
    c2_data = pd.read_csv(f"{OUTPUT_DIR}/c2_manifold_scaled.csv")
    network_data = pd.read_csv(f"{OUTPUT_DIR}/network_manifold_scaled.csv")
    physical_data = pd.read_csv(f"{OUTPUT_DIR}/physical_manifold_scaled.csv")
    labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")['label']
    
    print(f"✓ C2 manifold: {c2_data.shape}")
    print(f"✓ Network manifold: {network_data.shape}")
    print(f"✓ Physical manifold: {physical_data.shape}")
    print(f"✓ Labels: {len(labels):,} samples")
    print()
    
except FileNotFoundError as e:
    print(f"✗ ERROR: {e}")
    exit(1)

n_flows = len(c2_data)

# ==============================================================================
# COMPUTE PER-FLOW PERSISTENCE - C2 MANIFOLD
# ==============================================================================
print("=" * 80)
print("COMPUTING PER-FLOW PERSISTENCE: C2 MANIFOLD")
print("=" * 80)
print()

c2_diagrams = []
all_points_c2 = c2_data.values

start_time = datetime.now()
last_save = 0

for idx in range(n_flows):
    # Get neighborhood around this flow
    start_idx = max(0, idx - NEIGHBORHOOD_SIZE//2)
    end_idx = min(n_flows, idx + NEIGHBORHOOD_SIZE//2)
    neighborhood = all_points_c2[start_idx:end_idx]
    
    # Build Rips complex
    rips = gd.RipsComplex(points=neighborhood, max_edge_length=MAX_EDGE_LENGTH)
    simplex_tree = rips.create_simplex_tree(max_dimension=MAX_DIMENSION)
    simplex_tree.compute_persistence()
    
    diagram = simplex_tree.persistence()
    c2_diagrams.append(diagram)
    
    # Progress + periodic save
    if (idx + 1) % 1000 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (n_flows - (idx + 1)) / rate
        
        print(f"C2: {idx+1:,}/{n_flows:,} ({(idx+1)/n_flows*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {remaining/3600:.1f}h")
    
    # Save batch
    if (idx + 1) % BATCH_SIZE == 0:
        batch_num = (idx + 1) // BATCH_SIZE
        with open(f"{PERSISTENCE_DIR}/c2_batch_{batch_num}.pkl", 'wb') as f:
            pickle.dump(c2_diagrams[last_save:], f)
        print(f"  → Saved batch {batch_num} (flows {last_save+1:,}-{idx+1:,})")
        last_save = idx + 1

# Save final batch
if last_save < n_flows:
    batch_num = (n_flows // BATCH_SIZE) + 1
    with open(f"{PERSISTENCE_DIR}/c2_batch_{batch_num}.pkl", 'wb') as f:
        pickle.dump(c2_diagrams[last_save:], f)
    print(f"  → Saved final batch {batch_num}")

c2_time = (datetime.now() - start_time).total_seconds() / 60
print(f"\n✓ C2 complete in {c2_time:.1f} min ({c2_time/60:.2f} hours)\n")

# ==============================================================================
# COMPUTE PER-FLOW PERSISTENCE - NETWORK MANIFOLD
# ==============================================================================
print("=" * 80)
print("COMPUTING PER-FLOW PERSISTENCE: NETWORK MANIFOLD")
print("=" * 80)
print()

network_diagrams = []
all_points_network = network_data.values

start_time = datetime.now()
last_save = 0

for idx in range(n_flows):
    start_idx = max(0, idx - NEIGHBORHOOD_SIZE//2)
    end_idx = min(n_flows, idx + NEIGHBORHOOD_SIZE//2)
    neighborhood = all_points_network[start_idx:end_idx]
    
    rips = gd.RipsComplex(points=neighborhood, max_edge_length=MAX_EDGE_LENGTH)
    simplex_tree = rips.create_simplex_tree(max_dimension=MAX_DIMENSION)
    simplex_tree.compute_persistence()
    
    diagram = simplex_tree.persistence()
    network_diagrams.append(diagram)
    
    if (idx + 1) % 1000 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (n_flows - (idx + 1)) / rate
        
        print(f"Network: {idx+1:,}/{n_flows:,} ({(idx+1)/n_flows*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {remaining/3600:.1f}h")
    
    if (idx + 1) % BATCH_SIZE == 0:
        batch_num = (idx + 1) // BATCH_SIZE
        with open(f"{PERSISTENCE_DIR}/network_batch_{batch_num}.pkl", 'wb') as f:
            pickle.dump(network_diagrams[last_save:], f)
        print(f"  → Saved batch {batch_num}")
        last_save = idx + 1

if last_save < n_flows:
    batch_num = (n_flows // BATCH_SIZE) + 1
    with open(f"{PERSISTENCE_DIR}/network_batch_{batch_num}.pkl", 'wb') as f:
        pickle.dump(network_diagrams[last_save:], f)
    print(f"  → Saved final batch {batch_num}")

network_time = (datetime.now() - start_time).total_seconds() / 60
print(f"\n✓ Network complete in {network_time:.1f} min ({network_time/60:.2f} hours)\n")

# ==============================================================================
# COMPUTE PER-FLOW PERSISTENCE - PHYSICAL MANIFOLD
# ==============================================================================
print("=" * 80)
print("COMPUTING PER-FLOW PERSISTENCE: PHYSICAL MANIFOLD")
print("=" * 80)
print()

physical_diagrams = []
all_points_physical = physical_data.values

start_time = datetime.now()
last_save = 0

for idx in range(n_flows):
    start_idx = max(0, idx - NEIGHBORHOOD_SIZE//2)
    end_idx = min(n_flows, idx + NEIGHBORHOOD_SIZE//2)
    neighborhood = all_points_physical[start_idx:end_idx]
    
    rips = gd.RipsComplex(points=neighborhood, max_edge_length=MAX_EDGE_LENGTH)
    simplex_tree = rips.create_simplex_tree(max_dimension=MAX_DIMENSION)
    simplex_tree.compute_persistence()
    
    diagram = simplex_tree.persistence()
    physical_diagrams.append(diagram)
    
    if (idx + 1) % 1000 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (idx + 1) / elapsed
        remaining = (n_flows - (idx + 1)) / rate
        
        print(f"Physical: {idx+1:,}/{n_flows:,} ({(idx+1)/n_flows*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {remaining/3600:.1f}h")
    
    if (idx + 1) % BATCH_SIZE == 0:
        batch_num = (idx + 1) // BATCH_SIZE
        with open(f"{PERSISTENCE_DIR}/physical_batch_{batch_num}.pkl", 'wb') as f:
            pickle.dump(physical_diagrams[last_save:], f)
        print(f"  → Saved batch {batch_num}")
        last_save = idx + 1

if last_save < n_flows:
    batch_num = (n_flows // BATCH_SIZE) + 1
    with open(f"{PERSISTENCE_DIR}/physical_batch_{batch_num}.pkl", 'wb') as f:
        pickle.dump(physical_diagrams[last_save:], f)
    print(f"  → Saved final batch {batch_num}")

physical_time = (datetime.now() - start_time).total_seconds() / 60
print(f"\n✓ Physical complete in {physical_time:.1f} min ({physical_time/60:.2f} hours)\n")

# ==============================================================================
# SAVE CONSOLIDATED FILES
# ==============================================================================
print("=" * 80)
print("SAVING CONSOLIDATED PERSISTENCE DIAGRAMS")
print("=" * 80)

with open(f"{PERSISTENCE_DIR}/c2_all_diagrams.pkl", 'wb') as f:
    pickle.dump(c2_diagrams, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/c2_all_diagrams.pkl")

with open(f"{PERSISTENCE_DIR}/network_all_diagrams.pkl", 'wb') as f:
    pickle.dump(network_diagrams, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/network_all_diagrams.pkl")

with open(f"{PERSISTENCE_DIR}/physical_all_diagrams.pkl", 'wb') as f:
    pickle.dump(physical_diagrams, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/physical_all_diagrams.pkl")

# Save metadata
metadata = {
    'n_flows': n_flows,
    'neighborhood_size': NEIGHBORHOOD_SIZE,
    'max_edge_length': MAX_EDGE_LENGTH,
    'max_dimension': MAX_DIMENSION,
    'c2_time_minutes': c2_time,
    'network_time_minutes': network_time,
    'physical_time_minutes': physical_time,
    'total_time_hours': (c2_time + network_time + physical_time) / 60,
    'computation_date': datetime.now().isoformat()
}

with open(f"{PERSISTENCE_DIR}/metadata.pkl", 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Saved: {PERSISTENCE_DIR}/metadata.pkl")

total_time = (c2_time + network_time + physical_time) / 60

print("\n" + "=" * 80)
print("✓ PER-FLOW PERSISTENCE COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"\nTotal computation time: {total_time:.2f} hours")
print(f"  - C2 manifold: {c2_time/60:.2f} hours")
print(f"  - Network manifold: {network_time/60:.2f} hours")
print(f"  - Physical manifold: {physical_time/60:.2f} hours")
print(f"\nPersistence diagrams computed: {n_flows:,} × 3 manifolds = {n_flows*3:,} total")
print("\nNext: Run 4W_wasserstein_distances.py (30-60 min)")
print("=" * 80)
