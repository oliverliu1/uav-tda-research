"""
5W_wasserstein_detection.py

Z-Score Normalization and 3σ Anomaly Detection

Normalizes Wasserstein distances using Z-score and applies 3σ threshold
for anomaly detection per manifold.

As per methodology: "A time window is flagged as anomalous when any manifold's 
normalized Wasserstein distance exceeds μ + 3σ of the healthy baseline distribution."

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = "outputs"
WASSERSTEIN_DIR = f"{OUTPUT_DIR}/wasserstein"

print("=" * 80)
print("WASSERSTEIN APPROACH - STEP 5W: Z-SCORE & 3σ DETECTION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD WASSERSTEIN DISTANCES
# ==============================================================================
print("Loading Wasserstein distances...")
print("-" * 80)

try:
    results_df = pd.read_csv(f"{WASSERSTEIN_DIR}/wasserstein_distances.csv")
    
    print(f"✓ Loaded: {results_df.shape}")
    print(f"  - Flows: {len(results_df):,}")
    print(f"  - Columns: {list(results_df.columns)}")
    print()
    
except FileNotFoundError:
    print("✗ ERROR: Wasserstein distances not found")
    print("  Please run 4W_wasserstein_distances.py first")
    exit(1)

# ==============================================================================
# COMPUTE Z-SCORES (Using Normal Traffic as Baseline)
# ==============================================================================
print("=" * 80)
print("COMPUTING Z-SCORE NORMALIZATION")
print("=" * 80)
print()

print("Computing statistics from Normal Traffic baseline...")
print("-" * 80)

# Filter Normal Traffic for baseline statistics
normal_mask = results_df['label'] == 'Normal Traffic'
normal_df = results_df[normal_mask]

print(f"Normal Traffic samples: {len(normal_df):,}")
print()

# Compute mean and std for each manifold
manifolds = ['c2', 'network', 'physical']
stats = {}

for manifold in manifolds:
    col = f'{manifold}_distance'
    mean = normal_df[col].mean()
    std = normal_df[col].std()
    
    stats[manifold] = {'mean': mean, 'std': std}
    
    print(f"{manifold.upper()} manifold:")
    print(f"  - Mean: {mean:.6f}")
    print(f"  - Std:  {std:.6f}")
    print(f"  - 3σ threshold: {mean + 3*std:.6f}")
    print()

# ==============================================================================
# NORMALIZE DISTANCES
# ==============================================================================
print("-" * 80)
print("Normalizing distances (Z-score)...")
print("-" * 80)

for manifold in manifolds:
    col = f'{manifold}_distance'
    z_col = f'{manifold}_zscore'
    
    mean = stats[manifold]['mean']
    std = stats[manifold]['std']
    
    # Z-score normalization
    results_df[z_col] = (results_df[col] - mean) / std
    
    print(f"✓ {manifold.upper()}: Z-scores computed")

print()

# ==============================================================================
# APPLY 3σ THRESHOLD
# ==============================================================================
print("=" * 80)
print("APPLYING 3σ ANOMALY DETECTION THRESHOLD")
print("=" * 80)
print()

THRESHOLD = 3.0

print(f"Threshold: μ + {THRESHOLD}σ")
print()

# Flag anomalies per manifold
for manifold in manifolds:
    z_col = f'{manifold}_zscore'
    flag_col = f'{manifold}_anomaly'
    
    # Flag if Z-score exceeds threshold
    results_df[flag_col] = results_df[z_col] > THRESHOLD
    
    n_flagged = results_df[flag_col].sum()
    pct_flagged = (n_flagged / len(results_df)) * 100
    
    print(f"{manifold.upper()} manifold:")
    print(f"  - Flagged as anomalous: {n_flagged:,} / {len(results_df):,} ({pct_flagged:.2f}%)")

print()

# ==============================================================================
# COMBINED ANOMALY FLAG
# ==============================================================================
print("-" * 80)
print("Creating combined anomaly flag...")
print("-" * 80)

# Anomaly if ANY manifold flags it
results_df['anomaly_detected'] = (
    results_df['c2_anomaly'] | 
    results_df['network_anomaly'] | 
    results_df['physical_anomaly']
)

n_anomalies = results_df['anomaly_detected'].sum()
pct_anomalies = (n_anomalies / len(results_df)) * 100

print(f"✓ Combined anomaly detection:")
print(f"  - Total flagged: {n_anomalies:,} / {len(results_df):,} ({pct_anomalies:.2f}%)")
print()

# ==============================================================================
# MANIFOLD PATTERN ANALYSIS
# ==============================================================================
print("=" * 80)
print("ANOMALY PATTERN ANALYSIS")
print("=" * 80)
print()

print("Per-manifold detection patterns:")
print("-" * 80)

# Create pattern signature
results_df['pattern'] = (
    results_df['c2_anomaly'].astype(int).astype(str) + 
    results_df['network_anomaly'].astype(int).astype(str) + 
    results_df['physical_anomaly'].astype(int).astype(str)
)

pattern_counts = results_df.groupby('pattern').size().sort_values(ascending=False)

print("\nPattern codes (C2-Network-Physical):")
for pattern, count in pattern_counts.head(10).items():
    pct = (count / len(results_df)) * 100
    print(f"  {pattern}: {count:,} samples ({pct:.2f}%)")

print()

# ==============================================================================
# PER-ATTACK DETECTION STATISTICS
# ==============================================================================
print("-" * 80)
print("Per-attack type detection:")
print("-" * 80)
print()

for attack_type in results_df['label'].unique():
    attack_mask = results_df['label'] == attack_type
    attack_df = results_df[attack_mask]
    
    n_total = len(attack_df)
    n_detected = attack_df['anomaly_detected'].sum()
    pct_detected = (n_detected / n_total) * 100 if n_total > 0 else 0
    
    # Per-manifold detection for this attack
    c2_det = attack_df['c2_anomaly'].sum()
    net_det = attack_df['network_anomaly'].sum()
    phy_det = attack_df['physical_anomaly'].sum()
    
    print(f"{attack_type}:")
    print(f"  - Total samples: {n_total:,}")
    print(f"  - Detected: {n_detected:,} ({pct_detected:.2f}%)")
    print(f"  - C2 flags: {c2_det:,}, Network: {net_det:,}, Physical: {phy_det:,}")
    print()

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("=" * 80)
print("SAVING DETECTION RESULTS")
print("=" * 80)

results_df.to_csv(f"{WASSERSTEIN_DIR}/detection_results.csv", index=False)
print(f"✓ Saved: {WASSERSTEIN_DIR}/detection_results.csv")

# Save statistics
detection_stats = {
    'threshold': THRESHOLD,
    'baseline_stats': stats,
    'total_samples': len(results_df),
    'total_anomalies': int(n_anomalies),
    'anomaly_rate': float(pct_anomalies),
    'pattern_counts': pattern_counts.to_dict(),
    'computation_date': datetime.now().isoformat()
}

with open(f"{WASSERSTEIN_DIR}/detection_stats.pkl", 'wb') as f:
    pickle.dump(detection_stats, f)
print(f"✓ Saved: {WASSERSTEIN_DIR}/detection_stats.pkl")

# Create summary table
summary_rows = []
for attack_type in results_df['label'].unique():
    attack_mask = results_df['label'] == attack_type
    attack_df = results_df[attack_mask]
    
    summary_rows.append({
        'Attack Type': attack_type,
        'Total Samples': len(attack_df),
        'Detected': attack_df['anomaly_detected'].sum(),
        'Detection Rate (%)': (attack_df['anomaly_detected'].sum() / len(attack_df) * 100),
        'C2 Flags': attack_df['c2_anomaly'].sum(),
        'Network Flags': attack_df['network_anomaly'].sum(),
        'Physical Flags': attack_df['physical_anomaly'].sum()
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{WASSERSTEIN_DIR}/detection_summary.csv", index=False)
print(f"✓ Saved: {WASSERSTEIN_DIR}/detection_summary.csv")

print("\n" + "=" * 80)
print("✓ Z-SCORE NORMALIZATION & DETECTION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\nKey Findings:")
print(f"  - 3σ threshold: μ + {THRESHOLD}σ")
print(f"  - Total anomalies detected: {n_anomalies:,} / {len(results_df):,} ({pct_anomalies:.2f}%)")
print("\nPer-attack detection rates:")
print(summary_df.to_string(index=False))

print("\nNext: Run 6W_wasserstein_evaluation.py (15 min)")
print("=" * 80)
