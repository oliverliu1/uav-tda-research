"""
01_data_prep.py

Data Preparation and Manifold Partitioning

This script implements Phase 1 of the multi-manifold TDA pipeline:
1. Data ingestion (UAVIDS-2025 dataset)
2. Time windowing (treating each flow as independent)
3. Feature partitioning into three manifolds (C2, Network, Physical)
4. Preprocessing (IP octet extraction, binary encoding, scaling)

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
from datetime import datetime

# Paths - relative to where script is run from
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
RESULTS_DIR = "./results"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)

print("=" * 80)
print("MULTI-MANIFOLD TDA PIPELINE - STEP 1: DATA PREPARATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# STEP 1: DATA INGESTION
# ==============================================================================
print("Step 1: Loading UAVIDS-2025 dataset...")

try:
    df = pd.read_csv(f"{DATA_DIR}/UAVIDS-2025.csv")
    print(f"✓ Dataset loaded successfully")
    print(f"  - Shape: {df.shape}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  - Columns ({len(df.columns)}): {df.columns.tolist()}\n")
except FileNotFoundError:
    print(f"✗ ERROR: UAVIDS-2025.csv not found in {DATA_DIR}/")
    print("  Please place the dataset in the data/ directory and try again.")
    exit(1)

# Verify dataset integrity
print("Verifying dataset integrity...")
print(f"  - Missing values: {df.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df.duplicated().sum()}")

# Check label distribution
print("\nLabel distribution:")
label_counts = df["label"].value_counts()
for label, count in label_counts.items():
    print(f"  - {label}: {count:,} ({count/len(df)*100:.2f}%)")

# ==============================================================================
# STEP 2: TIME WINDOWING
# ==============================================================================
print("\n" + "-" * 80)
print("Step 2: Time Windowing")
print("-" * 80)

# FlowID is sequential row index, FlowDuration defines window boundaries
# For this analysis, we treat each flow record as an independent observation
print("✓ Using per-record analysis (each flow is a time window)")
print(f"  - Total time windows: {len(df):,}\n")

# ==============================================================================
# STEP 3: FEATURE PARTITIONING
# ==============================================================================
print("-" * 80)
print("Step 3: Feature Partitioning into Three Manifolds")
print("-" * 80)

# Drop FlowID (row index) and Protocol (all UDP)
print("\nDropping uninformative features:")
print("  - FlowID: Sequential row index with no semantic value")
print("  - Protocol: All records are UDP (zero variance)\n")

if "FlowID" in df.columns:
    df = df.drop("FlowID", axis=1)
if "Protocol" in df.columns:
    protocol_unique = df["Protocol"].unique()
    print(f"    Protocol unique values: {protocol_unique}")
    df = df.drop("Protocol", axis=1)

# Define manifold feature sets
C2_FEATURES = ["SrcAddr", "SrcPort", "DstAddr", "DstPort", "FlowDuration/s"]
NETWORK_FEATURES = [
    "TxPackets",
    "RxPackets",
    "LostPackets",
    "TxBytes",
    "RxBytes",
    "TxPacketRate/s",
    "RxPacketRate/s",
    "TxByteRate/s",
    "RxByteRate/s",
    "MeanPacketSize",
    "MeanDelay/s",
    "MeanJitter/s",
    "Throughput/Kbps",
    "PacketDropRate",
    "AverageHopCount",
]
PHYSICAL_FEATURES = ["MeanDelay/s", "AverageHopCount"]

print("Manifold definitions:")
print(f"  - C2 Manifold (5D): {C2_FEATURES}")
print(f"  - Network Manifold (15D): {NETWORK_FEATURES}")
print(f"  - Physical Manifold (2D): {PHYSICAL_FEATURES}\n")

# Extract labels for later use
labels = df["label"].copy()

# ==============================================================================
# STEP 4: PREPROCESSING
# ==============================================================================
print("-" * 80)
print("Step 4: Manifold-Specific Preprocessing")
print("-" * 80)

# ------------------------------------------------------------------------------
# C2 Manifold Preprocessing
# ------------------------------------------------------------------------------
print("\n[C2 Manifold]")
print("Applying domain-specific categorical encoding...")

c2_data = df[C2_FEATURES].copy()

# IP Address Encoding: Extract last octet from 192.168.0.X
print("  1. IP Address Encoding (last octet extraction):")


def extract_last_octet(ip_str):
    """Extract last octet from IP address string"""
    return int(ip_str.split(".")[-1])


c2_data["SrcAddr_octet"] = c2_data["SrcAddr"].apply(extract_last_octet)
c2_data["DstAddr_octet"] = c2_data["DstAddr"].apply(extract_last_octet)

print(
    f"     - SrcAddr range: {c2_data['SrcAddr_octet'].min()}-{c2_data['SrcAddr_octet'].max()}"
)
print(
    f"     - DstAddr range: {c2_data['DstAddr_octet'].min()}-{c2_data['DstAddr_octet'].max()}"
)

# Drop original IP string columns
c2_data = c2_data.drop(["SrcAddr", "DstAddr"], axis=1)

# Port Encoding: Binary flags for ports 9 (Discard) and 654 (AODV)
print("  2. Port Encoding (binary flags):")
unique_src_ports = sorted(c2_data["SrcPort"].unique())
unique_dst_ports = sorted(c2_data["DstPort"].unique())
print(f"     - Unique SrcPort values: {unique_src_ports}")
print(f"     - Unique DstPort values: {unique_dst_ports}")

# Map ports to binary: 9 -> 0, 654 -> 1
port_map = {9: 0, 654: 1}
c2_data["SrcPort_binary"] = c2_data["SrcPort"].map(port_map)
c2_data["DstPort_binary"] = c2_data["DstPort"].map(port_map)

# Drop original port columns
c2_data = c2_data.drop(["SrcPort", "DstPort"], axis=1)

# Reorder columns for clarity
c2_data = c2_data[
    [
        "SrcAddr_octet",
        "SrcPort_binary",
        "DstAddr_octet",
        "DstPort_binary",
        "FlowDuration/s",
    ]
]

print(f"  3. Standardization (StandardScaler)")
scaler_c2 = StandardScaler()
c2_scaled = scaler_c2.fit_transform(c2_data)
c2_scaled = pd.DataFrame(c2_scaled, columns=c2_data.columns)

print(f"     ✓ C2 manifold shape: {c2_scaled.shape}")

# ------------------------------------------------------------------------------
# Network Manifold Preprocessing
# ------------------------------------------------------------------------------
print("\n[Network Manifold]")
print("Applying standardization only (all features already numeric)...")

network_data = df[NETWORK_FEATURES].copy()
scaler_network = StandardScaler()
network_scaled = scaler_network.fit_transform(network_data)
network_scaled = pd.DataFrame(network_scaled, columns=NETWORK_FEATURES)

print(f"  ✓ Network manifold shape: {network_scaled.shape}")

# ------------------------------------------------------------------------------
# Physical Manifold Preprocessing
# ------------------------------------------------------------------------------
print("\n[Physical Manifold]")
print("Applying standardization only (spatial proxy features)...")

physical_data = df[PHYSICAL_FEATURES].copy()
scaler_physical = StandardScaler()
physical_scaled = scaler_physical.fit_transform(physical_data)
physical_scaled = pd.DataFrame(physical_scaled, columns=PHYSICAL_FEATURES)

print(f"  ✓ Physical manifold shape: {physical_scaled.shape}")

# ==============================================================================
# SAVE PREPROCESSED DATA
# ==============================================================================
print("\n" + "=" * 80)
print("Saving preprocessed manifolds...")
print("=" * 80)

# Save each manifold separately
c2_scaled.to_csv(f"{OUTPUT_DIR}/c2_manifold_scaled.csv", index=False)
network_scaled.to_csv(f"{OUTPUT_DIR}/network_manifold_scaled.csv", index=False)
physical_scaled.to_csv(f"{OUTPUT_DIR}/physical_manifold_scaled.csv", index=False)
labels.to_csv(f"{OUTPUT_DIR}/labels.csv", index=False)

print(f"✓ Saved: {OUTPUT_DIR}/c2_manifold_scaled.csv")
print(f"✓ Saved: {OUTPUT_DIR}/network_manifold_scaled.csv")
print(f"✓ Saved: {OUTPUT_DIR}/physical_manifold_scaled.csv")
print(f"✓ Saved: {OUTPUT_DIR}/labels.csv")

# Save scalers for potential inverse transform
with open(f"{OUTPUT_DIR}/scaler_c2.pkl", "wb") as f:
    pickle.dump(scaler_c2, f)
with open(f"{OUTPUT_DIR}/scaler_network.pkl", "wb") as f:
    pickle.dump(scaler_network, f)
with open(f"{OUTPUT_DIR}/scaler_physical.pkl", "wb") as f:
    pickle.dump(scaler_physical, f)

print(f"✓ Saved scalers as pickle files")

# Save original feature data for baseline models (before scaling, but with encoding)
original_features = pd.concat([c2_data, network_data, physical_data], axis=1)
original_features.to_csv(f"{OUTPUT_DIR}/original_features.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/original_features.csv (for baseline models)")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING SUMMARY")
print("=" * 80)

summary = {
    "Total Samples": len(df),
    "C2 Manifold Dimensions": c2_scaled.shape[1],
    "Network Manifold Dimensions": network_scaled.shape[1],
    "Physical Manifold Dimensions": physical_scaled.shape[1],
    "Total Feature Dimensions": c2_scaled.shape[1]
    + network_scaled.shape[1]
    + physical_scaled.shape[1],
    "Number of Classes": len(labels.unique()),
    "Class Labels": labels.unique().tolist(),
}

for key, value in summary.items():
    print(f"{key:.<40} {value}")

print("\n" + "=" * 80)
print(f"✓ DATA PREPARATION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nNext step: Run 02_baseline_models.py to train baseline classifiers")
print("=" * 80)
