import pandas as pd
import numpy as np

# Load the ORIGINAL dataset (not the processed one)
original_data = pd.read_csv('data/UAVIDS-2025.csv')

print("="*80)
print("WHY IS BASELINE 99.73% AUC?")
print("="*80)

print(f"\nDataset shape: {original_data.shape}")
print(f"Features: {original_data.columns.tolist()}")

# Check attack signatures
print("\n" + "="*80)
print("ATTACK SIGNATURES IN RAW DATA")
print("="*80)

key_features = ['PacketDropRate', 'TxPacketRate/s', 'RxPacketRate/s', 
                'Throughput/Kbps', 'LostPackets', 'MeanDelay/s']

print("\nMean values by attack type:")
for feature in key_features:
    if feature in original_data.columns:
        print(f"\n{feature}:")
        means = original_data.groupby('label')[feature].mean().sort_values(ascending=False)
        for label, val in means.items():
            print(f"  {label:20s}: {val:12.4f}")

# Separability check
print("\n" + "="*80)
print("ARE ATTACKS TRIVIALLY SEPARABLE?")
print("="*80)

# Blackhole - should drop packets
if 'PacketDropRate' in original_data.columns:
    print("\nPacketDropRate by attack:")
    for label in original_data['label'].unique():
        rate = original_data[original_data['label'] == label]['PacketDropRate'].mean()
        print(f"  {label:20s}: {rate:.6f}")

# Flooding - should have high traffic
if 'TxPacketRate/s' in original_data.columns:
    print("\nTxPacketRate/s by attack:")
    for label in original_data['label'].unique():
        rate = original_data[original_data['label'] == label]['TxPacketRate/s'].mean()
        print(f"  {label:20s}: {rate:.6f}")

# Check overlap
print("\n" + "="*80)
print("FEATURE OVERLAP ANALYSIS")
print("="*80)

if 'PacketDropRate' in original_data.columns:
    # Check if PacketDropRate perfectly separates Blackhole
    blackhole_drops = original_data[original_data['label'] == 'Blackhole Attack']['PacketDropRate']
    other_drops = original_data[original_data['label'] != 'Blackhole Attack']['PacketDropRate']
    
    print(f"\nPacketDropRate ranges:")
    print(f"  Blackhole:  [{blackhole_drops.min():.4f}, {blackhole_drops.max():.4f}]")
    print(f"  Others:     [{other_drops.min():.4f}, {other_drops.max():.4f}]")
    
    # Check overlap
    overlap = (blackhole_drops.min() <= other_drops.max()) and (other_drops.min() <= blackhole_drops.max())
    print(f"  Overlap: {overlap}")
    
    if not overlap:
        print("  ⚠️  PERFECT SEPARATION - Blackhole is trivially detectable!")

