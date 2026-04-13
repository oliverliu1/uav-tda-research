import pandas as pd
import numpy as np

# Check the original features
original = pd.read_csv('outputs/original_features.csv')
labels = pd.read_csv('outputs/labels.csv')

print("="*80)
print("WHY IS BASELINE SO GOOD?")
print("="*80)

print("\nOriginal Features (22 total):")
print(original.columns.tolist())

print("\n" + "="*80)
print("CHECKING FOR PERFECT SEPARATORS")
print("="*80)

# Check if any features perfectly separate classes
for col in original.columns:
    unique_per_class = labels.groupby('label')[col].apply(lambda x: x.nunique())
    total_unique = original[col].nunique()
    
    # If feature has many unique values per class, might be discriminative
    if total_unique > 100:
        print(f"\n{col}:")
        print(f"  Total unique values: {total_unique}")
        print(f"  Unique per class:")
        for label, count in unique_per_class.items():
            print(f"    {label}: {count}")

# Check key discriminative features
print("\n" + "="*80)
print("ATTACK SIGNATURES IN ORIGINAL FEATURES")
print("="*80)

# Group by label and show means
summary = original.copy()
summary['label'] = labels['label']

key_features = ['PacketDropRate', 'TxPacketRate/s', 'RxPacketRate/s', 
                'Throughput/Kbps', 'LostPackets', 'MeanDelay/s']

print("\nMean values by attack type (key features):")
for feature in key_features:
    if feature in summary.columns:
        print(f"\n{feature}:")
        means = summary.groupby('label')[feature].mean().sort_values(ascending=False)
        for label, val in means.items():
            print(f"  {label:20s}: {val:12.4f}")

# Check if attacks are trivially separable
print("\n" + "="*80)
print("SEPARABILITY CHECK")
print("="*80)

# Blackhole should have high packet drop
blackhole_drop = summary[summary['label'] == 'Blackhole Attack']['PacketDropRate'].mean()
normal_drop = summary[summary['label'] == 'Normal Traffic']['PacketDropRate'].mean()

print(f"\nPacketDropRate:")
print(f"  Blackhole Attack: {blackhole_drop:.4f}")
print(f"  Normal Traffic:   {normal_drop:.4f}")
print(f"  Ratio: {blackhole_drop/normal_drop if normal_drop > 0 else 'inf'}x")

# Flooding should have high packet rate
flooding_rate = summary[summary['label'] == 'Flooding Attack']['TxPacketRate/s'].mean()
normal_rate = summary[summary['label'] == 'Normal Traffic']['TxPacketRate/s'].mean()

print(f"\nTxPacketRate/s:")
print(f"  Flooding Attack:  {flooding_rate:.4f}")
print(f"  Normal Traffic:   {normal_rate:.4f}")
print(f"  Ratio: {flooding_rate/normal_rate if normal_rate > 0 else 'inf'}x")

