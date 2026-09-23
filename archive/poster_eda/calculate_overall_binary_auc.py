import pandas as pd
from sklearn.metrics import roc_auc_score

# Load detection results
detection = pd.read_csv('outputs/wasserstein/detection_results.csv')

# Create binary labels: 0 = Normal, 1 = Any Attack
y_true_binary = (detection['label'] != 'Normal Traffic').astype(int)
y_pred_binary = detection['anomaly_detected'].astype(int)

# Also try using the max z-score as the anomaly score
max_zscore = detection[['c2_zscore', 'network_zscore', 'physical_zscore']].max(axis=1)

# Binary AUC (Normal vs Any Attack)
binary_auc = roc_auc_score(y_true_binary, max_zscore)

print("="*80)
print("OVERALL BINARY DETECTION AUC")
print("="*80)
print(f"\nBinary AUC (Normal vs Any Attack): {binary_auc:.4f}")

# Breakdown
normal_count = (detection['label'] == 'Normal Traffic').sum()
attack_count = (detection['label'] != 'Normal Traffic').sum()

detected_attacks = detection[detection['label'] != 'Normal Traffic']['anomaly_detected'].sum()
detected_normal = detection[detection['label'] == 'Normal Traffic']['anomaly_detected'].sum()

print(f"\nDetection breakdown:")
print(f"  Normal flows: {normal_count:,}")
print(f"    Flagged as anomaly: {detected_normal} ({100*detected_normal/normal_count:.1f}% false positive)")
print(f"  Attack flows: {attack_count:,}")
print(f"    Flagged as anomaly: {detected_attacks} ({100*detected_attacks/attack_count:.1f}% true positive)")

print(f"\n✓ This is the HONEST unsupervised number to report")

