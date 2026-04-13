import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Load data
X_original = pd.read_csv('outputs/original_features.csv')
labels_df = pd.read_csv('outputs/labels.csv')
y_binary = (labels_df['label'] != 'Normal Traffic').astype(int)

# Train Isolation Forest on full dataset (unsupervised)
print("Training Isolation Forest (unsupervised baseline)...")
iso_forest = IsolationForest(contamination=0.2, random_state=42)
iso_forest.fit(X_original)

# Get anomaly scores (lower = more anomalous)
anomaly_scores = iso_forest.score_samples(X_original)

# Flip scores (higher = more anomalous for AUC calculation)
anomaly_scores = -anomaly_scores

# Compute AUC
baseline_auc = roc_auc_score(y_binary, anomaly_scores)

print(f"\n{'='*80}")
print(f"UNSUPERVISED BASELINE (Isolation Forest)")
print(f"{'='*80}")
print(f"Binary AUC (Normal vs Attack): {baseline_auc:.4f}")
print(f"\nComparison:")
print(f"  Isolation Forest (baseline): {baseline_auc:.4f}")
print(f"  TDA Wasserstein (ours):      0.8442")
print(f"  Improvement:                 {0.8442 - baseline_auc:+.4f}")

