import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

print("="*80)
print("COMPUTING AUC-ROC FOR TDA-ENHANCED MODELS")
print("="*80)

# Load data
X_combined = pd.read_csv('outputs/tda_features/combined_features.csv')
labels = pd.read_csv('outputs/labels.csv')['label']
train_indices = np.load('outputs/train_indices.npy')
test_indices = np.load('outputs/test_indices.npy')

X_test = X_combined.iloc[test_indices]
y_test = labels.iloc[test_indices]

# Load model
with open('results/models/tda_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Get predictions
y_pred_proba = model.predict_proba(X_test)
classes = model.classes_

print(f"\nTest set size: {len(y_test)}")
print(f"Classes: {classes}")

# Compute multi-class AUC (one-vs-rest)
y_test_binarized = label_binarize(y_test, classes=classes)

# Macro-average AUC
auc_macro = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
print(f"\n{'='*60}")
print(f"MACRO-AVERAGE AUC: {auc_macro:.4f}")
print(f"{'='*60}")

# Per-class AUC
print("\nPER-CLASS AUC (One-vs-Rest):")
print("-" * 60)
per_class_auc = {}
for i, class_name in enumerate(classes):
    auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
    per_class_auc[class_name] = auc
    print(f"{class_name:20s}: {auc:.4f}")

# Weighted average AUC
auc_weighted = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
print(f"\nWEIGHTED-AVERAGE AUC: {auc_weighted:.4f}")

# Save results
results_df = pd.DataFrame([
    {'Metric': 'Macro-Average AUC', 'Value': auc_macro},
    {'Metric': 'Weighted-Average AUC', 'Value': auc_weighted}
])

for class_name, auc in per_class_auc.items():
    results_df = pd.concat([results_df, pd.DataFrame([{'Metric': f'AUC - {class_name}', 'Value': auc}])], ignore_index=True)

results_df.to_csv('results/tables/supervised_auc_scores.csv', index=False)
print(f"\n✓ Saved to: results/tables/supervised_auc_scores.csv")

# Also compute for baseline
print("\n" + "="*80)
print("BASELINE MODEL AUC (for comparison)")
print("="*80)

X_original = pd.read_csv('outputs/original_features.csv')
X_test_baseline = X_original.iloc[test_indices]

with open('results/models/baseline_random_forest.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

y_pred_proba_baseline = baseline_model.predict_proba(X_test_baseline)
auc_baseline_macro = roc_auc_score(y_test_binarized, y_pred_proba_baseline, average='macro', multi_class='ovr')
auc_baseline_weighted = roc_auc_score(y_test_binarized, y_pred_proba_baseline, average='weighted', multi_class='ovr')

print(f"\nBaseline Macro-Average AUC: {auc_baseline_macro:.4f}")
print(f"Baseline Weighted-Average AUC: {auc_baseline_weighted:.4f}")

print(f"\nIMPROVEMENT:")
print(f"  Macro-Average: {auc_baseline_macro:.4f} → {auc_macro:.4f} (+{auc_macro - auc_baseline_macro:.4f})")
print(f"  Weighted-Average: {auc_baseline_weighted:.4f} → {auc_weighted:.4f} (+{auc_weighted - auc_baseline_weighted:.4f})")

