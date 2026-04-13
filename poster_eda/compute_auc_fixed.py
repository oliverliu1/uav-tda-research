import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("COMPUTING AUC-ROC FOR TDA-ENHANCED MODELS")
print("="*80)

# Load data
X_combined = pd.read_csv('outputs/tda_features/combined_features.csv')
labels_df = pd.read_csv('outputs/labels.csv')
y_all = labels_df['label'].values

train_indices = np.load('outputs/train_indices.npy')
test_indices = np.load('outputs/test_indices.npy')

X_test = X_combined.iloc[test_indices].values
y_test = y_all[test_indices]

# Load model
with open('results/models/tda_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"\nTest set size: {len(y_test)}")
print(f"Unique classes in test: {np.unique(y_test)}")
print(f"Test class distribution:")
for cls in np.unique(y_test):
    count = (y_test == cls).sum()
    print(f"  {cls}: {count}")

# Get predictions
y_pred_proba = model.predict_proba(X_test)

# Encode labels
le = LabelEncoder()
le.fit(y_all)
y_test_encoded = le.transform(y_test)

print(f"\nModel classes: {model.classes_}")
print(f"Label encoder classes: {le.classes_}")

# Compute AUC using encoded labels and probabilities
auc_macro = roc_auc_score(y_test_encoded, y_pred_proba, average='macro', multi_class='ovr', labels=model.classes_)
auc_weighted = roc_auc_score(y_test_encoded, y_pred_proba, average='weighted', multi_class='ovr', labels=model.classes_)

print(f"\n{'='*60}")
print(f"TDA-ENHANCED MODEL:")
print(f"MACRO-AVERAGE AUC: {auc_macro:.4f}")
print(f"WEIGHTED-AVERAGE AUC: {auc_weighted:.4f}")
print(f"{'='*60}")

# Per-class AUC
print("\nPER-CLASS AUC (One-vs-Rest):")
print("-" * 60)
per_class_auc = {}

for i, class_idx in enumerate(model.classes_):
    class_name = le.inverse_transform([class_idx])[0]
    # Binary classification: this class vs all others
    y_binary = (y_test_encoded == class_idx).astype(int)
    auc = roc_auc_score(y_binary, y_pred_proba[:, i])
    per_class_auc[class_name] = auc
    print(f"{class_name:20s}: {auc:.4f}")

# Now compute baseline
print("\n" + "="*80)
print("BASELINE MODEL AUC (for comparison)")
print("="*80)

X_original = pd.read_csv('outputs/original_features.csv')
X_test_baseline = X_original.iloc[test_indices].values

with open('results/models/baseline_random_forest.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

y_pred_proba_baseline = baseline_model.predict_proba(X_test_baseline)

auc_baseline_macro = roc_auc_score(y_test_encoded, y_pred_proba_baseline, average='macro', multi_class='ovr', labels=baseline_model.classes_)
auc_baseline_weighted = roc_auc_score(y_test_encoded, y_pred_proba_baseline, average='weighted', multi_class='ovr', labels=baseline_model.classes_)

print(f"\nBaseline Macro-Average AUC: {auc_baseline_macro:.4f}")
print(f"Baseline Weighted-Average AUC: {auc_baseline_weighted:.4f}")

# Per-class baseline AUC
print("\nPER-CLASS BASELINE AUC:")
print("-" * 60)
baseline_per_class = {}
for i, class_idx in enumerate(baseline_model.classes_):
    class_name = le.inverse_transform([class_idx])[0]
    y_binary = (y_test_encoded == class_idx).astype(int)
    auc = roc_auc_score(y_binary, y_pred_proba_baseline[:, i])
    baseline_per_class[class_name] = auc
    print(f"{class_name:20s}: {auc:.4f}")

print(f"\n{'='*80}")
print("IMPROVEMENT SUMMARY")
print(f"{'='*80}")
print(f"\nMacro-Average AUC:")
print(f"  Baseline: {auc_baseline_macro:.4f}")
print(f"  TDA:      {auc_macro:.4f}")
print(f"  Change:   {auc_macro - auc_baseline_macro:+.4f}")

print(f"\nWeighted-Average AUC:")
print(f"  Baseline: {auc_baseline_weighted:.4f}")
print(f"  TDA:      {auc_weighted:.4f}")
print(f"  Change:   {auc_weighted - auc_baseline_weighted:+.4f}")

print(f"\nPer-Class Improvements:")
print("-" * 60)
for class_name in sorted(per_class_auc.keys()):
    baseline_auc = baseline_per_class[class_name]
    tda_auc = per_class_auc[class_name]
    improvement = tda_auc - baseline_auc
    print(f"{class_name:20s}: {baseline_auc:.4f} → {tda_auc:.4f} ({improvement:+.4f})")

# Save results
results_df = pd.DataFrame([
    {'Model': 'Baseline', 'Metric': 'Macro-Avg AUC', 'Value': auc_baseline_macro},
    {'Model': 'Baseline', 'Metric': 'Weighted-Avg AUC', 'Value': auc_baseline_weighted},
    {'Model': 'TDA-Enhanced', 'Metric': 'Macro-Avg AUC', 'Value': auc_macro},
    {'Model': 'TDA-Enhanced', 'Metric': 'Weighted-Avg AUC', 'Value': auc_weighted},
])

for class_name in sorted(per_class_auc.keys()):
    results_df = pd.concat([results_df, pd.DataFrame([
        {'Model': 'Baseline', 'Metric': f'AUC - {class_name}', 'Value': baseline_per_class[class_name]},
        {'Model': 'TDA-Enhanced', 'Metric': f'AUC - {class_name}', 'Value': per_class_auc[class_name]}
    ])], ignore_index=True)

results_df.to_csv('results/tables/auc_comparison.csv', index=False)
print(f"\n✓ Saved to: results/tables/auc_comparison.csv")

