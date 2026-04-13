import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize

# Load everything
X_combined = pd.read_csv('outputs/tda_features/combined_features.csv')
labels_df = pd.read_csv('outputs/labels.csv')
y_all = labels_df['label'].values

train_indices = np.load('outputs/train_indices.npy')
test_indices = np.load('outputs/test_indices.npy')

X_test = X_combined.iloc[test_indices].values
y_test = y_all[test_indices]

# Load fixed model
with open('results/models/tda_random_forest_fixed.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred_proba = model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=model.classes_)

# Overall AUC
auc_macro = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
auc_weighted = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')

print("="*80)
print("FINAL TDA-ENHANCED RESULTS")
print("="*80)
print(f"\nMacro-Average AUC:    {auc_macro:.4f}")
print(f"Weighted-Average AUC: {auc_weighted:.4f}")

print("\nPER-CLASS AUC:")
print("-"*60)
per_class_results = []
for i, class_name in enumerate(model.classes_):
    y_binary = (y_test == class_name).astype(int)
    auc = roc_auc_score(y_binary, y_pred_proba[:, i])
    per_class_results.append({'Attack': class_name, 'AUC': auc})
    print(f"{class_name:20s}: {auc:.4f}")

# Compare to baseline
print("\n" + "="*80)
print("BASELINE COMPARISON")
print("="*80)

X_original = pd.read_csv('outputs/original_features.csv')
X_test_baseline = X_original.iloc[test_indices].values

with open('results/models/baseline_random_forest.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

y_pred_proba_baseline = baseline_model.predict_proba(X_test_baseline)
auc_baseline = roc_auc_score(y_test_bin, y_pred_proba_baseline, average='macro', multi_class='ovr')

print(f"\nBaseline AUC:     {auc_baseline:.4f}")
print(f"TDA-Enhanced AUC: {auc_macro:.4f}")
print(f"Improvement:      {auc_macro - auc_baseline:+.4f}")

# Save for poster
results_df = pd.DataFrame(per_class_results)
results_df.to_csv('results/tables/final_tda_auc_per_class.csv', index=False)

summary = pd.DataFrame([
    {'Model': 'Baseline RF', 'AUC': auc_baseline},
    {'Model': 'TDA-Enhanced RF', 'AUC': auc_macro},
    {'Model': 'Improvement', 'AUC': auc_macro - auc_baseline}
])
summary.to_csv('results/tables/final_auc_comparison.csv', index=False)

print(f"\n✓ Saved results to:")
print(f"  - results/tables/final_tda_auc_per_class.csv")
print(f"  - results/tables/final_auc_comparison.csv")

