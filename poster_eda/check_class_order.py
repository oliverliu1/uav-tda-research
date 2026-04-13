import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load labels
labels_df = pd.read_csv('outputs/labels.csv')
y_all = labels_df['label'].values

# Recreate the encoding from Script 7
train_indices = np.load('outputs/train_indices.npy')
test_indices = np.load('outputs/test_indices.npy')

y_train = y_all[train_indices]
y_test = y_all[test_indices]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("="*80)
print("LABEL ENCODING CHECK")
print("="*80)
print(f"\nLabelEncoder classes: {le.classes_}")
print(f"LabelEncoder mapping:")
for i, cls in enumerate(le.classes_):
    print(f"  {i} → {cls}")

# Load model
with open('results/models/tda_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"\nModel classes: {model.classes_}")

# Check if they match
if np.array_equal(le.classes_, model.classes_):
    print("\n✓ Label encoder and model classes MATCH")
else:
    print("\n✗ MISMATCH between label encoder and model classes!")
    print("This would cause incorrect AUC calculation")

# Now compute AUC correctly
from sklearn.metrics import roc_auc_score

X_combined = pd.read_csv('outputs/tda_features/combined_features.csv')
X_test = X_combined.iloc[test_indices].values

y_pred_proba = model.predict_proba(X_test)

# CORRECT AUC calculation
auc_correct = roc_auc_score(y_test_encoded, y_pred_proba, average='macro', multi_class='ovr')

print(f"\n" + "="*80)
print("CORRECTED AUC CALCULATION")
print("="*80)
print(f"Macro-Average AUC: {auc_correct:.4f}")

# Per-class AUC
print("\nPer-class AUC:")
for i, class_name in enumerate(le.classes_):
    y_binary = (y_test_encoded == i).astype(int)
    auc = roc_auc_score(y_binary, y_pred_proba[:, i])
    print(f"  {class_name:20s}: {auc:.4f}")

