import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

print("EMERGENCY RE-TRAIN (STRING LABELS)")

# Load data
X_combined = pd.read_csv('outputs/tda_features/combined_features.csv')
labels_df = pd.read_csv('outputs/labels.csv')
y_all = labels_df['label'].values

train_indices = np.load('outputs/train_indices.npy')
test_indices = np.load('outputs/test_indices.npy')

X_train = X_combined.iloc[train_indices].values
X_test = X_combined.iloc[test_indices].values
y_train = y_all[train_indices]  # Keep as strings!
y_test = y_all[test_indices]

# Train
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training...")
model.fit(X_train, y_train)

# Save
with open('results/models/tda_random_forest_fixed.pkl', 'wb') as f:
    pickle.dump(model, f)

# Quick metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
y_test_bin = label_binarize(y_test, classes=model.classes_)
auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')

print(f"\nFixed Model:")
print(f"  Accuracy: {acc:.4f}")
print(f"  AUC: {auc:.4f}")

