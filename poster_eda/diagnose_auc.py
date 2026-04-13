import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load everything
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

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Encode labels
le = LabelEncoder()
le.fit(y_all)
y_test_encoded = le.transform(y_test)
y_pred_encoded = le.transform(y_pred)

print("="*80)
print("DIAGNOSTIC: Why is AUC low when F1 is high?")
print("="*80)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

print(f"\nPrediction probability statistics:")
print(f"  Mean max probability: {y_pred_proba.max(axis=1).mean():.4f}")
print(f"  Min max probability: {y_pred_proba.max(axis=1).min():.4f}")
print(f"  Median max probability: {np.median(y_pred_proba.max(axis=1)):.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

print(f"\nProbability distributions per class:")
for i, class_name in enumerate(le.classes_):
    class_probs = y_pred_proba[y_test_encoded == i, i]
    if len(class_probs) > 0:
        print(f"\n{class_name}:")
        print(f"  Mean prob (for true positives): {class_probs.mean():.4f}")
        print(f"  Std: {class_probs.std():.4f}")
        print(f"  Min: {class_probs.min():.4f}")
        print(f"  Max: {class_probs.max():.4f}")

