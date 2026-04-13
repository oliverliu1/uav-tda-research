import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load data
labels_df = pd.read_csv('outputs/labels.csv')
y_all = labels_df['label'].values
train_indices = np.load('outputs/train_indices.npy')
test_indices = np.load('outputs/test_indices.npy')
y_test = y_all[test_indices]

# Encode
le = LabelEncoder()
le.fit(y_all)
y_test_encoded = le.transform(y_test)

# Load baseline model
X_original = pd.read_csv('outputs/original_features.csv')
X_test_baseline = X_original.iloc[test_indices]

with open('results/models/baseline_random_forest.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

y_pred_proba_baseline = baseline_model.predict_proba(X_test_baseline)

# Compute AUC (without labels parameter)
try:
    auc_baseline_macro = roc_auc_score(y_test_encoded, y_pred_proba_baseline, average='macro', multi_class='ovr')
    print(f"Baseline Macro-Average AUC: {auc_baseline_macro:.4f}")
    
    print("\nPer-class Baseline AUC:")
    for i in range(len(baseline_model.classes_)):
        class_name = le.classes_[i]
        y_binary = (y_test_encoded == i).astype(int)
        auc = roc_auc_score(y_binary, y_pred_proba_baseline[:, i])
        print(f"  {class_name:20s}: {auc:.4f}")
        
except Exception as e:
    print(f"Error: {e}")
    
