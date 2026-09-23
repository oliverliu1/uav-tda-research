import pandas as pd
import numpy as np
import pickle

# Check what data the TDA model was actually trained on
print("="*80)
print("CHECKING TDA MODEL TRAINING")
print("="*80)

# Load the combined features
combined = pd.read_csv('outputs/tda_features/combined_features.csv')
print(f"\nCombined features shape: {combined.shape}")
print(f"Columns: {combined.shape[1]}")
print(f"First few columns: {list(combined.columns[:10])}")
print(f"Last few columns: {list(combined.columns[-10:])}")

# Check for NaN or inf
print(f"\nNaN values: {combined.isna().sum().sum()}")
print(f"Inf values: {np.isinf(combined.values).sum()}")

# Check TDA feature statistics
tda_cols = [c for c in combined.columns if 'tda_' in c.lower() or 'h0' in c.lower() or 'h1' in c.lower() or 'h2' in c.lower()]
print(f"\nTDA feature columns found: {len(tda_cols)}")

if len(tda_cols) > 0:
    tda_data = combined[tda_cols]
    print(f"\nTDA feature statistics:")
    print(f"  Mean of means: {tda_data.mean().mean():.6f}")
    print(f"  Mean of stds: {tda_data.std().mean():.6f}")
    print(f"  Any all-zero columns: {(tda_data == 0).all().sum()}")
    print(f"  Any constant columns: {tda_data.nunique().min()}")

# Load models and check their training
with open('results/models/tda_random_forest.pkl', 'rb') as f:
    tda_model = pickle.load(f)
    
with open('results/models/baseline_random_forest.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

print(f"\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(f"\nBaseline model:")
print(f"  n_features: {baseline_model.n_features_in_}")
print(f"  n_classes: {len(baseline_model.classes_)}")

print(f"\nTDA model:")  
print(f"  n_features: {tda_model.n_features_in_}")
print(f"  n_classes: {len(tda_model.classes_)}")

# Check if they were trained on the same split
print(f"\nClass encoding:")
print(f"  Baseline classes: {baseline_model.classes_}")
print(f"  TDA classes: {tda_model.classes_}")

