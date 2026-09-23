import pandas as pd
import pickle
import os

print("="*80)
print("AUC-ROC RESULTS CHECK")
print("="*80)

# Check classification report
if os.path.exists('results/tables/tda_classification_report.csv'):
    print("\nTDA Classification Report:")
    report = pd.read_csv('results/tables/tda_classification_report.csv')
    print(report)

# Check for AUC in models
print("\n" + "="*80)
print("CHECKING MODEL FILES FOR AUC METRICS")
print("="*80)

try:
    with open('results/models/tda_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    print("\nRandom Forest model loaded successfully")
    print(f"Model type: {type(model)}")
    
    # Check if model has predict_proba (needed for AUC)
    if hasattr(model, 'predict_proba'):
        print("✓ Model supports probability predictions (can compute AUC)")
    
except Exception as e:
    print(f"Could not load model: {e}")

# List all available result files
print("\n" + "="*80)
print("AVAILABLE RESULT FILES:")
print("="*80)
import glob
for f in sorted(glob.glob('results/tables/*.csv')):
    print(f"  {f}")

