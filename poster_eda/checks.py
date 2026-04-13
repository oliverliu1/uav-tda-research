"""
Comprehensive validation checks for UAV TDA research project
Run with: python3 checks.py
"""

import pandas as pd
import numpy as np
import pickle
import os

print("\n" + "="*80)
print("UAV TDA PROJECT - COMPREHENSIVE VALIDATION CHECKS")
print("="*80)

# ============================================================================
# CHECK 1: CLASS DISTRIBUTION & PER-CLASS PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("CHECK 1: CLASS DISTRIBUTION & PER-CLASS PERFORMANCE")
print("="*80)

labels = pd.read_csv('outputs/labels.csv')

print("\nCLASS DISTRIBUTION:")
print("-" * 40)
dist = labels['label'].value_counts()
print(dist)
print("\nPercentages:")
pct = labels['label'].value_counts(normalize=True) * 100
print(pct)

# Check for severe imbalance
max_pct = pct.max()
min_pct = pct.min()
if max_pct > 95:
    print(f"\n⚠️  WARNING: Severe class imbalance! Max class: {max_pct:.1f}%, Min class: {min_pct:.1f}%")
else:
    print(f"\n✓ Class distribution reasonably balanced (Max: {max_pct:.1f}%, Min: {min_pct:.1f}%)")

# Load confusion matrix
print("\n" + "-"*80)
print("CONFUSION MATRIX (Random Forest with TDA):")
print("-" * 80)
cm = pd.read_csv('results/tables/tda_confusion_matrix_random_forest_(tda).csv', index_col=0)
print(cm)

# Calculate per-class metrics
print("\n" + "-"*80)
print("PER-CLASS PERFORMANCE METRICS:")
print("-" * 80)

classes = cm.index.tolist()
results = []

for cls in classes:
    tp = cm.loc[cls, cls]
    fp = cm[cls].sum() - tp
    fn = cm.loc[cls].sum() - tp
    tn = cm.sum().sum() - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'Class': cls,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': int(tp + fn)
    })
    
    print(f"\n{cls}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Support:   {int(tp + fn)}")
    
    if recall < 0.5:
        print(f"  ⚠️  LOW RECALL WARNING!")

# Summary
results_df = pd.DataFrame(results)
print("\n" + "-"*80)
print("SUMMARY:")
print(f"  Average Precision: {results_df['Precision'].mean():.4f}")
print(f"  Average Recall:    {results_df['Recall'].mean():.4f}")
print(f"  Average F1:        {results_df['F1-Score'].mean():.4f}")

# ============================================================================
# CHECK 2: DATA LEAKAGE VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("CHECK 2: DATA LEAKAGE VERIFICATION")
print("="*80)

try:
    with open('outputs/persistence_diagrams/c2_metadata.pkl', 'rb') as f:
        c2_meta = pickle.load(f)
    
    print("\nMetadata keys:", list(c2_meta.keys()))
    
    if 'train_indices' in c2_meta:
        print("\n✓ Train/test split was tracked during TDA computation")
        print(f"  Training samples: {len(c2_meta['train_indices'])}")
        if 'test_indices' in c2_meta:
            print(f"  Test samples: {len(c2_meta['test_indices'])}")
    else:
        print("\n⚠️  WARNING: No train/test tracking found in metadata")
        print("   This may indicate potential data leakage in neighborhood construction")
        
except Exception as e:
    print(f"\n⚠️  Could not load metadata: {e}")

# Check feature dimensions
orig_features = pd.read_csv('outputs/original_features.csv')
tda_features = pd.read_csv('outputs/tda_features/tda_features.csv')

print(f"\nFeature dimensions:")
print(f"  Original features: {orig_features.shape}")
print(f"  TDA features: {tda_features.shape}")

if len(orig_features) == len(tda_features):
    print("  ✓ Feature counts match")
else:
    print("  ⚠️  Feature count mismatch!")

# ============================================================================
# CHECK 3: BASELINE MODEL HYPERPARAMETERS
# ============================================================================
print("\n" + "="*80)
print("CHECK 3: BASELINE MODEL HYPERPARAMETERS")
print("="*80)

try:
    with open('results/models/baseline_random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    
    with open('results/models/baseline_svm.pkl', 'rb') as f:
        svm = pickle.load(f)
    
    with open('results/models/baseline_logistic_regression.pkl', 'rb') as f:
        lr = pickle.load(f)
    
    print("\nRandom Forest:")
    print(f"  n_estimators: {rf.n_estimators}")
    print(f"  max_depth: {rf.max_depth}")
    print(f"  min_samples_split: {rf.min_samples_split}")
    print(f"  class_weight: {rf.class_weight}")
    
    print("\nSVM:")
    print(f"  kernel: {svm.kernel}")
    print(f"  C: {svm.C}")
    print(f"  gamma: {svm.gamma}")
    print(f"  class_weight: {svm.class_weight}")
    
    print("\nLogistic Regression:")
    print(f"  C: {lr.C}")
    print(f"  penalty: {lr.penalty}")
    print(f"  solver: {lr.solver}")
    print(f"  class_weight: {lr.class_weight}")
    
    # Check for potential issues
    if svm.class_weight is None:
        print("\n⚠️  SVM: No class_weight balancing (may explain poor performance on imbalanced data)")
    if svm.C == 1.0 and svm.gamma == 'scale':
        print("⚠️  SVM: Using default hyperparameters (not tuned)")
        
except Exception as e:
    print(f"\n⚠️  Could not load models: {e}")

# Feature scaling check
print("\n" + "-"*80)
print("FEATURE SCALING CHECK:")
print("-" * 80)
print("\nOriginal features statistics (first 5 features):")
print(orig_features.iloc[:, :5].describe().loc[['mean', 'std', 'min', 'max']])

# ============================================================================
# CHECK 4: TDA FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("CHECK 4: TDA FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feat_imp = pd.read_csv('results/tables/tda_feature_importance_full.csv')

print("\nTOP 20 FEATURES:")
print("-" * 80)
print(feat_imp.head(20).to_string(index=False))

# Categorize features
tda_features_imp = feat_imp[feat_imp['feature'].str.contains('tda_', na=False)]
original_features_imp = feat_imp[~feat_imp['feature'].str.contains('tda_', na=False)]

print("\n" + "-"*80)
print("FEATURE TYPE BREAKDOWN:")
print("-" * 80)
print(f"TDA features in top 20: {len(tda_features_imp.head(20))}")
print(f"Original features in top 20: {len(original_features_imp.head(20))}")

# Break down by manifold
c2_feat = tda_features_imp[tda_features_imp['feature'].str.contains('c2', na=False)]
network_feat = tda_features_imp[tda_features_imp['feature'].str.contains('network', na=False)]
physical_feat = tda_features_imp[tda_features_imp['feature'].str.contains('physical', na=False)]

print(f"\nC2 manifold features in top 20: {len(c2_feat.head(20))}")
print(f"Network manifold features in top 20: {len(network_feat.head(20))}")
print(f"Physical manifold features in top 20: {len(physical_feat.head(20))}")

# Break down by homology dimension
h0_feat = tda_features_imp[tda_features_imp['feature'].str.contains('h0', na=False)]
h1_feat = tda_features_imp[tda_features_imp['feature'].str.contains('h1', na=False)]
h2_feat = tda_features_imp[tda_features_imp['feature'].str.contains('h2', na=False)]

print(f"\nH0 (components) features in top 20: {len(h0_feat.head(20))}")
print(f"H1 (loops) features in top 20: {len(h1_feat.head(20))}")
print(f"H2 (voids) features in top 20: {len(h2_feat.head(20))}")

print("\n" + "-"*80)
print("TOP 10 TDA FEATURES (for interpretation):")
print("-" * 80)
print(tda_features_imp.head(10).to_string(index=False))

# ============================================================================
# CHECK 5: WASSERSTEIN METHODOLOGY
# ============================================================================
print("\n" + "="*80)
print("CHECK 5: WASSERSTEIN METHODOLOGY VERIFICATION")
print("="*80)

try:
    with open('outputs/wasserstein/baseline_metadata.pkl', 'rb') as f:
        baseline_meta = pickle.load(f)
    
    print("\nBaseline metadata keys:", list(baseline_meta.keys()))
    
    if 'sample_indices' in baseline_meta:
        print(f"\nBaseline sample size: {len(baseline_meta['sample_indices'])}")
        print(f"First 10 indices: {baseline_meta['sample_indices'][:10]}")
    else:
        print("\n⚠️  Sample indices not tracked")
    
    if 'selection_method' in baseline_meta:
        print(f"\nSelection method: {baseline_meta['selection_method']}")
    else:
        print("\n⚠️  Selection method not documented")
        
except Exception as e:
    print(f"\n⚠️  Could not load baseline metadata: {e}")

# Detection threshold analysis
detection_results = pd.read_csv('outputs/wasserstein/detection_results.csv')

print("\n" + "-"*80)
print("DETECTION THRESHOLD ANALYSIS:")
print("-" * 80)

if 'c2_z_score' in detection_results.columns:
    print("\n✓ Z-score normalization used")
    print(f"  C2 z-score range: [{detection_results['c2_z_score'].min():.2f}, {detection_results['c2_z_score'].max():.2f}]")
    print(f"  Network z-score range: [{detection_results['network_z_score'].min():.2f}, {detection_results['network_z_score'].max():.2f}]")
    print(f"  Physical z-score range: [{detection_results['physical_z_score'].min():.2f}, {detection_results['physical_z_score'].max():.2f}]")

if 'detected' in detection_results.columns:
    detected = detection_results['detected'].sum()
    total = len(detection_results)
    print(f"\nTotal detected: {detected}/{total} ({100*detected/total:.2f}%)")

print("\n⚠️  ACTION REQUIRED: Verify if 3σ threshold was fixed a priori or tuned")
print("    Check script: scripts/5W_wasserstein_detection.py")

# ============================================================================
# CHECK 6: MULTI-MANIFOLD COMBINATION
# ============================================================================
print("\n" + "="*80)
print("CHECK 6: MULTI-MANIFOLD COMBINATION STRATEGY")
print("="*80)

print("\nDetection results columns:")
print(detection_results.columns.tolist())

if 'c2_detected' in detection_results.columns:
    print("\n✓ Per-manifold detection flags found")
    
    c2_only = (detection_results['c2_detected'] == 1) & (detection_results['network_detected'] == 0) & (detection_results['physical_detected'] == 0)
    net_only = (detection_results['c2_detected'] == 0) & (detection_results['network_detected'] == 1) & (detection_results['physical_detected'] == 0)
    phys_only = (detection_results['c2_detected'] == 0) & (detection_results['network_detected'] == 0) & (detection_results['physical_detected'] == 1)
    multiple = detection_results[['c2_detected', 'network_detected', 'physical_detected']].sum(axis=1) > 1
    
    print(f"\nDetected by C2 only: {c2_only.sum()}")
    print(f"Detected by Network only: {net_only.sum()}")
    print(f"Detected by Physical only: {phys_only.sum()}")
    print(f"Detected by multiple manifolds: {multiple.sum()}")
    
    # Check combination logic
    or_logic = (detection_results['detected'] == ((detection_results['c2_detected'] == 1) | 
                                            (detection_results['network_detected'] == 1) | 
                                            (detection_results['physical_detected'] == 1)).astype(int)).all()
    
    if or_logic:
        print(f"\n✓ Combination logic: OR (any manifold flags = anomaly)")
    else:
        print(f"\n⚠️  Combination logic: NOT simple OR (check script for details)")

# Per-attack AUC
print("\n" + "-"*80)
print("PER-ATTACK AUC BREAKDOWN:")
print("-" * 80)
auc_data = pd.read_csv('results/tables/wasserstein_per_attack_auc.csv')
print(auc_data.to_string(index=False))

# ============================================================================
# CHECK 7: COMPUTATIONAL PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("CHECK 7: COMPUTATIONAL PERFORMANCE")
print("="*80)

total_flows = 122171
runtime_minutes = 53
runtime_seconds = runtime_minutes * 60

samples_per_sec = total_flows / runtime_seconds
print(f"\nProcessing statistics:")
print(f"  Total flows: {total_flows:,}")
print(f"  Runtime (Script 3W): {runtime_minutes} min = {runtime_seconds:,} sec")
print(f"  Processing rate: {samples_per_sec:.1f} samples/sec")

supervised_neighborhood = 100
unsupervised_neighborhood = 50
print(f"\nNeighborhood sizes:")
print(f"  Supervised (100-pt neighborhoods): {supervised_neighborhood} points")
print(f"  Unsupervised (50-pt neighborhoods): {unsupervised_neighborhood} points")
print(f"  Ratio: {supervised_neighborhood/unsupervised_neighborhood:.1f}x more points")

print("\n⚠️  ACTION REQUIRED: Document why 50-pt vs 100-pt neighborhoods")
print("    Is this a computational constraint or design choice?")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print("\n✓ CHECKS COMPLETED:")
print("  [1] Class distribution & per-class performance")
print("  [2] Data leakage verification")
print("  [3] Baseline hyperparameters")
print("  [4] TDA feature importance")
print("  [5] Wasserstein baseline construction")
print("  [6] Multi-manifold combination logic")
print("  [7] Computational performance")

print("\n⚠️  ACTION ITEMS TO VERIFY MANUALLY:")
print("  [ ] Check scripts/5W_wasserstein_detection.py for threshold selection")
print("  [ ] Review neighborhood construction for train/test separation")
print("  [ ] Document justification for 50-pt vs 100-pt neighborhoods")
print("  [ ] Verify no cross-validation was performed (or add results)")

print("\n" + "="*80)
print("END OF VALIDATION CHECKS")
print("="*80 + "\n")

