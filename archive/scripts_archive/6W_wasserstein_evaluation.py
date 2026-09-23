"""
6W_wasserstein_evaluation.py

Wasserstein Approach Evaluation

Evaluates anomaly detection performance using AUC-ROC metrics
and generates visualizations for poster presentation.

Author: Oliver Liu  
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = "outputs"
WASSERSTEIN_DIR = f"{OUTPUT_DIR}/wasserstein"
RESULTS_DIR = "results"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
TABLES_DIR = f"{RESULTS_DIR}/tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("WASSERSTEIN APPROACH - STEP 6W: EVALUATION & VISUALIZATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD DETECTION RESULTS
# ==============================================================================
print("Loading detection results...")
print("-" * 80)

try:
    results_df = pd.read_csv(f"{WASSERSTEIN_DIR}/detection_results.csv")
    
    print(f"✓ Loaded: {results_df.shape}")
    print(f"  Columns: {list(results_df.columns)}")
    print()
    
except FileNotFoundError:
    print("✗ ERROR: Detection results not found")
    print("  Please run 5W_wasserstein_detection.py first")
    exit(1)

# ==============================================================================
# CREATE BINARY LABELS (Normal vs Attack)
# ==============================================================================
print("Creating binary labels (Normal=0, Attack=1)...")
print("-" * 80)

results_df['is_attack'] = (results_df['label'] != 'Normal Traffic').astype(int)
results_df['detected_binary'] = results_df['anomaly_detected'].astype(int)

n_normal = (results_df['is_attack'] == 0).sum()
n_attack = (results_df['is_attack'] == 1).sum()

print(f"✓ Normal Traffic: {n_normal:,}")
print(f"✓ Attack Traffic: {n_attack:,}")
print()

# ==============================================================================
# OVERALL AUC-ROC
# ==============================================================================
print("=" * 80)
print("COMPUTING AUC-ROC METRICS")
print("=" * 80)
print()

# Binary classification: Normal vs Attack
y_true = results_df['is_attack']
y_pred = results_df['detected_binary']

# Per-manifold Z-scores as prediction scores
z_scores_combined = results_df[['c2_zscore', 'network_zscore', 'physical_zscore']].max(axis=1)

overall_auc = roc_auc_score(y_true, z_scores_combined)

print(f"Overall AUC-ROC: {overall_auc:.4f}")
print()

# Per-manifold AUC
print("Per-manifold AUC-ROC:")
print("-" * 80)

manifold_aucs = {}
for manifold in ['c2', 'network', 'physical']:
    z_col = f'{manifold}_zscore'
    auc = roc_auc_score(y_true, results_df[z_col])
    manifold_aucs[manifold] = auc
    print(f"  {manifold.upper():8s}: {auc:.4f}")

print()

# ==============================================================================
# PER-ATTACK AUC-ROC
# ==============================================================================
print("Per-attack type AUC-ROC:")
print("-" * 80)

per_attack_results = []

for attack_type in results_df['label'].unique():
    if attack_type == 'Normal Traffic':
        continue
    
    # Binary: this attack vs normal
    y_binary = (results_df['label'] == attack_type).astype(int)
    
    # AUC for each manifold
    attack_aucs = {}
    for manifold in ['c2', 'network', 'physical']:
        z_col = f'{manifold}_zscore'
        try:
            auc = roc_auc_score(y_binary, results_df[z_col])
            attack_aucs[manifold] = auc
        except:
            attack_aucs[manifold] = 0.0
    
    # Overall (max of manifolds)
    overall_attack_auc = roc_auc_score(y_binary, z_scores_combined)
    
    per_attack_results.append({
        'Attack Type': attack_type,
        'AUC (Overall)': overall_attack_auc,
        'AUC (C2)': attack_aucs['c2'],
        'AUC (Network)': attack_aucs['network'],
        'AUC (Physical)': attack_aucs['physical']
    })
    
    print(f"\n{attack_type}:")
    print(f"  Overall: {overall_attack_auc:.4f}")
    print(f"  C2: {attack_aucs['c2']:.4f}, Network: {attack_aucs['network']:.4f}, Physical: {attack_aucs['physical']:.4f}")

per_attack_df = pd.DataFrame(per_attack_results)

print()

# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================
print("=" * 80)
print("CONFUSION MATRIX (Normal vs Attack)")
print("=" * 80)
print()

cm = confusion_matrix(y_true, y_pred)
print(cm)
print()

tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print()

# ==============================================================================
# SAVE METRICS
# ==============================================================================
print("-" * 80)
print("Saving evaluation metrics...")
print("-" * 80)

# Overall metrics
overall_metrics = {
    'Overall AUC': overall_auc,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'C2 AUC': manifold_aucs['c2'],
    'Network AUC': manifold_aucs['network'],
    'Physical AUC': manifold_aucs['physical']
}

overall_df = pd.DataFrame([overall_metrics])
overall_df.to_csv(f"{TABLES_DIR}/wasserstein_overall_metrics.csv", index=False)
print(f"✓ Saved: {TABLES_DIR}/wasserstein_overall_metrics.csv")

# Per-attack metrics
per_attack_df.to_csv(f"{TABLES_DIR}/wasserstein_per_attack_auc.csv", index=False)
print(f"✓ Saved: {TABLES_DIR}/wasserstein_per_attack_auc.csv")

print()

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

# ------------------------------------------------------------------------------
# 1. ROC Curves Per Attack
# ------------------------------------------------------------------------------
print("1. Creating ROC curves...")

fig, ax = plt.subplots(figsize=(10, 8))

for attack_type in results_df['label'].unique():
    if attack_type == 'Normal Traffic':
        continue
    
    y_binary = (results_df['label'] == attack_type).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, z_scores_combined)
    
    auc = roc_auc_score(y_binary, z_scores_combined)
    
    ax.plot(fpr, tpr, label=f'{attack_type} (AUC={auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Wasserstein Distance Anomaly Detection', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/wasserstein_roc_curves.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/wasserstein_roc_curves.png")
plt.close()

# ------------------------------------------------------------------------------
# 2. Distance Distributions
# ------------------------------------------------------------------------------
print("2. Creating distance distributions...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

manifolds = ['c2', 'network', 'physical']
titles = ['C2 Manifold', 'Network Manifold', 'Physical Manifold']

for idx, (manifold, title) in enumerate(zip(manifolds, titles)):
    ax = axes[idx]
    col = f'{manifold}_distance'
    
    # Normal vs attacks
    normal_dist = results_df[results_df['is_attack'] == 0][col]
    attack_dist = results_df[results_df['is_attack'] == 1][col]
    
    ax.hist(normal_dist, bins=50, alpha=0.6, label='Normal', color='green', density=True)
    ax.hist(attack_dist, bins=50, alpha=0.6, label='Attack', color='red', density=True)
    
    ax.set_xlabel('Wasserstein Distance', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/wasserstein_distance_distributions.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/wasserstein_distance_distributions.png")
plt.close()

# ------------------------------------------------------------------------------
# 3. Per-Attack AUC Heatmap
# ------------------------------------------------------------------------------
print("3. Creating per-attack AUC heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = per_attack_df.set_index('Attack Type')[['AUC (C2)', 'AUC (Network)', 'AUC (Physical)']]
heatmap_data.columns = ['C2', 'Network', 'Physical']

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
            cbar_kws={'label': 'AUC'}, ax=ax, linewidths=1)

ax.set_title('Per-Attack AUC by Manifold', fontsize=14, fontweight='bold')
ax.set_xlabel('Manifold', fontsize=12)
ax.set_ylabel('Attack Type', fontsize=12)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/wasserstein_attack_manifold_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/wasserstein_attack_manifold_heatmap.png")
plt.close()

# ------------------------------------------------------------------------------
# 4. Confusion Matrix Visualization
# ------------------------------------------------------------------------------
print("4. Creating confusion matrix visualization...")

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'})

ax.set_title('Confusion Matrix: Wasserstein Anomaly Detection', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/wasserstein_confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/wasserstein_confusion_matrix.png")
plt.close()

print()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("=" * 80)
print("✓ WASSERSTEIN EVALUATION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\n🎯 KEY RESULTS:")
print("-" * 80)
print(f"\nOverall Performance:")
print(f"  - AUC-ROC: {overall_auc:.4f}")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")

print(f"\nPer-Manifold AUC:")
print(f"  - C2: {manifold_aucs['c2']:.4f}")
print(f"  - Network: {manifold_aucs['network']:.4f}")
print(f"  - Physical: {manifold_aucs['physical']:.4f}")

print(f"\nPer-Attack AUC (Overall):")
for _, row in per_attack_df.iterrows():
    print(f"  - {row['Attack Type']:20s}: {row['AUC (Overall)']:.4f}")

print("\n" + "=" * 80)
print("FILES CREATED FOR POSTER:")
print("=" * 80)

print("\nTables:")
print(f"  - {TABLES_DIR}/wasserstein_overall_metrics.csv")
print(f"  - {TABLES_DIR}/wasserstein_per_attack_auc.csv")

print("\nFigures:")
print(f"  - {FIGURES_DIR}/wasserstein_roc_curves.png")
print(f"  - {FIGURES_DIR}/wasserstein_distance_distributions.png")
print(f"  - {FIGURES_DIR}/wasserstein_attack_manifold_heatmap.png")
print(f"  - {FIGURES_DIR}/wasserstein_confusion_matrix.png")

print("\n" + "=" * 80)
print("🎉 COMPLETE WASSERSTEIN PIPELINE FINISHED!")
print("=" * 80)
print("\nYou now have BOTH approaches for your poster:")
print("  ✓ Supervised (Scripts 1-8) - TDA feature extraction")
print("  ✓ Unsupervised (Scripts 2W-6W) - Wasserstein anomaly detection")
print("\nBoth approaches ready for Monday presentation!")
print("=" * 80)
