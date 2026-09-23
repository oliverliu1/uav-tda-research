"""
08_comparative_analysis.py

Comparative Analysis: Baseline vs. TDA-Enhanced Models

Compares performance of baseline models (original features only) against
TDA-enhanced models (original + topological features) to quantify the
value added by persistent homology.

Generates:
- Performance comparison tables
- Improvement metrics
- Statistical significance tests
- Visualizations for poster

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Paths
RESULTS_DIR = "results"
TABLES_DIR = f"{RESULTS_DIR}/tables"
FIGURES_DIR = f"{RESULTS_DIR}/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

print("=" * 80)
print("MULTI-MANIFOLD TDA PIPELINE - STEP 8: COMPARATIVE ANALYSIS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD BASELINE RESULTS
# ==============================================================================
print("Loading baseline model results...")
print("-" * 80)

try:
    baseline_metrics = pd.read_csv(f"{TABLES_DIR}/baseline_metrics.csv")
    print(f"✓ Baseline metrics loaded:")
    print(baseline_metrics.to_string(index=False))
    print()
except FileNotFoundError:
    print("✗ ERROR: Baseline metrics not found")
    print("  Please run Script 2 first.")
    exit(1)

# ==============================================================================
# LOAD TDA-ENHANCED RESULTS
# ==============================================================================
print("Loading TDA-enhanced model results...")
print("-" * 80)

try:
    tda_metrics = pd.read_csv(f"{TABLES_DIR}/tda_models_metrics.csv")
    print(f"✓ TDA metrics loaded:")
    print(tda_metrics.to_string(index=False))
    print()
except FileNotFoundError:
    print("✗ ERROR: TDA metrics not found")
    print("  Please run Script 7 first.")
    exit(1)

# ==============================================================================
# COMPARATIVE ANALYSIS
# ==============================================================================
print("=" * 80)
print("COMPARATIVE ANALYSIS: BASELINE vs TDA-ENHANCED")
print("=" * 80)
print()

# Standardize column names
# Baseline has: Test_Accuracy, Test_F1, Test_AUC
# TDA has: Test Accuracy, Test F1 (weighted), Test AUC
baseline_metrics = baseline_metrics.rename(
    columns={
        "Test_Accuracy": "Test Accuracy",
        "Test_F1": "Test F1 (weighted)",
        "Test_AUC": "Test AUC",
    }
)

# Rename for clarity
baseline_metrics["Type"] = "Baseline"
tda_metrics["Type"] = "TDA-Enhanced"

# Standardize model names
baseline_metrics["Model"] = baseline_metrics["Model"].str.replace(
    " \\(Baseline\\)", "", regex=True
)
tda_metrics["Model"] = tda_metrics["Model"].str.replace(" \\(TDA\\)", "", regex=True)

# Combine
combined_metrics = pd.concat([baseline_metrics, tda_metrics], ignore_index=True)

# ==============================================================================
# COMPUTE IMPROVEMENTS
# ==============================================================================
print("-" * 80)
print("Computing improvements (TDA vs Baseline)...")
print("-" * 80)

improvements = []

for model in baseline_metrics["Model"].unique():
    baseline_row = baseline_metrics[baseline_metrics["Model"] == model].iloc[0]
    tda_row = tda_metrics[tda_metrics["Model"] == model].iloc[0]

    acc_baseline = baseline_row["Test Accuracy"]
    acc_tda = tda_row["Test Accuracy"]
    acc_improvement = acc_tda - acc_baseline
    acc_pct_improvement = (acc_improvement / acc_baseline) * 100

    f1_baseline = baseline_row["Test F1 (weighted)"]
    f1_tda = tda_row["Test F1 (weighted)"]
    f1_improvement = f1_tda - f1_baseline
    f1_pct_improvement = (f1_improvement / f1_baseline) * 100

    auc_baseline = baseline_row["Test AUC"]
    auc_tda = tda_row["Test AUC"]
    auc_improvement = auc_tda - auc_baseline
    auc_pct_improvement = (auc_improvement / auc_baseline) * 100

    improvements.append(
        {
            "Model": model,
            "Baseline Accuracy": acc_baseline,
            "TDA Accuracy": acc_tda,
            "Accuracy Δ": acc_improvement,
            "Accuracy Δ%": acc_pct_improvement,
            "Baseline F1": f1_baseline,
            "TDA F1": f1_tda,
            "F1 Δ": f1_improvement,
            "F1 Δ%": f1_pct_improvement,
            "Baseline AUC": auc_baseline,
            "TDA AUC": auc_tda,
            "AUC Δ": auc_improvement,
            "AUC Δ%": auc_pct_improvement,
        }
    )

improvements_df = pd.DataFrame(improvements)

print("\nPerformance Improvements:")
print(
    improvements_df[["Model", "Accuracy Δ%", "F1 Δ%", "AUC Δ%"]].to_string(index=False)
)
print()

# Save
improvements_df.to_csv(f"{TABLES_DIR}/comparison_improvements.csv", index=False)
print(f"✓ Saved: {TABLES_DIR}/comparison_improvements.csv")
print()

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print("-" * 80)
print("Summary Statistics:")
print("-" * 80)

print(f"\nAverage improvements across all models:")
print(f"  - Accuracy: {improvements_df['Accuracy Δ%'].mean():+.2f}%")
print(f"  - F1 Score: {improvements_df['F1 Δ%'].mean():+.2f}%")
print(f"  - AUC: {improvements_df['AUC Δ%'].mean():+.2f}%")

print(f"\nBest improvement:")
best_acc_model = improvements_df.loc[improvements_df["Accuracy Δ%"].idxmax()]
print(f"  - Model: {best_acc_model['Model']}")
print(f"  - Accuracy improvement: {best_acc_model['Accuracy Δ%']:+.2f}%")
print(
    f"  - Baseline → TDA: {best_acc_model['Baseline Accuracy']*100:.2f}% → {best_acc_model['TDA Accuracy']*100:.2f}%"
)

print(f"\nBest overall model:")
best_tda = tda_metrics.loc[tda_metrics["Test Accuracy"].idxmax()]
print(f"  - Model: {best_tda['Model']}")
print(f"  - Accuracy: {best_tda['Test Accuracy']*100:.2f}%")
print(f"  - F1 Score: {best_tda['Test F1 (weighted)']*100:.2f}%")
print(f"  - AUC: {best_tda['Test AUC']*100:.2f}%")
print()

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
print("=" * 80)
print("GENERATING VISUALIZATIONS FOR POSTER")
print("=" * 80)
print()

# ------------------------------------------------------------------------------
# 1. Performance Comparison Bar Chart
# ------------------------------------------------------------------------------
print("1. Creating performance comparison chart...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["Test Accuracy", "Test F1 (weighted)", "Test AUC"]
titles = ["Accuracy Comparison", "F1 Score Comparison", "AUC Comparison"]

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]

    # Prepare data
    comparison_data = combined_metrics.pivot(
        index="Model", columns="Type", values=metric
    )

    # Plot
    comparison_data.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("")
    ax.legend(title="", fontsize=11)
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis="y", alpha=0.3)

    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    f"{FIGURES_DIR}/baseline_vs_tda_comparison.png", dpi=300, bbox_inches="tight"
)
print(f"✓ Saved: {FIGURES_DIR}/baseline_vs_tda_comparison.png")
plt.close()

# ------------------------------------------------------------------------------
# 2. Improvement Heatmap
# ------------------------------------------------------------------------------
print("2. Creating improvement heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = improvements_df[["Model", "Accuracy Δ%", "F1 Δ%", "AUC Δ%"]].set_index(
    "Model"
)
heatmap_data.columns = ["Accuracy Δ%", "F1 Δ%", "AUC Δ%"]

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    cbar_kws={"label": "Improvement (%)"},
    ax=ax,
    linewidths=1,
)

ax.set_title(
    "Performance Improvements: TDA vs Baseline", fontsize=14, fontweight="bold"
)
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/improvement_heatmap.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {FIGURES_DIR}/improvement_heatmap.png")
plt.close()

# ------------------------------------------------------------------------------
# 3. Model Performance Radar Chart (Best Model)
# ------------------------------------------------------------------------------
print("3. Creating radar chart for best model...")

from math import pi

# Get best baseline and best TDA models
best_baseline = baseline_metrics.loc[baseline_metrics["Test Accuracy"].idxmax()]
best_tda = tda_metrics.loc[tda_metrics["Test Accuracy"].idxmax()]

categories = ["Accuracy", "F1 Score", "AUC"]
baseline_values = [
    best_baseline["Test Accuracy"],
    best_baseline["Test F1 (weighted)"],
    best_baseline["Test AUC"],
]
tda_values = [
    best_tda["Test Accuracy"],
    best_tda["Test F1 (weighted)"],
    best_tda["Test AUC"],
]

# Number of variables
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
baseline_values += baseline_values[:1]
tda_values += tda_values[:1]
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
ax.plot(angles, baseline_values, "o-", linewidth=2, label="Baseline", color="#ff7f0e")
ax.fill(angles, baseline_values, alpha=0.25, color="#ff7f0e")
ax.plot(angles, tda_values, "o-", linewidth=2, label="TDA-Enhanced", color="#2ca02c")
ax.fill(angles, tda_values, alpha=0.25, color="#2ca02c")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=10)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title(
    f'Best Model Performance: {best_tda["Model"]}',
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/best_model_radar.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {FIGURES_DIR}/best_model_radar.png")
plt.close()

# ------------------------------------------------------------------------------
# 4. Create Summary Table for Poster
# ------------------------------------------------------------------------------
print("4. Creating summary table for poster...")

summary_table = improvements_df[
    ["Model", "Baseline Accuracy", "TDA Accuracy", "Accuracy Δ%"]
].copy()
summary_table["Baseline Accuracy"] = (summary_table["Baseline Accuracy"] * 100).round(2)
summary_table["TDA Accuracy"] = (summary_table["TDA Accuracy"] * 100).round(2)
summary_table["Accuracy Δ%"] = summary_table["Accuracy Δ%"].round(2)
summary_table.columns = ["Model", "Baseline (%)", "TDA (%)", "Improvement (%)"]

summary_table.to_csv(f"{TABLES_DIR}/poster_summary_table.csv", index=False)
print(f"✓ Saved: {TABLES_DIR}/poster_summary_table.csv")

print("\nSummary Table for Poster:")
print(summary_table.to_string(index=False))
print()

# ==============================================================================
# STATISTICAL SIGNIFICANCE (PLACEHOLDER)
# ==============================================================================
print("-" * 80)
print("Statistical Analysis:")
print("-" * 80)

print("\nNote: Full statistical significance testing requires cross-validation")
print("or multiple train-test splits. Current analysis uses single split.")
print()

# Simple paired t-test on improvements
improvements_list = improvements_df["Accuracy Δ%"].values
t_stat, p_value = stats.ttest_1samp(improvements_list, 0)

print(f"One-sample t-test (H0: mean improvement = 0):")
print(f"  - t-statistic: {t_stat:.4f}")
print(f"  - p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"  - Result: Improvements are statistically significant (p < 0.05)")
else:
    print(f"  - Result: Improvements are not statistically significant (p ≥ 0.05)")
print()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("=" * 80)
print("✓ COMPARATIVE ANALYSIS COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\n🎯 KEY FINDINGS FOR POSTER:")
print("-" * 80)

print(f"\n1. OVERALL IMPROVEMENT:")
print(f"   Average accuracy improvement: {improvements_df['Accuracy Δ%'].mean():+.2f}%")
if improvements_df["Accuracy Δ%"].mean() > 0:
    print(f"   ✓ TDA features improve classification performance")
else:
    print(f"   ⚠ TDA features did not improve performance (investigate further)")

print(f"\n2. BEST MODEL:")
print(f"   Model: {best_tda['Model']}")
print(f"   Accuracy: {best_tda['Test Accuracy']*100:.2f}%")
print(f"   Improvement over baseline: {best_acc_model['Accuracy Δ%']:+.2f}%")

print(f"\n3. MODEL COMPARISON:")
for _, row in improvements_df.iterrows():
    model = row["Model"]
    delta = row["Accuracy Δ%"]
    symbol = "✓" if delta > 0 else "✗"
    print(f"   {symbol} {model:20s} {delta:+.2f}%")

print("\n" + "=" * 80)
print("FILES CREATED FOR POSTER:")
print("=" * 80)
print("\nTables:")
print(f"  - {TABLES_DIR}/comparison_improvements.csv")
print(f"  - {TABLES_DIR}/poster_summary_table.csv")

print("\nFigures:")
print(f"  - {FIGURES_DIR}/baseline_vs_tda_comparison.png")
print(f"  - {FIGURES_DIR}/improvement_heatmap.png")
print(f"  - {FIGURES_DIR}/best_model_radar.png")

print("\n" + "=" * 80)
print("🎉 COMPLETE PIPELINE FINISHED!")
print("=" * 80)
print("\nYou now have everything you need for your poster presentation:")
print("  ✓ Baseline models (Script 2)")
print("  ✓ TDA persistence diagrams (Scripts 3-5)")
print("  ✓ TDA feature extraction (Script 6)")
print("  ✓ TDA-enhanced models (Script 7)")
print("  ✓ Comparative analysis (Script 8)")
print("\nNext steps:")
print("  1. Review results in results/tables/")
print("  2. Use visualizations in results/figures/ for poster")
print("  3. If time permits, run unsupervised Wasserstein approach")
print("=" * 80)
