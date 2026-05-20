"""
create_feature_importance_viz.py

Visualizes which TDA features are most important for classification.
Shows top features from the best Random Forest model.

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set style
sns.set_style("whitegrid")

print("Loading Random Forest model and feature names...")

# Load the trained Random Forest model
with open("results/models/tda_random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load combined features to get feature names
combined_features = pd.read_csv("outputs/tda_features/combined_features.csv")
feature_names = combined_features.columns.tolist()

print(f"✓ Loaded model with {len(feature_names)} features")

# Get feature importances
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

# Separate original vs TDA features
feature_importance_df["type"] = feature_importance_df["feature"].apply(
    lambda x: (
        "TDA Feature"
        if any(substr in x for substr in ["C2_", "Network_", "Physical_"])
        else "Original Feature"
    )
)

print(f"\nTop 10 Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Create visualization - Top 20 features
fig, ax = plt.subplots(figsize=(12, 8))

top_20 = feature_importance_df.head(20)

# Color by feature type
colors = ["#2E86AB" if t == "TDA Feature" else "#A23B72" for t in top_20["type"]]

bars = ax.barh(
    range(len(top_20)),
    top_20["importance"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
)

# Set labels
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20["feature"], fontsize=10)
ax.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
ax.set_title(
    "Top 20 Features by Importance (Random Forest TDA-Enhanced)",
    fontsize=14,
    fontweight="bold",
)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#2E86AB", label="TDA Feature"),
    Patch(facecolor="#A23B72", label="Original Feature"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig(
    "results/figures/tda_feature_importance_top20.png", dpi=300, bbox_inches="tight"
)
print("\n✓ Saved: results/figures/tda_feature_importance_top20.png")
plt.close()

# Create grouped comparison: TDA vs Original
tda_importance = feature_importance_df[feature_importance_df["type"] == "TDA Feature"][
    "importance"
].sum()
original_importance = feature_importance_df[
    feature_importance_df["type"] == "Original Feature"
]["importance"].sum()

fig, ax = plt.subplots(figsize=(8, 6))

categories = ["TDA Features", "Original Features"]
values = [tda_importance, original_importance]
colors_grouped = ["#2E86AB", "#A23B72"]

bars = ax.bar(
    categories, values, color=colors_grouped, alpha=0.8, edgecolor="black", linewidth=2
)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax.set_ylabel("Total Importance", fontsize=12, fontweight="bold")
ax.set_title(
    "Feature Importance: TDA vs Original Features", fontsize=14, fontweight="bold"
)
ax.set_ylim(0, max(values) * 1.15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "results/figures/tda_vs_original_importance.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: results/figures/tda_vs_original_importance.png")
plt.close()

# Per-manifold importance breakdown
manifold_importance = {
    "C2": feature_importance_df[
        feature_importance_df["feature"].str.contains("C2_", na=False)
    ]["importance"].sum(),
    "Network": feature_importance_df[
        feature_importance_df["feature"].str.contains("Network_", na=False)
    ]["importance"].sum(),
    "Physical": feature_importance_df[
        feature_importance_df["feature"].str.contains("Physical_", na=False)
    ]["importance"].sum(),
    "Original": original_importance,
}

fig, ax = plt.subplots(figsize=(10, 6))

manifolds = list(manifold_importance.keys())
values = list(manifold_importance.values())
colors_manifold = ["#E63946", "#F1A208", "#06FFA5", "#A23B72"]

bars = ax.bar(
    manifolds, values, color=colors_manifold, alpha=0.8, edgecolor="black", linewidth=2
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylabel("Total Importance", fontsize=12, fontweight="bold")
ax.set_title("Feature Importance by Manifold", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(values) * 1.15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "results/figures/manifold_feature_importance.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: results/figures/manifold_feature_importance.png")
plt.close()

# Save detailed feature importance table
feature_importance_df.to_csv(
    "results/tables/tda_feature_importance_full.csv", index=False
)
print("✓ Saved: results/tables/tda_feature_importance_full.csv")

print("\n" + "=" * 80)
print("✓ FEATURE IMPORTANCE VISUALIZATION COMPLETE")
print("=" * 80)
print("\nKey Findings:")
print(f"  - TDA features total importance: {tda_importance:.3f}")
print(f"  - Original features total importance: {original_importance:.3f}")
print(
    f"  - TDA contribution: {tda_importance/(tda_importance+original_importance)*100:.1f}%"
)
print(f"\nPer-manifold importance:")
for manifold, imp in manifold_importance.items():
    if manifold != "Original":
        print(f"  - {manifold}: {imp:.3f}")
print("=" * 80)
