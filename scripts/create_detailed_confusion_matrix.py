"""
create_detailed_confusion_matrix.py

Creates detailed multi-class confusion matrix showing
how well the TDA-enhanced Random Forest classifies each attack type.

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

print("Loading model and test data...")

# Load the trained Random Forest model
with open("results/models/tda_random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load test data
import joblib
from sklearn.preprocessing import LabelEncoder

# Load combined features
combined_features = pd.read_csv(
    "outputs/tda_features/combined_features_with_labels.csv"
)

# Recreate train-test split (same random_state as Script 7)
from sklearn.model_selection import train_test_split

X = combined_features.drop("label", axis=1)
y = combined_features["label"]

# Split (same as Script 7)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale test data (load scaler or recreate)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)

# Generate predictions
y_pred = rf_model.predict(X_test_scaled)

print(f"✓ Loaded {len(y_test):,} test samples")

# Get class names
classes = sorted(y_test.unique())
print(f"Classes: {classes}")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Raw counts
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    ax=axes[0],
    cbar_kws={"label": "Count"},
    linewidths=0.5,
    linecolor="gray",
)
axes[0].set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
axes[0].set_ylabel("True Label", fontsize=12, fontweight="bold")
axes[0].set_title(
    "Confusion Matrix: Raw Counts\n(Random Forest TDA-Enhanced)",
    fontsize=13,
    fontweight="bold",
)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

# 2. Normalized (percentage)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    xticklabels=classes,
    yticklabels=classes,
    ax=axes[1],
    cbar_kws={"label": "Percentage (%)"},
    vmin=0,
    vmax=100,
    linewidths=0.5,
    linecolor="gray",
)
axes[1].set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
axes[1].set_ylabel("True Label", fontsize=12, fontweight="bold")
axes[1].set_title(
    "Confusion Matrix: Normalized (%)\n(Random Forest TDA-Enhanced)",
    fontsize=13,
    fontweight="bold",
)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(
    "results/figures/detailed_confusion_matrix.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: results/figures/detailed_confusion_matrix.png")
plt.close()

# Create per-class performance table
precision_per_class = []
recall_per_class = []
f1_per_class = []

for i, cls in enumerate(classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    precision_per_class.append(precision)
    recall_per_class.append(recall)
    f1_per_class.append(f1)

performance_df = pd.DataFrame(
    {
        "Attack Type": classes,
        "Precision": precision_per_class,
        "Recall": recall_per_class,
        "F1-Score": f1_per_class,
        "Support": cm.sum(axis=1),
    }
)

# Visualize per-class performance
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(classes))
width = 0.25

bars1 = ax.bar(
    x - width,
    performance_df["Precision"],
    width,
    label="Precision",
    color="#2E86AB",
    alpha=0.8,
    edgecolor="black",
)
bars2 = ax.bar(
    x,
    performance_df["Recall"],
    width,
    label="Recall",
    color="#A23B72",
    alpha=0.8,
    edgecolor="black",
)
bars3 = ax.bar(
    x + width,
    performance_df["F1-Score"],
    width,
    label="F1-Score",
    color="#06FFA5",
    alpha=0.8,
    edgecolor="black",
)


# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

ax.set_xlabel("Attack Type", fontsize=12, fontweight="bold")
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title(
    "Per-Class Performance: Random Forest TDA-Enhanced", fontsize=14, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.legend(loc="lower right", fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/per_class_performance.png", dpi=300, bbox_inches="tight")
print("✓ Saved: results/figures/per_class_performance.png")
plt.close()

# Save performance table
performance_df.to_csv("results/tables/per_class_performance.csv", index=False)
print("✓ Saved: results/tables/per_class_performance.csv")

print("\n" + "=" * 80)
print("✓ DETAILED CONFUSION MATRIX COMPLETE")
print("=" * 80)
print("\nPer-Class Performance:")
print(performance_df.to_string(index=False))
print("=" * 80)
