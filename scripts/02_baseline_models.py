"""
02_baseline_models.py

Baseline Model Training (No TDA Features)

This script trains three baseline classifiers on the original feature space
WITHOUT topological features. Results serve as comparison baseline for
TDA-enhanced models.

Models:
- Logistic Regression (L2 regularization)
- Random Forest (200 trees)
- Support Vector Machine (RBF kernel, class-weighted)

Evaluation: 5-fold cross-validation, 80/20 train-test split

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
import pickle
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Paths
OUTPUT_DIR = "outputs"
RESULTS_DIR = "results"

print("=" * 80)
print("MULTI-MANIFOLD TDA PIPELINE - STEP 2: BASELINE MODELS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD PREPROCESSED DATA
# ==============================================================================
print("Loading preprocessed features and labels...")

try:
    # Load original features (before TDA)
    X = pd.read_csv(f"{OUTPUT_DIR}/original_features.csv")
    y = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")["label"]

    print(f"✓ Features loaded: {X.shape}")
    print(f"✓ Labels loaded: {y.shape}")
    print(f"\nFeature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")

except FileNotFoundError as e:
    print(f"✗ ERROR: Required files not found.")
    print("  Please run 01_data_prep.py first.")
    exit(1)

# ==============================================================================
# TRAIN-TEST SPLIT
# ==============================================================================
print("\n" + "-" * 80)
print("Creating train-test split...")
print("-" * 80)

RANDOM_STATE = 42
TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(
    f"  - Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)"
)
print(f"  - Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  - Random state: {RANDOM_STATE}")
print(f"  - Stratified split: Yes\n")

# Class distribution
print("Class distribution in splits:")
print("\nTraining set:")
train_dist = y_train.value_counts()
for label, count in train_dist.items():
    print(f"  - {label}: {count:,} ({count/len(y_train)*100:.2f}%)")

print("\nTest set:")
test_dist = y_test.value_counts()
for label, count in test_dist.items():
    print(f"  - {label}: {count:,} ({count/len(y_test)*100:.2f}%)")

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================
print("\n" + "-" * 80)
print("Defining baseline models...")
print("-" * 80)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=RANDOM_STATE,
        class_weight="balanced",
        probability=True,  # Enable probability estimates for AUC
        cache_size=1000,
    ),
}

print("Models configured:")
for name in models.keys():
    print(f"  ✓ {name}")

# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 1: 5-FOLD CROSS-VALIDATION ON TRAINING SET")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, model in models.items():
    print(f"\n[{name}]")
    print("-" * 40)
    print(f"  Running 5-fold CV...")

    # Cross-validation scoring
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )

    cv_results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores,
    }

    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ==============================================================================
# FINAL MODEL TRAINING
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 2: TRAINING FINAL MODELS ON FULL TRAINING SET")
print("=" * 80)

trained_models = {}

for name, model in models.items():
    print(f"\n[{name}]")
    print("-" * 40)
    print("  Training...")

    model.fit(X_train, y_train)
    trained_models[name] = model

    print(f"  ✓ Training complete")

# Save trained models
print("\n" + "-" * 80)
print("Saving trained models...")
print("-" * 80)

os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)

for name, model in trained_models.items():
    model_filename = name.lower().replace(" ", "_")
    with open(f"{RESULTS_DIR}/models/baseline_{model_filename}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: baseline_{model_filename}.pkl")

# ==============================================================================
# TEST SET EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 3: EVALUATION ON HELD-OUT TEST SET")
print("=" * 80)

results = []

for name, model in trained_models.items():
    print(f"\n[{name}]")
    print("-" * 40)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # AUC-ROC (multiclass)
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    try:
        auc = roc_auc_score(
            y_test_bin, y_pred_proba, average="weighted", multi_class="ovr"
        )
    except:
        auc = np.nan

    # Store results
    results.append(
        {
            "Model": name,
            "CV_Mean": cv_results[name]["cv_mean"],
            "CV_Std": cv_results[name]["cv_std"],
            "Test_Accuracy": accuracy,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F1": f1,
            "Test_AUC": auc,
        }
    )

    print(f"  Test Set Performance:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

    # Classification report (brief)
    print(f"\n  Per-class F1 scores:")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    for class_name, metrics in report.items():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            print(f"    {class_name:.<30} {metrics['f1-score']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
    cm_filename = name.lower().replace(" ", "_")
    cm_df.to_csv(f"{RESULTS_DIR}/tables/baseline_cm_{cm_filename}.csv")
    print(f"\n    ✓ Saved confusion matrix to tables/")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("Saving results summary...")
print("=" * 80)

os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)

results_df = pd.DataFrame(results)
results_df = results_df.round(4)

# Save to CSV
results_df.to_csv(f"{RESULTS_DIR}/tables/baseline_metrics.csv", index=False)
print(f"✓ Saved: {RESULTS_DIR}/tables/baseline_metrics.csv")

# Display results table
print("\n" + "=" * 80)
print("BASELINE MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("BASELINE MODEL TRAINING COMPLETE")
print("=" * 80)

best_model = results_df.loc[results_df["Test_Accuracy"].idxmax(), "Model"]
best_accuracy = results_df["Test_Accuracy"].max()

print(f"\nBest baseline model: {best_model}")
print(f"Best test accuracy: {best_accuracy:.4f}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nOutputs created:")
print(f"  - {RESULTS_DIR}/tables/baseline_metrics.csv")
print(f"  - {RESULTS_DIR}/models/baseline_*.pkl (3 models)")
print(f"  - {RESULTS_DIR}/tables/baseline_cm_*.csv (3 confusion matrices)")
print("\nNext steps:")
print("  1. Review baseline performance above")
print("  2. Run TDA computation scripts (03-05) - these will take several hours")
print("=" * 80)
