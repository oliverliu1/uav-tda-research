"""
07_tda_enhanced_models.py

TDA-Enhanced Models

Trains classification models using the combined feature set (original + TDA).
Compares performance against baseline models to quantify TDA's value.

Models: Logistic Regression, Random Forest, SVM

Author: Oliver Liu
Date: April 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")

# Paths
OUTPUT_DIR = "outputs"
FEATURES_DIR = f"{OUTPUT_DIR}/tda_features"
RESULTS_DIR = "results"
MODELS_DIR = f"{RESULTS_DIR}/models"
TABLES_DIR = f"{RESULTS_DIR}/tables"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 80)
print("MULTI-MANIFOLD TDA PIPELINE - STEP 7: TDA-ENHANCED MODELS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# LOAD PRE-SPLIT COMBINED FEATURES
#
# Script 06 saves separate train/test files that were produced WITHOUT any
# cross-split contamination in the TDA neighborhood computation.
# We load those directly — no train_test_split call needed here.
# ==============================================================================
print("Loading pre-split combined feature sets (Original + TDA)...")
print("-" * 80)

try:
    train_df = pd.read_csv(f"{FEATURES_DIR}/combined_features_train.csv")
    test_df = pd.read_csv(f"{FEATURES_DIR}/combined_features_test.csv")

    print(f"✓ Train features loaded: {train_df.shape}")
    print(f"✓ Test features loaded:  {test_df.shape}")
    print(f"  - Features per sample: {train_df.shape[1] - 1} (excluding label)")

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Keep X_combined column reference for feature importance output
    X_combined = X_train  # same columns

    print(f"\n✓ Train: {X_train.shape[0]:,} samples")
    print(f"✓ Test:  {X_test.shape[0]:,} samples")
    print(f"\nTrain label distribution:")
    print(y_train.value_counts())
    print(f"\nTest label distribution:")
    print(y_test.value_counts())
    print()

except FileNotFoundError as e:
    print(f"✗ ERROR: Pre-split combined features not found")
    print(f"  {e}")
    print("  Please run Script 6 first.")
    exit(1)

# Random state for reproducibility (matches split in Script 01)
RANDOM_STATE = 42

# ==============================================================================
# FEATURE SCALING
# ==============================================================================
print("-" * 80)
print("Scaling features...")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")
print()

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# ==============================================================================
# TRAIN MODELS
# ==============================================================================
print("=" * 80)
print("TRAINING TDA-ENHANCED MODELS")
print("=" * 80)
print("Training 3 models: Logistic Regression, Random Forest, SVM")
print("⚠️  Estimated time: 45-60 minutes")
print()

results = {}
start_time_all = datetime.now()

# ------------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
# ------------------------------------------------------------------------------
print("-" * 80)
print("1. LOGISTIC REGRESSION (TDA-Enhanced)")
print("-" * 80)

start_time = datetime.now()

lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)

print("Training...")
lr_model.fit(X_train_scaled, y_train_encoded)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)

# Metrics
acc_lr = accuracy_score(y_test_encoded, y_pred_lr)
f1_lr = f1_score(y_test_encoded, y_pred_lr, average="weighted")
auc_lr = roc_auc_score(
    y_test_encoded, y_pred_proba_lr, multi_class="ovr", average="weighted"
)

elapsed = (datetime.now() - start_time).total_seconds() / 60

print(f"✓ Training complete in {elapsed:.1f} minutes")
print(f"  - Test Accuracy: {acc_lr*100:.2f}%")
print(f"  - Test F1 (weighted): {f1_lr*100:.2f}%")
print(f"  - Test AUC: {auc_lr*100:.2f}%")
print()

results["Logistic Regression (TDA)"] = {
    "model": lr_model,
    "accuracy": acc_lr,
    "f1": f1_lr,
    "auc": auc_lr,
    "predictions": y_pred_lr,
    "probabilities": y_pred_proba_lr,
}

# Save model
with open(f"{MODELS_DIR}/tda_logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# ------------------------------------------------------------------------------
# 2. RANDOM FOREST
# ------------------------------------------------------------------------------
print("-" * 80)
print("2. RANDOM FOREST (TDA-Enhanced)")
print("-" * 80)

start_time = datetime.now()

rf_model = RandomForestClassifier(
    n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
)

print("Training...")
rf_model.fit(X_train_scaled, y_train_encoded)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

# Metrics
acc_rf = accuracy_score(y_test_encoded, y_pred_rf)
f1_rf = f1_score(y_test_encoded, y_pred_rf, average="weighted")
auc_rf = roc_auc_score(
    y_test_encoded, y_pred_proba_rf, multi_class="ovr", average="weighted"
)

elapsed = (datetime.now() - start_time).total_seconds() / 60

print(f"✓ Training complete in {elapsed:.1f} minutes")
print(f"  - Test Accuracy: {acc_rf*100:.2f}%")
print(f"  - Test F1 (weighted): {f1_rf*100:.2f}%")
print(f"  - Test AUC: {auc_rf*100:.2f}%")
print()

results["Random Forest (TDA)"] = {
    "model": rf_model,
    "accuracy": acc_rf,
    "f1": f1_rf,
    "auc": auc_rf,
    "predictions": y_pred_rf,
    "probabilities": y_pred_proba_rf,
}

# Save model
with open(f"{MODELS_DIR}/tda_random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": X_combined.columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

feature_importance.to_csv(f"{TABLES_DIR}/tda_feature_importance.csv", index=False)
print(f"✓ Feature importance saved")

# Top 10 most important features
print("\nTop 10 most important features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.4f}")
print()

# ------------------------------------------------------------------------------
# 3. SVM
# ------------------------------------------------------------------------------
print("-" * 80)
print("3. SUPPORT VECTOR MACHINE (TDA-Enhanced)")
print("-" * 80)

start_time = datetime.now()

svm_model = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)

print("Training...")
svm_model.fit(X_train_scaled, y_train_encoded)

# Predictions
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)

# Metrics
acc_svm = accuracy_score(y_test_encoded, y_pred_svm)
f1_svm = f1_score(y_test_encoded, y_pred_svm, average="weighted")
auc_svm = roc_auc_score(
    y_test_encoded, y_pred_proba_svm, multi_class="ovr", average="weighted"
)

elapsed = (datetime.now() - start_time).total_seconds() / 60

print(f"✓ Training complete in {elapsed:.1f} minutes")
print(f"  - Test Accuracy: {acc_svm*100:.2f}%")
print(f"  - Test F1 (weighted): {f1_svm*100:.2f}%")
print(f"  - Test AUC: {auc_svm*100:.2f}%")
print()

results["SVM (TDA)"] = {
    "model": svm_model,
    "accuracy": acc_svm,
    "f1": f1_svm,
    "auc": auc_svm,
    "predictions": y_pred_svm,
    "probabilities": y_pred_proba_svm,
}

# Save model
with open(f"{MODELS_DIR}/tda_svm.pkl", "wb") as f:
    pickle.dump(svm_model, f)

total_training_time = (datetime.now() - start_time_all).total_seconds() / 60

print("=" * 80)
print(f"✓ All models trained in {total_training_time:.1f} minutes")
print("=" * 80)
print()

# ==============================================================================
# SAVE RESULTS SUMMARY
# ==============================================================================
print("-" * 80)
print("Saving results summary...")
print("-" * 80)

results_summary = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "Test Accuracy": [results[m]["accuracy"] for m in results.keys()],
        "Test F1 (weighted)": [results[m]["f1"] for m in results.keys()],
        "Test AUC": [results[m]["auc"] for m in results.keys()],
    }
)

results_summary.to_csv(f"{TABLES_DIR}/tda_models_metrics.csv", index=False)
print(f"✓ Saved: {TABLES_DIR}/tda_models_metrics.csv")

# ==============================================================================
# PER-CLASS METRICS (Best Model)
# ==============================================================================
print("\n" + "-" * 80)
print("Computing per-class metrics for best model...")
print("-" * 80)

# Find best model by test accuracy
best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
best_model = results[best_model_name]

print(f"Best model: {best_model_name}")
print(f"Test Accuracy: {best_model['accuracy']*100:.2f}%")
print()

# Classification report
y_pred_best = best_model["predictions"]
class_names = le.classes_

print("Per-class metrics:")
print(
    classification_report(
        y_test_encoded, y_pred_best, target_names=class_names, digits=4
    )
)

# Save classification report
report_dict = classification_report(
    y_test_encoded, y_pred_best, target_names=class_names, output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{TABLES_DIR}/tda_classification_report.csv")
print(f"✓ Saved: {TABLES_DIR}/tda_classification_report.csv")

# Confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred_best)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(
    f"{TABLES_DIR}/tda_confusion_matrix_{best_model_name.replace(' ', '_').lower()}.csv"
)
print(f"✓ Saved confusion matrix")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("✓ TDA-ENHANCED MODELS COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\nModels trained:")
for model_name in results.keys():
    acc = results[model_name]["accuracy"]
    f1 = results[model_name]["f1"]
    auc = results[model_name]["auc"]
    print(
        f"  {model_name:30s} Acc: {acc*100:.2f}%  F1: {f1*100:.2f}%  AUC: {auc*100:.2f}%"
    )

print(f"\nBest model: {best_model_name}")
print(f"  Accuracy: {best_model['accuracy']*100:.2f}%")

print("\nFiles saved:")
print(f"  - {TABLES_DIR}/tda_models_metrics.csv")
print(f"  - {TABLES_DIR}/tda_classification_report.csv")
print(f"  - {TABLES_DIR}/tda_feature_importance.csv")
print(f"  - {MODELS_DIR}/tda_*.pkl (3 models)")

print("\nNext: Run 08_comparative_analysis.py (10-15 min)")
print("=" * 80)
