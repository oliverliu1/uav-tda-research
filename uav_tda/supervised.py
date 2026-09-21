"""Phase 5 — supervised pipeline, ported verbatim from pipeline.py SECTION 9.

For each (feature_set, model, seed) cell:
  1. Build the hyperparameter grid for the model.
  2. For each grid combo: fit on training set, score on validation set.
  3. Take the combo with the best val accuracy; refit on full training.
  4. Evaluate on the test set (accuracy, weighted/macro F1, weighted AUC,
     per-class precision/recall/F1, per-class AUC, confusion matrix).
RandomForest also runs a "curated" variant: the feature subset is the union
of (top-30 by RF feature_importances_, top-30 by mutual_info_classif),
frozen at seed=42 then evaluated across all seeds.
SVM training is capped at SVM_MAX_TRAIN_ROWS via stratified sampling because
SVC with RBF on the full 85k training set is intractable.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.supervised")


def load_feature_split(ws: Workspace, feature_set: str, split: str):
    """Return the feature DataFrame for one feature_set / split pair."""
    import pandas as pd

    if feature_set == "original":
        return pd.read_csv(ws.outputs_dir / f"original_features_{split}.csv")
    if feature_set == "summary_only":
        return pd.read_csv(ws.tda_features_dir / f"summary_{split}.csv")
    if feature_set == "summary_plus_images":
        summary = pd.read_csv(ws.tda_features_dir / f"summary_{split}.csv")
        images = pd.read_csv(ws.tda_features_dir / f"images_{split}.csv")
        return pd.concat([summary, images], axis=1)
    if feature_set == "combined":
        return pd.read_csv(ws.tda_features_dir / f"combined_{split}.csv")
    raise ValueError(f"unknown feature set: {feature_set}")


def load_labels_for(ws: Workspace, split: str):
    """Return the label array for one split."""
    import pandas as pd

    return pd.read_csv(ws.outputs_dir / f"labels_{split}.csv")[config.LABEL_COLUMN].values


def build_model(name: str, seed: int, **params):
    """Construct an unfitted estimator for one of {logreg, rf, svm}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    if name == "logreg":
        return LogisticRegression(
            C=params["C"], max_iter=2000, multi_class="auto", solver="lbfgs",
            random_state=seed, n_jobs=-1,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            random_state=seed, n_jobs=-1,
        )
    if name == "svm":
        return SVC(
            C=params["C"], gamma=params["gamma"], kernel="rbf",
            probability=True, random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def grid_combos(model_name: str) -> list:
    """Return the list of hyperparameter dicts to try for one model."""
    if model_name == "logreg":
        return [{"C": c} for c in config.LR_C_GRID]
    if model_name == "rf":
        return [
            {"n_estimators": n, "max_depth": d}
            for n in config.RF_N_ESTIMATORS_GRID for d in config.RF_MAX_DEPTH_GRID
        ]
    if model_name == "svm":
        return [
            {"C": c, "gamma": g}
            for c in config.SVM_C_GRID for g in config.SVM_GAMMA_GRID
        ]
    raise ValueError(f"unknown model: {model_name}")


def maybe_subsample_for_svm(X_train, y_train, model_name: str, seed: int):
    """Stratified-subsample training data when training SVM; otherwise pass through."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    if (
        model_name != "svm"
        or config.SVM_MAX_TRAIN_ROWS is None
        or len(X_train) <= config.SVM_MAX_TRAIN_ROWS
    ):
        return X_train, y_train
    keep_frac = config.SVM_MAX_TRAIN_ROWS / len(X_train)
    keep_idx, _ = train_test_split(
        np.arange(len(X_train)), train_size=keep_frac,
        stratify=y_train, random_state=seed,
    )
    return X_train.iloc[keep_idx], y_train[keep_idx]


def select_best_params(model_name: str, X_train, y_train, X_val, y_val, seed: int) -> dict:
    """Try every grid combo; return the one with the highest val accuracy."""
    from sklearn.metrics import accuracy_score

    X_fit, y_fit = maybe_subsample_for_svm(X_train, y_train, model_name, seed)
    best_acc = -1.0
    best_params: dict = {}
    for params in grid_combos(model_name):
        model = build_model(model_name, seed, **params)
        model.fit(X_fit, y_fit)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        log.debug("%s seed=%s params=%s val_acc=%.4f", model_name, seed, params, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
    log.info(
        "best %s (seed=%s): params=%s val_acc=%.4f",
        model_name, seed, best_params, best_acc,
    )
    return best_params


def fit_final_model(model_name: str, X_train, y_train, params: dict, seed: int):
    """Fit the model on full training data with the chosen hyperparameters."""
    X_fit, y_fit = maybe_subsample_for_svm(X_train, y_train, model_name, seed)
    model = build_model(model_name, seed, **params)
    model.fit(X_fit, y_fit)
    return model


def evaluate_model(model, X_test, y_test) -> tuple:
    """Compute headline metrics, per-class metrics, and the confusion matrix."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, roc_auc_score,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    # Per-class AUC (one-vs-rest)
    per_class_auc = {}
    y_test_arr = np.asarray(y_test)
    for i, cls in enumerate(classes):
        binary = (y_test_arr == cls).astype(int)
        try:
            per_class_auc[cls] = float(roc_auc_score(binary, y_proba[:, i]))
        except ValueError:
            per_class_auc[cls] = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_auc": float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        ),
    }
    for cls, val in per_class_auc.items():
        metrics[f"auc_{cls}"] = val
    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=classes), index=classes, columns=classes,
    )
    return y_pred, metrics, report, cm


def validate_supervised(y_pred, y_test, metrics: dict) -> None:
    """Raise on length mismatch or AUC out of [0, 1]; warn on zero-support classes.

    Zero-support is downgraded from a hard error: a model that fails to predict
    one of the rarer classes on a single seed is informative but not a reason
    to crash the whole pipeline. The warning still surfaces in logs so audits
    can spot persistent misses across seeds.
    """
    if len(y_pred) != len(y_test):
        raise AssertionError(f"len(y_pred)={len(y_pred)} != len(y_test)={len(y_test)}")
    missing = set(config.EXPECTED_CLASSES) - set(map(str, y_pred))
    if missing:
        log.warning("classes predicted with zero support: %s", sorted(missing))
    for key, val in metrics.items():
        if "auc" in key and val == val and not (0.0 <= val <= 1.0):  # NaN-safe
            raise AssertionError(f"{key}={val} outside [0, 1]")


def select_curated_features(rf_model, X_train, y_train) -> list:
    """Union of top-K RF importances and top-K mutual-info features."""
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif

    importances = np.asarray(rf_model.feature_importances_)
    top_rf = set(
        X_train.columns[np.argsort(importances)[::-1][:config.TOP_K_RF_IMPORTANCE]]
    )
    mi = mutual_info_classif(X_train, y_train, random_state=config.PRIMARY_SEED)
    top_mi = set(X_train.columns[np.argsort(mi)[::-1][:config.TOP_K_MUTUAL_INFO]])
    return sorted(top_rf | top_mi)


def run_one_combo(
    ws: Workspace, feature_set: str, model_name: str, seed: int,
    curated_cols: list | None = None,
) -> tuple:
    """Train + tune + evaluate one (feature_set, model, seed) cell; return artefacts."""
    X_train = load_feature_split(ws, feature_set, "train")
    X_val = load_feature_split(ws, feature_set, "val")
    X_test = load_feature_split(ws, feature_set, "test")
    if curated_cols is not None:
        X_train = X_train[curated_cols]
        X_val = X_val[curated_cols]
        X_test = X_test[curated_cols]
    y_train = load_labels_for(ws, "train")
    y_val = load_labels_for(ws, "val")
    y_test = load_labels_for(ws, "test")
    log.info(
        "fitting %s on %s (n_train=%d, n_features=%d, seed=%s)",
        model_name, feature_set, len(X_train), X_train.shape[1], seed,
    )
    best_params = select_best_params(model_name, X_train, y_train, X_val, y_val, seed)
    model = fit_final_model(model_name, X_train, y_train, best_params, seed)
    y_pred, metrics, report, cm = evaluate_model(model, X_test, y_test)
    validate_supervised(y_pred, y_test, metrics)
    return model, metrics, report, cm, best_params


def save_seed42_artifacts(
    ws: Workspace, feature_set: str, model_name: str, model, report, cm, curated: bool,
) -> None:
    """Pickle the seed=42 model and write per-class + confusion-matrix CSVs."""
    import pickle

    suffix = "_curated" if curated else ""
    stem = f"{feature_set}_{model_name}{suffix}"
    with (ws.models_dir / f"{stem}_seed42.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    report.to_csv(ws.tables_dir / f"per_class_metrics_{stem}.csv")
    cm.to_csv(ws.tables_dir / f"confusion_matrix_{stem}.csv")


def metrics_to_row(
    feature_set: str, model_name: str, seed: int, metrics: dict, params: dict, curated: bool,
) -> dict:
    """Flatten one cell's metrics + chosen params into a single-row dict."""
    row = {
        "feature_set": feature_set, "model": model_name, "seed": seed,
        "curated": curated, **metrics, "best_params": str(params),
    }
    return row


def run_seeds_for_combo(
    ws: Workspace, feature_set: str, model_name: str, curated_cols: list | None = None,
) -> list:
    """Run all three seeds for one (feature_set, model) cell; return list of rows."""
    rows: list = []
    for seed in config.SUPERVISED_SEEDS:
        model, metrics, report, cm, params = run_one_combo(
            ws, feature_set, model_name, seed, curated_cols=curated_cols,
        )
        if seed == config.PRIMARY_SEED:
            save_seed42_artifacts(
                ws, feature_set, model_name, model, report, cm,
                curated=(curated_cols is not None),
            )
        rows.append(
            metrics_to_row(
                feature_set, model_name, seed, metrics, params,
                curated=(curated_cols is not None),
            )
        )
    return rows


def summarize_metrics(rows: list):
    """Return a DataFrame of mean ± std for every numeric metric per (feature_set, model)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in
                   {"feature_set", "model", "seed", "curated", "best_params"}]
    group_cols = ["feature_set", "model", "curated"]
    means = df.groupby(group_cols)[metric_cols].mean().add_suffix("_mean")
    stds = df.groupby(group_cols)[metric_cols].std().add_suffix("_std")
    return means.join(stds).reset_index()


def plot_supervised_comparison(summary_df, out_path) -> None:
    """Grouped bar chart of accuracy by (feature_set, model) with std error bars."""
    import matplotlib.pyplot as plt
    import numpy as np

    base = summary_df[~summary_df["curated"]].copy()
    pivot_mean = base.pivot(index="feature_set", columns="model", values="accuracy_mean")
    pivot_std = base.pivot(index="feature_set", columns="model", values="accuracy_std")
    pivot_mean = pivot_mean.loc[list(config.FEATURE_SETS), list(config.MODEL_NAMES)]
    pivot_std = pivot_std.loc[list(config.FEATURE_SETS), list(config.MODEL_NAMES)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(config.FEATURE_SETS))
    width = 0.8 / len(config.MODEL_NAMES)
    for i, model in enumerate(config.MODEL_NAMES):
        ax.bar(
            x + i * width, pivot_mean[model].values, width,
            yerr=pivot_std[model].values, label=model, capsize=3,
        )
    ax.set_xticks(x + width * (len(config.MODEL_NAMES) - 1) / 2)
    ax.set_xticklabels(list(config.FEATURE_SETS), rotation=15)
    ax.set_ylabel("test accuracy")
    ax.set_title("Supervised accuracy by feature set and model (mean ± std over 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curated_importance(rf_model, feature_names: list, out_path) -> None:
    """Top-30 features of the curated RF, color-coded by feature origin."""
    import matplotlib.pyplot as plt
    import numpy as np

    importances = np.asarray(rf_model.feature_importances_)
    top_idx = np.argsort(importances)[::-1][:30]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]
    colors = []
    for name in top_names:
        if "_img_" in name:
            colors.append("tab:blue")
        elif any(stat in name for stat in config.SUMMARY_STAT_NAMES):
            colors.append("tab:orange")
        else:
            colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(range(len(top_names))[::-1], top_vals, color=colors)
    ax.set_yticks(range(len(top_names))[::-1])
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("RF feature importance")
    ax.set_title("Top-30 curated-RF features (blue=image, orange=summary, gray=original)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def freeze_curated_subset(ws: Workspace, feature_set: str) -> tuple:
    """Run seed-42 RF on all features, derive curated subset, refit, return (cols, model)."""
    X_train = load_feature_split(ws, feature_set, "train")
    y_train = load_labels_for(ws, "train")
    base_model = build_model("rf", config.PRIMARY_SEED, n_estimators=300, max_depth=None)
    base_model.fit(X_train, y_train)
    curated = select_curated_features(base_model, X_train, y_train)
    log.info(
        "curated subset for %s: %d features (top-30 RF ∪ top-30 MI)",
        feature_set, len(curated),
    )
    curated_model = build_model("rf", config.PRIMARY_SEED, n_estimators=300, max_depth=None)
    curated_model.fit(X_train[curated], y_train)
    return curated, curated_model


def log_headline_results(summary_df) -> None:
    """Print mean ± std test accuracy for the combined+RF / summary_plus_images+RF cells."""
    log.info("HEADLINE results (mean ± std test accuracy across 3 seeds):")
    for fs in ("combined", "summary_plus_images"):
        for curated in (False, True):
            sel = summary_df[
                (summary_df["feature_set"] == fs)
                & (summary_df["model"] == "rf")
                & (summary_df["curated"] == curated)
            ]
            if sel.empty:
                continue
            mu = float(sel["accuracy_mean"].iloc[0])
            sd = float(sel["accuracy_std"].iloc[0])
            tag = "rf_curated" if curated else "rf"
            log.info("  %s / %s: %.4f ± %.4f", fs, tag, mu, sd)


def run_supervised(ws: Workspace, debug: bool = False) -> None:
    """Phase 5: tune, train, evaluate classifiers across feature sets and seeds."""
    import pandas as pd

    # Monolith relied on CLI-level ensure_directories(); this phase owns
    # tables_dir/figures_dir/models_dir outputs, so mkdir them here.
    ws.tables_dir.mkdir(parents=True, exist_ok=True)
    ws.figures_dir.mkdir(parents=True, exist_ok=True)
    ws.models_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list = []
    curated_subsets: dict = {}
    curated_models: dict = {}

    for feature_set in config.FEATURE_SETS:
        for model_name in config.MODEL_NAMES:
            log.info("=== feature_set=%s model=%s ===", feature_set, model_name)
            all_rows.extend(run_seeds_for_combo(ws, feature_set, model_name))

        # Curated RF variant per feature set.
        curated_cols, curated_model = freeze_curated_subset(ws, feature_set)
        curated_subsets[feature_set] = curated_cols
        curated_models[feature_set] = curated_model
        log.info("=== feature_set=%s model=rf_curated ===", feature_set)
        all_rows.extend(
            run_seeds_for_combo(ws, feature_set, "rf", curated_cols=curated_cols)
        )

    metrics_df = pd.DataFrame(all_rows)
    summary_df = summarize_metrics(all_rows)
    metrics_df.to_csv(ws.tables_dir / "supervised_metrics.csv", index=False)
    summary_df.to_csv(ws.tables_dir / "supervised_summary.csv", index=False)

    plot_supervised_comparison(summary_df, ws.figures_dir / "supervised_comparison.png")
    # Use the combined feature-set's curated RF for the importance plot.
    plot_curated_importance(
        curated_models["combined"], curated_subsets["combined"],
        ws.figures_dir / "feature_importance_curated.png",
    )

    log_headline_results(summary_df)
    log.info("supervised complete -> tables=%s figures=%s", ws.tables_dir, ws.figures_dir)
