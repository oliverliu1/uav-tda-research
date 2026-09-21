"""Phase 7 — evaluation + reporting, ported verbatim from pipeline.py SECTION 11.

Distills Phases 5 + 6 outputs into paper-ready tables and figures. Runs two
ablations (drop each manifold, drop each feature group) on top of the
combined+RF supervised setup. Renders three figures (methodology flowchart,
persistence-barcode examples per class x {C2, Network}, and the composite
headline-results figure). Logs a final summary block of every number a paper
reviewer would expect.

Note: the monolith's SECTION 11 defines its own module-level ``SEEDS = (42, 7,
2024)`` constant; that is byte-identical to ``config.SUPERVISED_SEEDS`` (Phase
5's seeds), which is reused here instead of duplicating the tuple.
"""

from __future__ import annotations

import logging

from . import config
from .features import load_diagrams_pkl
from .supervised import build_model, load_feature_split, load_labels_for
from .tda import load_labels_for_split
from .workspace import Workspace

log = logging.getLogger("uav_tda.evaluate")


def classify_columns(columns: list) -> dict:
    """Bucket feature columns into original / summary / images / other."""
    original_set = set(config.C2_FEATURES) | set(config.NETWORK_FEATURES) | set(config.PHYSICAL_FEATURES)
    summary_suffixes = tuple(f"_{stat}" for stat in config.SUMMARY_STAT_NAMES)
    out: dict = {"original": [], "summary": [], "images": [], "other": []}
    for col in columns:
        if col in original_set:
            out["original"].append(col)
        elif "_img_" in col:
            out["images"].append(col)
        elif col.endswith(summary_suffixes):
            out["summary"].append(col)
        else:
            out["other"].append(col)
    return out


def columns_for_manifold(columns: list, manifold: str) -> list:
    """Return columns belonging to one manifold (original, summary, or image)."""
    original_lookup = {
        "c2": set(config.C2_FEATURES),
        "network": set(config.NETWORK_FEATURES),
        "physical": set(config.PHYSICAL_FEATURES),
    }
    originals = original_lookup[manifold]
    prefix = f"{manifold}_H"
    return [c for c in columns if c in originals or c.startswith(prefix)]


def best_supervised_per_feature_set(summary_df) -> "pandas.DataFrame":
    """Pick the (model, curated) row with the highest accuracy_mean per feature set."""
    non_curated = summary_df[~summary_df["curated"]]
    rows = []
    for fs in config.FEATURE_SETS:
        sel = non_curated[non_curated["feature_set"] == fs]
        if sel.empty:
            continue
        best = sel.loc[sel["accuracy_mean"].idxmax()].to_dict()
        rows.append(best)
    return __import__("pandas").DataFrame(rows)


def build_final_supervised(summary_df) -> "pandas.DataFrame":
    """Final paper table: best model per feature set with mean +/- std for headline metrics."""
    best = best_supervised_per_feature_set(summary_df)
    keep_cols = [
        "feature_set", "model",
        "accuracy_mean", "accuracy_std",
        "weighted_f1_mean", "weighted_f1_std",
        "macro_f1_mean", "macro_f1_std",
        "weighted_auc_mean", "weighted_auc_std",
    ]
    return best[[c for c in keep_cols if c in best.columns]].reset_index(drop=True)


def build_final_unsupervised(per_class_auc, overall) -> "pandas.DataFrame":
    """Per-attack AUC by manifold + pattern-classifier metrics on test."""
    import pandas as pd

    pivot = per_class_auc.pivot(index="attack_class", columns="manifold", values="auc")
    pivot = pivot.reset_index()
    pivot.columns.name = None
    summary_fields = {
        "binary_normal_vs_attack_auc": float(overall["binary_normal_vs_attack_auc"].iloc[0]),
        "multiclass_inference_accuracy": float(overall["multiclass_inference_accuracy"].iloc[0]),
    }
    if "unmapped_pattern_fraction_test" in overall.columns:
        summary_fields["unmapped_pattern_fraction_test"] = float(
            overall["unmapped_pattern_fraction_test"].iloc[0]
        )
    overall_row = pd.DataFrame(
        [{
            "attack_class": "<overall>",
            **{m: float("nan") for m in config.MANIFOLDS},
            **summary_fields,
        }]
    )
    return pd.concat([pivot, overall_row], ignore_index=True)


def fit_rf_with_seed(X_train, y_train, seed: int):
    """Fit an RF with the spec's typical-best params; used by ablations."""
    return build_model("rf", seed, n_estimators=300, max_depth=None).fit(X_train, y_train)


def evaluate_rf_on_test(model, X_test, y_test) -> dict:
    """Return accuracy + weighted F1 + weighted AUC for one fitted RF."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "weighted_auc": float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        ),
    }


def run_ablation_one_setting(ws: Workspace, label: str, drop_cols: list) -> list:
    """Train + evaluate RF on combined-minus-drop_cols across all seeds."""
    rows = []
    for seed in config.SUPERVISED_SEEDS:
        X_train = load_feature_split(ws, "combined", "train").drop(columns=drop_cols, errors="ignore")
        X_test = load_feature_split(ws, "combined", "test").drop(columns=drop_cols, errors="ignore")
        y_train = load_labels_for(ws, "train")
        y_test = load_labels_for(ws, "test")
        model = fit_rf_with_seed(X_train, y_train, seed)
        metrics = evaluate_rf_on_test(model, X_test, y_test)
        rows.append({"ablation": label, "seed": seed, "n_features": X_train.shape[1], **metrics})
        log.info("ablation %s seed=%d acc=%.4f", label, seed, metrics["accuracy"])
    return rows


def run_ablation_with_baseline(ws: Workspace, ablations: dict) -> "pandas.DataFrame":
    """Run a no-ablation reference (full `combined`) + each ablation setting.

    The reference cell uses the same fixed RF hyperparameters (n_estimators=300,
    max_depth=None) as the ablation cells. This makes deltas directly comparable
    without re-tuning per ablation -- and disambiguates "feature group matters"
    from "we just didn't tune for this subset".
    """
    import pandas as pd

    rows = run_ablation_one_setting(ws, "no_ablation", [])
    for label, drop_cols in ablations.items():
        rows.extend(run_ablation_one_setting(ws, label, drop_cols))
    return pd.DataFrame(rows)


def run_manifold_ablation(ws: Workspace) -> "pandas.DataFrame":
    """Drop each manifold's contribution from `combined`; refit RF; report deltas."""
    combined_cols = list(load_feature_split(ws, "combined", "train").columns)
    ablations = {
        f"drop_{manifold}": columns_for_manifold(combined_cols, manifold)
        for manifold in config.MANIFOLDS
    }
    return run_ablation_with_baseline(ws, ablations)


def run_feature_group_ablation(ws: Workspace) -> "pandas.DataFrame":
    """Drop each feature group (original / summary / images) from `combined`; refit RF."""
    combined_cols = list(load_feature_split(ws, "combined", "train").columns)
    classified = classify_columns(combined_cols)
    ablations = {f"drop_{group}": classified[group] for group in ("original", "summary", "images")}
    return run_ablation_with_baseline(ws, ablations)


def summarize_ablation(df) -> "pandas.DataFrame":
    """Collapse per-seed ablation rows to mean +/- std per `ablation` label."""
    metric_cols = [c for c in df.columns if c not in {"ablation", "seed", "n_features"}]
    means = df.groupby("ablation")[metric_cols].mean().add_suffix("_mean")
    stds = df.groupby("ablation")[metric_cols].std().add_suffix("_std")
    n_feats = df.groupby("ablation")["n_features"].first()
    return means.join(stds).join(n_feats).reset_index()


def plot_methodology_flowchart(out_path) -> None:
    """Render a phase-by-phase flowchart of the pipeline, no embedded numbers."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    phases = [
        ("Phase 2", "Data prep:\nencode, split,\nscale per manifold"),
        ("Phase 3", "Persistence:\nRips on\n{query} ∪ ref_500"),
        ("Phase 4", "Features:\nsummary stats +\npersistence images"),
        ("Phase 5", "Supervised:\nLR / RF / SVM\n× feature sets"),
        ("Phase 6", "Unsupervised:\nWasserstein vs\nbaseline barcode"),
        ("Phase 7", "Evaluation:\nablations +\nheadline figures"),
    ]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    box_w, box_h = 1.6, 1.0
    for i, (title, body) in enumerate(phases):
        x = i * 2.2
        rect = patches.FancyBboxPatch(
            (x, 0), box_w, box_h, boxstyle="round,pad=0.05",
            facecolor="#e8f0fe", edgecolor="#1a73e8", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, 0.8, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + box_w / 2, 0.35, body, ha="center", va="center", fontsize=8)
        if i + 1 < len(phases):
            ax.annotate("", xy=(x + 2.2, 0.5), xytext=(x + box_w, 0.5),
                        arrowprops={"arrowstyle": "->", "color": "#1a73e8"})
    ax.set_xlim(-0.2, len(phases) * 2.2)
    ax.set_ylim(-0.1, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_example_flow_per_class(labels) -> dict:
    """Return {class_name: first test-row index with that label}."""
    out: dict = {}
    labels_arr = labels.values if hasattr(labels, "values") else labels
    for cls in config.EXPECTED_CLASSES:
        matches = (labels_arr == cls).nonzero()[0]
        if len(matches) > 0:
            out[cls] = int(matches[0])
    return out


def plot_barcode_examples(ws: Workspace, out_path) -> None:
    """One test-flow barcode per class x {C2, Network}; bars colored by H-dim."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels_test = load_labels_for_split(ws, "test")
    examples = pick_example_flow_per_class(labels_test)
    manifolds_to_show = ["c2", "network"]
    fig, axes = plt.subplots(
        len(manifolds_to_show), len(config.EXPECTED_CLASSES), figsize=(20, 6), sharex="row",
    )
    cmap = plt.get_cmap("tab10")
    for r, manifold in enumerate(manifolds_to_show):
        diagrams = load_diagrams_pkl(ws, manifold, "test")
        for c, cls in enumerate(config.EXPECTED_CLASSES):
            ax = axes[r, c]
            if cls not in examples:
                ax.set_visible(False)
                continue
            diag = diagrams[examples[cls]]
            for i, (dim, birth, death) in enumerate(diag):
                if np.isinf(death):
                    death = max(birth + 0.5, birth * 1.1)
                ax.hlines(i, birth, death, color=cmap(int(dim)), lw=1.0)
            if r == 0:
                ax.set_title(cls, fontsize=9)
            if c == 0:
                ax.set_ylabel(manifold, fontsize=10)
            ax.set_yticks([])
    fig.suptitle("Persistence barcodes: one test flow per class x manifold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_headline_results(summary_df, metrics_df, out_path) -> None:
    """Composite paper Fig 4: baseline-vs-best-TDA bars per model + per-class AUC heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    non_curated = summary_df[~summary_df["curated"]]
    baseline = non_curated[non_curated["feature_set"] == "original"].set_index("model")
    tda_sets = [fs for fs in config.FEATURE_SETS if fs != "original"]
    best_tda = (
        non_curated[non_curated["feature_set"].isin(tda_sets)]
        .sort_values("accuracy_mean", ascending=False)
        .groupby("model").head(1).set_index("model")
    )

    auc_cols = [c for c in metrics_df.columns if c.startswith("auc_")]
    # Pick the overall best non-curated row for the heatmap.
    best_overall_idx = non_curated["accuracy_mean"].idxmax()
    best_overall = non_curated.loc[best_overall_idx]
    matching = metrics_df[
        (metrics_df["feature_set"] == best_overall["feature_set"])
        & (metrics_df["model"] == best_overall["model"])
        & (~metrics_df["curated"])
    ]
    per_class_means = matching[auc_cols].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(config.MODEL_NAMES))
    axes[0].bar(x - 0.2, [baseline.loc[m, "accuracy_mean"] for m in config.MODEL_NAMES],
                width=0.4, label="original", color="tab:gray")
    axes[0].bar(x + 0.2, [best_tda.loc[m, "accuracy_mean"] for m in config.MODEL_NAMES],
                width=0.4, label="best TDA", color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(config.MODEL_NAMES))
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Baseline vs best TDA feature set per model")
    axes[0].legend()

    heatmap_df = pd.DataFrame({
        "class": [c.replace("auc_", "") for c in auc_cols],
        "AUC": per_class_means.values,
    }).set_index("class")
    sns.heatmap(heatmap_df.T, annot=True, fmt=".3f", cmap="YlGn", ax=axes[1],
                cbar_kws={"label": "per-class AUC"}, vmin=0.5, vmax=1.0)
    axes[1].set_title(f"Per-class AUC — best cell ({best_overall['feature_set']} / {best_overall['model']})")
    fig.suptitle("Headline results")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def log_final_summary(
    final_sup, final_unsup, manifold_ablation_summary, feature_ablation_summary,
) -> None:
    """Emit a single human-readable block with every paper-abstract number."""
    log.info("=" * 70)
    log.info("FINAL SUMMARY — values are mean +/- std over 3 seeds where applicable")
    log.info("=" * 70)
    log.info("Supervised best per feature set:")
    for _, row in final_sup.iterrows():
        log.info(
            "  %-22s %-7s  acc=%.4f ± %.4f   w-AUC=%.4f ± %.4f",
            row["feature_set"], row["model"],
            row["accuracy_mean"], row["accuracy_std"],
            row["weighted_auc_mean"], row["weighted_auc_std"],
        )
    log.info("Unsupervised:")
    for _, row in final_unsup.iterrows():
        if row["attack_class"] == "<overall>":
            unmapped = row.get("unmapped_pattern_fraction_test", float("nan"))
            log.info(
                "  overall   binary AUC=%.4f   inference acc=%.4f   unmapped=%.4f",
                row["binary_normal_vs_attack_auc"],
                row["multiclass_inference_accuracy"],
                unmapped,
            )
        else:
            log.info(
                "  %-18s c2=%.3f net=%.3f phy=%.3f",
                row["attack_class"], row["c2"], row["network"], row["physical"],
            )
    log.info("Ablation — drop manifold (RF on combined minus that manifold):")
    for _, row in manifold_ablation_summary.iterrows():
        log.info(
            "  %-15s acc=%.4f ± %.4f  (n_features=%d)",
            row["ablation"], row["accuracy_mean"], row["accuracy_std"], row["n_features"],
        )
    log.info("Ablation — drop feature group:")
    for _, row in feature_ablation_summary.iterrows():
        log.info(
            "  %-15s acc=%.4f ± %.4f  (n_features=%d)",
            row["ablation"], row["accuracy_mean"], row["accuracy_std"], row["n_features"],
        )
    log.info("=" * 70)


def run_evaluate(ws: Workspace, debug: bool = False) -> None:
    """Phase 7: final tables, ablations, paper figures, and a printable summary block."""
    import pandas as pd

    # Monolith relied on CLI-level ensure_directories(); this phase owns
    # tables_dir/figures_dir outputs, so mkdir them here.
    ws.tables_dir.mkdir(parents=True, exist_ok=True)
    ws.figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(ws.tables_dir / "supervised_summary.csv")
    metrics_df = pd.read_csv(ws.tables_dir / "supervised_metrics.csv")
    per_class_auc = pd.read_csv(ws.tables_dir / "unsupervised_per_class_auc.csv")
    overall_unsup = pd.read_csv(ws.tables_dir / "unsupervised_overall_metrics.csv")

    final_sup = build_final_supervised(summary_df)
    final_unsup = build_final_unsupervised(per_class_auc, overall_unsup)
    final_sup.to_csv(ws.tables_dir / "final_supervised.csv", index=False)
    final_unsup.to_csv(ws.tables_dir / "final_unsupervised.csv", index=False)

    log.info("running manifold ablation")
    manifold_ablation = run_manifold_ablation(ws)
    log.info("running feature-group ablation")
    feature_ablation = run_feature_group_ablation(ws)
    manifold_ablation.to_csv(ws.tables_dir / "ablation_manifolds.csv", index=False)
    feature_ablation.to_csv(ws.tables_dir / "ablation_features.csv", index=False)
    manifold_ablation_summary = summarize_ablation(manifold_ablation)
    feature_ablation_summary = summarize_ablation(feature_ablation)

    plot_methodology_flowchart(ws.figures_dir / "methodology_flowchart.png")
    plot_barcode_examples(ws, ws.figures_dir / "persistence_barcode_examples.png")
    plot_headline_results(summary_df, metrics_df, ws.figures_dir / "headline_results.png")

    log_final_summary(
        final_sup, final_unsup, manifold_ablation_summary, feature_ablation_summary,
    )
    log.info("evaluate complete -> tables=%s figures=%s", ws.tables_dir, ws.figures_dir)
