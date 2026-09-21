"""Phase 6 — unsupervised pipeline, ported verbatim from pipeline.py SECTION 10.

Single baseline barcode per manifold = Rips on just the 500 reference points
(no query). Each flow's anomaly score per manifold = Wasserstein-2 distance
from its Phase 3 diagram to that manifold's baseline barcode, summed over
H-dims. Threshold per manifold = 95th percentile of val Normal distances.
A flow is flagged if ANY manifold's distance exceeds its threshold. The
attack-type inference rule maps each class to its most-common training
pattern of (c2_flag, network_flag, physical_flag) and is applied to test.

NOTE: combined detection here uses max-pool across manifolds plus the
per-manifold flag patterns above (the monolith's own Phase 6 semantics).
This is intentionally distinct from uav_tda.probe's sum-based semantics
(the paper's probe) and must not be merged with it.
"""

from __future__ import annotations

import logging

from . import config
from .features import diagram_dim_slice, load_diagrams_pkl
from .tda import load_labels_for_split, load_split_manifold
from .workspace import Workspace

log = logging.getLogger("uav_tda.unsupervised")


def get_wasserstein_backend() -> tuple:
    """Return (callable, backend_name) preferring GUDHI Hera, falling back to gudhi.wasserstein."""
    try:
        from gudhi.hera import wasserstein_distance as wdist
        return wdist, "hera"
    except ImportError:
        from gudhi.wasserstein import wasserstein_distance as wdist
        return wdist, "gudhi.wasserstein"


def compute_baseline_barcodes(ws: Workspace, max_edge_lengths: dict, reference_indices) -> dict:
    """Run Rips on each manifold's 500 reference points; return per-dim baseline diagrams."""
    import gudhi
    import numpy as np

    baselines: dict = {}
    for manifold in config.MANIFOLDS:
        train = load_split_manifold(ws, manifold, "train")
        ref = train[reference_indices]
        max_edge = max_edge_lengths[manifold]
        max_simplex_dim = config.MAX_HOM_DIM[manifold] + 1
        sparse = config.SPARSE_RIPS_EPSILON.get(manifold)
        kwargs: dict = {"points": ref, "max_edge_length": max_edge}
        if sparse is not None:
            kwargs["sparse"] = sparse
        rips = gudhi.RipsComplex(**kwargs)
        simplex_tree = rips.create_simplex_tree(max_dimension=max_simplex_dim)
        raw = simplex_tree.persistence()
        if raw:
            diag = np.array(
                [[float(d), float(b), float(de)] for d, (b, de) in raw], dtype=float,
            )
        else:
            diag = np.empty((0, 3), dtype=float)
        per_dim = {
            k: diagram_dim_slice(diag, k, max_edge)
            for k in range(config.MAX_HOM_DIM[manifold] + 1)
        }
        baselines[manifold] = per_dim
        log.info(
            "baseline barcode %s: per_dim shapes=%s",
            manifold, {k: v.shape[0] for k, v in per_dim.items()},
        )
    return baselines


def _wasserstein_for_flow(diagram, baseline_per_dim: dict, max_edge: float, max_hom_dim: int) -> float:
    """Per-flow Wasserstein-2 distance to baseline (summed across H-dims).

    Runs inside a joblib worker, so the backend is imported here (try Hera
    first, fall back to gudhi.wasserstein). This mirrors get_wasserstein_backend
    so the worker survives environments without Hera.
    """
    try:
        from gudhi.hera import wasserstein_distance as wdist
    except ImportError:
        from gudhi.wasserstein import wasserstein_distance as wdist

    total = 0.0
    for k in range(max_hom_dim + 1):
        flow_bd = diagram_dim_slice(diagram, k, max_edge)
        total += float(wdist(flow_bd, baseline_per_dim[k], order=2.0))
    return total


def distances_for_split(
    ws: Workspace, manifold: str, split: str, baseline_per_dim: dict, max_edge: float,
    max_hom_dim: int, n_jobs: int,
):
    """Compute per-flow Wasserstein distances for one (manifold, split) in parallel."""
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm import tqdm

    diagrams = load_diagrams_pkl(ws, manifold, split)
    log.info("computing W: manifold=%s split=%s flows=%d", manifold, split, len(diagrams))
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    gen = parallel(
        delayed(_wasserstein_for_flow)(d, baseline_per_dim, max_edge, max_hom_dim)
        for d in diagrams
    )
    return np.array(list(tqdm(gen, total=len(diagrams), desc=f"W/{manifold}/{split}")))


def validate_distances(distances: dict) -> None:
    """Raise if any computed Wasserstein distance is negative."""
    import numpy as np

    for key, arr in distances.items():
        if (np.asarray(arr) < 0).any():
            raise AssertionError(f"{key}: negative Wasserstein distance")


def compute_thresholds(distances_val: dict, val_labels) -> dict:
    """Per-manifold threshold = 95th percentile of val Normal-Traffic distances."""
    import numpy as np

    mask = (val_labels == "Normal Traffic").values
    return {
        manifold: float(np.percentile(distances_val[manifold][mask], config.THRESHOLD_PERCENTILE))
        for manifold in config.MANIFOLDS
    }


def flag_distances(distances: dict, thresholds: dict) -> dict:
    """Return {manifold: bool array} where True means distance > threshold."""
    import numpy as np

    return {
        manifold: (np.asarray(distances[manifold]) > thresholds[manifold])
        for manifold in config.MANIFOLDS
    }


def derive_inference_rule(train_flags: dict, train_labels) -> dict:
    """Return {class_name: (c2, network, physical) most common pattern in train}."""
    from collections import Counter

    rule: dict = {}
    for cls in config.EXPECTED_CLASSES:
        mask = (train_labels == cls).values
        if not mask.any():
            continue
        patterns = list(
            zip(
                train_flags["c2"][mask].astype(int).tolist(),
                train_flags["network"][mask].astype(int).tolist(),
                train_flags["physical"][mask].astype(int).tolist(),
            )
        )
        rule[cls] = tuple(int(x) for x in Counter(patterns).most_common(1)[0][0])
    return rule


def apply_inference_rule(rule: dict, flags: dict) -> list:
    """Map per-flow (c2, network, physical) flag patterns to predicted class names."""
    # Invert {class: pattern} → {pattern: class}; on collision keep first per insertion order
    # (Python dicts preserve insertion order from 3.7+).
    pattern_to_class: dict = {}
    for cls, pat in rule.items():
        pattern_to_class.setdefault(pat, cls)
    default = "Normal Traffic"
    c2 = flags["c2"].astype(int).tolist()
    net = flags["network"].astype(int).tolist()
    phy = flags["physical"].astype(int).tolist()
    return [pattern_to_class.get((c, n, p), default) for c, n, p in zip(c2, net, phy)]


def fraction_unmapped(rule: dict, flags: dict) -> float:
    """Return the fraction of flows whose flag pattern has no class in the rule.

    Flows in this set get bucketed into the 'Normal Traffic' default by
    apply_inference_rule; the rate is reported alongside accuracy so a
    reviewer can tell how often the rule actually fired vs fell through.
    """
    valid_patterns = set(rule.values())
    c2 = flags["c2"].astype(int).tolist()
    net = flags["network"].astype(int).tolist()
    phy = flags["physical"].astype(int).tolist()
    patterns = list(zip(c2, net, phy))
    if not patterns:
        return 0.0
    n_default = sum(1 for p in patterns if p not in valid_patterns)
    return n_default / len(patterns)


def build_distances_frame(
    distances_by_split: dict, flags_by_split: dict, labels_by_split: dict,
    predicted_by_split: dict,
):
    """Combine per-split distance / flag / label arrays into one long-form DataFrame."""
    import pandas as pd

    frames = []
    for split in ("train", "val", "test"):
        d = distances_by_split[split]
        f = flags_by_split[split]
        frames.append(
            pd.DataFrame({
                "split": split,
                "label": labels_by_split[split].values,
                "c2_distance": d["c2"],
                "network_distance": d["network"],
                "physical_distance": d["physical"],
                "c2_flag": f["c2"].astype(int),
                "network_flag": f["network"].astype(int),
                "physical_flag": f["physical"].astype(int),
                "predicted_class": predicted_by_split[split],
            })
        )
    return pd.concat(frames, ignore_index=True)


def compute_per_class_auc(distances: dict, labels) -> "pandas.DataFrame":
    """Per (manifold, attack class) AUC: distance vs Normal-vs-class binary labels."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    rows = []
    for manifold in config.MANIFOLDS:
        d = np.asarray(distances[manifold])
        for cls in config.EXPECTED_CLASSES:
            if cls == "Normal Traffic":
                continue
            mask = (labels == cls) | (labels == "Normal Traffic")
            y = (labels[mask] == cls).astype(int).values
            try:
                auc = float(roc_auc_score(y, d[mask.values]))
            except ValueError:
                auc = float("nan")
            rows.append({"manifold": manifold, "attack_class": cls, "auc": auc})
    return pd.DataFrame(rows)


def warn_low_auc(per_class_auc) -> None:
    """Log a warning for every (manifold, class) cell whose AUC is below 0.5."""
    suspect = per_class_auc[per_class_auc["auc"] < 0.5]
    if suspect.empty:
        return
    for _, row in suspect.iterrows():
        log.warning(
            "per-class AUC inversion: manifold=%s class=%s auc=%.4f (audit needed)",
            row["manifold"], row["attack_class"], row["auc"],
        )


def compute_overall_metrics(distances: dict, predicted, labels) -> dict:
    """Binary normal-vs-attack AUC + multi-class inference-rule accuracy."""
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score

    is_attack = (labels != "Normal Traffic").astype(int).values
    # Pool: per-flow max distance across manifolds (higher = more anomalous).
    pooled = np.maximum.reduce([np.asarray(distances[m]) for m in config.MANIFOLDS])
    try:
        bin_auc = float(roc_auc_score(is_attack, pooled))
    except ValueError:
        bin_auc = float("nan")
    return {
        "binary_normal_vs_attack_auc": bin_auc,
        "multiclass_inference_accuracy": float(accuracy_score(labels.values, predicted)),
        "n_flows": int(len(labels)),
    }


def plot_distance_distributions(distances_df, out_path) -> None:
    """3x5 grid of histograms: distance distribution per (manifold, label class)."""
    import matplotlib.pyplot as plt

    manifolds = list(config.MANIFOLDS)
    classes = list(config.EXPECTED_CLASSES)
    test_df = distances_df[distances_df["split"] == "test"]
    fig, axes = plt.subplots(
        len(manifolds), len(classes), figsize=(18, 9), sharex="row",
    )
    for i, manifold in enumerate(manifolds):
        col = f"{manifold}_distance"
        for j, cls in enumerate(classes):
            ax = axes[i, j]
            sub = test_df[test_df["label"] == cls][col]
            ax.hist(sub, bins=50, color="steelblue", alpha=0.85)
            if i == 0:
                ax.set_title(cls, fontsize=9)
            if j == 0:
                ax.set_ylabel(manifold, fontsize=10)
    fig.suptitle("Test-set Wasserstein distance distributions by manifold x class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_unsupervised_roc(distances_df, out_path) -> None:
    """Per-manifold ROC panels with one curve per attack class (test split only)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    manifolds = list(config.MANIFOLDS)
    test_df = distances_df[distances_df["split"] == "test"]
    fig, axes = plt.subplots(1, len(manifolds), figsize=(15, 5), sharey=True)
    for i, manifold in enumerate(manifolds):
        ax = axes[i]
        col = f"{manifold}_distance"
        for cls in config.EXPECTED_CLASSES:
            if cls == "Normal Traffic":
                continue
            mask = test_df["label"].isin(["Normal Traffic", cls])
            y = (test_df.loc[mask, "label"] == cls).astype(int).values
            score = test_df.loc[mask, col].values
            fpr, tpr, _ = roc_curve(y, score)
            ax.plot(fpr, tpr, label=cls)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(manifold)
        ax.set_xlabel("FPR")
        if i == 0:
            ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    fig.suptitle("Test-set ROC (Normal vs each attack) per manifold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pattern_heatmap(distances_df, out_path) -> None:
    """Heatmap of observed flag-pattern frequency per true class on the test split."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    test_df = distances_df[distances_df["split"] == "test"].copy()
    test_df["pattern"] = (
        test_df["c2_flag"].astype(str)
        + test_df["network_flag"].astype(str)
        + test_df["physical_flag"].astype(str)
    )
    counts = (
        test_df.groupby(["label", "pattern"]).size().unstack(fill_value=0)
        .reindex(list(config.EXPECTED_CLASSES), axis=0, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("flag pattern (c2 net phy)")
    ax.set_ylabel("true class")
    ax.set_title("Test-set flag-pattern frequency per class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_all_distances(ws: Workspace, baselines: dict, max_edge_lengths: dict, n_jobs: int) -> dict:
    """Run Wasserstein for every (manifold, split); return nested distances dict."""
    distances_by_split: dict = {s: {} for s in ("train", "val", "test")}
    for manifold in config.MANIFOLDS:
        baseline = baselines[manifold]
        max_edge = max_edge_lengths[manifold]
        max_hom_dim = config.MAX_HOM_DIM[manifold]
        for split in ("train", "val", "test"):
            distances_by_split[split][manifold] = distances_for_split(
                ws, manifold, split, baseline, max_edge, max_hom_dim, n_jobs=n_jobs,
            )
    log.info("all Wasserstein distances computed")
    return distances_by_split


def save_unsupervised_tables(
    ws: Workspace, distances_df, per_class_auc, overall_metrics: dict, rule: dict,
) -> None:
    """Persist all four output CSVs to results/tables/."""
    import pandas as pd

    distances_df.to_csv(ws.tables_dir / "unsupervised_distances.csv", index=False)
    per_class_auc.to_csv(ws.tables_dir / "unsupervised_per_class_auc.csv", index=False)
    pd.DataFrame([overall_metrics]).to_csv(
        ws.tables_dir / "unsupervised_overall_metrics.csv", index=False,
    )
    rule_rows = [
        {"class": cls, "c2_flag": p[0], "network_flag": p[1], "physical_flag": p[2]}
        for cls, p in rule.items()
    ]
    pd.DataFrame(rule_rows).to_csv(ws.tables_dir / "inference_rule.csv", index=False)


def fit_rule_and_predict(
    ws: Workspace, distances_by_split: dict, labels_by_split: dict, thresholds: dict,
) -> tuple:
    """Compute flags per split, derive rule on train, apply rule to every split."""
    import json

    flags_by_split = {
        split: flag_distances(distances_by_split[split], thresholds)
        for split in ("train", "val", "test")
    }
    rule = derive_inference_rule(flags_by_split["train"], labels_by_split["train"])
    with (ws.outputs_dir / "inference_rule.json").open("w") as fh:
        json.dump({cls: list(pat) for cls, pat in rule.items()}, fh, indent=2)
    log.info("inference rule: %s", rule)
    predicted_by_split = {
        split: apply_inference_rule(rule, flags_by_split[split])
        for split in ("train", "val", "test")
    }
    return rule, flags_by_split, predicted_by_split


def produce_unsupervised_plots(ws: Workspace, distances_df) -> None:
    """Render the three Phase 6 figures into results/figures/."""
    plot_distance_distributions(
        distances_df, ws.figures_dir / "unsupervised_distance_distributions.png",
    )
    plot_unsupervised_roc(distances_df, ws.figures_dir / "unsupervised_roc_curves.png")
    plot_pattern_heatmap(distances_df, ws.figures_dir / "unsupervised_pattern_heatmap.png")


def run_unsupervised(ws: Workspace, debug: bool = False, n_jobs: int = -1) -> None:
    """Phase 6: Wasserstein-distance anomaly detection and pattern-based attack typing."""
    import json

    import numpy as np

    # Monolith relied on CLI-level ensure_directories(); this phase owns
    # tables_dir/figures_dir outputs, so mkdir them here.
    ws.tables_dir.mkdir(parents=True, exist_ok=True)
    ws.figures_dir.mkdir(parents=True, exist_ok=True)

    _, backend_name = get_wasserstein_backend()
    log.info("Wasserstein backend: %s", backend_name)

    with (ws.outputs_dir / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)
    reference_indices = np.load(ws.outputs_dir / "reference_indices.npy")

    baselines = compute_baseline_barcodes(ws, max_edge_lengths, reference_indices)
    distances_by_split = compute_all_distances(ws, baselines, max_edge_lengths, n_jobs=n_jobs)
    for split, dists in distances_by_split.items():
        validate_distances({f"{split}_{m}": dists[m] for m in config.MANIFOLDS})

    labels_by_split = {s: load_labels_for_split(ws, s) for s in ("train", "val", "test")}
    thresholds = compute_thresholds(distances_by_split["val"], labels_by_split["val"])
    with (ws.outputs_dir / "thresholds.json").open("w") as fh:
        json.dump(thresholds, fh, indent=2)
    log.info("thresholds: %s", thresholds)

    rule, flags_by_split, predicted_by_split = fit_rule_and_predict(
        ws, distances_by_split, labels_by_split, thresholds,
    )
    distances_df = build_distances_frame(
        distances_by_split, flags_by_split, labels_by_split, predicted_by_split,
    )

    test_labels = labels_by_split["test"]
    per_class_auc = compute_per_class_auc(distances_by_split["test"], test_labels)
    warn_low_auc(per_class_auc)
    overall = compute_overall_metrics(
        distances_by_split["test"], predicted_by_split["test"], test_labels,
    )
    overall["unmapped_pattern_fraction_test"] = fraction_unmapped(
        rule, flags_by_split["test"],
    )
    log.info("test overall: %s", overall)
    save_unsupervised_tables(ws, distances_df, per_class_auc, overall, rule)
    produce_unsupervised_plots(ws, distances_df)

    log.info("unsupervised complete -> tables=%s figures=%s", ws.tables_dir, ws.figures_dir)
