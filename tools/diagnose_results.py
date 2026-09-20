"""Diagnostic reporter for the multi-manifold TDA pipeline.

Reads the tables and JSONs produced by ``pipeline.py evaluate`` and emits a
single markdown report at ``paper/DIAGNOSTIC_REPORT.md`` covering everything a
reviewer or advisor would care about: a run summary, headline numbers, a
five-question triage with explicit verdicts, per-class supervised performance
for the headline cell, ablation deltas, an overall verdict, and a sanity-check
appendix.

Reads are strictly read-only from ``results/`` and ``outputs/``. The script
writes exactly one file (the markdown report). If any required input is
missing, prints the missing paths to stderr and exits with code 2 without
writing a partial report.

Usage:
    python tools/diagnose_results.py
    python tools/diagnose_results.py --results-dir results
    python tools/diagnose_results.py --output paper/DIAGNOSTIC_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ==== SECTION 1: CONFIG ====

# Q1: minimum per-(manifold, class) unsupervised AUC.
Q1_STRONG_MIN_AUC = 0.60
Q1_OK_MIN_AUC = 0.50

# Q2: supervised lift of best TDA + RF over original + RF, in percentage points.
Q2_STRONG_DELTA_PP = 1.0
Q2_OK_LOWER_DELTA_PP = -0.5

# Q3: per-manifold ablation accuracy drop, in percentage points.
Q3_STRONG_DELTA_PP = 0.5

# Q4: inference-rule unmapped-pattern fraction on test.
Q4_STRONG_UNMAPPED = 0.10
Q4_OK_UNMAPPED = 0.30

# Per-class F1 below this is flagged in the headline cell.
PER_CLASS_F1_FLAG = 0.85

# Q5: expected dominant manifold per attack class (Zeng et al. UAVIDS-2025).
Q5_EXPECTED_DOMINANT_MANIFOLD = {
    "Sybil Attack": "c2",
    "Blackhole Attack": "physical",
    "Flooding Attack": "network",
    "Wormhole Attack": "physical",
}

# Default paths (resolved relative to the repo root one level up from this file).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "DIAGNOSTIC_REPORT.md"

MANIFOLDS = ("c2", "network", "physical")
SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = (
    "Normal Traffic",
    "Blackhole Attack",
    "Wormhole Attack",
    "Sybil Attack",
    "Flooding Attack",
)


# ==== SECTION 2: I/O ====

def required_table_paths(results_dir: Path) -> list[Path]:
    """Return the paths under results/tables/ that the report always reads."""
    t = results_dir / "tables"
    return [
        t / "supervised_summary.csv",
        t / "supervised_metrics.csv",
        t / "unsupervised_per_class_auc.csv",
        t / "unsupervised_overall_metrics.csv",
        t / "final_supervised.csv",
        t / "final_unsupervised.csv",
        t / "ablation_manifolds.csv",
        t / "ablation_features.csv",
        t / "inference_rule.csv",
        t / "unsupervised_distances.csv",
    ]


def required_outputs_paths(outputs_dir: Path) -> list[Path]:
    """Return required JSON + per-split label paths under outputs/ (read-only)."""
    return [
        outputs_dir / "thresholds.json",
        outputs_dir / "max_edge_lengths.json",
    ] + [outputs_dir / f"labels_{s}.csv" for s in SPLITS]


def cell_specific_paths(
    results_dir: Path, feature_set: str, model: str,
) -> list[Path]:
    """Return CSV paths whose names depend on which cell is the headline winner."""
    stem = f"{feature_set}_{model}"
    t = results_dir / "tables"
    return [
        t / f"per_class_metrics_{stem}.csv",
        t / f"confusion_matrix_{stem}.csv",
    ]


def find_missing(paths: list[Path]) -> list[Path]:
    """Return only those paths that do not currently exist."""
    return [p for p in paths if not p.exists()]


def load_inputs(results_dir: Path, outputs_dir: Path) -> dict[str, Any]:
    """Load every always-required CSV/JSON, plus per-split row counts."""
    import pandas as pd

    t = results_dir / "tables"
    data: dict[str, Any] = {
        "supervised_summary": pd.read_csv(t / "supervised_summary.csv"),
        "supervised_metrics": pd.read_csv(t / "supervised_metrics.csv"),
        "unsupervised_per_class_auc": pd.read_csv(t / "unsupervised_per_class_auc.csv"),
        "unsupervised_overall": pd.read_csv(t / "unsupervised_overall_metrics.csv"),
        "final_supervised": pd.read_csv(t / "final_supervised.csv"),
        "final_unsupervised": pd.read_csv(t / "final_unsupervised.csv"),
        "ablation_manifolds": pd.read_csv(t / "ablation_manifolds.csv"),
        "ablation_features": pd.read_csv(t / "ablation_features.csv"),
        "inference_rule": pd.read_csv(t / "inference_rule.csv"),
        "unsupervised_distances": pd.read_csv(t / "unsupervised_distances.csv"),
    }
    with (outputs_dir / "thresholds.json").open() as fh:
        data["thresholds"] = json.load(fh)
    with (outputs_dir / "max_edge_lengths.json").open() as fh:
        data["max_edge_lengths"] = json.load(fh)
    data["split_sizes"] = {
        s: int(len(pd.read_csv(outputs_dir / f"labels_{s}.csv"))) for s in SPLITS
    }
    return data


def load_cell_inputs(
    results_dir: Path, feature_set: str, model: str,
) -> dict[str, Any]:
    """Load per-class metrics and confusion-matrix CSVs for the headline cell."""
    import pandas as pd

    stem = f"{feature_set}_{model}"
    t = results_dir / "tables"
    return {
        "per_class_metrics": pd.read_csv(
            t / f"per_class_metrics_{stem}.csv", index_col=0,
        ),
        "confusion_matrix": pd.read_csv(
            t / f"confusion_matrix_{stem}.csv", index_col=0,
        ),
    }


# ==== SECTION 3: ANALYSIS ====

def find_headline_cell(final_supervised) -> tuple[str, str]:
    """Return the (feature_set, model) row with the highest accuracy_mean."""
    row = final_supervised.loc[final_supervised["accuracy_mean"].idxmax()]
    return str(row["feature_set"]), str(row["model"])


def q1_unsupervised_auc(per_class_auc) -> dict[str, Any]:
    """Triage Q1: minimum per-(manifold, class) unsupervised AUC vs thresholds."""
    valid = per_class_auc.dropna(subset=["auc"])
    if valid.empty:
        return {"verdict": "WEAK", "min_auc": float("nan"), "weak_pairs": []}
    min_auc = float(valid["auc"].min())
    weak_pairs = [
        (str(r["manifold"]), str(r["attack_class"]), float(r["auc"]))
        for _, r in valid.iterrows() if r["auc"] < Q1_OK_MIN_AUC
    ]
    if min_auc >= Q1_STRONG_MIN_AUC:
        verdict = "STRONG"
    elif min_auc >= Q1_OK_MIN_AUC:
        verdict = "OK"
    else:
        verdict = "WEAK"
    return {"verdict": verdict, "min_auc": min_auc, "weak_pairs": weak_pairs}


def q2_supervised_lift(supervised_summary) -> dict[str, Any]:
    """Triage Q2: best TDA + RF accuracy lift over original + RF."""
    non_cur = supervised_summary[~supervised_summary["curated"]]
    baseline = non_cur[
        (non_cur["feature_set"] == "original") & (non_cur["model"] == "rf")
    ]
    tda = non_cur[
        (non_cur["feature_set"].isin(["combined", "summary_plus_images"]))
        & (non_cur["model"] == "rf")
    ]
    if baseline.empty or tda.empty:
        return {
            "verdict": "WEAK",
            "baseline": float("nan"), "best_tda": float("nan"),
            "best_tda_set": "?", "delta_pp": float("nan"),
        }
    baseline_acc = float(baseline["accuracy_mean"].iloc[0])
    best_row = tda.loc[tda["accuracy_mean"].idxmax()]
    best_tda = float(best_row["accuracy_mean"])
    delta_pp = (best_tda - baseline_acc) * 100
    if delta_pp >= Q2_STRONG_DELTA_PP:
        verdict = "STRONG"
    elif delta_pp >= Q2_OK_LOWER_DELTA_PP:
        verdict = "OK"
    else:
        verdict = "WEAK"
    return {
        "verdict": verdict,
        "baseline": baseline_acc, "best_tda": best_tda,
        "best_tda_set": str(best_row["feature_set"]), "delta_pp": delta_pp,
    }


def q3_manifold_ablation(ablation_manifolds) -> dict[str, Any]:
    """Triage Q3: accuracy drop when each manifold is removed from `combined`."""
    means = ablation_manifolds.groupby("ablation")["accuracy"].mean()
    if "no_ablation" not in means.index:
        return {
            "verdict": "WEAK",
            "baseline": float("nan"),
            "deltas": {m: float("nan") for m in MANIFOLDS},
            "weak_manifolds": list(MANIFOLDS),
            "note": "no_ablation baseline row missing (pre-audit-fix output)",
        }
    baseline = float(means.loc["no_ablation"])
    deltas: dict[str, float] = {}
    for m in MANIFOLDS:
        key = f"drop_{m}"
        deltas[m] = (
            (baseline - float(means.loc[key])) * 100
            if key in means.index else float("nan")
        )
    above_thresh = [m for m, d in deltas.items() if d == d and d >= Q3_STRONG_DELTA_PP]
    n_above = len(above_thresh)
    if n_above == 3:
        verdict = "STRONG"
    elif n_above == 2:
        verdict = "OK"
    else:
        verdict = "WEAK"
    weak = [m for m, d in deltas.items() if not (d == d) or d < Q3_STRONG_DELTA_PP]
    return {
        "verdict": verdict, "baseline": baseline,
        "deltas": deltas, "weak_manifolds": weak, "note": "",
    }


def q4_unmapped_fraction(unsupervised_overall) -> dict[str, Any]:
    """Triage Q4: inference-rule fallback-to-default rate on test."""
    col = "unmapped_pattern_fraction_test"
    if col not in unsupervised_overall.columns:
        return {
            "verdict": "WEAK", "unmapped": float("nan"),
            "note": f"{col} column missing (pre-audit-fix output)",
        }
    unmapped = float(unsupervised_overall[col].iloc[0])
    if unmapped < Q4_STRONG_UNMAPPED:
        verdict = "STRONG"
    elif unmapped <= Q4_OK_UNMAPPED:
        verdict = "OK"
    else:
        verdict = "WEAK"
    return {"verdict": verdict, "unmapped": unmapped, "note": ""}


def q5_attack_signatures(per_class_auc) -> dict[str, Any]:
    """Triage Q5: dominant manifold per attack vs Zeng et al. expected signatures."""
    matches: list[tuple[str, str, str]] = []
    mismatches: list[tuple[str, str, str]] = []
    for attack, expected_m in Q5_EXPECTED_DOMINANT_MANIFOLD.items():
        sub = per_class_auc[per_class_auc["attack_class"] == attack].dropna(subset=["auc"])
        if sub.empty:
            mismatches.append((attack, expected_m, "no data"))
            continue
        actual_m = str(sub.loc[sub["auc"].idxmax(), "manifold"])
        if actual_m == expected_m:
            matches.append((attack, expected_m, actual_m))
        else:
            mismatches.append((attack, expected_m, actual_m))
    n_match = len(matches)
    if n_match == 4:
        verdict = "STRONG"
    elif n_match == 3:
        verdict = "OK"
    else:
        verdict = "WEAK"
    return {
        "verdict": verdict, "matches": matches,
        "mismatches": mismatches, "n_match": n_match,
    }


def final_verdict(triage: dict[str, dict]) -> str:
    """Apply the four-tier verdict rule, most-severe first."""
    verdicts = [triage[k]["verdict"] for k in ("q1", "q2", "q3", "q4", "q5")]
    strong = verdicts.count("STRONG")
    ok = verdicts.count("OK")
    weak = verdicts.count("WEAK")
    critical_weak = any(triage[k]["verdict"] == "WEAK" for k in ("q1", "q2", "q4"))
    if weak >= 2:
        return "MAJOR REWORK"
    if critical_weak:
        return "INVESTIGATE BEFORE WRITING"
    if strong == 5 or (strong == 4 and ok == 1):
        return "READY TO WRITE"
    return "WRITE WITH CAVEATS"


def per_class_rows(
    per_class_metrics, supervised_summary, feature_set: str, model: str,
) -> list[dict[str, Any]]:
    """Build per-class precision/recall/F1/AUC rows for the headline cell."""
    summary_row = supervised_summary[
        (supervised_summary["feature_set"] == feature_set)
        & (supervised_summary["model"] == model)
        & (~supervised_summary["curated"])
    ]
    rows: list[dict[str, Any]] = []
    for cls in EXPECTED_CLASSES:
        if cls not in per_class_metrics.index:
            continue
        r = per_class_metrics.loc[cls]
        auc_col = f"auc_{cls}_mean"
        auc_mean = (
            float(summary_row[auc_col].iloc[0])
            if (not summary_row.empty and auc_col in summary_row.columns)
            else float("nan")
        )
        f1 = float(r["f1-score"])
        rows.append({
            "class": cls,
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": f1,
            "support": int(float(r["support"])),
            "auc_mean": auc_mean,
            "flagged": f1 < PER_CLASS_F1_FLAG,
        })
    return rows


def ablation_deltas(ablation_df) -> list[dict[str, Any]]:
    """Aggregate per-seed ablation rows to mean ± std and deltas vs no_ablation."""
    grouped = ablation_df.groupby("ablation")
    means_acc = grouped["accuracy"].mean()
    stds_acc = grouped["accuracy"].std()
    n_feat = grouped["n_features"].first()
    if "no_ablation" in means_acc.index:
        baseline_mean = float(means_acc.loc["no_ablation"])
        order = ["no_ablation"] + sorted(a for a in means_acc.index if a != "no_ablation")
    else:
        baseline_mean = float("nan")
        order = sorted(means_acc.index)
    rows: list[dict[str, Any]] = []
    for ablation in order:
        m = float(means_acc.loc[ablation])
        s_raw = stds_acc.loc[ablation]
        s = 0.0 if (s_raw != s_raw) else float(s_raw)
        delta_pp = (
            0.0 if ablation == "no_ablation" else (baseline_mean - m) * 100
        )
        rows.append({
            "ablation": ablation,
            "n_features": int(n_feat.loc[ablation]),
            "acc_mean": m, "acc_std": s,
            "delta_pp_vs_baseline": delta_pp,
        })
    return rows


def sanity_check_summary(
    unsupervised_distances, supervised_metrics, confusion_matrix,
) -> dict[str, Any]:
    """Compute the four sanity counts for the appendix."""
    test_df = unsupervised_distances[unsupervised_distances["split"] == "test"]
    n_neg = int(
        ((test_df["c2_distance"] < 0)
         | (test_df["network_distance"] < 0)
         | (test_df["physical_distance"] < 0)).sum()
    )
    out_of_range = []
    for col in [c for c in supervised_metrics.columns if c.startswith("auc_")]:
        bad = supervised_metrics[col].dropna()
        bad = bad[(bad < 0.0) | (bad > 1.0)]
        if not bad.empty:
            out_of_range.append((col, bad.tolist()))
    zero_pred_classes = [
        str(c) for c in confusion_matrix.columns
        if int(confusion_matrix[c].sum()) == 0
    ]
    patterns = set(
        zip(test_df["c2_flag"], test_df["network_flag"], test_df["physical_flag"])
    )
    return {
        "n_negative_distances": n_neg,
        "out_of_range_aucs": out_of_range,
        "zero_pred_classes": zero_pred_classes,
        "n_unique_patterns": len(patterns),
    }


# ==== SECTION 4: RENDERING ====

def render_section_1(data: dict[str, Any], now: datetime) -> str:
    """Render section 1: run summary, split sizes, max-edge, thresholds."""
    lines = [
        "## 1. Run summary",
        "",
        f"- **Report generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Train rows:** {data['split_sizes']['train']:,}",
        f"- **Val rows:** {data['split_sizes']['val']:,}",
        f"- **Test rows:** {data['split_sizes']['test']:,}",
        "",
        "**Per-manifold MAX_EDGE_LENGTH (Rips filtration cutoff):**",
        "",
    ]
    for m in MANIFOLDS:
        v = data["max_edge_lengths"].get(m, float("nan"))
        lines.append(f"- `{m}`: {v:.6f}")
    lines.extend([
        "",
        "**Per-manifold thresholds (95th-percentile val Normal Wasserstein):**",
        "",
    ])
    for m in MANIFOLDS:
        v = data["thresholds"].get(m, float("nan"))
        lines.append(f"- `{m}`: {v:.6f}")
    return "\n".join(lines)


def render_section_2(final_supervised) -> str:
    """Render section 2: headline numbers; the highest-accuracy row is bolded."""
    rows = final_supervised.reset_index(drop=True)
    best_idx = int(rows["accuracy_mean"].idxmax())
    lines = [
        "## 2. Headline numbers",
        "",
        "Best non-curated model per feature set (mean ± std over 3 seeds).",
        "The highest-accuracy row is bolded.",
        "",
        "| Feature set | Model | Accuracy | Weighted F1 | Weighted AUC |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for i, r in rows.iterrows():
        acc = f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
        f1 = f"{r['weighted_f1_mean']:.4f} ± {r['weighted_f1_std']:.4f}"
        auc = f"{r['weighted_auc_mean']:.4f} ± {r['weighted_auc_std']:.4f}"
        fs, mod = r["feature_set"], r["model"]
        if i == best_idx:
            lines.append(f"| **{fs}** | **{mod}** | **{acc}** | **{f1}** | **{auc}** |")
        else:
            lines.append(f"| {fs} | {mod} | {acc} | {f1} | {auc} |")
    return "\n".join(lines)


def render_q(name: str, question: str, verdict: str, body: list[str]) -> str:
    """Render one Q block of the triage section."""
    lines = [f"### {name}. {question}", "", f"**Verdict: {verdict}**", ""]
    lines.extend(body)
    return "\n".join(lines)


def _q1_body(q: dict) -> list[str]:
    """Body lines for Q1 (unsupervised AUC)."""
    body = [f"- Minimum per-(manifold, class) unsupervised AUC: **{q['min_auc']:.4f}**"]
    if q["weak_pairs"]:
        body.append("- Pairs below 0.50:")
        for m, c, v in q["weak_pairs"]:
            body.append(f"    - {m} / {c}: {v:.4f}")
    return body


def _q2_body(q: dict) -> list[str]:
    """Body lines for Q2 (supervised lift)."""
    return [
        f"- Baseline (original + RF): **{q['baseline']:.4f}**",
        f"- Best TDA (`{q['best_tda_set']}` + RF): **{q['best_tda']:.4f}**",
        f"- Δ accuracy: **{q['delta_pp']:+.2f} pp**",
    ]


def _q3_body(q: dict) -> list[str]:
    """Body lines for Q3 (manifold ablation)."""
    body = [f"- no_ablation baseline accuracy: **{q['baseline']:.4f}**"]
    if q.get("note"):
        body.append(f"- *Note: {q['note']}*")
    for m in MANIFOLDS:
        d = q["deltas"].get(m, float("nan"))
        body.append(f"- drop_{m} Δ vs baseline: **{d:+.2f} pp**")
    if q["weak_manifolds"]:
        body.append(
            f"- Manifolds below {Q3_STRONG_DELTA_PP:+.2f} pp threshold: {q['weak_manifolds']}"
        )
    return body


def _q4_body(q: dict) -> list[str]:
    """Body lines for Q4 (unmapped pattern fraction)."""
    body = [f"- Inference-rule unmapped-pattern fraction on test: **{q['unmapped']:.4f}**"]
    if q.get("note"):
        body.append(f"- *Note: {q['note']}*")
    return body


def _q5_body(q: dict) -> list[str]:
    """Body lines for Q5 (attack signature match)."""
    body = [f"- Attacks whose dominant manifold matches expected: **{q['n_match']}/4**"]
    if q["matches"]:
        body.append("- Matches:")
        for attack, expected, actual in q["matches"]:
            body.append(f"    - {attack}: expected `{expected}`, got `{actual}`")
    if q["mismatches"]:
        body.append("- Mismatches:")
        for attack, expected, actual in q["mismatches"]:
            body.append(f"    - {attack}: expected `{expected}`, got `{actual}`")
    return body


def render_section_3(triage: dict[str, dict]) -> str:
    """Render section 3: five-question triage."""
    blocks = ["## 3. Five-question triage", ""]
    blocks.append(render_q(
        "Q1", "All per-(manifold, class) unsupervised AUCs above 0.5?",
        triage["q1"]["verdict"], _q1_body(triage["q1"]),
    ))
    blocks.append("")
    blocks.append(render_q(
        "Q2", "Best TDA-augmented model beats original + RF?",
        triage["q2"]["verdict"], _q2_body(triage["q2"]),
    ))
    blocks.append("")
    blocks.append(render_q(
        "Q3", "Does each manifold contribute to combined-RF accuracy?",
        triage["q3"]["verdict"], _q3_body(triage["q3"]),
    ))
    blocks.append("")
    blocks.append(render_q(
        "Q4", "Inference-rule unmapped-pattern fraction low?",
        triage["q4"]["verdict"], _q4_body(triage["q4"]),
    ))
    blocks.append("")
    blocks.append(render_q(
        "Q5", "Per-attack AUC patterns match UAVIDS-2025 signatures?",
        triage["q5"]["verdict"], _q5_body(triage["q5"]),
    ))
    return "\n".join(blocks)


def render_section_4(
    per_class: list[dict], feature_set: str, model: str,
) -> str:
    """Render section 4: per-class metrics for the headline winning cell."""
    lines = [
        "## 4. Per-class supervised performance",
        "",
        f"Headline cell: `{feature_set}` / `{model}` (non-curated). "
        "AUC is the mean across 3 seeds; precision/recall/F1 are from seed 42.",
        "",
        "| Class | Precision | Recall | F1 | AUC (mean) | Support |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    flagged: list[str] = []
    for r in per_class:
        marker = "  **(LOW)**" if r["flagged"] else ""
        if r["flagged"]:
            flagged.append(r["class"])
        lines.append(
            f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f}{marker} | {r['auc_mean']:.3f} | {r['support']:,} |"
        )
    if flagged:
        lines.extend(["", f"Classes with F1 below {PER_CLASS_F1_FLAG}: {flagged}"])
    return "\n".join(lines)


def render_ablation_table(rows: list[dict], title: str) -> str:
    """Render one ablation table with mean ± std and Δ-pp-vs-baseline columns."""
    lines = [
        f"**{title}**",
        "",
        "| Ablation | n_features | Accuracy | Δ vs no_ablation (pp) |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        acc = f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}"
        if r["ablation"] == "no_ablation":
            delta_str = "—"
        else:
            delta_str = f"{r['delta_pp_vs_baseline']:+.2f}"
        lines.append(f"| {r['ablation']} | {r['n_features']} | {acc} | {delta_str} |")
    return "\n".join(lines)


def render_section_5(
    manifold_rows: list[dict], feature_rows: list[dict],
) -> str:
    """Render section 5: both ablation tables with delta columns."""
    lines = ["## 5. Ablation deltas", ""]
    lines.append(render_ablation_table(manifold_rows, "Manifold ablation"))
    lines.append("")
    lines.append(render_ablation_table(feature_rows, "Feature-group ablation"))
    return "\n".join(lines)


def render_section_6(triage: dict[str, dict], verdict: str) -> str:
    """Render section 6: overall verdict."""
    summary = " · ".join(
        f"Q{i + 1}: {triage[f'q{i + 1}']['verdict']}" for i in range(5)
    )
    return "\n".join([
        "## 6. Verdict",
        "",
        f"Triage results — {summary}.",
        "",
        f"**Overall verdict: {verdict}**",
    ])


def render_section_7(sanity: dict[str, Any]) -> str:
    """Render section 7: sanity-check appendix."""
    lines = [
        "## 7. Sanity check appendix",
        "",
        f"- Negative Wasserstein distances in test (should be 0): "
        f"**{sanity['n_negative_distances']}**",
    ]
    if sanity["out_of_range_aucs"]:
        lines.append("- Per-class test AUCs outside [0, 1]:")
        for col, vals in sanity["out_of_range_aucs"]:
            lines.append(f"    - `{col}`: {vals}")
    else:
        lines.append("- Per-class test AUCs outside [0, 1]: **none**")
    if sanity["zero_pred_classes"]:
        lines.append(
            f"- Classes with zero predictions in seed-42 confusion matrix: "
            f"**{sanity['zero_pred_classes']}**"
        )
    else:
        lines.append(
            "- Classes with zero predictions in seed-42 confusion matrix: **none**"
        )
    lines.append(
        f"- Unique flag patterns observed in test split: "
        f"**{sanity['n_unique_patterns']}** of 8 possible"
    )
    return "\n".join(lines)


def render_report(
    data: dict, feature_set: str, model: str, triage: dict, verdict: str,
    per_class: list[dict], manifold_rows: list[dict],
    feature_rows: list[dict], sanity: dict, now: datetime,
) -> str:
    """Compose the full markdown report."""
    sections = [
        "# UAV TDA Pipeline — Diagnostic Report",
        "",
        "_Auto-generated by `tools/diagnose_results.py`._",
        "",
        render_section_1(data, now),
        "",
        render_section_2(data["final_supervised"]),
        "",
        render_section_3(triage),
        "",
        render_section_4(per_class, feature_set, model),
        "",
        render_section_5(manifold_rows, feature_rows),
        "",
        render_section_6(triage, verdict),
        "",
        render_section_7(sanity),
        "",
    ]
    return "\n".join(sections) + "\n"


# ==== SECTION 5: CLI ====

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(
        prog="diagnose_results.py",
        description=(
            "Build paper/DIAGNOSTIC_REPORT.md from pipeline.py evaluate outputs. "
            "Reads results/tables/ and outputs/ read-only; writes one markdown file."
        ),
    )
    p.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Path to results/ directory (read-only).",
    )
    p.add_argument(
        "--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR,
        help="Path to outputs/ directory (read-only).",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Markdown report path (will create parent dirs).",
    )
    return p


def print_missing(label: str, missing: list[Path]) -> None:
    """Print a missing-input block to stderr in a standard format."""
    print(f"{label}", file=sys.stderr)
    for p in missing:
        print(f"  {p}", file=sys.stderr)


def check_and_load(args: argparse.Namespace) -> tuple[dict, str, str, dict] | None:
    """Run both rounds of input-existence checks; load everything or return None."""
    needed = (
        required_table_paths(args.results_dir)
        + required_outputs_paths(args.outputs_dir)
    )
    missing = find_missing(needed)
    if missing:
        print_missing("Missing required input files:", missing)
        return None
    data = load_inputs(args.results_dir, args.outputs_dir)
    feature_set, model = find_headline_cell(data["final_supervised"])
    cell_missing = find_missing(
        cell_specific_paths(args.results_dir, feature_set, model)
    )
    if cell_missing:
        print_missing(
            f"Missing input files for headline cell ({feature_set}/{model}):",
            cell_missing,
        )
        return None
    cell_data = load_cell_inputs(args.results_dir, feature_set, model)
    return data, feature_set, model, cell_data


def run_diagnostics(
    data: dict, cell_data: dict, feature_set: str, model: str,
) -> dict:
    """Run every analysis pass and bundle the results for rendering."""
    triage = {
        "q1": q1_unsupervised_auc(data["unsupervised_per_class_auc"]),
        "q2": q2_supervised_lift(data["supervised_summary"]),
        "q3": q3_manifold_ablation(data["ablation_manifolds"]),
        "q4": q4_unmapped_fraction(data["unsupervised_overall"]),
        "q5": q5_attack_signatures(data["unsupervised_per_class_auc"]),
    }
    return {
        "triage": triage,
        "verdict": final_verdict(triage),
        "per_class": per_class_rows(
            cell_data["per_class_metrics"], data["supervised_summary"],
            feature_set, model,
        ),
        "manifold_rows": ablation_deltas(data["ablation_manifolds"]),
        "feature_rows": ablation_deltas(data["ablation_features"]),
        "sanity": sanity_check_summary(
            data["unsupervised_distances"], data["supervised_metrics"],
            cell_data["confusion_matrix"],
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse args, check inputs, run analyses, render markdown, write atomically."""
    args = build_parser().parse_args(argv)
    loaded = check_and_load(args)
    if loaded is None:
        return 2
    data, feature_set, model, cell_data = loaded
    bundle = run_diagnostics(data, cell_data, feature_set, model)
    markdown = render_report(
        data, feature_set, model, bundle["triage"], bundle["verdict"],
        bundle["per_class"], bundle["manifold_rows"], bundle["feature_rows"],
        bundle["sanity"], datetime.now(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
