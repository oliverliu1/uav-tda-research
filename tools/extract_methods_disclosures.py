"""Render paper-ready methods, limitations, and future-work paragraphs.

Emits a single markdown file (``paper/METHODS_DISCLOSURES.md``) consolidating
the eight deviations (D1-D8) and eight limitations (M1-M8) from the project
summary into three flowing-prose paragraph blocks suitable for direct paste
into the manuscript's Methods, Limitations, and Future Work sections.

The script reads no pipeline output. It is a one-shot writing aid: the
deviation and limitation catalog is hardcoded below, and three curated
paragraphs render that catalog as prose. A traceability source-index table
appended below the paragraphs maps every ID to where it is covered (and
flags the two limitations resolved by the audit fix).

Voice: ``we``, past tense, no contractions, no emoji.

Usage:
    python tools/extract_methods_disclosures.py
    python tools/extract_methods_disclosures.py --output paper/METHODS.md
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ==== SECTION 1: PATHS ====

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "METHODS_DISCLOSURES.md"


# ==== SECTION 2: SOURCE INDEX ====
# Structured catalogue of every deviation and limitation. The paragraph
# blocks below are curated to cover these entries in flowing prose; the
# dicts exist for traceability and future revision.

DEVIATIONS: dict[str, dict[str, str]] = {
    "D1": {
        "title": "Debug-mode stratified head sample",
        "note": (
            "UAVIDS-2025 is row-sorted by class, so a literal head(5000) "
            "yields a single class. Debug mode reads the full CSV and keeps "
            "the first N/n_classes rows per class."
        ),
    },
    "D2": {
        "title": "MAX_EDGE_PERCENTILE lowered from 95 to 25",
        "note": (
            "At the 95th percentile the simplex tree exceeded 19 million "
            "simplices per query and exhausted memory at simplex-dimension "
            "three. The 25th percentile keeps the tree small enough that "
            "sparse Rips reaches sdim=3 in 1-2 seconds per call on C2."
        ),
    },
    "D3": {
        "title": "Sparse Rips approximation (epsilon = 0.5) for C2 and Network",
        "note": (
            "Physical at sdim=2 stays exact. Persistence values differ from "
            "exact Rips by at most epsilon in interleaving distance."
        ),
    },
    "D4": {
        "title": "Simplex-tree max_dimension = MAX_HOM_DIM + 1",
        "note": (
            "Reliable H_k via Rips needs (k+1)-simplices to fill k-cycles, "
            "so sdim=3 for C2/Network and sdim=2 for Physical so the "
            "highest-order homology features have real finite death times."
        ),
    },
    "D5": {
        "title": "SVM trained on a stratified 5,000-row subsample",
        "note": (
            "SVC(RBF) scales O(N^2)-O(N^3); at N=85,519 with 72 SVM fits "
            "per run it is intractable. LR and RF still see the full "
            "training set."
        ),
    },
    "D6": {
        "title": "Grid search picks best by val accuracy, no inner CV",
        "note": (
            "Single training fit per hyperparameter combination, selected "
            "by validation accuracy. Avoids the 3x compute cost of 3-fold "
            "CV inside grid search."
        ),
    },
    "D7": {
        "title": "Ablations use fixed RF; no_ablation row included",
        "note": (
            "Manifold and feature-group ablations train RF "
            "(n_estimators=300, max_depth=None) on the ablated subset. "
            "A no_ablation row computed with the same RF lives in each "
            "ablation table so deltas are apples-to-apples within the table."
        ),
    },
    "D8": {
        "title": "Curated RF subset uses fixed hyperparameters",
        "note": (
            "The base RF that derives feature importances and the final "
            "curated RF both use n_estimators=300, max_depth=None rather "
            "than re-running the grid search on the curated subset."
        ),
    },
}

LIMITATIONS: dict[str, dict[str, str]] = {
    "M1": {
        "title": "Reference cloud sampled by KMedoids on C2 only",
        "covered_in": "Limitations",
        "note": (
            "The 500 reference medoids are diverse in C2 space, then reused "
            "positionally across Network and Physical. The latter manifolds "
            "have no guarantee of reference diversity in their own metrics."
        ),
    },
    "M2": {
        "title": "Essential H_0 truncated to MAX_EDGE_LENGTH",
        "covered_in": "Limitations",
        "note": (
            "The one essential H_0 feature per Rips diagram (the global "
            "connected component, death = infinity) is replaced by "
            "MAX_EDGE_LENGTH so summary statistics are finite. Mean and "
            "max persistence are biased upward by a manifold-specific "
            "constant."
        ),
    },
    "M3": {
        "title": "Wasserstein aggregation by sum across homology dimensions",
        "covered_in": "Limitations",
        "note": (
            "Per-flow Wasserstein-2 score is sum_k W_2(D_k(flow), "
            "D_k(baseline)). Alternative aggregations (max-pooling, "
            "multi-dimensional Wasserstein) could change rankings."
        ),
    },
    "M4": {
        "title": "Inference-rule tie-break by class-name order",
        "covered_in": "Limitations",
        "note": (
            "If two attack classes share the same modal flag pattern, the "
            "alphabetically earlier class claims it. May systematically "
            "disadvantage rarer attack types."
        ),
    },
    "M5": {
        "title": "Inference-rule unmapped-pattern fraction now reported",
        "covered_in": "Audit fix M5 (not a paper limitation)",
        "note": (
            "Resolved during the post-refactor audit: the fraction of test "
            "flows that fall through to the Normal Traffic default is now "
            "an explicit reported metric."
        ),
    },
    "M6": {
        "title": "SVM training subsample (5,000 rows) vs LR/RF full 85k",
        "covered_in": "Limitations",
        "note": (
            "Cross-model accuracy comparisons should note this asymmetry. "
            "SVM's relative position may understate its competitiveness on "
            "the full training set."
        ),
    },
    "M7": {
        "title": "Ablations now compared against no_ablation baseline row",
        "covered_in": "Audit fix C3+C4+M7 (not a paper limitation)",
        "note": (
            "Resolved during the post-refactor audit: each ablation table "
            "carries a no_ablation row trained with the same fixed RF, so "
            "deltas are computed apples-to-apples."
        ),
    },
    "M8": {
        "title": "Variance estimated over only three seeds",
        "covered_in": "Limitations",
        "note": (
            "Standard deviations across 3 samples are noisy estimators with "
            "large standard error. Additional seeds would tighten "
            "confidence intervals."
        ),
    },
}


# ==== SECTION 3: CURATED PARAGRAPHS ====
# Order in the deviations paragraph (per the brief):
#   filtration parameters (D2, D3, D4)
#   training-set scope (D5)
#   hyperparameter protocol (D6, D8)
#   ablation comparability (D7)
#   debug mode (D1)

METHODS_DEVIATIONS_PARAGRAPH = (
    "Several methodology choices warrant explicit disclosure. We set "
    "MAX_EDGE_LENGTH to the 25th percentile of pairwise distances within each "
    "reference cloud rather than the 95th percentile of our initial design, "
    "because the 95th percentile produced simplex trees exceeding 19 million "
    "simplices per query and exhausted memory at simplex-dimension three. We "
    "applied a sparse Rips approximation (epsilon = 0.5) for C2 and Network "
    "while Physical remained exact, and set the simplex-tree max_dimension to "
    "MAX_HOM_DIM + 1 so the highest-order homology features had finite death "
    "times. The Support Vector Machine was trained on a stratified 5,000-row "
    "subsample for tractability; Logistic Regression and Random Forest used "
    "all 85,519 training rows. We searched the hyperparameter grid by single "
    "fit per combination, selecting by validation accuracy; the curated "
    "Random Forest variant used fixed (n_estimators = 300, max_depth = None) "
    "rather than re-tuning. Ablation deltas were reported against a "
    "no_ablation row trained with the same fixed RF, so deltas reflect "
    "feature contribution rather than hyperparameter mismatch. Debug mode "
    "used a per-class stratified head sample because UAVIDS-2025 is "
    "row-sorted by class."
)

# Limitations paragraph covers M1, M2, M3, M4, M6, M8 (M5 and M7 are
# audit-fix improvements, not paper limitations).

LIMITATIONS_PARAGRAPH = (
    "The reference cloud was sampled by k-medoids on the C2 manifold only; "
    "the same row positions were reused for Network and Physical, "
    "guaranteeing a shared scaffold but not diversity in the latter two "
    "feature spaces. The single essential H_0 feature per Rips diagram was "
    "truncated to MAX_EDGE_LENGTH before summary statistics, biasing mean "
    "and max persistence upward by a manifold-specific constant. Per-flow "
    "Wasserstein distances were aggregated by summing across homology "
    "dimensions; max-pooling or multi-dimensional Wasserstein could change "
    "relative rankings. The pattern-based inference rule resolves ties by "
    "class-name order, potentially disadvantaging rarer attack types. The "
    "Support Vector Machine was trained on a 5,000-row stratified subsample "
    "because the RBF kernel does not scale to 85,519 rows, making its "
    "comparison with the linear and random-forest models approximate. "
    "Variance was estimated over three seeds, so reported standard "
    "deviations carry wide uncertainty."
)

FUTURE_WORK_PARAGRAPH = (
    "Three immediate extensions follow from these limitations. First, a "
    "larger filtration radius combined with a denser sparse Rips "
    "approximation would capture topological features at coarser scales "
    "than the present 25th-percentile cutoff permits. Second, per-manifold "
    "reference-cloud sampling, rather than reusing C2-derived medoids "
    "across all three manifolds, would ensure that the reference scaffold "
    "is locally representative of each manifold's geometry. Third, "
    "alternative Wasserstein aggregations across homology dimensions, "
    "including max-pooling for sensitivity to single-dimension anomalies "
    "and concatenated multi-dimensional Wasserstein for joint structure, "
    "merit empirical comparison against the current sum-aggregation."
)


# ==== SECTION 4: RENDERING ====

def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in ``text``."""
    return len(text.split())


def render_source_index() -> str:
    """Render the traceability section that appears below the paragraphs."""
    lines = [
        "## Source index (not for paper text)",
        "",
        "Below maps every catalogued deviation and limitation to its "
        "coverage in the paragraphs above. Two limitations (M5, M7) were "
        "resolved during the post-refactor audit and are not paper "
        "limitations.",
        "",
        "### Deviations (D1-D8) — all covered in Methods deviations",
        "",
        "| ID | Title |",
        "| :--- | :--- |",
    ]
    for did, entry in DEVIATIONS.items():
        lines.append(f"| {did} | {entry['title']} |")
    lines.extend([
        "",
        "### Limitations (M1-M8)",
        "",
        "| ID | Title | Coverage |",
        "| :--- | :--- | :--- |",
    ])
    for mid, entry in LIMITATIONS.items():
        lines.append(f"| {mid} | {entry['title']} | {entry['covered_in']} |")
    return "\n".join(lines)


def render_markdown() -> str:
    """Compose the full markdown document with three paragraphs and a source index."""
    blocks = [
        "# Methodology Disclosures",
        "",
        "_Three paragraph blocks suitable for direct paste into the paper's "
        "Methods, Limitations, and Future Work sections. The source-index "
        "table below the paragraphs is for traceability only and should not "
        "appear in the manuscript._",
        "",
        "## Methods deviations",
        "",
        METHODS_DEVIATIONS_PARAGRAPH,
        "",
        "## Limitations",
        "",
        LIMITATIONS_PARAGRAPH,
        "",
        "## Future work",
        "",
        FUTURE_WORK_PARAGRAPH,
        "",
        "---",
        "",
        render_source_index(),
        "",
    ]
    return "\n".join(blocks) + "\n"


# ==== SECTION 5: CLI ====

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(
        prog="extract_methods_disclosures.py",
        description=(
            "Render three paper-ready paragraphs (Methods deviations, "
            "Limitations, Future Work) as paper/METHODS_DISCLOSURES.md. "
            "Reads nothing; writes one markdown file."
        ),
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Markdown output path (parent dirs created as needed).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Render the markdown document and write it to ``args.output``."""
    args = build_parser().parse_args(argv)
    markdown = render_markdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    counts = {
        "methods_deviations": word_count(METHODS_DEVIATIONS_PARAGRAPH),
        "limitations": word_count(LIMITATIONS_PARAGRAPH),
        "future_work": word_count(FUTURE_WORK_PARAGRAPH),
    }
    print(f"wrote {args.output}")
    for name, n in counts.items():
        print(f"  {name}: {n} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
