"""Probe unsupervised Wasserstein-2 anomaly detection on a stratified test subset.

Standalone replacement for the unsupervised phase. Recomputes baseline barcodes
from the reference indices (the killed pipeline never persisted them), then
computes per-flow W_2 distances on a stratified subset of test flows and
reports binary + per-class AUCs.

Reads (read-only):
  outputs/{manifold}_train.csv                              (reference cloud points)
  outputs/persistence_diagrams/{manifold}_test.pkl          (per-flow diagrams)
  outputs/labels_test.csv                                   (class labels)
  outputs/reference_indices.npy                             (500 medoid indices)
  outputs/max_edge_lengths.json                             (filtration radii)

Writes:
  paper/PROBE_RESULTS.md                 (paste-ready markdown summary)
  results/tables/probe_distances.csv     (raw per-flow distances for inspection)

Usage:
  python tools/quick_unsup_probe.py
  python tools/quick_unsup_probe.py --per-class 200 --n-jobs 8
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ==== SECTION 1: CONFIG ====

# Default to the main project repo (script may run from a worktree).
DEFAULT_REPO = Path("/Users/oliverliu/uav-tda-research")

MANIFOLDS = ("c2", "network", "physical")
# MAX_HOM_DIM per manifold (inferred from saved diagram dim values).
MAX_HOM_DIM = {"c2": 2, "network": 2, "physical": 1}
# Sparse Rips epsilon per manifold (None = exact). Matches D3 disclosure.
SPARSE_EPS = {"c2": 0.5, "network": 0.5, "physical": None}


# ==== SECTION 2: BASELINE COMPUTATION ====

def load_reference_clouds(outputs_dir: Path) -> dict[str, np.ndarray]:
    """Load the 500-point reference cloud for each manifold."""
    ref_idx = np.load(outputs_dir / "reference_indices.npy")
    clouds: dict[str, np.ndarray] = {}
    for m in MANIFOLDS:
        df = pd.read_csv(outputs_dir / f"{m}_train.csv")
        clouds[m] = df.iloc[ref_idx].to_numpy(dtype=float)
    return clouds


def compute_baseline_barcode(
    cloud: np.ndarray, manifold: str, max_edge: float
) -> dict[int, np.ndarray]:
    """Compute Rips persistence on the reference cloud for one manifold.

    Returns dict mapping homology dim -> (n_bars, 2) array of (birth, death).
    Inf death values are clipped to max_edge per the M2 disclosure.
    """
    import gudhi  # noqa: PLC0415

    eps = SPARSE_EPS[manifold]
    if eps is None:
        rips = gudhi.RipsComplex(points=cloud, max_edge_length=max_edge)
    else:
        rips = gudhi.RipsComplex(points=cloud, max_edge_length=max_edge, sparse=eps)
    st = rips.create_simplex_tree(max_dimension=MAX_HOM_DIM[manifold] + 1)
    st.compute_persistence()

    out: dict[int, np.ndarray] = {}
    for k in range(MAX_HOM_DIM[manifold] + 1):
        bars = np.asarray(st.persistence_intervals_in_dimension(k), dtype=float)
        if bars.size == 0:
            out[k] = np.empty((0, 2), dtype=float)
            continue
        bars = bars.reshape(-1, 2)
        # Clip essential infinities to max_edge (M2 disclosure).
        bars[~np.isfinite(bars[:, 1]), 1] = float(max_edge)
        out[k] = bars
    return out


# ==== SECTION 3: PER-FLOW WASSERSTEIN ====

def diagram_to_per_dim(
    diag_array: np.ndarray, manifold: str, max_edge: float
) -> dict[int, np.ndarray]:
    """Slice an (n_bars, 3) [dim, birth, death] array into per-dim (n_bars, 2) arrays."""
    out: dict[int, np.ndarray] = {}
    if diag_array.size == 0:
        for k in range(MAX_HOM_DIM[manifold] + 1):
            out[k] = np.empty((0, 2), dtype=float)
        return out
    arr = np.asarray(diag_array, dtype=float)
    for k in range(MAX_HOM_DIM[manifold] + 1):
        mask = arr[:, 0] == k
        bars = arr[mask, 1:3].copy()
        if bars.size:
            bars[~np.isfinite(bars[:, 1]), 1] = float(max_edge)
        else:
            bars = np.empty((0, 2), dtype=float)
        out[k] = bars
    return out


def top_k_by_persistence(bars: np.ndarray, k: int | None) -> np.ndarray:
    """Keep the K most-persistent bars (largest death - birth). None disables."""
    if k is None or bars.shape[0] <= k:
        return bars
    lifetimes = bars[:, 1] - bars[:, 0]
    keep = np.argsort(lifetimes)[-k:]
    return bars[keep]


def _hera_subprocess_target(d1, d2, order, internal_p, delta, q):  # noqa: PLR0913
    """Child-process entry point used by ``_w2_with_timeout``.

    Imported inside the function so this module remains importable in
    environments where gudhi isn't available (e.g., the diagnose_c2_scaling
    script that explicitly avoids the gudhi dep).
    """
    try:
        from gudhi.hera import wasserstein_distance  # noqa: PLC0415

        result = wasserstein_distance(
            d1, d2, order=order, internal_p=internal_p, delta=delta,
        )
        q.put(float(result))
    except BaseException as exc:  # noqa: BLE001
        q.put(("error", repr(exc)))


def _w2_with_timeout(
    d1: np.ndarray,
    d2: np.ndarray,
    order: float,
    internal_p: float,
    delta: float,
    timeout_sec: float,
) -> float:
    """Hera W_2 wrapped in a fork+terminate timeout.

    Diagnostic C addition (2026-05-24): pathological persistence diagrams
    cause hera's auction LP to loop for many minutes without releasing the
    GIL, so signal.alarm-based interruption does not work (verified
    empirically against gudhi.hera 3.x). Each W_2 call runs in a forked
    subprocess; if wall time exceeds ``timeout_sec``, the child is killed
    and NaN is returned. The caller (``wasserstein_score_for_flow``) treats
    NaN as a zero contribution to the cross-dim sum and increments the
    timeout counter for the markdown disclosure.
    """
    import multiprocessing as mp  # noqa: PLC0415

    ctx = mp.get_context("fork")
    q: "mp.Queue" = ctx.Queue()
    p = ctx.Process(
        target=_hera_subprocess_target,
        args=(d1, d2, order, internal_p, delta, q),
        daemon=True,
    )
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(0.5)
        if p.is_alive():
            p.kill()
            p.join(0.5)
        return float("nan")
    try:
        value = q.get(timeout=0.5)
    except Exception:  # noqa: BLE001
        return float("nan")
    if isinstance(value, tuple) and value and value[0] == "error":
        return float("nan")
    return float(value)


def wasserstein_score_for_flow(
    flow_per_dim: dict[int, np.ndarray],
    baseline_per_dim: dict[int, np.ndarray],
    delta: float,
    top_k: int | None,
    w2_timeout: float | None = None,
) -> tuple[float, int]:
    """Sum of W_2 across homology dims for one (flow, baseline) pair.

    Uses the hera backend the production pipeline uses. ``delta`` is hera's
    relative-error tolerance (higher = faster, less exact). ``top_k`` keeps only
    the K most-persistent bars per dimension (None = keep all). When
    ``w2_timeout`` is set, each hera call runs in a forked subprocess and is
    killed if it exceeds the timeout; timed-out dims contribute zero to the
    sum.

    Returns ``(total_score, n_timeouts)``.
    """
    from gudhi.hera import wasserstein_distance  # noqa: PLC0415

    total = 0.0
    n_timeouts = 0
    for k, base_bars in baseline_per_dim.items():
        flow_bars = flow_per_dim.get(k, np.empty((0, 2), dtype=float))
        flow_sub = top_k_by_persistence(flow_bars, top_k)
        base_sub = top_k_by_persistence(base_bars, top_k)
        if w2_timeout is None:
            w = wasserstein_distance(
                flow_sub, base_sub, order=2, internal_p=2.0, delta=delta,
            )
        else:
            w = _w2_with_timeout(
                flow_sub, base_sub, order=2, internal_p=2.0,
                delta=delta, timeout_sec=w2_timeout,
            )
        if not np.isfinite(w):
            n_timeouts += 1
            continue  # treat as zero contribution
        total += float(w)
    return total, n_timeouts


def _worker_one_flow(
    payload: tuple[
        int,
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        float,
        int | None,
        float | None,
    ],
) -> tuple[int, float, int]:
    """Worker entry point for joblib.Parallel."""
    i, flow_per_dim, baseline_per_dim, delta, top_k, w2_timeout = payload
    score, n_to = wasserstein_score_for_flow(
        flow_per_dim, baseline_per_dim, delta, top_k, w2_timeout,
    )
    return i, score, n_to


# ==== SECTION 4: AUC HELPERS ====

def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with NaN fallback when only one class is present."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


# ==== SECTION 5: MAIN ====

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quick_unsup_probe.py",
        description=(
            "Compute W_2 anomaly scores on a stratified subset of test flows. "
            "Reports binary and per-class AUC per manifold and combined."
        ),
    )
    parser.add_argument(
        "--per-class", type=int, default=200,
        help="Number of test flows to sample per class (default: 200).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=7,
        help="Joblib n_jobs (default: 7; leaves 1 core for OS on 8-core Macs).",
    )
    parser.add_argument(
        "--delta", type=float, default=0.5,
        help=(
            "Hera Wasserstein approximation tolerance (default: 0.5 = 50%% slack; "
            "bounds runtime on pathological diagrams). Reduce to 0.1 for more "
            "accuracy at the risk of multi-minute hangs on edge cases."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help=(
            "Keep only the K most-persistent bars per (manifold, dim) before W_2 "
            "(default: 20). Defense against runaway hera iterations on "
            "high-bar-count H_0 diagrams. None to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repo", type=Path, default=DEFAULT_REPO,
        help=f"Path to the project root containing outputs/, paper/, results/ (default: {DEFAULT_REPO}).",
    )
    # Diagnostic C addition (2026-05-24): allow callers to redirect the per-flow
    # distances CSV to a custom path so multi-seed sweeps don't clobber each
    # other. When --output-path is set, the auto-written paper/PROBE_RESULTS.md
    # is also redirected to a sibling .md file alongside the CSV so the seed=42
    # report on disk stays untouched. Default behavior (flag omitted) is
    # unchanged from prior versions: CSV -> results/tables/probe_distances.csv
    # and markdown -> paper/PROBE_RESULTS.md.
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help=(
            "Optional path for the per-flow distances CSV. When set, the "
            "markdown report is written next to it (same basename, .md "
            "extension). When omitted, defaults to "
            "results/tables/probe_distances.csv and paper/PROBE_RESULTS.md."
        ),
    )
    # Diagnostic C addition (2026-05-24): wall-clock timeout per hera W_2 call.
    # Defense against pathological persistence diagrams where the auction LP
    # loops without releasing the GIL (signal.alarm-based interruption verified
    # empirically not to work). When None (default), behavior unchanged from
    # prior versions. When set, each W_2 call runs in a forked subprocess and
    # is killed if wall time exceeds the value; timed-out dims contribute zero
    # to the per-flow sum. Per-manifold timeout counts are reported to stdout.
    parser.add_argument(
        "--w2-timeout", type=float, default=None,
        help=(
            "Optional per-W_2-call wall-clock timeout in seconds. When set, "
            "each hera Wasserstein call runs in a forked subprocess and is "
            "killed if it exceeds this duration; timed-out homology "
            "dimensions contribute zero to the per-flow sum. Default None "
            "(no timeout) preserves prior behavior. Typical value: 5 seconds."
        ),
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    outputs_dir = repo / "outputs"
    paper_dir = repo / "paper"
    results_tables_dir = repo / "results" / "tables"

    print("=== Quick Unsupervised W_2 Probe ===", flush=True)
    print(f"repo: {repo}", flush=True)
    print(
        f"per-class: {args.per_class}  n_jobs: {args.n_jobs}  "
        f"delta: {args.delta}  top_k: {args.top_k}  seed: {args.seed}",
        flush=True,
    )
    print(flush=True)

    # Load filtration params
    with open(outputs_dir / "max_edge_lengths.json") as f:
        max_edges = json.load(f)
    print(f"max_edge_lengths: {max_edges}")
    print()

    # Stratified subsample of test flows
    labels_df = pd.read_csv(outputs_dir / "labels_test.csv")
    rng = np.random.default_rng(args.seed)
    sampled_idx: list[int] = []
    for cls in sorted(labels_df["label"].unique()):
        cls_idx = labels_df.index[labels_df["label"] == cls].to_numpy()
        n = min(args.per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n, replace=False)
        sampled_idx.extend(int(i) for i in chosen)
    sampled_idx.sort()
    sampled_labels = labels_df.iloc[sampled_idx].reset_index(drop=True)
    print(f"Sampled {len(sampled_idx)} test flows:")
    for cls, n in sampled_labels["label"].value_counts().items():
        print(f"  {cls}: {n}")
    print()

    # Compute baselines fresh from reference cloud
    print("Computing baseline barcodes from reference clouds...")
    t0 = time.perf_counter()
    clouds = load_reference_clouds(outputs_dir)
    baselines: dict[str, dict[int, np.ndarray]] = {}
    for m in MANIFOLDS:
        t1 = time.perf_counter()
        baselines[m] = compute_baseline_barcode(clouds[m], m, max_edges[m])
        sizes = {k: int(v.shape[0]) for k, v in baselines[m].items()}
        print(f"  {m}: per_dim_sizes={sizes}  ({time.perf_counter()-t1:.1f}s)")
    print(f"  baselines total: {time.perf_counter()-t0:.1f}s")
    print()

    # Load test diagrams (large files; one per manifold)
    print("Loading test diagrams (large files, ~250-350 MB each)...")
    test_diags: dict[str, list[np.ndarray]] = {}
    for m in MANIFOLDS:
        t1 = time.perf_counter()
        with open(outputs_dir / "persistence_diagrams" / f"{m}_test.pkl", "rb") as f:
            test_diags[m] = pickle.load(f)
        print(f"  {m}: n={len(test_diags[m])} ({time.perf_counter()-t1:.1f}s)")
    print()

    # Compute W_2 in parallel per manifold
    print(f"Computing W_2 distances for {len(sampled_idx)} flows × 3 manifolds...")
    from joblib import Parallel, delayed  # noqa: PLC0415

    scores: dict[str, np.ndarray] = {m: np.zeros(len(sampled_idx)) for m in MANIFOLDS}
    timeout_counts: dict[str, int] = {m: 0 for m in MANIFOLDS}  # Diagnostic C addition
    grand_t0 = time.perf_counter()
    for m in MANIFOLDS:
        t1 = time.perf_counter()
        # Pre-split each diagram into per-dim arrays (cheap) so workers only do W_2.
        tasks = []
        for i, test_i in enumerate(sampled_idx):
            flow_per_dim = diagram_to_per_dim(test_diags[m][test_i], m, max_edges[m])
            tasks.append(
                (i, flow_per_dim, baselines[m], args.delta, args.top_k, args.w2_timeout),
            )
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_worker_one_flow)(t) for t in tasks
        )
        for i, score, n_to in results:
            scores[m][i] = score
            timeout_counts[m] += n_to
        print(
            f"  {m}: done in {time.perf_counter()-t1:.1f}s  "
            f"mean_score={scores[m].mean():.4f}  "
            f"median={np.median(scores[m]):.4f}  "
            f"timeouts={timeout_counts[m]}"
        )
    print(f"  W_2 total wall time: {time.perf_counter()-grand_t0:.1f}s")
    if args.w2_timeout is not None:
        total_to = sum(timeout_counts.values())
        print(
            f"  W_2 timeouts (>{args.w2_timeout}s per call): total={total_to}, "
            f"per_manifold={timeout_counts}",
            flush=True,
        )
    print()

    # Combined score = sum across manifolds (all three available)
    scores["combined"] = sum(scores[m] for m in MANIFOLDS)

    # Seven manifold-dropout combinations for onboard / denied-environment story.
    # Score = sum of W_2 across whichever manifolds are "available" in that scenario.
    # Names mirror the deployment scenario, not the math.
    dropout_combos: dict[str, tuple[str, ...]] = {
        "c2_only": ("c2",),
        "network_only": ("network",),
        "physical_only": ("physical",),
        "c2_network": ("c2", "network"),               # GPS-denied / no Physical
        "c2_physical": ("c2", "physical"),             # Network-side compromised
        "network_physical": ("network", "physical"),   # No C2 link visibility
        "all_three": ("c2", "network", "physical"),    # Full availability
    }
    for name, mfs in dropout_combos.items():
        if name == "all_three":
            scores[name] = scores["combined"]
        else:
            scores[name] = sum(scores[m] for m in mfs)

    # Compute AUCs
    y_str = sampled_labels["label"].to_numpy()
    is_attack = (y_str != "Normal Traffic").astype(int)

    score_cols = list(MANIFOLDS) + ["combined"] + list(dropout_combos.keys())

    print("=== BINARY AUC (Normal vs Attack) ===", flush=True)
    print(f"  {'Score column':<20} {'AUC':>8}", flush=True)
    binary_aucs: dict[str, float] = {}
    for m in score_cols:
        auc = safe_auc(is_attack, scores[m])
        binary_aucs[m] = auc
        print(f"  {m:<20} {auc:>8.4f}", flush=True)
    print(flush=True)

    print("=== PER-CLASS AUC (one-vs-rest) ===", flush=True)
    classes = sorted(labels_df["label"].unique())
    header_cols = list(MANIFOLDS) + ["combined"]
    print(f"  {'Class':<22} " + " ".join(f"{c:>10}" for c in header_cols), flush=True)
    per_class_aucs: dict[str, dict[str, float]] = {}
    for cls in classes:
        y_cls = (y_str == cls).astype(int)
        row_aucs: dict[str, float] = {}
        line = f"  {cls:<22} "
        for m in header_cols:
            auc = safe_auc(y_cls, scores[m])
            row_aucs[m] = auc
            line += f"{auc:>10.4f} "
        per_class_aucs[cls] = row_aucs
        print(line, flush=True)
    print(flush=True)

    # Dropout binary AUCs printed as a dedicated block for easy reading
    print("=== MANIFOLD DROPOUT (Binary AUC) ===", flush=True)
    print(f"  {'Available manifolds':<26} {'AUC':>8}", flush=True)
    dropout_binary: dict[str, float] = {}
    for name in dropout_combos:
        auc = safe_auc(is_attack, scores[name])
        dropout_binary[name] = auc
        print(f"  {name:<26} {auc:>8.4f}", flush=True)
    print(flush=True)

    # Per-class AUC for dropout combinations (Zeng-attribution check)
    print("=== PER-CLASS AUC under dropout ===", flush=True)
    print(
        f"  {'Class':<22} "
        + " ".join(f"{n:>17}" for n in dropout_combos.keys()),
        flush=True,
    )
    dropout_per_class: dict[str, dict[str, float]] = {}
    for cls in classes:
        y_cls = (y_str == cls).astype(int)
        row: dict[str, float] = {}
        line = f"  {cls:<22} "
        for name in dropout_combos:
            auc = safe_auc(y_cls, scores[name])
            row[name] = auc
            line += f"{auc:>17.4f} "
        dropout_per_class[cls] = row
        print(line, flush=True)
    print(flush=True)

    # Persist distances for inspection (per-manifold + combined + all dropout combos)
    # Diagnostic C addition: honor --output-path when set so multi-seed sweeps
    # write to per-seed files; otherwise keep the original default.
    if args.output_path is not None:
        distances_path = args.output_path.resolve()
        distances_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_tables_dir.mkdir(parents=True, exist_ok=True)
        distances_path = results_tables_dir / "probe_distances.csv"
    df_out = pd.DataFrame(
        {"test_idx": sampled_idx, "label": y_str},
    )
    for m in score_cols:
        df_out[f"W2_{m}"] = scores[m]
    df_out.to_csv(distances_path, index=False)
    print(f"wrote {distances_path}")

    # Markdown report
    paper_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Quick Unsupervised W_2 Probe",
        "",
        (
            f"_Generated by `tools/quick_unsup_probe.py` on a stratified subset of "
            f"{len(sampled_idx)} test flows ({args.per_class} per class, seed={args.seed}). "
            f"Baselines recomputed from `reference_indices.npy`. "
            f"hera delta={args.delta}, top_k={args.top_k}._"
        ),
        "",
        "## Binary AUC (Normal Traffic vs any Attack)",
        "",
        "### Per manifold and combined (all three)",
        "",
        "| Score column | AUC |",
        "| :--- | ---: |",
    ]
    for m in list(MANIFOLDS) + ["combined"]:
        md.append(f"| {m} | {binary_aucs[m]:.4f} |")
    md += [
        "",
        "### Manifold-dropout combinations (onboard / denied-environment)",
        "",
        "| Available manifolds | Scenario | Binary AUC |",
        "| :--- | :--- | ---: |",
        f"| c2_only | C2 link only (no telemetry, no network captures) | {dropout_binary['c2_only']:.4f} |",
        f"| network_only | Network captures only (no C2 visibility, no sensors) | {dropout_binary['network_only']:.4f} |",
        f"| physical_only | Sensor / telemetry only (no traffic capture) | {dropout_binary['physical_only']:.4f} |",
        f"| c2_network | **GPS / sensor denied (no Physical)** | {dropout_binary['c2_network']:.4f} |",
        f"| c2_physical | Mid-network compromised (no Network) | {dropout_binary['c2_physical']:.4f} |",
        f"| network_physical | C2 unobservable (e.g., encrypted control) | {dropout_binary['network_physical']:.4f} |",
        f"| all_three | Full instrumentation (baseline) | {dropout_binary['all_three']:.4f} |",
        "",
        "## Per-class AUC (one-vs-rest)",
        "",
        "### Per manifold and combined",
        "",
        "| Class | C2 | Network | Physical | Combined |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for cls in classes:
        row = per_class_aucs[cls]
        md.append(
            f"| {cls} | {row['c2']:.4f} | {row['network']:.4f} | "
            f"{row['physical']:.4f} | {row['combined']:.4f} |"
        )
    md += [
        "",
        "### Per-class AUC under dropout (Zeng-attribution check)",
        "",
        "| Class | c2_only | network_only | physical_only | c2_network | c2_physical | network_physical | all_three |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cls in classes:
        r = dropout_per_class[cls]
        md.append(
            f"| {cls} | {r['c2_only']:.4f} | {r['network_only']:.4f} | "
            f"{r['physical_only']:.4f} | {r['c2_network']:.4f} | "
            f"{r['c2_physical']:.4f} | {r['network_physical']:.4f} | "
            f"{r['all_three']:.4f} |"
        )
    md += [
        "",
        "## Sample composition",
        "",
        "| Class | Count |",
        "| :--- | ---: |",
    ]
    for cls in classes:
        n = int((y_str == cls).sum())
        md.append(f"| {cls} | {n} |")
    md += [
        "",
        "## Verdict heuristic",
        "",
        f"- **Binary AUC (all_three) = {binary_aucs['combined']:.4f}**",
        f"- **Binary AUC (c2_network — GPS-denied) = {dropout_binary['c2_network']:.4f}**",
        "",
        "Thresholds (probe-only; full-test results may differ by +-0.02-0.05):",
        "",
        "- **>= 0.85** -> label-free anomaly detection is a viable paper headline.",
        "- **0.75-0.85** -> story is solid; relaunch full unsupervised on val+test only.",
        "- **0.65-0.75** -> marginal; consider negative-result / diagnostic framing.",
        "- **< 0.65** -> pivot to negative-result paper.",
        "",
    ]
    # Diagnostic C addition: when --output-path is set, write the markdown
    # next to the CSV (same stem, .md extension) so multi-seed sweeps do not
    # clobber the canonical paper/PROBE_RESULTS.md from the seed=42 run.
    if args.output_path is not None:
        report_path = distances_path.with_suffix(".md")
    else:
        report_path = paper_dir / "PROBE_RESULTS.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
