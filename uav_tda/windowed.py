"""Phase 5 — time-windowed multi-manifold persistence (core).

Implements the windowed variant of docs/superpowers/specs/
2026-09-21-windowed-variant-design.md §2: non-overlapping FlowID-order
windows of W consecutive flows (trailing partial dropped), one
Vietoris-Rips complex per (window, manifold) where the W standardized
feature vectors ARE the point cloud (Bruillard et al. 2016 construction),
plus the exact-Rips benchmark gate that fixes config.WINDOWED_SPARSE.

`window_diagram` mirrors the body of `tda._persistence_for_point` minus
the query-point stacking, so diagram conventions ((n, 3) [dim, birth,
death] float rows; empty -> (0, 3)) match the per-flow arm exactly.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.windowed")


def make_windows(n_rows: int, w: int, order=None) -> list:
    """Return index arrays for non-overlapping windows of w rows.

    Windows are consecutive slices of `order` (identity row order when
    None, e.g. FlowID order; a permutation for the shuffle control).
    The trailing partial window is dropped, per spec §2.1.
    """
    import numpy as np

    if order is None:
        order = np.arange(n_rows)
    else:
        order = np.asarray(order)
    n_windows = n_rows // w
    return [order[i * w:(i + 1) * w] for i in range(n_windows)]


def window_diagram(points, max_edge: float, max_hom_dim: int, sparse):
    """Compute one (sparse) Rips persistence diagram for a window's point cloud.

    Mirrors tda._persistence_for_point minus the query-point stacking:
    the window's points are the whole cloud. Returns an (n, 3) float
    array of [dim, birth, death] rows; empty diagram -> (0, 3).
    """
    import gudhi
    import numpy as np

    kwargs: dict = {"points": points, "max_edge_length": max_edge}
    if sparse is not None:
        kwargs["sparse"] = sparse
    rips = gudhi.RipsComplex(**kwargs)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_hom_dim + 1)
    raw = simplex_tree.persistence()
    if not raw:
        return np.empty((0, 3), dtype=float)
    return np.array(
        [[float(dim), float(birth), float(death)] for dim, (birth, death) in raw],
        dtype=float,
    )


def w2_distance(d1, d2, max_edge: float, max_hom_dim: int) -> float:
    """Hera Wasserstein-2 distance between two diagrams, summed over dims.

    Per-dim slices (birth, death) via probe._slice_dim (inf deaths clamped
    to max_edge, matching the per-flow arm), dims 0..max_hom_dim inclusive.
    """
    from gudhi.hera import wasserstein_distance

    from .probe import _slice_dim

    total = 0.0
    for dim in range(max_hom_dim + 1):
        a = _slice_dim(d1, dim, max_edge)
        b = _slice_dim(d2, dim, max_edge)
        total += float(wasserstein_distance(a, b, order=2, internal_p=2.0))
    return total


def baseline_medoid_diagram(diagrams: list, max_edge: float, max_hom_dim: int):
    """W2-medoid of a list of diagrams: the one minimizing summed W2 to the rest.

    Builds the full symmetric pairwise W2 distance matrix (zero diagonal)
    via w2_distance, then returns (argmin-row-sum index, that diagram).
    """
    import numpy as np

    n = len(diagrams)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = w2_distance(diagrams[i], diagrams[j], max_edge, max_hom_dim)
            dist[i, j] = d
            dist[j, i] = d
    medoid_idx = int(np.argmin(dist.sum(axis=1)))
    return medoid_idx, diagrams[medoid_idx]


_NORMAL_LABEL = "Normal Traffic"


def window_majority_label(labels) -> str:
    """Modal label of a window.

    Ties are broken: attack labels beat "Normal Traffic", then
    alphabetically among tied attack labels (arbitrary but deterministic;
    documented per the Task-2 brief).
    """
    from collections import Counter

    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [lbl for lbl, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    attack_candidates = sorted(c for c in candidates if c != _NORMAL_LABEL)
    if attack_candidates:
        return attack_candidates[0]
    return _NORMAL_LABEL


def window_attack_frac(labels) -> float:
    """Fraction of non-"Normal Traffic" flows in a window."""
    import numpy as np

    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels != _NORMAL_LABEL))


def run_windowed(ws: Workspace, w: int, order_seed=None, repeat: int = 0):
    """Coupled single-invocation windowed pass (spec §2.3).

    (a) val-Normal rows (original order) chunked into windows -> per-manifold
        diagrams -> W2-medoid baseline + val-window score distribution ->
        z-stats (mean/std, degenerate-sigma guard).
    (b) test split (permuted iff order_seed is not None) -> windows ->
        diagrams -> per-manifold W2 scores vs the SAME baselines.
    (c) window_df: window_idx, start_pos, majority_label, attack_frac,
        W2_<manifold> x3, W2_<subset> x7, and Z2_ variants of all 10
        (z-scored manifolds then subset sums recomputed, matching
        metrics.apply_znorm semantics).
    (d) timing: total_s, n_windows, per_window_s, rips_s, w2_s, baseline_s.

    ``repeat`` is metadata only; it is not consumed by any RNG here.
    """
    import json
    import time

    import numpy as np
    import pandas as pd

    from .metrics import MANIFOLD_SUBSETS
    from .tda import load_labels_for_split, load_split_manifold

    t_total0 = time.perf_counter()
    timers = {"rips_s": 0.0, "w2_s": 0.0, "baseline_s": 0.0}

    with (ws.outputs_dir / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)
    max_hom_dim = config.MAX_HOM_DIM

    # --- (a) val-Normal baseline ------------------------------------------------
    labels_val = load_labels_for_split(ws, "val").to_numpy()
    normal_mask = labels_val == _NORMAL_LABEL

    baseline_diagrams: dict = {}
    stats: dict = {}
    for m in config.MANIFOLDS:
        points_val = load_split_manifold(ws, m, "val")[normal_mask]
        max_edge = max_edge_lengths[m]
        hd = max_hom_dim[m]
        sparse = config.WINDOWED_SPARSE[m]

        val_windows = make_windows(len(points_val), w)
        val_diagrams = []
        for idx in val_windows:
            t0 = time.perf_counter()
            diag = window_diagram(points_val[idx], max_edge, hd, sparse)
            timers["rips_s"] += time.perf_counter() - t0
            val_diagrams.append(diag)

        t0 = time.perf_counter()
        _, baseline_diag = baseline_medoid_diagram(val_diagrams, max_edge, hd)
        timers["baseline_s"] += time.perf_counter() - t0
        baseline_diagrams[m] = baseline_diag

        val_scores = []
        for diag in val_diagrams:
            t0 = time.perf_counter()
            score = w2_distance(diag, baseline_diag, max_edge, hd)
            timers["w2_s"] += time.perf_counter() - t0
            val_scores.append(score)
        val_scores = np.asarray(val_scores, dtype=float)
        mean, std = float(np.mean(val_scores)), float(np.std(val_scores))
        if std < 1e-12:
            log.warning("run_windowed: manifold %s has degenerate val-window "
                        "std=%.3g; using 1.0", m, std)
            std = 1.0
        stats[m] = (mean, std)

    # --- (b) test split, scored against the same baselines ---------------------
    labels_test = load_labels_for_split(ws, "test").to_numpy()
    n_test = len(labels_test)
    if order_seed is None:
        order = None
    else:
        order = np.random.default_rng(order_seed).permutation(n_test)
    test_windows = make_windows(n_test, w, order=order)

    points_test = {m: load_split_manifold(ws, m, "test") for m in config.MANIFOLDS}

    rows = []
    for i, idx in enumerate(test_windows):
        window_labels = labels_test[idx]
        row = {
            "window_idx": i,
            "start_pos": i * w,
            "majority_label": window_majority_label(window_labels),
            "attack_frac": window_attack_frac(window_labels),
        }
        for m in config.MANIFOLDS:
            max_edge = max_edge_lengths[m]
            hd = max_hom_dim[m]
            sparse = config.WINDOWED_SPARSE[m]

            t0 = time.perf_counter()
            diag = window_diagram(points_test[m][idx], max_edge, hd, sparse)
            timers["rips_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            score = w2_distance(diag, baseline_diagrams[m], max_edge, hd)
            timers["w2_s"] += time.perf_counter() - t0
            row[f"W2_{m}"] = score
        rows.append(row)

    window_df = pd.DataFrame(rows)
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        window_df[f"W2_{subset}"] = sum(window_df[f"W2_{m}"] for m in manifolds)
    for m in config.MANIFOLDS:
        mean, std = stats[m]
        window_df[f"Z2_{m}"] = (window_df[f"W2_{m}"] - mean) / std
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        window_df[f"Z2_{subset}"] = sum(window_df[f"Z2_{m}"] for m in manifolds)

    n_windows = len(test_windows)
    total_s = time.perf_counter() - t_total0
    timing = {
        "total_s": total_s,
        "n_windows": n_windows,
        "per_window_s": (total_s / n_windows) if n_windows else float("nan"),
        "rips_s": timers["rips_s"],
        "w2_s": timers["w2_s"],
        "baseline_s": timers["baseline_s"],
    }
    log.info("run_windowed: w=%d order_seed=%s repeat=%d -> %d windows in %.1fs",
              w, order_seed, repeat, n_windows, total_s)

    return window_df, stats, timing


def benchmark_exact_rips(ws: Workspace, w: int = 200, n_trials: int = 3) -> dict:
    """Time exact (sparse=None) Rips calls per manifold on real test windows.

    For each manifold, takes the first n_trials windows of size w from the
    test split and times one exact `window_diagram` call per window with
    the frozen per-manifold max_edge_lengths and MAX_HOM_DIM. Returns
    {manifold: {"median_s": float, "w": int, "trials_s": [float, ...]}}.
    Used by the Task-1 benchmark gate to fix config.WINDOWED_SPARSE.
    """
    import json
    import time

    import numpy as np

    from .tda import load_split_manifold

    with (ws.outputs_dir / "max_edge_lengths.json").open() as fh:
        max_edge_lengths = json.load(fh)

    results: dict = {}
    for manifold in config.MANIFOLDS:
        points = load_split_manifold(ws, manifold, "test")
        windows = make_windows(len(points), w)[:n_trials]
        max_edge = max_edge_lengths[manifold]
        max_hom_dim = config.MAX_HOM_DIM[manifold]
        trials = []
        for idx in windows:
            start = time.perf_counter()
            window_diagram(points[idx], max_edge, max_hom_dim, sparse=None)
            trials.append(time.perf_counter() - start)
        results[manifold] = {
            "median_s": float(np.median(trials)),
            "w": w,
            "trials_s": trials,
        }
        log.info("benchmark %s: median %.3fs over %d trials (w=%d)",
                 manifold, results[manifold]["median_s"], len(trials), w)
    return results


# --- Task 3: campaign artifacts, tables, CLI plumbing -----------------------
# Grid: config.WINDOW_SIZES x {("ordered", k) for k in range(WINDOWED_REPEATS)}
# U {("shuffled", k) for k in WINDOWED_SHUFFLE_SEEDS} -- 4 x (10 + 10) = 80
# entries. All builder functions below work off however many runs are
# ACTUALLY present under results/tables/rebuild/windowed/ (not necessarily
# the full 80), which is what makes them unit-testable on tiny synthetic
# fixtures without a real campaign.

_WINDOWED_SUBDIR = ("rebuild", "windowed")


def _windowed_dir(ws: Workspace):
    d = ws.tables_dir
    for part in _WINDOWED_SUBDIR:
        d = d / part
    return d


def _run_paths(ws: Workspace, w: int, arm: str, k: int) -> dict:
    d = _windowed_dir(ws)
    return {
        "csv": d / f"run_w{w}_{arm}{k}.csv",
        "meta": d / f"run_w{w}_{arm}{k}_meta.json",
    }


def write_run_artifacts(ws: Workspace, w: int, arm: str, k: int,
                         window_df, stats: dict, timing: dict) -> dict:
    """Write one campaign run's window_df + stats/timing/params to disk.

    Writes `run_w{W}_{arm}{k}.csv` (the window_df, verbatim) and
    `run_w{W}_{arm}{k}_meta.json` (``{"stats", "timing", "params"}``, stats
    serialized as ``{manifold: {"mean", "std"}}``) under
    `results/tables/rebuild/windowed/`, each with a `write_provenance`
    sidecar. Returns ``{"csv": Path, "meta": Path}``.
    """
    import json

    from .provenance import write_provenance

    paths = _run_paths(ws, w, arm, k)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)

    params = {"w": w, "arm": arm, "k": k}

    window_df.to_csv(paths["csv"], index=False)
    write_provenance(paths["csv"], params)

    meta = {
        "stats": {m: {"mean": mean, "std": std} for m, (mean, std) in stats.items()},
        "timing": timing,
        "params": params,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True))
    write_provenance(paths["meta"], params)
    return paths


def load_run(ws: Workspace, w: int, arm: str, k: int):
    """Load one campaign run's (window_df, stats, timing) written by ``write_run_artifacts``."""
    import json

    import pandas as pd

    paths = _run_paths(ws, w, arm, k)
    window_df = pd.read_csv(paths["csv"])
    meta = json.loads(paths["meta"].read_text())
    stats = {m: (v["mean"], v["std"]) for m, v in meta["stats"].items()}
    return window_df, stats, meta["timing"]


def missing_runs(ws: Workspace) -> list:
    """(w, arm, k) triples in the full campaign grid lacking both artifacts.

    Grid: `config.WINDOW_SIZES` x {("ordered", k) for k in
    range(WINDOWED_REPEATS)} U {("shuffled", k) for k in
    WINDOWED_SHUFFLE_SEEDS} -- 4 x 20 = 80 entries when
    `results/tables/rebuild/windowed/` is empty.
    """
    missing = []
    for w in config.WINDOW_SIZES:
        for k in range(config.WINDOWED_REPEATS):
            paths = _run_paths(ws, w, "ordered", k)
            if not (paths["csv"].exists() and paths["meta"].exists()):
                missing.append((w, "ordered", k))
        for k in config.WINDOWED_SHUFFLE_SEEDS:
            paths = _run_paths(ws, w, "shuffled", k)
            if not (paths["csv"].exists() and paths["meta"].exists()):
                missing.append((w, "shuffled", k))
    return missing


def run_campaign_entry(ws: Workspace, w: int, arm: str, k: int) -> dict:
    """Run one campaign grid entry (real, slow) and write its artifacts.

    ``arm="ordered"`` -> ``run_windowed(order_seed=None, repeat=k)`` (FlowID
    order; k is metadata distinguishing the WINDOWED_REPEATS repeats -- if
    exact Rips holds for a manifold these repeats are bit-identical, per
    spec 2.2/3.5). ``arm="shuffled"`` -> ``run_windowed(order_seed=k)`` (the
    shuffle control, permutation seeded by k).
    """
    if arm == "ordered":
        window_df, stats, timing = run_windowed(ws, w, order_seed=None, repeat=k)
    elif arm == "shuffled":
        window_df, stats, timing = run_windowed(ws, w, order_seed=k, repeat=k)
    else:
        raise ValueError(f"unknown arm {arm!r}; expected 'ordered' or 'shuffled'")
    return write_run_artifacts(ws, w, arm, k, window_df, stats, timing)


def run_missing_campaign(ws: Workspace, entry_fn=run_campaign_entry) -> None:
    """Sequentially run every missing campaign grid entry (real, slow path).

    This is the 80-run, multi-hour Phase-5 campaign. It is invoked ONLY
    from `uav-tda windowed-report`, one entry at a time (no parallelism, no
    detachment) -- the campaign itself is explicitly out of scope for this
    task (see task-3-brief: "do NOT launch the real campaign in this
    task"). Tests must monkeypatch `entry_fn` (or this function itself) so
    no real windowed run ever executes under pytest.
    """
    for w, arm, k in missing_runs(ws):
        entry_fn(ws, w, arm, k)


def load_all_runs(ws: Workspace) -> list:
    """Load every campaign run currently present under `.../windowed/`.

    Returns a list of ``{"w", "arm", "k", "window_df", "stats", "timing"}``
    dicts, discovered from `run_w{W}_{arm}{k}.csv` filenames already on
    disk -- however many are present, not necessarily the full 80-run grid.
    """
    import re

    d = _windowed_dir(ws)
    runs = []
    if not d.exists():
        return runs
    pattern = re.compile(r"^run_w(\d+)_(ordered|shuffled)(\d+)\.csv$")
    for csv_path in sorted(d.glob("run_w*.csv")):
        m = pattern.match(csv_path.name)
        if not m:
            continue
        w, arm, k = int(m.group(1)), m.group(2), int(m.group(3))
        window_df, stats, timing = load_run(ws, w, arm, k)
        runs.append({"w": w, "arm": arm, "k": k, "window_df": window_df,
                     "stats": stats, "timing": timing})
    return runs


# --- Detection table ---------------------------------------------------------

def build_detection_table(ws: Workspace, B: int = 2000, bootstrap_seed: int = 0):
    """Window-level binary (Normal-vs-any-attack) AUC, per W x arm x subset x scoring.

    Groups loaded runs by (w, arm); within each group, treats each run as
    one "seed" in the `manuscript.bootstrap_mean_auc_ci` sense: per-run AUC
    on ``y = majority_label != "Normal Traffic"`` vs. that run's
    ``W2_<subset>`` column (scoring="raw") or ``Z2_<subset>`` column
    (scoring="znorm"). mean/std (ddof=1) across runs; CI via
    `bootstrap_mean_auc_ci` with ``labels_per_seed`` = each run's
    `majority_label` array (stratified resampling, avoiding single-class
    bootstrap replicates when a run has few attack windows). Columns: w,
    arm, subset, scoring, mean, std, ci_lo, ci_hi, n_runs.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from .manuscript import bootstrap_mean_auc_ci
    from .metrics import MANIFOLD_SUBSETS

    groups: dict = {}
    for r in load_all_runs(ws):
        groups.setdefault((r["w"], r["arm"]), []).append(r)

    rows = []
    for (w, arm), group_runs in sorted(groups.items()):
        for subset in MANIFOLD_SUBSETS:
            for scoring, prefix in (("raw", "W2_"), ("znorm", "Z2_")):
                per_run, labels_per_run, aucs = [], [], []
                for r in group_runs:
                    df = r["window_df"]
                    y = (df["majority_label"] != _NORMAL_LABEL).astype(int).to_numpy()
                    scores = df[f"{prefix}{subset}"].to_numpy()
                    per_run.append((y, scores))
                    labels_per_run.append(df["majority_label"].to_numpy())
                    aucs.append(float(roc_auc_score(y, scores)))
                aucs_arr = np.asarray(aucs, dtype=float)
                mean = float(aucs_arr.mean())
                std = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
                ci_lo, ci_hi = bootstrap_mean_auc_ci(
                    per_run, B=B, bootstrap_seed=bootstrap_seed,
                    labels_per_seed=labels_per_run)
                rows.append({
                    "w": w, "arm": arm, "subset": subset, "scoring": scoring,
                    "mean": mean, "std": std, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "n_runs": len(group_runs),
                })
    return pd.DataFrame(rows)


# --- Attribution table --------------------------------------------------------

def build_windowed_attribution_table(ws: Workspace, B: int = 2000, bootstrap_seed: int = 0):
    """One-vs-rest window-level AUC per (W, attack, manifold), ORDERED arm only.

    Ordered-arm runs only (spec 3.3: attribution survival is a property of
    the FlowID-order windowing, not the shuffle control). Raw `W2_<manifold>`
    scores -- per-manifold AUC is invariant to the per-manifold znorm affine
    rescaling, matching `manuscript.build_attribution_table`'s rationale, so
    there is no separate znorm variant here. `dominant` marks the single
    highest-mean manifold within each (w, attack) group.

    Robustness note: a real campaign can produce an attack class that is
    NEVER a window majority at some W (spec 1: Blackhole is interleaved
    with Normal at flow granularity, so Blackhole-era windows are always
    mixed) -- then y is single-class and AUC is undefined. Such
    (w, attack, manifold) cells get `mean=NaN`/`ci=(NaN, NaN)` rather than
    raising, and are excluded from the per-(w, attack) dominance vote (a
    group with no finite mean gets `dominant=False` everywhere).
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from .manuscript import bootstrap_mean_auc_ci
    from .metrics import ATTACK_CLASSES

    groups: dict = {}
    for r in load_all_runs(ws):
        if r["arm"] != "ordered":
            continue
        groups.setdefault(r["w"], []).append(r)

    rows = []
    for w, group_runs in sorted(groups.items()):
        per_cell_aucs: dict = {}
        per_cell_ci: dict = {}
        for attack in ATTACK_CLASSES:
            for m in config.MANIFOLDS:
                per_run, labels_per_run, aucs = [], [], []
                for r in group_runs:
                    df = r["window_df"]
                    y = (df["majority_label"].to_numpy() == attack).astype(int)
                    if len(np.unique(y)) < 2:
                        continue
                    scores = df[f"W2_{m}"].to_numpy()
                    per_run.append((y, scores))
                    labels_per_run.append(df["majority_label"].to_numpy())
                    aucs.append(float(roc_auc_score(y, scores)))
                per_cell_aucs[(attack, m)] = aucs
                per_cell_ci[(attack, m)] = (
                    bootstrap_mean_auc_ci(per_run, B=B, bootstrap_seed=bootstrap_seed,
                                           labels_per_seed=labels_per_run)
                    if per_run else (float("nan"), float("nan"))
                )

        for attack in ATTACK_CLASSES:
            means = {}
            for m in config.MANIFOLDS:
                aucs = per_cell_aucs[(attack, m)]
                means[m] = float(np.mean(aucs)) if aucs else float("nan")
            finite_means = {m: v for m, v in means.items() if np.isfinite(v)}
            dominant_manifold = max(finite_means, key=finite_means.get) if finite_means else None
            for m in config.MANIFOLDS:
                aucs_arr = np.asarray(per_cell_aucs[(attack, m)], dtype=float)
                std = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
                ci_lo, ci_hi = per_cell_ci[(attack, m)]
                rows.append({
                    "w": w, "attack_class": attack, "manifold": m,
                    "mean": means[m], "std": std, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "dominant": m == dominant_manifold, "n_runs": len(per_cell_aucs[(attack, m)]),
                })
    return pd.DataFrame(rows)


# --- Contamination curve -----------------------------------------------------

def _contamination_bin_labels() -> list:
    edges = config.CONTAMINATION_BINS
    return ["0"] + [f"({lo:.2f},{hi:.2f}]" for lo, hi in zip(edges[:-1], edges[1:])]


def _assign_contamination_bin(attack_frac) -> list:
    import numpy as np

    edges = config.CONTAMINATION_BINS
    labels = _contamination_bin_labels()
    attack_frac = np.asarray(attack_frac, dtype=float)
    out = np.empty(len(attack_frac), dtype=object)
    out[:] = labels[-1]
    is_zero = attack_frac == 0.0
    out[is_zero] = labels[0]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (~is_zero) & (attack_frac > lo) & (attack_frac <= hi)
        out[mask] = labels[i + 1]
    return out


def build_contamination_table(ws: Workspace):
    """Window attack_frac contamination curve, per W x bin (+ per-majority-class rows).

    Bins from `config.CONTAMINATION_BINS`: bin "0" = attack_frac exactly 0,
    then (0, .25], (.25, .5], (.5, .75], (.75, 1] -- a strict partition of
    every window's attack_frac in [0, 1]. Per W, pools ALL loaded runs
    (both arms) into one window population.

    THRESHOLD-DESIGN NOTE (documented per the task-3 controller ruling):
    `run_windowed`'s val-window score distribution (the natural source of a
    "val 95th-percentile" detection threshold) is not persisted past a
    run's own coupled invocation, so it cannot be reconstructed post-hoc
    from `run_meta.json` (which stores only mean/std). This function
    substitutes an EMPIRICAL near-normal reference population: per W, pools
    `Z2_all_three` over every loaded window with `attack_frac <= 0.05`
    (bin 0 plus near-zero contamination) and takes ITS 95th percentile as
    that W's detection threshold. `detection_rate` per bin/class is the
    fraction of that bin's windows scoring above this threshold. This is an
    empirical-negative-reference substitute for the canonical Phase-3-style
    val threshold, not the val threshold itself -- flag this explicitly
    wherever `contamination_curve.csv` is cited in the report.

    Columns: w, bin, bin_lo, bin_hi, majority_class ("all", or a specific
    class name for the per-class breakdown when n >= 10), n_windows,
    mean_raw_all_three, mean_znorm_all_three, threshold, detection_rate.
    """
    import numpy as np
    import pandas as pd

    groups: dict = {}
    for r in load_all_runs(ws):
        groups.setdefault(r["w"], []).append(r["window_df"])

    bin_labels = _contamination_bin_labels()
    edges = config.CONTAMINATION_BINS
    bin_edges_map = {bin_labels[0]: (0.0, 0.0)}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        bin_edges_map[bin_labels[i + 1]] = (lo, hi)

    rows = []
    for w, dfs in sorted(groups.items()):
        all_df = pd.concat(dfs, ignore_index=True).copy()
        all_df["_bin"] = _assign_contamination_bin(all_df["attack_frac"].to_numpy())

        near_normal = all_df[all_df["attack_frac"] <= 0.05]
        threshold = (float(np.percentile(near_normal["Z2_all_three"].to_numpy(), 95))
                     if len(near_normal) else float("nan"))

        def _row(label, sub, majority_class):
            n = len(sub)
            det = (float((sub["Z2_all_three"] > threshold).mean())
                   if n and np.isfinite(threshold) else float("nan"))
            lo, hi = bin_edges_map[label]
            return {
                "w": w, "bin": label, "bin_lo": lo, "bin_hi": hi,
                "majority_class": majority_class, "n_windows": n,
                "mean_raw_all_three": float(sub["W2_all_three"].mean()) if n else float("nan"),
                "mean_znorm_all_three": float(sub["Z2_all_three"].mean()) if n else float("nan"),
                "threshold": threshold, "detection_rate": det,
            }

        for label in bin_labels:
            sub = all_df[all_df["_bin"] == label]
            rows.append(_row(label, sub, "all"))
            for cls in sorted(sub["majority_label"].unique()):
                cls_sub = sub[sub["majority_label"] == cls]
                if len(cls_sub) >= 10:
                    rows.append(_row(label, cls_sub, cls))
    return pd.DataFrame(rows)


# --- Matched-compute frontier -------------------------------------------------

_PER_FLOW_ROW = "per_flow"


def _time_per_flow_probe(seed: int = 42, per_class: int = 4) -> float:
    """Time a small per-flow `probe.run_probe` sample; return seconds/flow (real, slow).

    Two-point marginal estimate, mirroring the FRONTIER COST ruling applied
    to the windowed arm: times ``probe.run_probe(seed, per_class=per_class)``
    (5 classes x per_class flows; per_class=4 -> 20 flows) against
    ``probe.run_probe(seed, per_class=1)`` (5 flows), and returns the SLOPE
    ``(t_big - t_small) / (n_big - n_small)``. This nets out the shared
    one-time cost every `run_probe` call pays (loading persistence diagrams
    + baseline barcodes from disk), isolating the true per-decision
    marginal cost rather than an amortized-with-setup number. Never called
    by tests -- `build_frontier_table` takes an injectable `time_probe_fn`.
    """
    import time

    from . import probe

    t0 = time.perf_counter()
    n_big = len(probe.run_probe(seed=seed, per_class=per_class))
    t_big = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_small = len(probe.run_probe(seed=seed, per_class=1))
    t_small = time.perf_counter() - t0

    if n_big == n_small:
        return t_big / n_big
    return (t_big - t_small) / (n_big - n_small)


def build_frontier_table(ws: Workspace, B: int = 2000, bootstrap_seed: int = 0,
                          binary_auc_csv=None, time_probe_fn=None):
    """Matched-compute frontier: znorm all_three AUC vs marginal per-decision seconds.

    One row per W present among the loaded ORDERED-arm runs (the shuffle
    control is not part of the primary frontier), plus one "per_flow" row.

    Windowed rows: AUC mean/std/ci_lo/ci_hi from `build_detection_table`
    restricted to (arm="ordered", subset="all_three", scoring="znorm").
    `marginal_s` = mean over that W's ordered runs of
    ``(timing["total_s"] - timing["baseline_s"]) / timing["n_windows"]`` --
    the FRONTIER COST controller ruling (marginal cost net of the baseline
    val-medoid setup, NOT `run_windowed`'s amortized `per_window_s`).
    `baseline_s` = mean fixed one-time val-baseline setup cost, recorded as
    a separate column (not folded into `marginal_s`).

    per_flow row: AUC mean/std/ci_lo/ci_hi read VERBATIM (not recomputed)
    from Phase-4's `binary_auc.csv` (default
    `ws.tables_dir/"rebuild"/"binary_auc.csv"`; override via
    `binary_auc_csv` for tests), filtered to subset="all_three",
    scoring="znorm". `marginal_s` from `time_probe_fn()` (default
    `_time_per_flow_probe`, a REAL timed probe sample -- injectable so
    tests never invoke it); `baseline_s` is NaN (the per-flow arm has no
    equivalent fixed setup cost).
    """
    import numpy as np
    import pandas as pd

    if binary_auc_csv is None:
        binary_auc_csv = ws.tables_dir / "rebuild" / "binary_auc.csv"
    if time_probe_fn is None:
        time_probe_fn = _time_per_flow_probe

    detection = build_detection_table(ws, B=B, bootstrap_seed=bootstrap_seed)
    ordered_znorm = detection[
        (detection["arm"] == "ordered")
        & (detection["subset"] == "all_three")
        & (detection["scoring"] == "znorm")
    ]

    run_groups: dict = {}
    for r in load_all_runs(ws):
        if r["arm"] == "ordered":
            run_groups.setdefault(r["w"], []).append(r)

    rows = []
    for _, row in ordered_znorm.sort_values("w").iterrows():
        w = int(row["w"])
        group_runs = run_groups.get(w, [])
        marginals = [
            (r["timing"]["total_s"] - r["timing"]["baseline_s"]) / r["timing"]["n_windows"]
            for r in group_runs if r["timing"]["n_windows"]
        ]
        baselines = [r["timing"]["baseline_s"] for r in group_runs]
        rows.append({
            "row": str(w), "w": w,
            "auc_mean": float(row["mean"]), "auc_std": float(row["std"]),
            "ci_lo": float(row["ci_lo"]), "ci_hi": float(row["ci_hi"]),
            "marginal_s": float(np.mean(marginals)) if marginals else float("nan"),
            "baseline_s": float(np.mean(baselines)) if baselines else float("nan"),
            "n_runs": int(row["n_runs"]),
        })

    binary_df = pd.read_csv(binary_auc_csv)
    pf = binary_df[(binary_df["subset"] == "all_three")
                    & (binary_df["scoring"] == "znorm")].iloc[0]
    rows.append({
        "row": _PER_FLOW_ROW, "w": None,
        "auc_mean": float(pf["mean"]), "auc_std": float(pf["std"]),
        "ci_lo": float(pf["ci_lo"]), "ci_hi": float(pf["ci_hi"]),
        "marginal_s": float(time_probe_fn()), "baseline_s": float("nan"),
        "n_runs": int(pf["n_seeds"]) if "n_seeds" in pf else None,
    })
    return pd.DataFrame(rows)


# --- Aggregate builder + campaign-report CLI plumbing ------------------------

def build_windowed_tables(ws: Workspace, B: int = 2000, bootstrap_seed: int = 0) -> dict:
    """Build all four windowed-variant report tables from whatever campaign
    runs are present under `results/tables/rebuild/windowed/`.

    Returns ``{"detection", "attribution", "contamination", "frontier"}``,
    one `pd.DataFrame` each (see the individual `build_*` docstrings).
    """
    return {
        "detection": build_detection_table(ws, B=B, bootstrap_seed=bootstrap_seed),
        "attribution": build_windowed_attribution_table(ws, B=B, bootstrap_seed=bootstrap_seed),
        "contamination": build_contamination_table(ws),
        "frontier": build_frontier_table(ws, B=B, bootstrap_seed=bootstrap_seed),
    }


_TABLE_FILENAMES = {
    "detection": "windowed_detection.csv",
    "attribution": "windowed_attribution.csv",
    "contamination": "contamination_curve.csv",
    "frontier": "compute_frontier.csv",
}


def write_windowed_tables(ws: Workspace, tables: dict, B: int = None,
                           bootstrap_seed: int = None) -> dict:
    """Write the four windowed tables to `.../windowed/*.csv` + provenance sidecars."""
    from .provenance import write_provenance

    d = _windowed_dir(ws)
    d.mkdir(parents=True, exist_ok=True)
    params = {"bootstrap": B, "bootstrap_seed": bootstrap_seed}
    paths = {}
    for name, df in tables.items():
        out = d / _TABLE_FILENAMES[name]
        df.to_csv(out, index=False)
        write_provenance(out, params)
        paths[name] = out
    return paths
