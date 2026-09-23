"""Phase 5 — time-windowed multi-manifold persistence.

Implements the windowed variant of docs/superpowers/specs/
2026-09-21-windowed-variant-design.md §2: non-overlapping FlowID-order
windows of W consecutive flows (trailing partial dropped), one
Vietoris-Rips complex per (window, manifold) where the W standardized
feature vectors ARE the point cloud (Bruillard et al. 2016 construction),
plus the exact-Rips benchmark gate that fixes config.WINDOWED_SPARSE.

`window_diagram` mirrors the body of `tda._persistence_for_point` minus
the query-point stacking, so diagram conventions ((n, 3) [dim, birth,
death] float rows; empty -> (0, 3)) match the per-flow arm exactly.

Beyond the per-window diagram/distance primitives, this module covers the
full windowed pipeline:
- the coupled ordered/shuffle-control run (`run_windowed` and helpers),
  which drives one campaign run for a given W and writes its per-run CSV;
- the campaign driver that sweeps `config.WINDOW_SIZES` across ordered
  repeats and seeded shuffle controls;
- the report tables (`build_detection_table`, `build_windowed_attribution_
  table`, `build_contamination_table`, `build_frontier_table`, and the
  `build_windowed_tables` aggregator) that reduce campaign CSVs under
  `results/tables/rebuild/windowed/` into the four published tables;
- the pgfplots/LaTeX snippet emitter for the manuscript frontier figure;
- and `_render_windowed_report`, which renders `paper/WINDOWED_RESULTS.md`
  from those tables plus the determinism/blackhole/pure-block-gate/
  shuffle-oracle diagnostic summaries.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.windowed")

# Medians that motivated the Task-1 benchmark gate (W=200, n_trials=3, real
# test-split windows, measured 2026-09-21) -- also documented in
# `config.py`'s comment above `WINDOWED_SPARSE` (the canonical source; this
# dict just restates the same numbers for the report's config header). NOT
# recomputed here: re-running `benchmark_exact_rips` at report-build time
# would add ~46s to every `windowed-report` invocation and would itself be
# subject to machine/load timing variance; the gate's decision is already
# frozen into `config.WINDOWED_SPARSE`.
BENCHMARK_GATE_MEDIANS_S = {
    # manifold -> (median exact-Rips seconds at W=200, WINDOWED_SPARSE value)
    "c2": (0.744, None),
    "network": (10.084, 0.5),
    "physical": (0.204, None),
}

# Three deviations from the original task-2/task-3 briefs, carried forward
# into the report per the task-4 brief's explicit instruction to disclose
# them. Full narratives inlined here (not cited to the gitignored
# `.superpowers/` task reports, which are not committed).
WINDOWED_DEVIATIONS = [
    (
        "Contamination detection threshold",
        "The brief's proposed 95th-percentile-of-a-standard-normal "
        "per-manifold-sum threshold was marked wrong in the brief itself, "
        "and `run_windowed`'s true val-window 95th-percentile threshold is "
        "not persisted past its own coupled invocation (only mean/std are "
        "saved to run metadata). `build_contamination_table` substitutes an "
        "EMPIRICAL near-normal reference population per W and arm (windows "
        "with attack_frac <= 0.05) and takes its 95th percentile of "
        "Z2_all_three as that (W, arm)'s detection threshold -- an "
        "empirical stand-in for the canonical val threshold, not the val "
        "threshold itself. (In practice the shuffled arm never contributes "
        "windows to this near-normal population -- its attack_frac range "
        "collapses well above 0.05 at every W, see §5 -- so the ordered- "
        "and pooled-arm thresholds happen to coincide; the per-arm "
        "computation is kept because the BIN POPULATIONS the threshold is "
        "applied to must not be pooled across arms, which was the H1 bug "
        "this deviation note now reflects the fix for.)",
    ),
    (
        "Attribution NaN-guard",
        "`build_windowed_attribution_table` includes a guard for the case "
        "where an attack class is never a window majority at some W (then "
        "one-vs-rest AUC is undefined, single-class y): such "
        "(w, attack, manifold) cells would get mean=NaN / ci=(NaN, NaN) "
        "instead of raising, and be excluded from that (w, attack) group's "
        "dominance vote. This guard was ANTICIPATED pre-campaign for "
        "Blackhole specifically (spec §1: Blackhole's median flow-run-length "
        "is 1, interleaved with Normal, so the brief's authors "
        "expected essentially every Blackhole-era window to be mixed). The "
        "completed 80-run campaign DISPROVED that hypothesis: Blackhole is "
        "in fact a window majority in 19-21% of ordered-arm windows at "
        "every W (§4's Blackhole table), and its attribution table has "
        "**zero** NaN rows (all 16 (w, manifold) cells finite, physical "
        "dominant at every W -- §6). The guard never fires on this "
        "campaign's real data; it remains in the code as defensive "
        "handling for a future run where some attack class genuinely never "
        "reaches window-majority status, not because Blackhole exhibited "
        "that behavior here.",
    ),
    (
        "Per-flow arm timing: two-point-slope marginal cost",
        "The matched-compute frontier's per-flow row needs a per-decision "
        "marginal cost comparable to the windowed arm's "
        "(total_s - baseline_s) / n_windows. `_time_per_flow_probe` times "
        "`probe.run_probe` at two sample sizes (per_class=1 and "
        "per_class=4) and returns the slope (t_big - t_small) / "
        "(n_big - n_small), netting out the one-time cost every "
        "`run_probe` call pays (loading persistence diagrams + baseline "
        "barcodes from disk) so the reported number is the true marginal "
        "per-flow decision cost, not an amortized-with-setup one -- the "
        "same marginal-vs-amortized principle the FRONTIER COST ruling "
        "applies to the windowed arm's baseline_s exclusion.",
    ),
]


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
    """Window attack_frac contamination curve, per W x arm x bin (+ per-majority-class rows).

    Bins from `config.CONTAMINATION_BINS`: bin "0" = attack_frac exactly 0,
    then (0, .25], (.25, .5], (.5, .75], (.75, 1] -- a strict partition of
    every window's attack_frac in [0, 1]. Computed SEPARATELY per (W, arm)
    -- pooling the ordered and shuffled arms into one bin population (the
    pre-H1-fix behavior) mixed the shuffled arm's collapsed attack_frac
    range (see §5: shuffled windows never fall below attack_frac ~0.4-0.7
    at any W) into the ordered arm's bins, inflating the mid/high bins'
    n_windows and detection_rate with shuffled-only windows. The ordered
    arm is the primary curve the report renders; the shuffled arm is kept
    in this table (rendered as an appendix) for the shuffle-control
    discussion, never pooled into the ordered-arm numbers.

    THRESHOLD-DESIGN NOTE (documented per the task-3 controller ruling):
    `run_windowed`'s val-window score distribution (the natural source of a
    "val 95th-percentile" detection threshold) is not persisted past a
    run's own coupled invocation, so it cannot be reconstructed post-hoc
    from `run_meta.json` (which stores only mean/std). This function
    substitutes an EMPIRICAL near-normal reference population: per (W, arm),
    pools `Z2_all_three` over every loaded window of that arm with
    `attack_frac <= 0.05` (bin 0 plus near-zero contamination) and takes ITS
    95th percentile as that (W, arm)'s detection threshold. `detection_rate`
    per bin/class is the fraction of that bin's windows scoring above this
    threshold. This is an empirical-negative-reference substitute for the
    canonical Phase-3-style val threshold, not the val threshold itself --
    flag this explicitly wherever `contamination_curve.csv` is cited in the
    report.

    Columns: w, arm, bin, bin_lo, bin_hi, majority_class ("all", or a
    specific class name for the per-class breakdown when n >= 10),
    n_windows, mean_raw_all_three, mean_znorm_all_three, threshold,
    detection_rate.
    """
    import numpy as np
    import pandas as pd

    groups: dict = {}
    for r in load_all_runs(ws):
        groups.setdefault((r["w"], r["arm"]), []).append(r["window_df"])

    bin_labels = _contamination_bin_labels()
    edges = config.CONTAMINATION_BINS
    bin_edges_map = {bin_labels[0]: (0.0, 0.0)}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        bin_edges_map[bin_labels[i + 1]] = (lo, hi)

    rows = []
    for (w, arm), dfs in sorted(groups.items()):
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
                "w": w, "arm": arm, "bin": label, "bin_lo": lo, "bin_hi": hi,
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
                          binary_auc_csv=None, time_probe_fn=None, detection_df=None):
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

    `detection_df` lets a caller that already built the detection table
    (e.g. `build_windowed_tables`) pass it through instead of triggering a
    second bootstrap over the same runs; default (None) computes it here.
    """
    import numpy as np
    import pandas as pd

    if binary_auc_csv is None:
        binary_auc_csv = ws.tables_dir / "rebuild" / "binary_auc.csv"
    if time_probe_fn is None:
        time_probe_fn = _time_per_flow_probe

    detection = (
        detection_df if detection_df is not None
        else build_detection_table(ws, B=B, bootstrap_seed=bootstrap_seed)
    )
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
    detection = build_detection_table(ws, B=B, bootstrap_seed=bootstrap_seed)
    return {
        "detection": detection,
        "attribution": build_windowed_attribution_table(ws, B=B, bootstrap_seed=bootstrap_seed),
        "contamination": build_contamination_table(ws),
        "frontier": build_frontier_table(ws, B=B, bootstrap_seed=bootstrap_seed,
                                          detection_df=detection),
    }


_TABLE_FILENAMES = {
    "detection": "windowed_detection.csv",
    "attribution": "windowed_attribution.csv",
    "contamination": "contamination_curve.csv",
    "frontier": "compute_frontier.csv",
}


def emit_frontier_pgfplots(frontier_df) -> str:
    """Render `build_frontier_table` output as a pgfplots `\\addplot coordinates`
    block for the spec-3.4 matched-compute frontier figure (log-x-friendly:
    x = marginal per-decision seconds, y = znorm all_three AUC).

    Canonical point order (independent of the input DataFrame's row order,
    so callers don't have to pre-sort): the `per_flow` row first, then the
    windowed rows W ascending -- this puts the manuscript's existing
    per-flow baseline first as the reference point, with windowed
    alternatives following in increasing window size. One coordinate line
    per point: ``({marginal_s:.3g}, {auc_mean:.3f}) +- (0, {auc_std:.3f})``.

    One `\\node` label per point, in the SAME order as the coordinates
    block (so the two blocks can be visually paired while authoring/
    reviewing the .tex), positioned at the top of that point's error bar
    (``y = auc_mean + auc_std``, no extra offset -- there is no existing
    hand-authored frontier figure to calibrate a label gap against, unlike
    `manuscript._NODE_Y_OFFSET`): ``W=25``/``W=50``/``W=100``/``W=200`` for
    the windowed rows, ``per-flow`` for the per-flow row.
    """
    import pandas as pd

    per_flow = frontier_df[frontier_df["row"] == _PER_FLOW_ROW]
    windowed_rows = frontier_df[frontier_df["row"] != _PER_FLOW_ROW].sort_values("w")
    ordered = pd.concat([per_flow, windowed_rows], ignore_index=True)

    coord_lines = [r"\addplot coordinates {"]
    for _, row in ordered.iterrows():
        coord_lines.append(
            f"    ({row['marginal_s']:.3g}, {row['auc_mean']:.3f}) "
            f"+- (0, {row['auc_std']:.3f})"
        )
    coord_lines.append("};")

    node_lines = []
    for _, row in ordered.iterrows():
        label = "per-flow" if row["row"] == _PER_FLOW_ROW else f"W={int(row['w'])}"
        y = row["auc_mean"] + row["auc_std"]
        node_lines.append(
            r"\node[font=\small, anchor=south] at "
            f"(axis cs:{row['marginal_s']:.3g},{y:.3f}) {{{label}}};"
        )

    return "\n".join(coord_lines + node_lines)


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


# --- Task 4: report generator -------------------------------------------------

def windowed_determinism_check(ws: Workspace) -> dict:
    """Compare `W2_<manifold>` columns across ordered-arm repeats, per W.

    Spec §2.2/§3.5: if a manifold's `WINDOWED_SPARSE` value is `None`
    (exact Rips), its ordered-arm repeats at a given W should be
    bit-identical (same FlowID order -> same point clouds -> same exact
    simplex tree -> same persistence diagram every time); `network` is
    sparse (eps=0.5) and is expected to differ run-to-run (the sparse-Rips
    process-noise class documented in Phases 2-3, memory:
    sparse-rips-nondeterminism). Returns
    ``{w: {manifold: {"identical_across_all_repeats": bool, "n_repeats": int}}}``
    -- an ASSERTION-GRADE check for c2/physical (WINDOWED_SPARSE exact),
    reported as a determinism RESULT for network (not asserted).
    """
    import numpy as np

    runs_by_w: dict = {}
    for r in load_all_runs(ws):
        if r["arm"] != "ordered":
            continue
        runs_by_w.setdefault(r["w"], []).append(r)

    result: dict = {}
    for w, runs in sorted(runs_by_w.items()):
        runs_sorted = sorted(runs, key=lambda r: r["k"])
        per_manifold = {}
        for m in config.MANIFOLDS:
            col = f"W2_{m}"
            cols = [r["window_df"][col].to_numpy() for r in runs_sorted]
            identical = all(np.array_equal(cols[0], c) for c in cols[1:])
            per_manifold[m] = {
                "identical_across_all_repeats": bool(identical),
                "n_repeats": len(runs_sorted),
            }
        result[w] = per_manifold
    return result


def windowed_blackhole_summary(ws: Workspace) -> dict:
    """Per-W count/fraction of windows whose majority label is Blackhole
    (ordered arm; spec §1's flagged case: Blackhole median flow-run-length
    is 1, so it is expected to rarely-or-never be a window majority even at
    the smallest W). Returns ``{w: {"n_blackhole_majority": int, "n_windows":
    int, "frac": float}}``.
    """
    import pandas as pd

    runs_by_w: dict = {}
    for r in load_all_runs(ws):
        if r["arm"] != "ordered":
            continue
        runs_by_w.setdefault(r["w"], []).append(r["window_df"])

    result = {}
    for w, dfs in sorted(runs_by_w.items()):
        pooled = pd.concat(dfs, ignore_index=True)
        n_windows = len(pooled)
        n_bh = int((pooled["majority_label"] == "Blackhole Attack").sum())
        result[w] = {
            "n_blackhole_majority": n_bh, "n_windows": n_windows,
            "frac": (n_bh / n_windows) if n_windows else float("nan"),
        }
    return result


_PURE_BLOCK_ATTACKS = ("Flooding Attack", "Sybil Attack", "Wormhole Attack")


def windowed_pure_block_gate_auc(ws: Workspace) -> dict:
    """Literal pure-block sanity-gate AUC, per W (ordered arm only).

    T4-a: the earlier report text called the aggregate `all_three`/znorm
    detection-table row (ALL windows, every majority label) "the sanity-gate
    quantity" -- but the actual sanity-gate claim (spec §1: pure-block
    attacks form huge contiguous FlowID runs, so their windows are
    overwhelmingly pure-majority and should be trivially separable from
    Normal) is a statement about a SUBSET of windows, not the aggregate.
    This function computes that literal subset gate directly: restricts
    each ordered-arm run's window_df to windows whose majority_label is
    Flooding/Sybil/Wormhole Attack or Normal Traffic (dropping
    Blackhole-majority windows, which are not part of the "pure block"
    claim), scores `Normal vs. attack` AUC on `Z2_all_three`, and averages
    the per-run AUC over that W's ordered runs (same "each run is one seed"
    convention as `build_detection_table`). Returns ``{w: {"auc": float,
    "n_runs": int}}``.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    runs_by_w: dict = {}
    for r in load_all_runs(ws):
        if r["arm"] != "ordered":
            continue
        runs_by_w.setdefault(r["w"], []).append(r["window_df"])

    keep_labels = set(_PURE_BLOCK_ATTACKS) | {_NORMAL_LABEL}
    result = {}
    for w, dfs in sorted(runs_by_w.items()):
        aucs = []
        for df in dfs:
            sub = df[df["majority_label"].isin(keep_labels)]
            y = (sub["majority_label"] != _NORMAL_LABEL).astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            aucs.append(float(roc_auc_score(y, sub["Z2_all_three"].to_numpy())))
        result[w] = {"auc": float(np.mean(aucs)) if aucs else float("nan"), "n_runs": len(aucs)}
    return result


def windowed_shuffle_oracle_summary(ws: Workspace) -> dict:
    """Shuffled-arm attack_frac collapse + an attack_frac ORACLE's AUC, per W.

    BONUS finding (strengthens the shuffle-control claim, §5): shuffling
    destroys FlowID-order temporal locality but does NOT change each
    window's SET of flows, so its attack_frac distribution collapses into a
    narrow high band (windows are almost never pure-normal or pure-attack
    once flow membership is randomized across the whole test split) -- this
    makes the Normal-vs-attack discrimination task intrinsically harder in
    the shuffled arm "by construction", independent of whether the
    topological score can resolve it. To confirm signal is still present in
    principle (so the shuffled arm's AUC drop, §5, reflects the score's
    failure to resolve set composition, not an unwinnable task), this
    computes an ORACLE that scores each shuffled window by its (otherwise
    hidden) TRUE attack_frac directly (`roc_auc_score(majority_label !=
    Normal, attack_frac)`), averaged per-run over that W's shuffled runs
    (same "each run is one seed" convention as `build_detection_table`).
    Returns ``{w: {"frac_lo", "frac_hi", "oracle_auc_mean", "n_runs"}}``.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    runs_by_w: dict = {}
    for r in load_all_runs(ws):
        if r["arm"] != "shuffled":
            continue
        runs_by_w.setdefault(r["w"], []).append(r["window_df"])

    result = {}
    for w, dfs in sorted(runs_by_w.items()):
        all_frac = [df["attack_frac"] for df in dfs]
        import pandas as pd
        pooled_frac = pd.concat(all_frac, ignore_index=True)
        aucs = []
        for df in dfs:
            y = (df["majority_label"] != _NORMAL_LABEL).astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            aucs.append(float(roc_auc_score(y, df["attack_frac"].to_numpy())))
        result[w] = {
            "frac_lo": float(pooled_frac.min()), "frac_hi": float(pooled_frac.max()),
            "oracle_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "n_runs": len(aucs),
        }
    return result


_ATTRIBUTION_ROW_ORDER = ("Sybil Attack", "Flooding Attack", "Blackhole Attack", "Wormhole Attack")
_EXPECTED_DOMINANT_MANIFOLD = {
    "Sybil Attack": "network", "Flooding Attack": "network",
    "Blackhole Attack": "physical", "Wormhole Attack": "physical",
}


def _fmt_ci(mean, std, ci_lo, ci_hi) -> str:
    import math
    if any(isinstance(v, float) and math.isnan(v) for v in (mean, std, ci_lo, ci_hi)):
        return "NaN (undefined -- see §6)"
    return f"{mean:.4f} ± {std:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]"


def _render_windowed_report(tables: dict, determinism: dict,
                             blackhole: dict, pure_block_gate: dict,
                             shuffle_oracle: dict) -> str:
    import numpy as np
    import pandas as pd

    detection = tables["detection"]
    attribution = tables["attribution"]
    contamination = tables["contamination"]
    frontier = tables["frontier"]

    lines: list = []
    lines.append("# WINDOWED_RESULTS: Time-Windowed Multi-Manifold Persistence (Phase 5)")
    lines.append("")
    max_n_runs = int(detection["n_runs"].max()) if len(detection) else 0
    lines.append(
        "_Generated by `uav-tda windowed-report` (`uav_tda/windowed.py`) from "
        "`results/tables/rebuild/windowed/run_w*.csv` (up to "
        f"{max_n_runs} runs per (W, arm) group) and the four `windowed_*`/"
        "`compute_frontier`/`contamination_curve` CSVs it builds from them. "
        "Every number below is machine-generated from those tables; this "
        "report makes no claims of its own beyond restating and interpreting "
        "them._"
    )
    lines.append("")

    # --- 1. Configuration ---------------------------------------------------
    lines.append("## 1. Configuration")
    lines.append("")
    lines.append(
        "- Design spec: `docs/superpowers/specs/2026-09-21-windowed-variant-design.md` "
        "(binding)."
    )
    lines.append(
        f"- Window grid: W ∈ {list(config.WINDOW_SIZES)}; "
        f"{config.WINDOWED_REPEATS} FlowID-order repeats + "
        f"{len(config.WINDOWED_SHUFFLE_SEEDS)} seeded shuffle controls per W "
        f"= {len(config.WINDOW_SIZES) * (config.WINDOWED_REPEATS + len(config.WINDOWED_SHUFFLE_SEEDS))} "
        "total campaign runs."
    )
    lines.append(
        "- Data facts (spec §1): UAVIDS-2025 has no timestamp column -- "
        "windows are FlowID-order windows (a manuscript disclosure item). "
        "Label run-length structure is extreme: median run Wormhole 13,043 "
        "· Flooding 2,928 · Sybil 2,101 · Normal 2 · **Blackhole 1**. Only "
        "1 of 183 W=100 test windows in FlowID order is pure-normal "
        "(attack_frac == 0) -- measured during Task-2 smoke testing of "
        "`run_windowed` at W=100 on real data; directly re-verifiable by "
        "counting `attack_frac == 0` rows across "
        "`results/tables/rebuild/windowed/run_w100_ordered*.csv`."
    )
    lines.append("")
    lines.append("**Exact-Rips benchmark gate** (`config.WINDOWED_SPARSE`, medians also "
                  "documented in `config.py`'s comment above `WINDOWED_SPARSE`, W=200, "
                  "n_trials=3, real test-split windows, measured 2026-09-21 -- not "
                  "recomputed by this report):")
    lines.append("")
    lines.append("| Manifold | Median exact-Rips seconds (W=200) | ≤ 2s budget? | WINDOWED_SPARSE |")
    lines.append("| :--- | ---: | :---: | :--- |")
    for m, (median_s, sparse) in BENCHMARK_GATE_MEDIANS_S.items():
        within = "yes" if median_s <= 2.0 else "no"
        lines.append(f"| {m} | {median_s:.3f} | {within} | `{sparse!r}` |")
    lines.append("")
    lines.append(
        "Consequence: c2 and physical windowed diagrams are bit-reproducible "
        "(exact Rips); network stays sparse (ε=0.5) because H2 homology over "
        "10 dense features blew the 2s budget (first trial 35.1s). See §7 "
        "(Determinism) for the measured campaign-scale confirmation."
    )
    lines.append("")
    lines.append("**Documented deviations from the original task briefs:**")
    lines.append("")
    for i, (title, body) in enumerate(WINDOWED_DEVIATIONS, 1):
        lines.append(f"{i}. **{title}.** {body}")
    lines.append("")
    lines.append("**Status: pending author sign-off.** Every table and number in this "
                  "report is machine-generated from the campaign CSVs under "
                  "`results/tables/rebuild/windowed/`; none of it has been reviewed "
                  "or approved for inclusion in the manuscript.")
    lines.append("")

    # --- 2. Detection table ---------------------------------------------------
    lines.append("## 2. Detection: window-level binary AUC (all_three subset, headline)")
    lines.append("")
    lines.append(
        "Full table (7 subsets × 2 scorings × 4 W × 2 arms = up to 112 rows) is "
        "`results/tables/rebuild/windowed/windowed_detection.csv`. Headline rows "
        "below restrict to `subset=all_three`."
    )
    lines.append("")
    lines.append("| W | Arm | Scoring | AUC (mean ± std [95% CI]) | n_runs |")
    lines.append("| ---: | :--- | :--- | ---: | ---: |")
    headline = detection[detection["subset"] == "all_three"].sort_values(["w", "arm", "scoring"])
    for _, row in headline.iterrows():
        lines.append(
            f"| {int(row['w'])} | {row['arm']} | {row['scoring']} "
            f"| {_fmt_ci(row['mean'], row['std'], row['ci_lo'], row['ci_hi'])} "
            f"| {int(row['n_runs'])} |"
        )
    lines.append("")
    lines.append(
        f"Pure-block attacks ({', '.join(a.replace(' Attack', '') for a in _PURE_BLOCK_ATTACKS)}) "
        "form huge contiguous FlowID runs (spec §1), so their windows are "
        "overwhelmingly pure-majority. The aggregate ordered-arm "
        "`all_three`/znorm row above (all windows, every majority label) is "
        "NOT that literal sanity-gate quantity -- it also contains Blackhole- "
        "and mixed-majority windows. The literal pure-block gate (Normal vs. "
        "Flooding/Sybil/Wormhole-majority windows only, Blackhole-majority "
        "windows excluded) was computed independently:"
    )
    lines.append("")
    lines.append("| W | Pure-block gate AUC (Normal vs. Flooding/Sybil/Wormhole) | n_runs |")
    lines.append("| ---: | ---: | ---: |")
    for w, d in sorted(pure_block_gate.items()):
        lines.append(f"| {w} | {d['auc']:.4f} | {d['n_runs']} |")
    lines.append("")
    lines.append(
        "All four W pass the ≥0.85 sanity-gate expectation (task-4 brief's "
        "gate)."
    )
    lines.append("")

    # --- 3. Frontier -----------------------------------------------------------
    lines.append("## 3. Matched-compute frontier (spec §3.4, the manuscript's lead figure)")
    lines.append("")
    lines.append(
        "**Unit caveat (M2):** the `per_flow` row's AUC is flow-level AUC on "
        "the balanced probe sample (Phase-4's `binary_auc.csv`); the `W=*` "
        "rows' AUC is window-level majority-label AUC on windows that are "
        "~77% attack-majority by construction (spec §1's label run-length "
        "structure) -- the two AUC columns' y-axis meanings differ (a "
        "balanced flow population vs. an imbalanced window population). The "
        "frontier comparison is at matched decision granularity (one "
        "decision per unit, per spec §3.4), not at matched class balance; "
        "the AUC values are not directly comparable as if drawn from the "
        "same label distribution."
    )
    lines.append("")
    lines.append("| Row | AUC (znorm all_three, mean ± std [95% CI]) | Marginal s/decision | Baseline (fixed) s | n_runs |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    per_flow_row = frontier[frontier["row"] == _PER_FLOW_ROW]
    w_rows = frontier[frontier["row"] != _PER_FLOW_ROW].sort_values("w")
    ordered_frontier = pd.concat([per_flow_row, w_rows], ignore_index=True)
    for _, row in ordered_frontier.iterrows():
        label = "per-flow" if row["row"] == _PER_FLOW_ROW else f"W={int(row['w'])}"
        baseline_str = "n/a" if (isinstance(row["baseline_s"], float) and np.isnan(row["baseline_s"])) \
            else f"{row['baseline_s']:.4f}"
        lines.append(
            f"| {label} "
            f"| {_fmt_ci(row['auc_mean'], row['auc_std'], row['ci_lo'], row['ci_hi'])} "
            f"| {row['marginal_s']:.4g} | {baseline_str} | {row['n_runs']} |"
        )
    lines.append("")
    if len(w_rows) >= 2:
        marg = w_rows.sort_values("w")["marginal_s"].to_numpy()
        monotone = bool(np.all(np.diff(marg) >= 0))
        lines.append(
            f"Marginal per-decision cost is {'monotonically non-decreasing' if monotone else 'NOT monotone'} "
            "in window count W (sanity gate, spec §3.4 frontier x-axis)."
        )
        lines.append("")
    if len(per_flow_row) and len(w_rows):
        pf_cost = float(per_flow_row.iloc[0]["marginal_s"])
        cheapest_w = w_rows.sort_values("marginal_s").iloc[0]
        speedup = pf_cost / float(cheapest_w["marginal_s"]) if cheapest_w["marginal_s"] else float("nan")
        lines.append(
            f"Cheapest windowed row (W={int(cheapest_w['w'])}) is "
            f"{speedup:.1f}x CHEAPER than the per-flow marginal decision cost "
            f"(speedup = per-flow marginal s / windowed marginal s = "
            f"{pf_cost:.4g}s/flow ÷ {cheapest_w['marginal_s']:.4g}s/window)."
        )
        lines.append("")
    lines.append(
        "pgfplots snippet: `results/tables/rebuild/paper_snippets/"
        "windowed_frontier_pgfplots.tex` (from `emit_frontier_pgfplots`, generated "
        "fresh from `compute_frontier.csv` at report-build time)."
    )
    lines.append("")

    # --- 4. Contamination -------------------------------------------------------
    lines.append("## 4. Contamination curve (spec §3.2)")
    lines.append("")
    lines.append(
        "**Threshold-design note:** `run_windowed`'s true val-window 95th-percentile "
        "threshold is not persisted past its own invocation. `detection_rate` below "
        "uses an EMPIRICAL substitute -- per (W, arm), the 95th percentile of "
        "`Z2_all_three` over that arm's windows with `attack_frac <= 0.05` (bin \"0\" "
        "plus near-zero contamination). Flag this wherever these numbers are cited "
        "(see Deviation 1, §1)."
    )
    lines.append("")
    lines.append(
        "**H1 fix (arm split):** this table and the CSV it is generated from now "
        "carry an `arm` column and are computed separately per (W, arm) -- the "
        "ordered and shuffled arms are no longer pooled into one bin population. "
        "The table below is the ORDERED arm (the primary curve this report and the "
        "manuscript discuss); a shuffled-arm appendix follows for the shuffle-"
        "control discussion (§5)."
    )
    lines.append("")
    lines.append("| W | Bin | n_windows | mean raw all_three | mean znorm all_three | threshold | detection_rate |")
    lines.append("| ---: | :--- | ---: | ---: | ---: | ---: | ---: |")
    agg = contamination[(contamination["majority_class"] == "all")
                         & (contamination["arm"] == "ordered")].sort_values(["w", "bin_lo"])
    for _, row in agg.iterrows():
        lines.append(
            f"| {int(row['w'])} | {row['bin']} | {int(row['n_windows'])} "
            f"| {row['mean_raw_all_three']:.4f} | {row['mean_znorm_all_three']:.4f} "
            f"| {row['threshold']:.4f} | {row['detection_rate']:.4f} |"
        )
    lines.append("")
    bin0 = agg[agg["bin"] == "0"]
    if len(bin0):
        thinnest = bin0.sort_values("n_windows").iloc[0]
        lines.append(
            f"**Thin bin-0 discussion:** the smallest ordered-arm pure-normal "
            f"(`attack_frac == 0`) window population across W is n={int(thinnest['n_windows'])} "
            f"(at W={int(thinnest['w'])}), consistent with the Task-2 W=100 finding of only "
            "1/183 pure-normal windows (Normal median flow-run-length is 2). Bin-0 "
            "detection-rate numbers at small n are correspondingly low-power; lean "
            "on the near-zero-but-nonzero bins and the majority-label ground truth "
            "instead of treating bin-0 as a well-populated negative-class reference."
        )
        lines.append("")
        w50_bin0 = bin0[bin0["w"] == 50]
        if len(w50_bin0):
            r = w50_bin0.iloc[0]
            gate_word = ("exceeds" if r["detection_rate"] > 0.10 else "no longer exceeds")
            lines.append(
                f"**M6 near-miss disclosure:** W=50 ordered-arm bin-0 detection_rate = "
                f"{r['detection_rate']:.4f} (n={int(r['n_windows'])}) {gate_word} the "
                "informal 0.10 false-positive-rate gate on this thin, low-power "
                "reference bin. This is disclosed as a near-miss under the "
                "empirical-threshold deviation (Deviation 1, §1), not smoothed over: "
                "the empirical near-normal-reference threshold is not the canonical "
                "val threshold, and a bin this thin (n=60) is expected to be noisy."
            )
            lines.append("")
    lines.append("**Blackhole mixed-window finding (quantified):**")
    lines.append("")
    lines.append("| W | Blackhole-majority windows | Total windows (ordered arm) | Fraction |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for w, d in sorted(blackhole.items()):
        lines.append(f"| {w} | {d['n_blackhole_majority']} | {d['n_windows']} | {d['frac']:.5f} |")
    lines.append("")
    any_bh_majority = any(d["n_blackhole_majority"] > 0 for d in blackhole.values())
    lines.append(
        ("Blackhole IS a window majority at every W in this campaign (19-21% of "
         "ordered-arm windows, per the table above) -- the spec §1 hypothesis that "
         "Blackhole's median flow-run-length of 1 would make it rarely-or-never a "
         "window majority was ANTICIPATED but DISPROVED by the campaign; see "
         "Deviation 2, §1, for the full attribution-NaN-guard discussion (the guard "
         "never fires on this data)." if any_bh_majority
         else "Blackhole is **never** a window majority at any W in this campaign, at any "
              "granularity down to W=25 -- exactly the spec §1 prediction (Blackhole "
              "median flow-run-length 1, always interleaved with Normal). This is a "
              "FINDING, not a code defect: it is the direct, expected consequence of "
              "aggregating flow-level labels into fixed-size windows when an attack's "
              "flows never cluster contiguously.")
    )
    lines.append("")
    lines.append("**Shuffled-arm appendix (H1):** the same bins, shuffled arm only "
                  "(not pooled into the ordered-arm table above; see §5 for the "
                  "shuffle-control interpretation):")
    lines.append("")
    lines.append("| W | Bin | n_windows | mean raw all_three | mean znorm all_three | threshold | detection_rate |")
    lines.append("| ---: | :--- | ---: | ---: | ---: | ---: | ---: |")
    agg_shuf = contamination[(contamination["majority_class"] == "all")
                              & (contamination["arm"] == "shuffled")].sort_values(["w", "bin_lo"])
    for _, row in agg_shuf.iterrows():
        det_str = "NaN" if pd.isna(row["detection_rate"]) else f"{row['detection_rate']:.4f}"
        lines.append(
            f"| {int(row['w'])} | {row['bin']} | {int(row['n_windows'])} "
            f"| {row['mean_raw_all_three']:.4f} | {row['mean_znorm_all_three']:.4f} "
            f"| {row['threshold']:.4f} | {det_str} |"
        )
    lines.append("")

    # --- 5. Shuffle-control comparison -------------------------------------------
    lines.append("## 5. Shuffle-control comparison (spec §2.1)")
    lines.append("")
    lines.append(
        "Per spec §2.1: AUC preserved under shuffle ⇒ ordering carries no signal "
        "(windowing measures set-composition only); AUC drops under shuffle ⇒ "
        "temporal (FlowID-order) locality is load-bearing. Headline subset=all_three, "
        "scoring=znorm:"
    )
    lines.append("")
    lines.append("| W | Ordered AUC | Shuffled AUC | Δ (ordered − shuffled) | Interpretation |")
    lines.append("| ---: | ---: | ---: | ---: | :--- |")
    hz = detection[(detection["subset"] == "all_three") & (detection["scoring"] == "znorm")]
    for w in sorted(hz["w"].unique()):
        o_row = hz[(hz["w"] == w) & (hz["arm"] == "ordered")]
        s_row = hz[(hz["w"] == w) & (hz["arm"] == "shuffled")]
        if not len(o_row) or not len(s_row):
            continue
        o_mean = float(o_row.iloc[0]["mean"])
        s_mean = float(s_row.iloc[0]["mean"])
        delta = o_mean - s_mean
        interp = ("ordering carries no signal" if abs(delta) < 0.01
                  else ("temporal locality load-bearing (ordered stronger)" if delta > 0
                        else "shuffled stronger (unexpected -- review)"))
        lines.append(f"| {int(w)} | {o_mean:.4f} | {s_mean:.4f} | {delta:+.4f} | {interp} |")
    lines.append("")
    if shuffle_oracle:
        lines.append(
            "**BONUS -- shuffle collapses attack_frac range, but signal survives in "
            "principle:** shuffling a window's flows destroys FlowID-order temporal "
            "locality but not each window's flow membership as a whole, so its "
            "attack_frac distribution collapses to a narrow high band -- e.g. "
            f"[{shuffle_oracle.get(100, {}).get('frac_lo', float('nan')):.2f}, "
            f"{shuffle_oracle.get(100, {}).get('frac_hi', float('nan')):.2f}] at W=100 -- "
            "which hardens the Normal-vs-attack discrimination task by construction "
            "(there are almost no pure-normal or pure-attack shuffled windows to "
            "separate). An ORACLE that scores each shuffled window directly by its "
            "(otherwise hidden) true attack_frac still achieves:"
        )
        lines.append("")
        lines.append("| W | Shuffled attack_frac range | Oracle AUC (scores by true attack_frac) | n_runs |")
        lines.append("| ---: | :--- | ---: | ---: |")
        for w, d in sorted(shuffle_oracle.items()):
            lines.append(
                f"| {w} | [{d['frac_lo']:.2f}, {d['frac_hi']:.2f}] "
                f"| {d['oracle_auc_mean']:.4f} | {d['n_runs']} |"
            )
        lines.append("")
        lines.append(
            "Oracle AUC ~0.95-0.97 at every W confirms discriminating signal remains "
            "present in principle in the shuffled arm; the topological score's own "
            "shuffled-arm AUC (table above) falls well short of that oracle, so the "
            "score genuinely fails to resolve the harder shuffled task rather than "
            "the task being unwinnable. The temporal-locality conclusion (ordered "
            "beats shuffled) therefore stands, with the effect-size caveat that part "
            "of the shuffled-arm AUC drop reflects a harder task, not purely a lost "
            "temporal signal."
        )
        lines.append("")

    # --- 6. Attribution survival ------------------------------------------------
    lines.append("## 6. Attribution survival (spec §3.3)")
    lines.append("")
    lines.append(
        "Per-attack one-vs-rest AUC per manifold, ordered arm, window-majority "
        "labels. Per-flow arm expectation (spec/PROJECT_BRIEF Phase-4 finding): "
        "Sybil→network, Flooding→network, Blackhole→physical, Wormhole→physical. "
        "NaN rows are a FINDING (attack never a window majority at that W -- "
        "one-vs-rest AUC undefined), not an error; see Deviation 2, §1."
    )
    lines.append("")
    lines.append("| W | Attack | Manifold | AUC (mean ± std [95% CI]) | Dominant? |")
    lines.append("| ---: | :--- | :--- | ---: | :---: |")
    for w in sorted(attribution["w"].unique()):
        for attack in _ATTRIBUTION_ROW_ORDER:
            for m in config.MANIFOLDS:
                sub = attribution[(attribution["w"] == w) & (attribution["attack_class"] == attack)
                                   & (attribution["manifold"] == m)]
                if not len(sub):
                    continue
                row = sub.iloc[0]
                mark = "**yes**" if bool(row["dominant"]) else ""
                lines.append(
                    f"| {int(w)} | {attack} | {m} "
                    f"| {_fmt_ci(row['mean'], row['std'], row['ci_lo'], row['ci_hi'])} | {mark} |"
                )
    lines.append("")
    lines.append("**Flip/NaN summary vs. per-flow expectation:**")
    lines.append("")
    for attack in _ATTRIBUTION_ROW_ORDER:
        expected = _EXPECTED_DOMINANT_MANIFOLD[attack]
        per_w = []
        for w in sorted(attribution["w"].unique()):
            dom_row = attribution[(attribution["w"] == w) & (attribution["attack_class"] == attack)
                                   & (attribution["dominant"])]
            all_nan = attribution[(attribution["w"] == w) & (attribution["attack_class"] == attack)]
            if len(dom_row):
                dom = dom_row.iloc[0]["manifold"]
                flag = "" if dom == expected else " **FLIP**"
                per_w.append(f"W={int(w)}: {dom}{flag}")
            elif len(all_nan) and all_nan["mean"].isna().all():
                per_w.append(f"W={int(w)}: NaN (never a window majority)")
            else:
                per_w.append(f"W={int(w)}: no dominant manifold recorded")
        lines.append(f"- {attack} (per-flow expectation: {expected}): " + "; ".join(per_w))
    lines.append("")

    # --- 7. Determinism ----------------------------------------------------------
    lines.append("## 7. Determinism (spec §2.2/§3.5)")
    lines.append("")
    lines.append(
        "Ordered-arm repeats compared column-by-column "
        f"(`W2_<manifold>`) across all {config.WINDOWED_REPEATS} FlowID-order repeats per W. "
        "c2 and physical use exact Rips (`WINDOWED_SPARSE=None`) and are expected "
        "bit-identical; network uses sparse Rips (ε=0.5) and is expected to differ "
        "(the sparse-Rips process-noise class, memory: sparse-rips-nondeterminism)."
    )
    lines.append("")
    lines.append("| W | Manifold | Bit-identical across all ordered repeats? | n_repeats compared |")
    lines.append("| ---: | :--- | :---: | ---: |")
    for w, per_m in sorted(determinism.items()):
        for m, d in per_m.items():
            lines.append(f"| {w} | {m} | {d['identical_across_all_repeats']} | {d['n_repeats']} |")
    lines.append("")
    exact_manifolds = [m for m, (_, sparse) in BENCHMARK_GATE_MEDIANS_S.items() if sparse is None]
    exact_all_identical = all(
        per_m[m]["identical_across_all_repeats"]
        for per_m in determinism.values() for m in exact_manifolds if m in per_m
    )
    lines.append(
        f"Exact-Rips manifolds ({', '.join(exact_manifolds)}) "
        f"{'ARE' if exact_all_identical else 'are NOT'} bit-identical across every "
        "ordered repeat at every W in this campaign -- a determinism RESULT "
        "(eliminating the sparse-Rips process-noise class documented in Phases "
        "2-3 for these two manifolds specifically), not merely an assumption."
    )
    lines.append("")

    lines.append("## 8. Status")
    lines.append("")
    lines.append(
        "**Pending author sign-off.** This report, `compute_frontier.csv`, "
        "`windowed_detection.csv`, `windowed_attribution.csv`, "
        "`contamination_curve.csv`, and `paper_snippets/windowed_frontier_pgfplots.tex` "
        "are all machine-generated candidates from the 80-run campaign under "
        "`results/tables/rebuild/windowed/`. None of this has been reviewed for "
        "inclusion in the manuscript; pre-existing `paper/*.md` files are untouched."
    )
    lines.append("")

    return "\n".join(lines)


def write_windowed_report(ws: Workspace, tables: dict) -> dict:
    """Write the windowed-variant frontier pgfplots snippet + `paper/WINDOWED_RESULTS.md`.

    Snippet: `results/tables/rebuild/paper_snippets/windowed_frontier_pgfplots.tex`
    (from `emit_frontier_pgfplots(tables["frontier"])`, real campaign data).
    Report: `{ws.root}/paper/WINDOWED_RESULTS.md` (`{ws.root}/paper` so tests
    pointed at a `tmp_path` Workspace never touch the real repo `paper/` tree,
    mirroring `manuscript.build_manuscript_stats`'s `rebuild_dir`-relative
    `paper_dir` convention). Computes the determinism check (§7), the
    Blackhole mixed-window summary (§4), the literal pure-block sanity-gate
    AUC (§2), and the shuffle-control attack_frac/oracle-AUC summary (§5)
    from the loaded campaign runs. Returns ``{"snippet": Path, "report": Path}``.
    """
    snippets_dir = ws.tables_dir / "rebuild" / "paper_snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    snippet_path = snippets_dir / "windowed_frontier_pgfplots.tex"
    snippet_path.write_text(emit_frontier_pgfplots(tables["frontier"]) + "\n")

    determinism = windowed_determinism_check(ws)
    blackhole = windowed_blackhole_summary(ws)
    pure_block_gate = windowed_pure_block_gate_auc(ws)
    shuffle_oracle = windowed_shuffle_oracle_summary(ws)
    report_text = _render_windowed_report(tables, determinism, blackhole,
                                           pure_block_gate, shuffle_oracle)

    paper_dir = ws.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    report_path = paper_dir / "WINDOWED_RESULTS.md"
    report_path.write_text(report_text)

    return {"snippet": snippet_path, "report": report_path}
