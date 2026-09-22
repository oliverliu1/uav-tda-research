"""Phase 6 Track A: sharded, resumable high-precision-Wasserstein-2 runner.

Replaces the probe approximation (top-K truncation + delta=0.2 relative
error) with definitive full-diagram, high-precision (delta<=0.01) hera
Wasserstein-2 on the full test split. Reuses `probe._load_baseline_barcodes`,
`probe._slice_dim`, and `probe._w2_with_timeout` (probe.py itself is
read-only / locked).

Same-run baseline coupling (Global Constraint): the per-manifold baseline
barcodes are computed ONCE and persisted under `exact_dir` (npy + manifest +
provenance). Every subsequent call in the SAME campaign, and every resume
after interruption, must LOAD the persisted baselines rather than
recomputing them -- `probe._load_baseline_barcodes` runs sparse-Rips on the
500 reference points with no seed control, so recomputing would silently
change the baseline realization that val Normal-znorm-stats and test
distances are scored against.

CAMPAIGN DEFINITION (revised 2026-09-22, controller ruling after a T3
pre-launch re-benchmark -- see paper/EXACT_RESULTS.md config header and
`.superpowers/sdd/2026-09-22-plan6-exact-w2-latency/task-3-report.md` for the
full record): hera's true exact mode (`delta=0.0`, the exact auction-LP
solution rather than a (1+delta)-approximate one) was measured on 13 real
test flows across all 3 manifolds and timed out on the 120s fork-timeout for
AT LEAST ONE homology dim of EVERY sampled flow (13/13), driving the
projected full-campaign wall clock to ~30 days at `--n-jobs 7` -- hera's
exact auction-LP is intractable on these diagrams at any practical timeout.
The alternative exact backend (`gudhi.wasserstein.wasserstein_distance`, an
assignment/LAP-based exact solver) was evaluated next and found unusable in
this environment without adding a new dependency: it has an unconditional
top-level `import ot` (POT) with no scipy-only fallback in the installed
gudhi 3.11.0, POT is not installed, and it was not installed per the
controller's explicit instruction and the plan's "no new dependencies"
constraint.

The campaign therefore targets **high-precision, NOT literally exact**
Wasserstein-2: hera `order=2.0, internal_p=2.0, delta=0.01` (hera's own
default tolerance -- a 1%-relative-error bound) as the FIRST attempt, with a
`delta=0.05` retry-once on timeout (both under the existing 120s fork-timeout
machinery, reused unmodified). Confirmed by re-benchmark on real diagrams
with the SAME persisted baselines: delta=0.01 is fast and never timed out
(mean s/flow c2=1.53, network=1.97, physical=1.18; max 3.36s, all well
inside the 120s budget) -- consistent with the paper's original ~3.3s/flow
estimate. This removes the probe's per-class sampling and top-K(=50)
truncation and tightens the relative-error tolerance 20x (0.2 -> 0.01)
relative to the probe's production config; `insensitivity_check` (below)
quantifies the residual delta=0.01-vs-delta=0.05 AUC sensitivity on a test
subsample so the report can state the residual approximation's measured
impact rather than merely assert it is small.

CAMPAIGN CALL PATH (revised 2026-09-22, second controller intervention): the
first real campaign launch nested `probe._w2_with_timeout`'s per-call
`fork()` inside joblib/loky worker processes, which is unstable on macOS
(loky workers silently died; only ~2/7 stayed alive; zero shards completed
in 47 minutes). `run_shard` now calls `direct_w2_flow` (below) -- a direct,
unwrapped in-process hera call mirroring `pipeline.py`'s production
`_wasserstein_for_flow` -- instead of `exact_w2_flow`. `exact_w2_flow`'s
fork-timeout+retry machinery is kept (own tests, optional/interactive use)
but is not on the campaign's hot path; shard-level resumability via
`manifest.json` is the hang-recovery story instead.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, manuscript, metrics, probe
from .metrics import MANIFOLD_SUBSETS, NORMAL
from .provenance import write_provenance
from .workspace import Workspace

log = logging.getLogger("uav_tda.exact")

MANIFEST_NAME = "manifest.json"
BASELINES_MANIFEST_NAME = "baselines_manifest.json"

# paper/MULTI_SEED_VARIANCE.md, 3-seed published extended-abstract numbers
# (raw scoring only -- the published abstract predates znorm scoring).
PUBLISHED3_RAW = {
    "c2_only": (0.7488, 0.0269),
    "network_only": (0.7610, 0.0163),
    "physical_only": (0.6114, 0.0152),
    "c2_network": (0.8304, 0.0211),
    "c2_physical": (0.7542, 0.0339),
    "network_physical": (0.8594, 0.0043),
    "all_three": (0.8577, 0.0276),
}


# ---------------------------------------------------------------------------
# Baseline persistence (resume-critical: compute once, load thereafter)
# ---------------------------------------------------------------------------


def persist_baselines(ws: Workspace, exact_dir: Path) -> dict:
    """Compute per-manifold per-dim baseline barcodes ONCE and persist them.

    Saves each (manifold, dim) slice as `baselines_{m}_dim{k}.npy` plus a
    `baselines_manifest.json` (shapes, creation date) and a provenance
    sidecar on the manifest. Returns the freshly computed baselines dict
    (same shape as `probe._load_baseline_barcodes`'s return value).
    """
    exact_dir.mkdir(parents=True, exist_ok=True)
    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    baselines = probe._load_baseline_barcodes(max_edge_lengths)

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifolds": {},
    }
    for manifold, per_dim in baselines.items():
        manifest["manifolds"][manifold] = {}
        for dim, arr in per_dim.items():
            fname = f"baselines_{manifold}_dim{dim}.npy"
            np.save(exact_dir / fname, np.asarray(arr))
            manifest["manifolds"][manifold][str(dim)] = {
                "file": fname,
                "shape": list(np.asarray(arr).shape),
            }

    manifest_path = exact_dir / BASELINES_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_provenance(manifest_path, {"max_edge_lengths": max_edge_lengths})
    return baselines


def load_baselines(exact_dir: Path) -> dict:
    """Reload the persisted baselines dict exactly as `persist_baselines` saved it."""
    manifest = json.loads((exact_dir / BASELINES_MANIFEST_NAME).read_text())
    baselines: dict = {}
    for manifold, per_dim in manifest["manifolds"].items():
        baselines[manifold] = {
            int(dim_str): np.load(exact_dir / info["file"])
            for dim_str, info in per_dim.items()
        }
    return baselines


def ensure_baselines(ws: Workspace, exact_dir: Path) -> dict:
    """Load persisted baselines if present, else compute-and-persist once.

    RESUME-CRITICAL: every call after the first campaign run must take the
    LOAD branch, never recompute -- see module docstring.
    """
    if (exact_dir / BASELINES_MANIFEST_NAME).exists():
        return load_baselines(exact_dir)
    return persist_baselines(ws, exact_dir)


# ---------------------------------------------------------------------------
# Per-flow exact W2 (per manifold, summed over homology dims)
# ---------------------------------------------------------------------------


# High-precision campaign definition (revised 2026-09-22 -- see module
# docstring): hera delta=0.0 (true exact) is intractable at 120s on real
# diagrams (13/13 sampled flows timed out); gudhi.wasserstein (POT-backed
# exact LAP) is unavailable without a new, disallowed dependency. The first
# attempt uses hera's own default tolerance; on timeout, retry once at a
# looser tolerance.
PRIMARY_DELTA = 0.01
RETRY_DELTA = 0.05


def exact_w2_flow(
    diagram: np.ndarray,
    baselines_m: dict,
    max_edge: float,
    max_hom_dim: int,
    timeout_s: float = 120.0,
) -> "tuple[float, int, bool]":
    """High-precision (delta<=0.05) per-dim Wasserstein-2 distance, summed.

    NOT literally exact -- see module docstring for why (`delta=0.0` hera
    exact-LP is intractable at 120s on these diagrams; the POT-backed exact
    LAP alternative, `gudhi.wasserstein`, is unavailable without a new
    dependency). Per homology dim: `probe._slice_dim` both sides, then
    `probe._w2_with_timeout(order=2.0, internal_p=2.0, delta=PRIMARY_DELTA,
    ...)` (hera's own default 1%-relative-error tolerance). If that call
    times out (NaN), retry ONCE with `delta=RETRY_DELTA` (5%) under the same
    timeout; if the retry also times out, the dim contributes 0.0 and is
    counted in `n_timeouts`.

    Returns ``(total, n_timeouts, approx_flag)`` where `approx_flag` is True
    iff ANY dim used the `RETRY_DELTA` retry at all -- whether or not that
    retry itself succeeded (a successful 5% retry is a looser bound than the
    campaign's primary 1% tolerance).
    """
    total = 0.0
    n_timeouts = 0
    approx_flag = False
    for dim in range(max_hom_dim + 1):
        flow_bd = probe._slice_dim(diagram, dim, max_edge)
        base_bd = baselines_m[dim]
        w = probe._w2_with_timeout(
            flow_bd, base_bd, order=2.0, internal_p=2.0, delta=PRIMARY_DELTA, timeout_sec=timeout_s,
        )
        if np.isfinite(w):
            total += w
            continue
        approx_flag = True
        w_retry = probe._w2_with_timeout(
            flow_bd, base_bd, order=2.0, internal_p=2.0, delta=RETRY_DELTA, timeout_sec=timeout_s,
        )
        if np.isfinite(w_retry):
            total += w_retry
        else:
            n_timeouts += 1
    return float(total), n_timeouts, approx_flag


def direct_w2_flow(
    diagram: np.ndarray,
    baselines_m: dict,
    max_edge: float,
    max_hom_dim: int,
    delta: float = PRIMARY_DELTA,
    row_idx: "int | None" = None,
    manifold: "str | None" = None,
) -> "tuple[float, int, bool]":
    """Direct in-process high-precision (delta<=0.01) W2, summed over dims.

    THE CAMPAIGN PATH (revised 2026-09-22, controller intervention). NO
    fork-timeout wrapper: `probe._w2_with_timeout`'s per-call
    `multiprocessing.get_context("fork")` nested INSIDE an already-parallel
    joblib/loky worker process was found unstable on macOS in the first
    real campaign launch -- loky workers silently died ("A worker stopped
    while some jobs were given to the executor"), leaving only ~2/7 workers
    alive and ZERO shards completed after 47 minutes. This mirrors
    `pipeline.py`'s `_wasserstein_for_flow` (the May production pass's own
    design, and the `w2_timeout is None` branch already used by
    `probe._probe_distances`): import the hera backend directly inside the
    worker and call it with no further forking.

    `exact_w2_flow`'s fork-timeout+delta-retry machinery is KEPT (its own
    tests still cover it) for optional/interactive use, but is no longer
    the campaign's per-flow call -- `run_shard` calls this function instead.
    The campaign's hang-recovery story is shard-level resumability
    (`manifest.json`): if a worker ever hangs, kill the campaign process and
    relaunch `uav-tda exact`, which resumes from the last completed shard
    against the SAME persisted baselines (`ensure_baselines` always takes
    the load branch on resume). Measured `delta=0.01` hang risk on real
    diagrams is empirically ~0 (0/13 sampled flows timed out at 120s in the
    T3 pre-launch re-benchmark).

    Per-dim hera calls are wrapped in try/except (2026-09-22, third
    intervention -- isolation probe for the worker-churn investigation): a
    Python-level exception from `gudhi.hera.wasserstein_distance` is caught,
    logged (with `row_idx`/`manifold`/`dim` context when given) via
    `uav_tda.exact`'s logger, contributes 0.0 to that dim, and is counted
    exactly like a timeout (`n_timeouts` incremented, `approx_flag=True`) --
    this is diagnostic AND safe: it distinguishes a catchable Python
    exception (caught here, shard keeps going) from an uncatchable native
    crash (segfault -- would still kill the worker process outright, since
    no Python exception handler can intercept that).

    Returns ``(total, n_timeouts, approx_flag)`` for row-schema
    compatibility with `exact_w2_flow`.
    """
    from gudhi.hera import wasserstein_distance as wdist  # noqa: PLC0415

    total = 0.0
    n_timeouts = 0
    approx_flag = False
    for dim in range(max_hom_dim + 1):
        flow_bd = probe._slice_dim(diagram, dim, max_edge)
        base_bd = baselines_m[dim]
        try:
            total += float(wdist(flow_bd, base_bd, order=2.0, internal_p=2.0, delta=delta))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "direct_w2_flow: hera exception row_idx=%s manifold=%s dim=%d: %r",
                row_idx, manifold, dim, exc,
            )
            n_timeouts += 1
            approx_flag = True
    return float(total), n_timeouts, approx_flag


def _free_memory_mb() -> "float | None":
    """Best-effort free-memory reading for shard-cadence telemetry.

    Tries macOS `vm_stat` first (matches the diagnostic used to confirm
    memory pressure as the worker-churn root cause), falling back to
    `psutil.virtual_memory().available` if `vm_stat` is unavailable or
    unparseable (e.g. non-macOS). Returns None if neither works --
    telemetry is best-effort and must never be fatal to the campaign.
    """
    import re
    import subprocess

    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=5.0)
        page_size_match = re.search(r"page size of (\d+) bytes", out)
        free_match = re.search(r"Pages free:\s+(\d+)\.", out)
        if page_size_match and free_match:
            page_size = int(page_size_match.group(1))
            free_pages = int(free_match.group(1))
            return free_pages * page_size / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None


def _process_chunk_for_manifold(
    ws_root: str,
    split: str,
    manifold: str,
    chunk_indices: "list[int]",
    baselines_m: dict,
    max_edge: float,
    max_hom_dim: int,
    delta: float = PRIMARY_DELTA,
) -> "list[dict]":
    """Runs INSIDE a joblib worker (or in-process at n_jobs=1): opens ONLY
    `{manifold}_{split}.pkl` itself, for ONLY this ~100-flow chunk, then
    lets it be freed when the call returns.

    Low-memory campaign profile (2026-09-22, third intervention). Root
    cause confirmed via an n_jobs=1 isolation-probe smoke shard (completed
    clean, 500/500 rows, 0 exceptions -- see
    `.superpowers/sdd/2026-09-22-plan6-exact-w2-latency/task-3-report.md`,
    "Update 3"): the PARENT process previously loaded the full split's
    diagrams for all 3 manifolds (~250-350MB pickled EACH) and captured
    that dict in a closure (`_flow_row`) submitted as ~500 individual
    per-flow `delayed()` tasks. joblib/loky's automatic array-memmapping
    only covers ndarrays passed as direct `delayed()` arguments, NOT
    objects captured by closure -- so each of those ~500 closures had to be
    independently cloudpickled, repeatedly re-serializing the large shared
    dict and exhausting this workstation's already memory-pressured RAM
    (confirmed: ~60MB free of 16GB, 4.9/6GB swap in use during the failed
    launches), which is what actually killed/reaped loky worker processes
    -- NOT a native crash in `gudhi.hera` (the isolation probe's
    try/except in `direct_w2_flow` never fired).

    This function instead takes only cheap-to-pickle arguments (a path
    string, a manifold name, ~100 flow indices, one manifold's tiny
    baseline dict) and loads the (large) diagram pkl itself, so the parent
    never holds or transmits it and each worker's peak memory is bounded to
    ~one manifold's pkl for the duration of one ~100-flow chunk.

    Returns one dict per flow in `chunk_indices`:
    `{row_idx, W2, n_timeouts, approx_flag}` (W2 for THIS manifold only --
    `run_shard` combines the 3 manifolds' per-chunk results into rows).
    """
    import pickle
    from pathlib import Path as _Path

    ws = Workspace.at(_Path(ws_root))
    diagrams = pickle.loads((ws.persistence_dir / f"{manifold}_{split}.pkl").read_bytes())
    rows = []
    for i in chunk_indices:
        total, n_t, approx = direct_w2_flow(
            diagrams[i], baselines_m, max_edge, max_hom_dim,
            delta=delta, row_idx=i, manifold=manifold,
        )
        rows.append({"row_idx": int(i), "W2": total, "n_timeouts": n_t, "approx_flag": approx})
    return rows


# ---------------------------------------------------------------------------
# Sharded runner + resumable manifest
# ---------------------------------------------------------------------------


def _load_split_inputs(ws: Workspace, split: str) -> "tuple[np.ndarray, dict]":
    import pickle

    labels = pd.read_csv(ws.outputs_dir / f"labels_{split}.csv")["label"].to_numpy()
    per_manifold_diagrams = {
        m: pickle.loads((ws.persistence_dir / f"{m}_{split}.pkl").read_bytes())
        for m in config.MANIFOLDS
    }
    return labels, per_manifold_diagrams


def _shard_key(split: str, start: int) -> str:
    return f"{split}_{start:05d}"


def _read_manifest(exact_dir: Path) -> dict:
    path = exact_dir / MANIFEST_NAME
    if not path.exists():
        return {"shards": {}}
    return json.loads(path.read_text())


def _write_manifest_atomic(exact_dir: Path, manifest: dict) -> None:
    path = exact_dir / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def run_shard(
    ws: Workspace,
    exact_dir: Path,
    split: str,
    start: int,
    size: int,
    baselines: dict,
    class_filter: "str | None" = None,
    n_jobs: int = -1,
    chunk_size: int = 100,
) -> Path:
    """Compute high-precision-W2 rows for flows [start, start+size) of ``split``.

    Rows: `[row_idx, label, W2_c2, W2_network, W2_physical, n_timeouts,
    approx_flag]`. ``class_filter`` restricts (order-preserved) to one
    class before slicing [start, start+size). Writes
    `shard_{split}_{start:05d}.csv` and updates `manifest.json` atomically
    (write tmp, rename) marking the shard complete with its row count.

    LOW-MEMORY CHUNKED PROFILE (2026-09-22, third intervention -- see
    `_process_chunk_for_manifold`'s docstring for the full root-cause
    story): this function no longer loads the split's diagrams itself.
    ``shard_idx`` is split into ``chunk_size``-flow chunks (default ~100);
    for each of the 3 manifolds, one `delayed(_process_chunk_for_manifold)`
    task per chunk is submitted -- each task opens ONLY that manifold's pkl,
    for ONLY its chunk, inside the worker, then lets it be freed. Only
    labels (a small CSV column) and each manifold's tiny baseline dict are
    held/transmitted by the parent. `direct_w2_flow`'s per-dim try/except
    means `n_timeouts`/`approx_flag` are no longer always 0/False on this
    path -- they also count a caught Python-level hera exception (see its
    docstring), on top of remaining in the schema for compatibility with
    `exact_w2_flow`'s interface.

    `row_idx` indexes into the ORIGINAL unfiltered `labels_{split}.csv`
    (i.e. `outputs/labels_{split}.csv` row position), so it is
    non-contiguous within a class-filtered shard (e.g. the val Normal-only
    shards).
    """
    from joblib import Parallel, delayed

    exact_dir.mkdir(parents=True, exist_ok=True)
    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    labels = pd.read_csv(ws.outputs_dir / f"labels_{split}.csv")["label"].to_numpy()

    if class_filter is not None:
        order_idx = np.where(labels == class_filter)[0]
    else:
        order_idx = np.arange(len(labels))
    shard_idx = order_idx[start:start + size]

    chunks = [shard_idx[c:c + chunk_size] for c in range(0, len(shard_idx), chunk_size)]

    parallel = Parallel(n_jobs=n_jobs)
    per_manifold_rows: "dict[str, dict[int, dict]]" = {}
    for m in config.MANIFOLDS:
        chunk_results = parallel(
            delayed(_process_chunk_for_manifold)(
                str(ws.root), split, m, chunk.tolist(), baselines[m],
                max_edge_lengths[m], config.MAX_HOM_DIM[m],
            )
            for chunk in chunks
        )
        per_manifold_rows[m] = {
            r["row_idx"]: r for chunk_rows in chunk_results for r in chunk_rows
        }

    rows = []
    for i in shard_idx:
        i = int(i)
        row: dict = {"row_idx": i, "label": labels[i]}
        n_timeouts_total = 0
        approx_flag_total = False
        for m in config.MANIFOLDS:
            r = per_manifold_rows[m][i]
            row[f"W2_{m}"] = r["W2"]
            n_timeouts_total += r["n_timeouts"]
            approx_flag_total = approx_flag_total or r["approx_flag"]
        row["n_timeouts"] = n_timeouts_total
        row["approx_flag"] = approx_flag_total
        rows.append(row)

    columns = ["row_idx", "label", *[f"W2_{m}" for m in config.MANIFOLDS], "n_timeouts", "approx_flag"]
    df = pd.DataFrame(rows, columns=columns)

    shard_path = exact_dir / f"shard_{split}_{start:05d}.csv"
    df.to_csv(shard_path, index=False)
    write_provenance(shard_path, {
        "split": split, "start": start, "size": size, "class_filter": class_filter,
    })

    manifest = _read_manifest(exact_dir)
    manifest.setdefault("shards", {})[_shard_key(split, start)] = {
        "split": split,
        "start": start,
        "size": size,
        "n_rows": len(df),
        "path": shard_path.name,
        "class_filter": class_filter,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest_atomic(exact_dir, manifest)
    return shard_path


def run_exact_campaign(ws: Workspace, n_jobs: int = -1, shard_size: int = 500) -> None:
    """Full exact-W2 campaign: baselines once, then val-Normal shards, then test shards.

    Resumable by construction: shards already recorded complete in
    `manifest.json` are skipped, so re-invoking after any interruption
    picks up exactly where it left off, against the SAME persisted
    baselines.

    Prints a shard-cadence + free-memory telemetry line after each shard
    (2026-09-22, third intervention -- `_free_memory_mb`) so a `nohup`-ed
    campaign's log carries a running record of s/flow and available RAM
    (severe memory pressure was the confirmed root cause of the earlier
    worker-churn failures; this makes any recurrence visible without
    needing a separate diagnostic pass).
    """
    import time

    exact_dir = ws.tables_dir / "rebuild" / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    baselines = ensure_baselines(ws, exact_dir)

    plan: list = []
    val_labels = pd.read_csv(ws.outputs_dir / "labels_val.csv")["label"].to_numpy()
    n_val = int((val_labels == "Normal Traffic").sum())
    for start in range(0, n_val, shard_size):
        plan.append(("val", start, min(shard_size, n_val - start), "Normal Traffic"))

    test_labels = pd.read_csv(ws.outputs_dir / "labels_test.csv")["label"].to_numpy()
    n_test = len(test_labels)
    for start in range(0, n_test, shard_size):
        plan.append(("test", start, min(shard_size, n_test - start), None))

    manifest = _read_manifest(exact_dir)
    completed = set(manifest.get("shards", {}))
    for split, start, size, class_filter in plan:
        if _shard_key(split, start) in completed:
            continue
        t0 = time.time()
        run_shard(ws, exact_dir, split, start, size, baselines, class_filter=class_filter, n_jobs=n_jobs)
        dt = time.time() - t0
        free_mb = _free_memory_mb()
        free_str = f"{free_mb:.0f}MB" if free_mb is not None else "unknown"
        print(
            f"[exact] shard {split}_{start:05d} complete in {dt:.1f}s "
            f"({dt / max(size, 1):.3f} s/flow) free_mem={free_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Assembly (Task 2)
# ---------------------------------------------------------------------------


def _exact_dir(ws: Workspace) -> Path:
    return ws.tables_dir / "rebuild" / "exact"


def assemble_distances(exact_dir: Path, split: str) -> pd.DataFrame:
    """Concatenate every completed `shard_{split}_*.csv` in start order.

    Reads shard membership and row counts from `manifest.json` (rather
    than a bare directory glob) so ordering is exact and resume-safe.
    Raises ``ValueError`` if no shards are recorded for ``split``, or if
    the concatenated row count doesn't match the manifest's recorded row
    counts (a corrupted/partial shard file). Adds the 7 manifold-subset
    columns (`metrics.MANIFOLD_SUBSETS`) as SUMS of the per-manifold `W2_*`
    columns.
    """
    manifest = _read_manifest(exact_dir)
    shard_infos = sorted(
        (info for info in manifest.get("shards", {}).values() if info["split"] == split),
        key=lambda info: info["start"],
    )
    if not shard_infos:
        raise ValueError(f"assemble_distances: no completed shards recorded for split={split!r}")

    frames = [pd.read_csv(exact_dir / info["path"]) for info in shard_infos]
    df = pd.concat(frames, ignore_index=True)

    expected_rows = sum(info["n_rows"] for info in shard_infos)
    if len(df) != expected_rows:
        raise ValueError(
            f"assemble_distances: row count mismatch for split={split!r}: "
            f"concatenated {len(df)} rows from {len(shard_infos)} shard(s) but "
            f"manifest records {expected_rows}"
        )

    for subset, manifolds in MANIFOLD_SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)

    return df


# ---------------------------------------------------------------------------
# Definitive tables (Task 2)
# ---------------------------------------------------------------------------


def _binary_auc_table(scored_frames: "dict[str, pd.DataFrame]", B: int, bootstrap_seed: int) -> pd.DataFrame:
    """7 subsets x {raw, znorm}: AUC + 95% CI over a single test-set cluster.

    ``scored_frames`` maps scoring name ("raw"/"znorm") to the (already
    z-normalized where applicable) test frame carrying `label`, the
    per-manifold `W2_*` columns, the `MANIFOLD_SUBSETS` sum columns, and
    `approx_flag`.
    """
    rows = []
    for scoring, df in scored_frames.items():
        y = (df["label"] != NORMAL).astype(int).to_numpy()
        labels_multiclass = df["label"].to_numpy()
        n_flows = len(df)
        n_approx = int(df["approx_flag"].sum())
        for subset in MANIFOLD_SUBSETS:
            scores = df[f"W2_{subset}"].to_numpy()
            auc = float(roc_auc_score(y, scores))
            ci_lo, ci_hi = manuscript.bootstrap_mean_auc_ci(
                per_seed=[(y, scores)], B=B, bootstrap_seed=bootstrap_seed,
                labels_per_seed=[labels_multiclass],
            )
            rows.append({
                "subset": subset, "scoring": scoring, "auc": auc,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n_flows": n_flows, "n_approx_flagged": n_approx,
            })
    # subset-major, scoring-minor ordering (matches paper/probe table conventions).
    df_out = pd.DataFrame(rows)
    subset_order = {s: i for i, s in enumerate(MANIFOLD_SUBSETS)}
    df_out["_order"] = df_out["subset"].map(subset_order)
    df_out = df_out.sort_values(["_order", "scoring"]).drop(columns="_order").reset_index(drop=True)
    return df_out


def _per_attack_auc_table(test_df: pd.DataFrame, B: int, bootstrap_seed: int) -> pd.DataFrame:
    """4 attacks x 3 manifolds one-vs-rest AUC (raw scores) + CI + dominant flag."""
    base = metrics.per_attack_auc(test_df)  # columns: attack_class, manifold, auc
    label = test_df["label"].to_numpy()

    rows = []
    for _, r in base.iterrows():
        attack, m = r["attack_class"], r["manifold"]
        y = (label == attack).astype(int)
        scores = test_df[f"W2_{m}"].to_numpy()
        ci_lo, ci_hi = manuscript.bootstrap_mean_auc_ci(
            per_seed=[(y, scores)], B=B, bootstrap_seed=bootstrap_seed,
            labels_per_seed=[label],
        )
        rows.append({
            "attack_class": attack, "manifold": m, "auc": float(r["auc"]),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    df_out = pd.DataFrame(rows)
    df_out["dominant"] = False
    for attack in df_out["attack_class"].unique():
        mask = df_out["attack_class"] == attack
        best_idx = df_out.loc[mask, "auc"].idxmax()
        df_out.loc[best_idx, "dominant"] = True
    return df_out


def _exact_vs_probe_table(binary_raw_and_znorm: pd.DataFrame, ws: Workspace) -> pd.DataFrame:
    """Join exact/probe-10-seed/published-3-seed binary AUCs per subset x scoring."""
    probe_path = ws.tables_dir / "rebuild" / "binary_auc.csv"
    probe_df = pd.read_csv(probe_path)

    rows = []
    for _, r in binary_raw_and_znorm.iterrows():
        subset, scoring, exact_auc = r["subset"], r["scoring"], float(r["auc"])
        probe_match = probe_df[(probe_df["subset"] == subset) & (probe_df["scoring"] == scoring)]
        if len(probe_match):
            probe_mean = float(probe_match["mean"].iloc[0])
            probe_std = float(probe_match["std"].iloc[0])
        else:
            probe_mean, probe_std = float("nan"), float("nan")

        if scoring == "raw":
            pub_mean, pub_std = PUBLISHED3_RAW[subset]
        else:
            pub_mean, pub_std = float("nan"), float("nan")

        rows.append({
            "subset": subset, "scoring": scoring, "exact_auc": exact_auc,
            "probe10_mean": probe_mean, "probe10_std": probe_std,
            "published3_mean": pub_mean, "published3_std": pub_std,
            "delta_exact_minus_probe": exact_auc - probe_mean,
        })
    return pd.DataFrame(rows)


def build_exact_tables(ws: Workspace, B: int = 2000, bootstrap_seed: int = 0) -> "dict[str, pd.DataFrame]":
    """Assemble the exact-W2 campaign shards into the definitive report tables.

    Reads the val (Normal-only, by campaign construction) and test frames
    via `assemble_distances`, derives znorm stats from val
    (`metrics.znorm_stats_from_val`), and builds:

    - `exact_binary_auc`: 7 subsets x {raw, znorm} AUC + 95% bootstrap CI
      (single-cluster bootstrap over the full test set).
    - `exact_per_attack_auc`: 4 attacks x 3 manifolds one-vs-rest AUC (raw)
      + CI + a `dominant` flag (the highest-AUC manifold per attack).
    - `exact_vs_probe`: exact vs the Phase-4 10-seed probe mean/std
      (`results/tables/rebuild/binary_auc.csv`) vs the published 3-seed
      literals (raw scoring only; NaN for znorm), plus the exact-minus-probe
      delta.

    If any test flow has `approx_flag` set (a delta=0.01 retry fallback was
    used on at least one homology dim), also returns
    `exact_binary_auc_excl_flagged` -- the same binary table recomputed
    with those flows dropped. Omitted entirely when no flow was flagged.
    """
    exact_dir = _exact_dir(ws)
    val_df = assemble_distances(exact_dir, "val")
    test_df = assemble_distances(exact_dir, "test")

    stats = metrics.znorm_stats_from_val(val_df)
    test_znorm_df = metrics.apply_znorm(test_df, stats)

    binary_df = _binary_auc_table(
        {"raw": test_df, "znorm": test_znorm_df}, B=B, bootstrap_seed=bootstrap_seed,
    )
    per_attack_df = _per_attack_auc_table(test_df, B=B, bootstrap_seed=bootstrap_seed)
    vs_probe_df = _exact_vs_probe_table(binary_df, ws)

    tables: "dict[str, pd.DataFrame]" = {
        "exact_binary_auc": binary_df,
        "exact_per_attack_auc": per_attack_df,
        "exact_vs_probe": vs_probe_df,
    }

    n_flagged = int(test_df["approx_flag"].sum())
    if n_flagged > 0:
        clean_test = test_df[~test_df["approx_flag"]].reset_index(drop=True)
        clean_test_znorm = metrics.apply_znorm(clean_test, stats)
        tables["exact_binary_auc_excl_flagged"] = _binary_auc_table(
            {"raw": clean_test, "znorm": clean_test_znorm}, B=B, bootstrap_seed=bootstrap_seed,
        )

    return tables


_TABLE_FILENAMES = {
    "exact_binary_auc": "exact_binary_auc.csv",
    "exact_per_attack_auc": "exact_per_attack_auc.csv",
    "exact_vs_probe": "exact_vs_probe.csv",
    "exact_binary_auc_excl_flagged": "exact_binary_auc_excl_flagged.csv",
}


def write_exact_tables(ws: Workspace, tables: "dict[str, pd.DataFrame]") -> "dict[str, Path]":
    """Write `build_exact_tables` output to `exact_dir/*.csv` + provenance sidecars.

    Report generation (`paper/EXACT_RESULTS.md`) lands in Task 3 -- this
    function is the seam it will call after (or alongside) writing tables.
    """
    exact_dir = _exact_dir(ws)
    exact_dir.mkdir(parents=True, exist_ok=True)
    paths: "dict[str, Path]" = {}
    for name, df in tables.items():
        out = exact_dir / _TABLE_FILENAMES[name]
        df.to_csv(out, index=False)
        write_provenance(out, {"table": name})
        paths[name] = out
    return paths


# ---------------------------------------------------------------------------
# delta=0.01-vs-delta=0.05 insensitivity check (Task 3, controller ruling
# 2026-09-22 -- quantifies the residual approximation the high-precision
# campaign definition carries, since it is no longer literally exact).
# ---------------------------------------------------------------------------


def _w2_sum_at_delta(
    diagram: np.ndarray, baselines_m: dict, max_edge: float, max_hom_dim: int,
    delta: float, timeout_s: float,
) -> float:
    """Per-dim Wasserstein-2 sum at a FIXED delta (no retry) -- helper for
    `insensitivity_check`, which compares two fixed-delta variants directly
    rather than the primary/retry-on-timeout logic in `exact_w2_flow`.
    """
    total = 0.0
    for dim in range(max_hom_dim + 1):
        flow_bd = probe._slice_dim(diagram, dim, max_edge)
        base_bd = baselines_m[dim]
        w = probe._w2_with_timeout(
            flow_bd, base_bd, order=2.0, internal_p=2.0, delta=delta, timeout_sec=timeout_s,
        )
        total += w if np.isfinite(w) else 0.0
    return float(total)


def insensitivity_check(
    ws: Workspace, exact_dir: Path, n: int = 500, rng_seed: int = 0,
    timeout_s: float = 120.0, n_jobs: int = -1,
) -> pd.DataFrame:
    """Binary AUC at `delta=0.01` vs `delta=0.05` on an ``n``-flow test subsample.

    Uses the SAME persisted baselines as the main campaign (`ensure_baselines`
    -- never recomputes). Draws a deterministic subsample of ``n`` test flows
    (`rng_seed`), computes the 7 `MANIFOLD_SUBSETS` W2 sums at each fixed
    delta (no retry -- `_w2_sum_at_delta`), and returns one row per subset:
    `auc_delta01`, `auc_delta05`, `delta_auc` (= auc_delta05 - auc_delta01),
    `n_flows`. A small `|delta_auc|` demonstrates the AUC rank statistic is
    insensitive to the residual approximation the high-precision campaign
    definition (delta<=0.01, with a delta=0.05 timeout fallback) carries.
    """
    from joblib import Parallel, delayed

    baselines = ensure_baselines(ws, exact_dir)
    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    labels, per_manifold_diagrams = _load_split_inputs(ws, "test")

    rng = np.random.default_rng(rng_seed)
    n_sample = min(n, len(labels))
    idx = np.sort(rng.choice(len(labels), size=n_sample, replace=False))

    def _row(i: int, delta: float) -> dict:
        row: dict = {"row_idx": int(i), "label": labels[i]}
        for m in config.MANIFOLDS:
            row[f"W2_{m}"] = _w2_sum_at_delta(
                per_manifold_diagrams[m][i], baselines[m], max_edge_lengths[m],
                config.MAX_HOM_DIM[m], delta, timeout_s,
            )
        return row

    frames: dict = {}
    for delta in (0.01, 0.05):
        parallel = Parallel(n_jobs=n_jobs)
        rows = parallel(delayed(_row)(int(i), delta) for i in idx)
        df = pd.DataFrame(rows, columns=["row_idx", "label", *[f"W2_{m}" for m in config.MANIFOLDS]])
        for subset, manifolds in MANIFOLD_SUBSETS.items():
            df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)
        frames[delta] = df

    y = (frames[0.01]["label"].to_numpy() != NORMAL).astype(int)
    rows_out = []
    for subset in MANIFOLD_SUBSETS:
        auc01 = float(roc_auc_score(y, frames[0.01][f"W2_{subset}"].to_numpy()))
        auc05 = float(roc_auc_score(y, frames[0.05][f"W2_{subset}"].to_numpy()))
        rows_out.append({
            "subset": subset, "auc_delta01": auc01, "auc_delta05": auc05,
            "delta_auc": auc05 - auc01, "n_flows": n_sample,
        })
    return pd.DataFrame(rows_out)


def write_insensitivity_check(ws: Workspace, df: pd.DataFrame) -> Path:
    """Write `insensitivity_check` output to `exact_dir/insensitivity_check.csv` + provenance."""
    exact_dir = _exact_dir(ws)
    exact_dir.mkdir(parents=True, exist_ok=True)
    out = exact_dir / "insensitivity_check.csv"
    df.to_csv(out, index=False)
    write_provenance(out, {"table": "insensitivity_check"})
    return out
