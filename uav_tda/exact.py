"""Phase 6 Track A: sharded, resumable exact-Wasserstein-2 runner.

Replaces the probe approximation (top-K + delta=0.2 relative error) with
definitive full-diagram, delta=0.0 (exact) hera Wasserstein-2 on the full
test split. Reuses `probe._load_baseline_barcodes`, `probe._slice_dim`, and
`probe._w2_with_timeout` (probe.py itself is read-only / locked).

Same-run baseline coupling (Global Constraint): the per-manifold baseline
barcodes are computed ONCE and persisted under `exact_dir` (npy + manifest +
provenance). Every subsequent call in the SAME campaign, and every resume
after interruption, must LOAD the persisted baselines rather than
recomputing them -- `probe._load_baseline_barcodes` runs sparse-Rips on the
500 reference points with no seed control, so recomputing would silently
change the baseline realization that val Normal-znorm-stats and test
distances are scored against.

Exact means exact: hera `order=2.0, internal_p=2.0, delta=0.0` (NOT hera's
own default delta=0.01) with NO top-K truncation. `delta=0.0` requests
hera's exact auction-LP solution rather than a (1+delta)-approximate one;
verified interactively that `delta=0.0` is markedly slower / can hang on
pathological inputs (consistent with `probe._w2_with_timeout`'s existing
fork-timeout machinery, which this module reuses unmodified) while the
paper's measured ~3.3s/flow on the real, non-pathological clean diagrams
stays well within the 120s per-call budget.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, probe
from .provenance import write_provenance
from .workspace import Workspace

MANIFEST_NAME = "manifest.json"
BASELINES_MANIFEST_NAME = "baselines_manifest.json"


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


def exact_w2_flow(
    diagram: np.ndarray,
    baselines_m: dict,
    max_edge: float,
    max_hom_dim: int,
    timeout_s: float = 120.0,
) -> "tuple[float, int, bool]":
    """Exact per-dim Wasserstein-2 distance from ``diagram`` to a baseline, summed.

    Per homology dim: `probe._slice_dim` both sides, then
    `probe._w2_with_timeout(order=2.0, internal_p=2.0, delta=0.0, ...)`
    (hera's exact mode). If that call times out (NaN), retry ONCE with
    `delta=0.01` under the same timeout; if the retry also times out, the
    dim contributes 0.0 and is counted in `n_timeouts`.

    Returns ``(total, n_timeouts, approx_flag)`` where `approx_flag` is True
    iff ANY dim used the delta retry at all -- whether or not that retry
    itself succeeded (a successful delta=0.01 retry is still an
    approximation, not the exact value).
    """
    total = 0.0
    n_timeouts = 0
    approx_flag = False
    for dim in range(max_hom_dim + 1):
        flow_bd = probe._slice_dim(diagram, dim, max_edge)
        base_bd = baselines_m[dim]
        w = probe._w2_with_timeout(
            flow_bd, base_bd, order=2.0, internal_p=2.0, delta=0.0, timeout_sec=timeout_s,
        )
        if np.isfinite(w):
            total += w
            continue
        approx_flag = True
        w_retry = probe._w2_with_timeout(
            flow_bd, base_bd, order=2.0, internal_p=2.0, delta=0.01, timeout_sec=timeout_s,
        )
        if np.isfinite(w_retry):
            total += w_retry
        else:
            n_timeouts += 1
    return float(total), n_timeouts, approx_flag


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
) -> Path:
    """Compute exact-W2 rows for flows [start, start+size) of ``split``.

    Rows: `[row_idx, label, W2_c2, W2_network, W2_physical, n_timeouts,
    approx_flag]`. ``class_filter`` restricts (order-preserved) to one
    class before slicing [start, start+size). Writes
    `shard_{split}_{start:05d}.csv` and updates `manifest.json` atomically
    (write tmp, rename) marking the shard complete with its row count.
    """
    from joblib import Parallel, delayed

    exact_dir.mkdir(parents=True, exist_ok=True)
    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    labels, per_manifold_diagrams = _load_split_inputs(ws, split)

    if class_filter is not None:
        order_idx = np.where(labels == class_filter)[0]
    else:
        order_idx = np.arange(len(labels))
    shard_idx = order_idx[start:start + size]

    def _flow_row(i: int) -> dict:
        row: dict = {"row_idx": int(i), "label": labels[i]}
        n_timeouts_total = 0
        approx_flag_total = False
        for m in config.MANIFOLDS:
            diagram = per_manifold_diagrams[m][i]
            total, n_t, approx = exact_w2_flow(
                diagram, baselines[m], max_edge_lengths[m], config.MAX_HOM_DIM[m],
            )
            row[f"W2_{m}"] = total
            n_timeouts_total += n_t
            approx_flag_total = approx_flag_total or approx
        row["n_timeouts"] = n_timeouts_total
        row["approx_flag"] = approx_flag_total
        return row

    parallel = Parallel(n_jobs=n_jobs)
    rows = parallel(delayed(_flow_row)(int(i)) for i in shard_idx)

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
    """
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
        run_shard(ws, exact_dir, split, start, size, baselines, class_filter=class_filter, n_jobs=n_jobs)
