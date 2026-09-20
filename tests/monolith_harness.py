"""Hermetic test harness for running the `pipeline.py` monolith in isolated workspaces.

Phase 2 of the strangle-monolith migration ports each pipeline phase into the
`uav_tda` package, gated by equivalence tests against FRESH monolith runs. This
module provides the plumbing: build a hermetic copy of the monolith (so its
self-relative path resolution stays intact), run debug-mode phases in it via
subprocess, cache the resulting workspace on disk keyed by pipeline.py content
hash (so repeated test sessions don't re-pay the ~15s+ cost per phase), and
compare output artifacts (CSV / pickled persistence diagrams / JSON) with
floating-point tolerance.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Phase chain in dependency order. `ensure_debug_phase` runs every not-yet-done
# prerequisite up to and including the requested phase.
PHASE_CHAIN: tuple[str, ...] = (
    "prep",
    "tda",
    "features",
    "supervised",
    "unsupervised",
    "evaluate",
)


def pipeline_hash(repo_root: Path) -> str:
    """First 12 hex chars of the sha256 of pipeline.py's bytes.

    Used as a cache key: a fresh workspace is built whenever pipeline.py's
    content changes, so the on-disk debug cache never silently serves stale
    monolith outputs.
    """
    data = (repo_root / "pipeline.py").read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def make_monolith_workspace(dest: Path, repo_root: Path) -> Path:
    """Create a hermetic copy of the monolith at `dest`.

    pipeline.py resolves all paths relative to its own file location
    (`REPO_ROOT = Path(__file__).resolve().parent`), so copying pipeline.py
    into `dest` and symlinking `repo_root/"data"` to `dest/"data"` fully
    relocates its I/O — outputs/results/logs get created fresh under `dest`,
    never touching the real repo's outputs/results directories.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pipeline.py").write_bytes((repo_root / "pipeline.py").read_bytes())
    data_link = dest / "data"
    if not data_link.exists():
        data_link.symlink_to(repo_root / "data")
    return dest


def run_monolith_phase(ws_root: Path, phase: str, *extra: str) -> None:
    """Run `python3 pipeline.py <phase> --debug <extra...>` inside `ws_root`.

    Raises with the last 30 lines of stderr appended for debuggability if the
    subprocess exits non-zero.
    """
    cmd = [sys.executable, "pipeline.py", phase, "--debug", *extra]
    try:
        subprocess.run(cmd, cwd=ws_root, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        tail = "\n".join(stderr.splitlines()[-30:])
        raise RuntimeError(
            f"monolith phase {phase!r} failed (cmd={cmd!r}, cwd={ws_root}):\n{tail}"
        ) from exc


def ensure_debug_phase(cache_root: Path, repo_root: Path, phase: str) -> Path:
    """Idempotently ensure `phase` (and its prerequisites) have run in debug mode.

    The workspace lives at `cache_root/<pipeline_hash>` and is reused across
    test sessions. Completion of each phase is tracked by a `.done-<phase>`
    marker file in the workspace; only phases without a marker are (re-)run,
    in chain order, up to and including the requested phase.
    """
    if phase not in PHASE_CHAIN:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASE_CHAIN}")

    ws = cache_root / pipeline_hash(repo_root)
    if not (ws / "pipeline.py").is_file():
        make_monolith_workspace(ws, repo_root)

    target_idx = PHASE_CHAIN.index(phase)
    for p in PHASE_CHAIN[: target_idx + 1]:
        marker = ws / f".done-{p}"
        if marker.exists():
            continue
        run_monolith_phase(ws, p)
        marker.touch()

    return ws


def assert_csvs_equal(a: Path, b: Path, float_rtol: float = 1e-9) -> None:
    """Assert two CSVs are equal, with floating-point tolerance."""
    df_a = pd.read_csv(a)
    df_b = pd.read_csv(b)
    pd.testing.assert_frame_equal(df_a, df_b, check_exact=False, rtol=float_rtol, atol=1e-12)


def _sort_rows_lexicographic(arr: np.ndarray) -> np.ndarray:
    """Sort a 2D array's rows lexicographically (row order, not element-wise sort).

    `np.sort(arr, axis=0)` sorts each column independently and destroys row
    identity — wrong here. `np.lexsort` needs keys in last-column-major order,
    so we transpose and reverse to sort primarily by column 0, then column 1, etc.
    """
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    order = np.lexsort(arr.T[::-1])
    return arr[order]


def assert_diagram_pkls_equal(a: Path, b: Path) -> None:
    """Assert two pickled lists of persistence diagrams are equal.

    Each diagram (a 2D array of birth/death pairs) is compared after sorting
    its rows lexicographically, since diagram row order is not semantically
    meaningful (joblib-parallel order is stable but arbitrary).
    """
    with open(a, "rb") as fh:
        diagrams_a = pickle.load(fh)
    with open(b, "rb") as fh:
        diagrams_b = pickle.load(fh)

    assert len(diagrams_a) == len(diagrams_b), (
        f"diagram list length mismatch: {len(diagrams_a)} != {len(diagrams_b)}"
    )
    for i, (da, db) in enumerate(zip(diagrams_a, diagrams_b)):
        arr_a = np.asarray(da)
        arr_b = np.asarray(db)
        assert arr_a.shape == arr_b.shape, (
            f"diagram {i} shape mismatch: {arr_a.shape} != {arr_b.shape}"
        )
        sorted_a = _sort_rows_lexicographic(arr_a)
        sorted_b = _sort_rows_lexicographic(arr_b)
        assert np.allclose(sorted_a, sorted_b, equal_nan=True), (
            f"diagram {i} values differ after row-sort"
        )


def _floats_close(x: object, y: object, rtol: float) -> bool:
    return abs(x - y) <= rtol * max(abs(x), abs(y), 1e-12)


def _json_equal(x: object, y: object, rtol: float) -> bool:
    if isinstance(x, float) or isinstance(y, float):
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return False
        return _floats_close(float(x), float(y), rtol)
    if isinstance(x, dict) and isinstance(y, dict):
        if x.keys() != y.keys():
            return False
        return all(_json_equal(x[k], y[k], rtol) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return False
        return all(_json_equal(xi, yi, rtol) for xi, yi in zip(x, y))
    return x == y


def assert_json_equal(a: Path, b: Path, float_rtol: float = 1e-9) -> None:
    """Assert two JSON files are equal, with floating-point tolerance."""
    with open(a) as fh:
        json_a = json.load(fh)
    with open(b) as fh:
        json_b = json.load(fh)
    assert _json_equal(json_a, json_b, float_rtol), f"JSON mismatch: {json_a!r} != {json_b!r}"
