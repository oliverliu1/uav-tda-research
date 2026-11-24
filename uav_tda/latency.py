"""Phase 6 Track B: portable latency harness (paper §V onboard-latency claim).

Measures wall-clock cost of the FULL production inference path -- standardize
-> Rips -> slice/truncate -> Wasserstein-2 -- per sampled decision, for two
arms:

- **per-flow**: the paper's DEPLOYED probe-style scoring config
  (`config.SPARSE_RIPS_EPSILON`, `config.PROBE_TOP_K`=50,
  `config.PROBE_DELTA`=0.2, matching `probe.run_probe`'s production scoring)
  -- deliberately NOT the Track-A exact-campaign config (delta<=0.01, no
  top-K), per the plan's Global Constraints.
- **windowed**: `config.WINDOWED_SPARSE` at each `config.WINDOW_SIZES`,
  matching `windowed.run_windowed`'s production construction.

One-command runnable on this machine (`uav-tda latency`) and designed to be
one-command runnable later on companion-class embedded hardware (Jetson/
RPi) against the same shipped `outputs/` artifacts -- see `run_latency`'s
report for exactly which files a companion box needs (NOT the multi-hundred-
MB persistence-diagram pickles).

STAGE-1 (scaler transform) measurement note: `outputs/{m}_test.csv` holds
the ALREADY-STANDARDIZED per-manifold features -- no raw, pre-scaler
per-manifold CSV is persisted anywhere on disk (`outputs/original_features_
*.csv`, despite its name, is the STANDARDIZED per-manifold columns
concatenated, not raw input -- see `data.save_combined_originals`). Since
`sklearn.preprocessing.StandardScaler.transform` is an exact, invertible
affine map ((x - mean) / std), this module recovers the GENUINE pre-scaler
feature vector for each sampled test flow via `scaler.inverse_transform` on
its stored standardized row (mathematically exact up to float64 round-trip
precision -- not a synthetic stand-in) and times the FORWARD `scaler.
transform` call on that recovered vector as stage 1. This needs only the
already-shipped `outputs/{m}_test.csv` + scaler artifacts; no raw dataset
file is required, keeping the harness itself lean on this memory-tight
machine (no full-split diagram loads anywhere in this module).
"""
from __future__ import annotations

import faulthandler
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, probe
from .provenance import write_provenance
from .workspace import Workspace

log = logging.getLogger("uav_tda.latency")

_NORMAL_LABEL = "Normal Traffic"

DEFAULT_WATCHDOG_S = 900.0


# ---------------------------------------------------------------------------
# Progress + watchdog
# ---------------------------------------------------------------------------


class Watchdog:
    """Print each step to stderr and kill the process if one step never ends.

    hera can loop forever inside native code on near-identical diagrams (see
    `uav_tda.hangcheck`), where no Python-level timeout can interrupt it.
    `faulthandler.dump_traceback_later(exit=True)` fires from a C thread, so
    it still works: the traceback says whether Rips or Wasserstein is stuck,
    and the last printed step says which stage/flow. Steps are only ever
    announced OUTSIDE the timed regions. ``timeout_s`` <= 0 disables the
    timer but keeps the progress lines.
    """

    def __init__(self, timeout_s: float = DEFAULT_WATCHDOG_S):
        self.timeout_s = float(timeout_s)
        self.armed = False

    def step(self, label: str) -> None:
        print(f"[latency] {label}", file=sys.stderr, flush=True)
        if self.timeout_s > 0:
            # re-arming replaces the previous timer
            faulthandler.dump_traceback_later(
                self.timeout_s, exit=True, file=sys.__stderr__)
            self.armed = True

    def cancel(self) -> None:
        if self.armed:
            faulthandler.cancel_dump_traceback_later()
            self.armed = False


# ---------------------------------------------------------------------------
# machine_info
# ---------------------------------------------------------------------------


def _cpu_brand() -> str:
    """CPU brand string: macOS `sysctl -n machdep.cpu.brand_string`, falling
    back to `/proc/cpuinfo`'s "model name" line (Linux ARM boards -- Jetson,
    Raspberry Pi -- where `sysctl` does not exist).
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5.0,
        ).strip()
        if out:
            return out
    except Exception:  # noqa: BLE001
        pass
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _physical_cores() -> "int | None":
    try:
        import psutil  # noqa: PLC0415

        return psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        return None


def _translated_under_rosetta() -> bool:
    """True iff THIS process is running translated under Rosetta 2
    (macOS `sysctl -n sysctl.proc_translated` == "1").

    Returns False (never raises) when the sysctl doesn't exist (non-macOS,
    or a macOS without Rosetta installed) or the call otherwise fails --
    both cases genuinely mean "not translated" for this process.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "sysctl.proc_translated"], text=True, timeout=5.0,
        ).strip()
        return out == "1"
    except Exception:  # noqa: BLE001
        return False


def _hardware_arch() -> str:
    """True physical CPU architecture, independent of THIS Python process's
    own build/personality.

    `platform.machine()` reports the architecture of the RUNNING PROCESS's
    personality, which can differ from the underlying hardware's on macOS
    under Rosetta 2 translation. A plain `uname -m` subprocess call is NOT a
    reliable independent check of this: `uname` is itself translated by
    Rosetta when launched from a translated process, so it inherits the
    SAME x86_64 personality and reports `x86_64` too -- confirmed on this
    machine (anaconda Python 3.9.7, x86_64 build): shelling `uname -m` from
    it returned `x86_64` despite the hardware being genuine Apple Silicon,
    making that check self-refuting rather than independent.

    The reliable macOS signal is `sysctl -n hw.optional.arm64`, a
    kernel-level hardware-capability query (not a subprocess personality)
    that reports `1` iff the PHYSICAL CPU is ARM64, regardless of which
    architecture the calling (or any spawned) process was built for.
    `machine_info` additionally surfaces `rosetta_translated`
    (`sysctl -n sysctl.proc_translated`) so the report can present coherent,
    non-self-refuting evidence: e.g. `cpu_brand` = "Apple M1 Pro" +
    `rosetta_translated` = True + `hardware_arch` = "arm64" together mean
    "Rosetta 2 translation on ARM64 silicon", verified via a kernel query
    independent of the translated process's own personality.

    Falls back to `/proc/cpuinfo` on Linux (no `sysctl`, and no Rosetta-style
    translation layer to worry about there), and finally to
    `platform.machine()` if neither signal is available.
    """
    import platform
    import subprocess

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.optional.arm64"], text=True, timeout=5.0,
        ).strip()
        if out == "1":
            return "arm64"
        if out == "0":
            return "x86_64"
    except Exception:  # noqa: BLE001
        pass
    try:
        text = Path("/proc/cpuinfo").read_text()
        if any(line.lower().startswith("cpu architecture") for line in text.splitlines()):
            return "arm64"
    except Exception:  # noqa: BLE001
        pass
    return platform.machine() or "unknown"


def machine_info() -> dict:
    """Machine-header dict for the latency report: platform, process arch,
    true hardware arch, CPU brand, core counts, library versions, hostname,
    timestamp.
    """
    import platform
    import socket
    from datetime import datetime, timezone
    from importlib.metadata import PackageNotFoundError, version

    try:
        gudhi_version = version("gudhi")
    except PackageNotFoundError:
        gudhi_version = "unknown"

    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "hardware_arch": _hardware_arch(),
        "rosetta_translated": _translated_under_rosetta(),
        "cpu_brand": _cpu_brand(),
        "physical_cores": _physical_cores(),
        "logical_cores": os.cpu_count(),
        "python_version": platform.python_version(),
        "gudhi_version": gudhi_version,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Shared setup (loaded ONCE, outside every timed region)
# ---------------------------------------------------------------------------


def _load_scalers(ws: Workspace) -> "dict[str, object]":
    """Per-manifold StandardScaler, from the combined `outputs/scalers.pkl`.

    INSPECTED both candidates named in the plan (`outputs/scaler_{m}.pkl`
    and `outputs/scalers.pkl`): on this repo, the per-manifold
    `outputs/scaler_{m}.pkl` files are STALE -- fitted on a DIFFERENT,
    older feature schema (e.g. `c2`'s scaler expects 5 columns named
    `SrcAddr_octet`/`SrcPort_binary`/..., not the current `config.
    C2_FEATURES`'s 7 columns `SrcAddr_last_octet`/`SrcPort_9`/`SrcPort_654`/
    ...). Neither `pipeline.py` nor any `uav_tda` module writes
    `scaler_{m}.pkl` -- it is leftover from an earlier, incompatible run.
    The combined `outputs/scalers.pkl` IS written by `data.run_prep`
    (`uav_tda/data.py`) against the CURRENT encoding and its per-manifold
    scalers' `n_features_in_` match `len(config.MANIFOLDS[m])` exactly
    (verified: c2=7, network=10, physical=5) -- it is the only correct
    source and is used unconditionally.
    """
    combined = pickle.loads((ws.outputs_dir / "scalers.pkl").read_bytes())
    scalers: dict = {}
    for m, cols in config.MANIFOLDS.items():
        scaler = combined[m]
        n_expected = len(cols)
        n_actual = getattr(scaler, "n_features_in_", n_expected)
        if n_actual != n_expected:
            raise AssertionError(
                f"latency._load_scalers: outputs/scalers.pkl['{m}'] expects "
                f"{n_actual} features but config.MANIFOLDS[{m!r}] has "
                f"{n_expected} -- stale/mismatched scaler artifact"
            )
        scalers[m] = scaler
    return scalers


def baseline_source(ws: Workspace, fresh: bool = False) -> str:
    """"committed" (the exact campaign's persisted baselines) or "fresh" (built here)."""
    manifest = ws.tables_dir / "rebuild" / "exact" / "baselines_manifest.json"
    return "committed" if manifest.exists() and not fresh else "fresh"


def _compute_fresh_baselines(ws: Workspace) -> dict:
    """Baseline barcodes of the workspace's own 500-point reference cloud.

    The same construction as `probe._load_baseline_barcodes` (Rips on exactly
    the reference points, production sparse config, per-dim slices), but
    relative to ``ws`` instead of the repo-fixed outputs directory.
    """
    from . import tda as tda_module

    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    clouds = tda_module.compute_reference_clouds(
        ws, np.load(ws.outputs_dir / "reference_indices.npy"))
    baselines: dict = {}
    for m in config.MANIFOLDS:
        hd = config.MAX_HOM_DIM[m]
        # {all but the last point} u {last point} == the reference cloud itself
        diag = tda_module._persistence_for_point(
            clouds[m][-1], clouds[m][:-1], max_edge_lengths[m], hd + 1,
            config.SPARSE_RIPS_EPSILON.get(m),
        )
        baselines[m] = {
            dim: probe._slice_dim(diag, dim, max_edge_lengths[m]) for dim in range(hd + 1)
        }
    return baselines


def _load_baselines_for_harness(ws: Workspace, fresh: bool = False) -> dict:
    """Per-manifold per-dim baseline barcodes for the per-flow W2 stage.

    Reuses the Track-A campaign's persisted baselines if present, unless
    ``fresh``; otherwise builds them on this machine.

    Use ``fresh=True`` on any machine other than the one that persisted the
    baselines. Diagrams that are bit-identical on one machine can differ in
    the last bit from another machine's, and hera may then never terminate
    (`uav_tda.hangcheck`; first seen on a Raspberry Pi 5, 2026-11-23).
    """
    if baseline_source(ws, fresh) == "committed":
        from . import exact as exact_module  # noqa: PLC0415

        return exact_module.load_baselines(ws.tables_dir / "rebuild" / "exact")
    return _compute_fresh_baselines(ws)


def _truncate(diagram_dim: np.ndarray, top_k: "int | None") -> np.ndarray:
    return diagram_dim if top_k is None else probe.truncate_top_k(diagram_dim, top_k)


def _raw_manifold_vectors(
    ws: Workspace, test_positions: np.ndarray, scalers: "dict[str, object]",
) -> "dict[str, pd.DataFrame]":
    """Genuine pre-scaler feature vectors for sampled test-split rows.

    Reads each manifold's ALREADY-STANDARDIZED `outputs/{m}_test.csv` at the
    sampled row positions and inverse-transforms via the loaded
    `StandardScaler` -- an exact affine inverse, not an approximation. See
    module docstring for why this is the leanest genuinely-faithful source
    of raw vectors (no raw dataset file needed). Returned as a DataFrame
    (named columns, matching what the scaler was originally fit on) so the
    later timed `scaler.transform` call carries feature names too and
    doesn't trip sklearn's "X does not have valid feature names" warning
    inside the timed region.
    """
    out: dict = {}
    for m, cols in config.MANIFOLDS.items():
        standardized = pd.read_csv(ws.outputs_dir / f"{m}_test.csv").iloc[test_positions]
        raw = scalers[m].inverse_transform(standardized)
        out[m] = pd.DataFrame(raw, columns=list(cols))
    return out


# ---------------------------------------------------------------------------
# Per-flow arm
# ---------------------------------------------------------------------------


def time_per_flow_decision(
    ws: Workspace, n: int = 30, rng_seed: int = 0, *,
    top_k: "int | None" = config.PROBE_TOP_K, delta: float = config.PROBE_DELTA,
    baselines: "dict | None" = None, progress=None, label: str = "per_flow",
) -> pd.DataFrame:
    """Full per-flow production inference path, timed per stage per manifold.

    Per sampled test flow, per manifold, the PAPER'S DEPLOYED config
    (`config.SPARSE_RIPS_EPSILON`, `config.PROBE_TOP_K`=50,
    `config.PROBE_DELTA`=0.2 -- matching `probe.run_probe`, NOT the Track-A
    exact-campaign's delta<=0.01/no-top-K config):

    1. `scaler.transform` on the genuine pre-scaler feature vector
       (`_raw_manifold_vectors` -- see module docstring).
    2. `tda._persistence_for_point({query} u 500 reference points,
       production sparse config)` -- the SAME per-flow Rips construction
       `uav-tda tda` uses in production.
    3. `probe._slice_dim` + `probe.truncate_top_k(k=PROBE_TOP_K)` per
       homology dimension.
    4. hera Wasserstein-2 (`delta=PROBE_DELTA`) vs the (once-loaded)
       baseline barcode, summed over dims.

    ``top_k=None, delta=exact.PRIMARY_DELTA`` times the full-test campaign's
    scoring configuration instead (full diagrams, high-precision W2 -- the
    configuration behind the 0.9082 headline AUC); Rips is identical.
    ``progress(label)`` is called before each flow, outside the timed calls.

    Setup (scalers, the 500-point reference cloud, baseline barcodes) is
    loaded ONCE before the timed loop; only the transform/Rips/slice/W2
    CALLS themselves are timed. Columns: `row_idx` (0-indexed position in
    the test split) + `{m}_scaler_s` / `{m}_rips_s` / `{m}_slice_s` /
    `{m}_w2_s` / `{m}_total_s` per manifold + `total_s` (sum of the 3
    manifolds' `{m}_total_s` -- the full multi-manifold per-flow decision
    cost).
    """
    import time

    from gudhi.hera import wasserstein_distance as wdist

    from . import tda as tda_module

    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    reference_idx = np.load(ws.outputs_dir / "reference_indices.npy")
    reference_clouds = tda_module.compute_reference_clouds(ws, reference_idx)
    if baselines is None:
        baselines = _load_baselines_for_harness(ws)
    scalers = _load_scalers(ws)

    n_test = len(pd.read_csv(ws.outputs_dir / "labels_test.csv"))
    n_sample = min(n, n_test)
    rng = np.random.default_rng(rng_seed)
    test_positions = np.sort(rng.choice(n_test, size=n_sample, replace=False))

    raw_vectors = _raw_manifold_vectors(ws, test_positions, scalers)

    rows = []
    for row_i, pos in enumerate(test_positions):
        if progress is not None:
            progress(f"{label} {row_i + 1}/{n_sample}: flow {int(pos)}")
        row: dict = {"row_idx": int(pos)}
        total_all = 0.0
        for m in config.MANIFOLDS:
            max_edge = max_edge_lengths[m]
            hd = config.MAX_HOM_DIM[m]
            sparse = config.SPARSE_RIPS_EPSILON.get(m)
            scaler = scalers[m]

            t0 = time.perf_counter()
            scaled = scaler.transform(raw_vectors[m].iloc[row_i:row_i + 1])[0]
            scaler_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            diag = tda_module._persistence_for_point(
                scaled, reference_clouds[m], max_edge, hd + 1, sparse,
            )
            rips_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            flow_slices = [
                _truncate(probe._slice_dim(diag, dim, max_edge), top_k)
                for dim in range(hd + 1)
            ]
            slice_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            total_w2 = 0.0
            for dim, flow_k in enumerate(flow_slices):
                base_k = _truncate(baselines[m][dim], top_k)
                total_w2 += float(
                    wdist(flow_k, base_k, order=2.0, internal_p=2.0, delta=delta)
                )
            w2_s = time.perf_counter() - t0

            manifold_total = scaler_s + rips_s + slice_s + w2_s
            row[f"{m}_scaler_s"] = scaler_s
            row[f"{m}_rips_s"] = rips_s
            row[f"{m}_slice_s"] = slice_s
            row[f"{m}_w2_s"] = w2_s
            row[f"{m}_total_s"] = manifold_total
            total_all += manifold_total
        row["total_s"] = total_all
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Windowed arm
# ---------------------------------------------------------------------------


def time_windowed_decision(
    ws: Workspace, w: int, n: int = 30, rng_seed: int = 0, *, progress=None,
) -> pd.DataFrame:
    """Full windowed production inference path, timed per stage per manifold.

    Baseline medoid setup (val-Normal windows -> per-manifold W2-medoid,
    matching `windowed.run_windowed`'s baseline construction exactly) is
    built ONCE, OUTSIDE the timed per-window loop; its wall-clock cost is
    recorded as a CONSTANT `baseline_setup_s` column on every returned row
    -- the FRONTIER-COST convention `windowed.build_frontier_table` already
    uses (marginal per-decision cost reported separately from the one-time
    fixed setup cost, never folded together).

    Per sampled TEST-split window (FlowID order, `windowed.make_windows`,
    the production "ordered" configuration):
    1. `windowed.window_diagram` per manifold (`config.WINDOWED_SPARSE`).
    2. `windowed.w2_distance` vs the ONE medoid baseline built above.

    Columns: `window_idx`, `start_pos`, `baseline_setup_s` (constant) +
    `{m}_rips_s` / `{m}_w2_s` / `{m}_total_s` per manifold + `total_s` (sum
    of the 3 manifolds' `{m}_total_s` -- the MARGINAL per-window cost, net
    of the fixed baseline setup).
    """
    import time

    from . import windowed as windowed_module
    from .tda import load_labels_for_split, load_split_manifold

    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())

    if progress is not None:
        progress(f"windowed W={w}: building val-Normal baseline")
    t_setup0 = time.perf_counter()
    labels_val = load_labels_for_split(ws, "val").to_numpy()
    normal_mask = labels_val == _NORMAL_LABEL
    baseline_diagrams: dict = {}
    for m in config.MANIFOLDS:
        max_edge = max_edge_lengths[m]
        hd = config.MAX_HOM_DIM[m]
        sparse = config.WINDOWED_SPARSE[m]
        points_val = load_split_manifold(ws, m, "val")[normal_mask]
        val_windows = windowed_module.make_windows(len(points_val), w)
        val_diagrams = [
            windowed_module.window_diagram(points_val[idx], max_edge, hd, sparse)
            for idx in val_windows
        ]
        _, baseline_diag = windowed_module.baseline_medoid_diagram(val_diagrams, max_edge, hd)
        baseline_diagrams[m] = baseline_diag
    baseline_setup_s = time.perf_counter() - t_setup0

    labels_test = load_labels_for_split(ws, "test").to_numpy()
    test_windows = windowed_module.make_windows(len(labels_test), w)
    n_windows = len(test_windows)
    n_sample = min(n, n_windows)
    rng = np.random.default_rng(rng_seed)
    sample_positions = np.sort(rng.choice(n_windows, size=n_sample, replace=False))

    points_test = {m: load_split_manifold(ws, m, "test") for m in config.MANIFOLDS}

    if progress is not None:
        progress(f"windowed W={w}: timing {n_sample} windows")
    rows = []
    for pos in sample_positions:
        idx = test_windows[pos]
        row: dict = {
            "window_idx": int(pos), "start_pos": int(pos) * w,
            "baseline_setup_s": baseline_setup_s,
        }
        total_all = 0.0
        for m in config.MANIFOLDS:
            max_edge = max_edge_lengths[m]
            hd = config.MAX_HOM_DIM[m]
            sparse = config.WINDOWED_SPARSE[m]

            t0 = time.perf_counter()
            diag = windowed_module.window_diagram(points_test[m][idx], max_edge, hd, sparse)
            rips_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            score = windowed_module.w2_distance(diag, baseline_diagrams[m], max_edge, hd)
            w2_s = time.perf_counter() - t0
            del score  # score itself is not part of the timing table

            manifold_total = rips_s + w2_s
            row[f"{m}_rips_s"] = rips_s
            row[f"{m}_w2_s"] = w2_s
            row[f"{m}_total_s"] = manifold_total
            total_all += manifold_total
        row["total_s"] = total_all
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------


def _summary_rows_for_frame(mode: str, w, df: pd.DataFrame) -> list:
    rows = []
    stage_cols = [c for c in df.columns if c.endswith("_s")]
    n_rows = len(df)
    for col in stage_cols:
        vals = df[col].to_numpy(dtype=float)
        rows.append({
            "mode": mode, "w": w, "stage": col,
            "mean_s": float(np.mean(vals)), "median_s": float(np.median(vals)),
            "p95_s": float(np.percentile(vals, 95)), "n": n_rows,
        })
    return rows


def _build_summary_table(
    per_flow_df: pd.DataFrame, windowed_frames: "dict[int, pd.DataFrame]",
    per_flow_exact_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    rows = _summary_rows_for_frame("per_flow", None, per_flow_df)
    if per_flow_exact_df is not None:
        rows.extend(_summary_rows_for_frame("per_flow_exact", None, per_flow_exact_df))
    for w, df_w in sorted(windowed_frames.items()):
        rows.extend(_summary_rows_for_frame("windowed", w, df_w))
    return pd.DataFrame(rows)


_PAPER_CLAIM_TEXT = (
    "Per-flow inference latency, measured on a 2020-era laptop, is on the "
    "order of 1-3 seconds; we do not characterize onboard performance in "
    "this extended abstract."
)
_PAPER_CLAIM_LO, _PAPER_CLAIM_HI = 1.0, 3.0

_COMPANION_ARTIFACTS = [
    "outputs/{m}_train.csv, outputs/{m}_test.csv for m in {c2, network, physical} "
    "(the 500-point reference cloud is drawn from *_train.csv; the sampled "
    "flows are drawn from *_test.csv)",
    "outputs/{m}_val.csv for m in {c2, network, physical} (windowed arm only "
    "-- val-Normal windows build the per-W medoid baseline)",
    "outputs/reference_indices.npy",
    "outputs/max_edge_lengths.json",
    "outputs/labels_test.csv, outputs/labels_val.csv",
    "outputs/scalers.pkl (the combined per-manifold StandardScaler dict written by "
    "data.run_prep; the per-manifold outputs/scaler_{m}.pkl files on this repo are "
    "STALE -- fitted on a different, older feature schema -- and are NOT used)",
    "results/tables/rebuild/exact/baselines_*.npy + baselines_manifest.json "
    "(OPTIONAL, and NOT to be used on a different machine: pass --fresh-baselines "
    "there, see uav_tda/hangcheck.py)",
]

_EXCLUDED_ARTIFACTS = (
    "The harness needs NONE of outputs/persistence_diagrams/*.pkl (hundreds "
    "of MB to 1.5GB each) -- every diagram it times is computed FRESH, "
    "on the fly, from the tiny CSV/npy/json artifacts above, exactly as a "
    "deployed detector would."
)


def _fmt_stat_row(r: pd.Series) -> str:
    return f"| {r['stage']} | {r['mean_s']:.4f} | {r['median_s']:.4f} | {r['p95_s']:.4f} | {int(r['n'])} |"


def load_latency_hosts(ws: Workspace) -> "list[dict]":
    """Every host with a committed latency summary, primary host first.

    Each entry: `info` (machine_info from the summary's sidecar), `n`,
    `summary` (the summary CSV), `exact` (0.9082-configuration stats, from
    the detail CSV when present, else the summary's own rows, else None),
    `exact_meta` (that arm's sidecar), `params` (the summary's sidecar params).
    """
    rebuild = ws.tables_dir / "rebuild"
    hosts = []
    for path in sorted(rebuild.glob("latency_*.csv")):
        if path.read_text().split("\n", 1)[0].split(",")[:3] != ["mode", "w", "stage"]:
            continue
        side = json.loads(Path(str(path) + ".provenance.json").read_text())
        params = side["params"]
        summary = pd.read_csv(path, float_precision="round_trip")
        name = params["hostname"]
        exact, exact_meta = None, None
        detail = rebuild / f"latency_{name}_per_flow_exact.csv"
        if detail.exists():
            exact = pd.DataFrame(_summary_rows_for_frame(
                "per_flow_exact", None, pd.read_csv(detail, float_precision="round_trip")))
            dside = json.loads(Path(str(detail) + ".provenance.json").read_text())
            exact_meta = {**dside["params"], "created_utc": dside["created_utc"]}
        elif (summary["mode"] == "per_flow_exact").any():
            exact = summary[summary["mode"] == "per_flow_exact"]
            exact_meta = dict(params)
        reg = config.LATENCY_HOSTS.get(name, {})
        hosts.append({"host": name, "label": reg.get("label", name), "role": reg.get("role"),
                      "info": params["machine_info"], "n": params["n"], "summary": summary,
                      "exact": exact, "exact_meta": exact_meta, "params": params,
                      "created_utc": side["created_utc"]})
    hosts.sort(key=lambda h: (h["host"] != config.LATENCY_PRIMARY_HOST, h["host"]))
    return hosts


_STAGE_ORDER = ("scaler_s", "rips_s", "slice_s", "w2_s", "total_s")
_PER_FLOW_COLS = [f"{m}_{s}" for m in config.MANIFOLDS for s in _STAGE_ORDER] + ["total_s"]


def _stat_table(rows: pd.DataFrame, cols) -> list:
    out = ["| Stage | mean (s) | median (s) | p95 (s) | n |",
           "| :--- | ---: | ---: | ---: | ---: |"]
    for col in cols:
        row = rows[rows["stage"] == col]
        if len(row):
            out.append(_fmt_stat_row(row.iloc[0]))
    return out + [""]


def _arch_disclosure(info: dict) -> str:
    machine, hw = info["machine"], info["hardware_arch"]
    translated = bool(info.get("rosetta_translated", False))
    if hw.lower() in ("arm64", "aarch64") and translated:
        return (
            f"This Python process reports `{machine}`, not ARM64: the Python "
            f"{info['python_version']} build is a {machine} binary running under Rosetta 2 "
            f"on ARM64 hardware (`hardware_arch` = `{hw}` from a kernel-level sysctl; "
            f"`rosetta_translated` = True). The measurements ran under x86_64-on-ARM64 "
            "emulation; a native build on the same hardware could measure differently, and "
            "this report does not assume faster or slower.")
    if hw.lower() in ("arm64", "aarch64"):
        return (f"`platform.machine()` reports `{machine}` (ARM64) directly and "
                "`rosetta_translated` = False: the process ran natively.")
    return (f"`hardware_arch` is `{hw}`, a native non-ARM host "
            f"(`rosetta_translated` = {translated}).")


def _role_paragraph(h: dict) -> str:
    info, role = h["info"], h["role"]
    if role == "companion":
        text = ("This is a companion-class board, so these are measured companion-hardware "
                "latencies, not a lower bound inferred from another machine. They are for this "
                "board and software stack only, on mains power with its own cooler; they say "
                "nothing about a battery-powered airframe or other boards.")
    elif role == "workstation" or info["cpu_brand"].strip() == "Apple M1 Pro":
        text = ("This is a workstation-class machine: its power budget, thermal envelope, core "
                "count and clock behaviour exceed a companion board's, so these numbers are a "
                "LOWER BOUND on onboard latency, not a companion-hardware measurement.")
    else:
        text = ("This host is not classified in `config.LATENCY_HOSTS`, so no claim is made "
                "here about how it relates to companion-class hardware.")
    return f"{_arch_disclosure(info)} {text}"


def _band_verdict(x: float) -> str:
    if _PAPER_CLAIM_LO <= x <= _PAPER_CLAIM_HI:
        return "within the claimed 1-3s band"
    if x < _PAPER_CLAIM_LO:
        return f"BELOW the claimed 1-3s band ({x:.4f}s < {_PAPER_CLAIM_LO}s)"
    return f"ABOVE the claimed 1-3s band ({x:.4f}s > {_PAPER_CLAIM_HI}s)"


def _host_section(h: dict, idx: int) -> list:
    info, s = h["info"], h["summary"]
    pf = s[s["mode"] == "per_flow"]
    L = [f"## {idx}. Host: {h['label']}", ""]
    if h["params"].get("rerender_reason"):
        L += [f"_Sidecar note: {h['params']['rerender_reason']}_", ""]
    L += ["| Field | Value |", "| :--- | :--- |"]
    for key in ("platform", "machine", "hardware_arch", "rosetta_translated", "cpu_brand",
                "physical_cores", "logical_cores", "python_version", "gudhi_version",
                "hostname", "timestamp_utc"):
        L.append(f"| {key} | {info[key]} |")
    L += ["", _role_paragraph(h), ""]

    L += [f"### {idx}.1 Per-flow arm, probe configuration", "",
          f"n={h['n']} sampled test flows, "
          f"`SPARSE_RIPS_EPSILON={config.SPARSE_RIPS_EPSILON}`, "
          f"`PROBE_TOP_K={config.PROBE_TOP_K}`, `PROBE_DELTA={config.PROBE_DELTA}` (pairs with "
          "the 0.8930 ten-seed probe AUC, not with 0.9082). The scaler stage is timed on the "
          "genuine pre-scaler features, recovered by the scaler's exact inverse.", ""]
    L += _stat_table(pf, _PER_FLOW_COLS)
    src = h["params"].get("baseline_source", "committed")
    L += [f"Baseline source: **{src}** (committed = the exact campaign's persisted barcodes; "
          "fresh = built on this machine before timing).", ""]

    if h["exact"] is not None:
        meta = h["exact_meta"] or {}
        L += [f"### {idx}.2 Per-flow arm, 0.9082 configuration", "",
              "Same sampled flows and Rips construction, scored like the full-test campaign "
              "behind the 0.9082 headline AUC (`paper/EXACT_RESULTS.md`): full diagrams (no "
              "top-K) and hera delta<=0.01. Rips is re-run, not reused. Baseline source: "
              f"**{meta.get('baseline_source', src)}**.", ""]
        created = str(meta.get("created_utc", ""))
        if created[:10] and created[:10] != str(h["created_utc"])[:10]:
            L += [f"_Timed in a separate session from the probe arm ({created[:10]} vs "
                  f"{str(h['created_utc'])[:10]}); compare the two arms with that in mind._", ""]
        L += _stat_table(h["exact"], _PER_FLOW_COLS)

    L += [f"### {idx}.3 Windowed arm", "",
          f"n={h['n']} sampled test-split windows per W (FlowID order). The val-Normal medoid "
          "baseline is built once per W outside the timed loop and reported separately as "
          "`baseline_setup_s`.", ""]
    wm = s[s["mode"] == "windowed"]
    for w in sorted(wm["w"].unique()):
        ws_ = wm[wm["w"] == w]
        setup = float(ws_.loc[ws_["stage"] == "baseline_setup_s", "mean_s"].iloc[0])
        L += [f"#### W={int(w)} (baseline_setup_s = {setup:.4f}s, fixed one-time cost)", ""]
        L += _stat_table(ws_, [f"{m}_{st}" for m in config.MANIFOLDS
                               for st in ("rips_s", "w2_s", "total_s")] + ["total_s"])

    total = pf[pf["stage"] == "total_s"].iloc[0]
    L += [f"### {idx}.4 The paper's \"1-3 s per flow\" claim vs measured", "",
          f"Published claim (`CLAUDE/PROJECT_BRIEF.md`): \"{_PAPER_CLAIM_TEXT}\"", "",
          f"Per-flow decision (probe configuration, all three manifolds): mean "
          f"**{float(total['mean_s']):.4f}s**, median {float(total['median_s']):.4f}s, p95 "
          f"{float(total['p95_s']):.4f}s -- {_band_verdict(float(total['mean_s']))}."]
    if h["exact"] is not None:
        ex = h["exact"][h["exact"]["stage"] == "total_s"].iloc[0]
        L += [f" 0.9082 configuration: mean **{float(ex['mean_s']):.4f}s**, p95 "
              f"{float(ex['p95_s']):.4f}s -- {_band_verdict(float(ex['mean_s']))}."]
    L += ["", "| Manifold | mean total_s | median | p95 | vs 1-3s band |",
          "| :--- | ---: | ---: | ---: | :--- |"]
    for m in config.MANIFOLDS:
        r = pf[pf["stage"] == f"{m}_total_s"].iloc[0]
        L.append(f"| {m} | {float(r['mean_s']):.4f} | {float(r['median_s']):.4f} | "
                 f"{float(r['p95_s']):.4f} | {_band_verdict(float(r['mean_s']))} |")
    return L + [""]


def _cross_host_section(hosts: list) -> list:
    prim = hosts[0]
    L = ["## 1. Per-decision cost by host (mean total_s / p95, seconds)", "",
         "Ratios are each host's mean divided by the first host's. They are not pure hardware "
         "comparisons: software stacks and baseline sources differ (see each host's section). "
         "Speedups inside a host compare one window with one flow.", "",
         "| Decision | " + " | ".join(f"{h['label']}" for h in hosts)
         + " | " + " | ".join(f"{h['label']} / {prim['label']}" for h in hosts[1:])
         + " |",
         "| :--- | " + " | ".join("---:" for _ in range(2 * len(hosts) - 1)) + " |"]

    def stat(h, mode, w):
        if mode == "per_flow_exact":
            rows = h["exact"] if h["exact"] is not None else h["summary"].iloc[0:0]
        else:
            rows = h["summary"][h["summary"]["mode"] == mode]
        sel = rows[rows["stage"] == "total_s"]
        if mode == "windowed":
            sel = sel[sel["w"] == w]
        return None if sel.empty else sel.iloc[0]

    ws_all = sorted({int(w) for h in hosts for w in
                     h["summary"].loc[h["summary"]["mode"] == "windowed", "w"].unique()})
    items = [("Per-flow, probe (0.8930)", "per_flow", None),
             ("Per-flow, full diagrams (0.9082)", "per_flow_exact", None)]
    items += [(f"Windowed W={w}", "windowed", w) for w in ws_all]
    for name, mode, w in items:
        cells, means = [], []
        for h in hosts:
            r = stat(h, mode, w)
            means.append(None if r is None else float(r["mean_s"]))
            cells.append("not measured" if r is None
                         else f"{float(r['mean_s']):.4f} / {float(r['p95_s']):.4f}")
        ratios = ["–" if m is None or means[0] is None else f"{m / means[0]:.2f}"
                  for m in means[1:]]
        L.append(f"| {name} | " + " | ".join(cells + ratios) + " |")
    return L + [""]


def write_latency_report(ws: Workspace, render_note: "str | None" = None) -> Path:
    """Render `paper/LATENCY_RESULTS.md` for every measured host, from committed CSVs only."""
    hosts = load_latency_hosts(ws)
    L = ["# LATENCY_RESULTS: Portable Onboard-Latency Harness (Phase 6, Track B)", "",
         "_Rendered by `uav-tda latency-report` (`uav_tda/latency.py`) from the per-host CSVs "
         "and sidecars in `results/tables/rebuild/`; each host's numbers are machine-measured "
         "by `uav-tda latency` on that host (full inference path: standardize -> Rips -> "
         "slice/truncate -> Wasserstein-2). Nothing here is re-measured or re-typed._", ""]
    if render_note:
        L += [render_note, ""]
    if len(hosts) > 1:
        L += _cross_host_section(hosts)
    for i, h in enumerate(hosts, start=2 if len(hosts) > 1 else 1):
        L += _host_section(h, i)

    L += ["## Companion-hardware one-command instructions", "",
          "The harness runs unmodified on a companion-class board given these artifacts "
          "copied from this workspace (small: no persistence-diagram pickles):", ""]
    L += [f"- `{a}`" for a in _COMPANION_ARTIFACTS]
    L += ["", _EXCLUDED_ARTIFACTS, "", "```bash", "pip install -e .",
          "uav-tda latency --fresh-baselines --exact-arm", "```", "",
          "Output files are keyed to the host name, so hosts never overwrite each other; "
          "`uav-tda latency-report` then renders every host's section here.", "",
          "## Pending sign-off", "",
          "These latencies are candidates for the manuscript's §V characterization and are "
          "pending author sign-off. Every number traces to "
          "`results/tables/rebuild/latency_<host>.csv`, its per-mode detail CSVs and "
          "sidecars.", ""]
    paper_dir = ws.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    report_path = paper_dir / "LATENCY_RESULTS.md"
    report_path.write_text("\n".join(L))
    return report_path


def _time_exact_arm(ws, n, baselines, src, info, rebuild_dir, dog) -> pd.DataFrame:
    """Time the 0.9082 configuration and write its detail CSV + sidecar."""
    from . import exact as exact_module

    dog.step("per_flow_exact: start")
    df = time_per_flow_decision(
        ws, n=n, baselines=baselines, progress=dog.step, label="per_flow_exact",
        top_k=None, delta=exact_module.PRIMARY_DELTA,
    )
    path = rebuild_dir / f"latency_{info['hostname']}_per_flow_exact.csv"
    df.to_csv(path, index=False)
    write_provenance(path, {
        "mode": "per_flow_exact", "n": n, "baseline_source": src,
        "top_k": None, "delta": exact_module.PRIMARY_DELTA, "machine_info": info,
    })
    return df


def run_latency(
    ws: Workspace, n: int = 30, fresh_baselines: bool = False, exact_arm: bool = False,
    watchdog_s: float = DEFAULT_WATCHDOG_S, exact_only: bool = False,
) -> Path:
    """Full latency harness: the per-flow arm(s) + every W in `config.WINDOW_SIZES`.

    Writes `results/tables/rebuild/latency_{hostname}.csv` (mean/median/p95
    summary per stage per mode) + one detail CSV per mode
    (`latency_{hostname}_per_flow.csv`, `latency_{hostname}_w{W}.csv`, and
    with ``exact_arm`` `latency_{hostname}_per_flow_exact.csv`), each with a
    provenance sidecar, and `paper/LATENCY_RESULTS.md`. Returns the report
    path. ``fresh_baselines`` builds the per-flow baselines on this machine;
    ``watchdog_s`` kills the run if any single step takes longer (0 = off).
    ``exact_only`` adds just the 0.9082 arm to a host that already has a
    summary: it writes that arm's detail CSV + sidecar and leaves every other
    file of the earlier run untouched.
    """
    ws.ensure()
    info = machine_info()
    hostname = info["hostname"]
    rebuild_dir = ws.tables_dir / "rebuild"
    rebuild_dir.mkdir(parents=True, exist_ok=True)

    src = baseline_source(ws, fresh_baselines)
    if exact_only and not (rebuild_dir / f"latency_{hostname}.csv").exists():
        raise FileNotFoundError(
            f"--exact-only adds to an existing run, but there is no summary "
            f"latency_{hostname}.csv in {rebuild_dir}")
    dog = Watchdog(watchdog_s)
    try:
        dog.step(f"baselines ({src})")
        baselines = _load_baselines_for_harness(ws, fresh=fresh_baselines)

        if exact_only:
            _time_exact_arm(ws, n, baselines, src, info, rebuild_dir, dog)
            dog.cancel()
            return write_latency_report(ws)

        dog.step("per_flow: start")
        per_flow_df = time_per_flow_decision(ws, n=n, baselines=baselines, progress=dog.step)
        per_flow_path = rebuild_dir / f"latency_{hostname}_per_flow.csv"
        per_flow_df.to_csv(per_flow_path, index=False)
        write_provenance(per_flow_path, {
            "mode": "per_flow", "n": n, "baseline_source": src,
            "top_k": config.PROBE_TOP_K, "delta": config.PROBE_DELTA,
        })

        per_flow_exact_df = None
        if exact_arm:
            per_flow_exact_df = _time_exact_arm(ws, n, baselines, src, info, rebuild_dir, dog)

        windowed_frames: "dict[int, pd.DataFrame]" = {}
        for w in config.WINDOW_SIZES:
            dog.step(f"windowed W={w}: start")
            df_w = time_windowed_decision(ws, w, n=n, progress=dog.step)
            windowed_frames[w] = df_w
            w_path = rebuild_dir / f"latency_{hostname}_w{w}.csv"
            df_w.to_csv(w_path, index=False)
            write_provenance(w_path, {"mode": f"windowed_w{w}", "n": n, "w": w})
    finally:
        dog.cancel()

    summary_df = _build_summary_table(per_flow_df, windowed_frames, per_flow_exact_df)
    summary_path = rebuild_dir / f"latency_{hostname}.csv"
    summary_df.to_csv(summary_path, index=False)
    write_provenance(summary_path, {
        "n": n, "hostname": hostname, "machine_info": info, "baseline_source": src,
        "exact_arm": exact_arm,
    })

    report_path = write_latency_report(ws)
    log.info("latency harness complete: %s, %s", summary_path, report_path)
    return report_path
