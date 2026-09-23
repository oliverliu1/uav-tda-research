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

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, probe
from .provenance import write_provenance
from .workspace import Workspace

log = logging.getLogger("uav_tda.latency")

_NORMAL_LABEL = "Normal Traffic"


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


def _load_baselines_for_harness(ws: Workspace) -> dict:
    """Per-manifold per-dim baseline barcodes for the per-flow W2 stage.

    Prefers the Track-A campaign's already-persisted baselines
    (`results/tables/rebuild/exact/baselines_*.npy`, loaded via
    `exact.load_baselines` -- same shape/convention as `probe.
    _load_baseline_barcodes`'s return value, computed once and reused here
    rather than re-running sparse-Rips) if present; otherwise computes fresh
    via `probe._load_baseline_barcodes` (e.g. on a companion box without a
    prior exact campaign).
    """
    exact_dir = ws.tables_dir / "rebuild" / "exact"
    if (exact_dir / "baselines_manifest.json").exists():
        from . import exact as exact_module  # noqa: PLC0415

        return exact_module.load_baselines(exact_dir)
    max_edge_lengths = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    return probe._load_baseline_barcodes(max_edge_lengths)


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


def time_per_flow_decision(ws: Workspace, n: int = 30, rng_seed: int = 0) -> pd.DataFrame:
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
    baselines = _load_baselines_for_harness(ws)
    scalers = _load_scalers(ws)

    n_test = len(pd.read_csv(ws.outputs_dir / "labels_test.csv"))
    n_sample = min(n, n_test)
    rng = np.random.default_rng(rng_seed)
    test_positions = np.sort(rng.choice(n_test, size=n_sample, replace=False))

    raw_vectors = _raw_manifold_vectors(ws, test_positions, scalers)

    rows = []
    for row_i, pos in enumerate(test_positions):
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
                probe.truncate_top_k(probe._slice_dim(diag, dim, max_edge), config.PROBE_TOP_K)
                for dim in range(hd + 1)
            ]
            slice_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            total_w2 = 0.0
            for dim, flow_k in enumerate(flow_slices):
                base_k = probe.truncate_top_k(baselines[m][dim], config.PROBE_TOP_K)
                total_w2 += float(
                    wdist(flow_k, base_k, order=2.0, internal_p=2.0, delta=config.PROBE_DELTA)
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


def time_windowed_decision(ws: Workspace, w: int, n: int = 30, rng_seed: int = 0) -> pd.DataFrame:
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


def _build_summary_table(per_flow_df: pd.DataFrame, windowed_frames: "dict[int, pd.DataFrame]") -> pd.DataFrame:
    rows = _summary_rows_for_frame("per_flow", None, per_flow_df)
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
    "(OPTIONAL -- reused if present; otherwise the harness computes fresh "
    "baseline barcodes via probe._load_baseline_barcodes on first run)",
]

_EXCLUDED_ARTIFACTS = (
    "The harness needs NONE of outputs/persistence_diagrams/*.pkl (hundreds "
    "of MB to 1.5GB each) -- every diagram it times is computed FRESH, "
    "on the fly, from the tiny CSV/npy/json artifacts above, exactly as a "
    "deployed detector would."
)


def _fmt_stat_row(r: pd.Series) -> str:
    return f"| {r['stage']} | {r['mean_s']:.4f} | {r['median_s']:.4f} | {r['p95_s']:.4f} | {int(r['n'])} |"


def _write_latency_report(
    ws: Workspace, info: dict, per_flow_df: pd.DataFrame,
    windowed_frames: "dict[int, pd.DataFrame]", summary_df: pd.DataFrame, n: int,
    render_note: "str | None" = None,
) -> Path:
    """Render `paper/LATENCY_RESULTS.md` from already-computed frames.

    ``render_note``, when given, is inserted as an extra disclosure
    paragraph right after the intro -- used for a text-only RE-RENDER of an
    already-measured report (e.g. wording/arch-detection fixes) so the
    document itself discloses that no measurement was re-run. Production
    runs (`run_latency`) never pass it.
    """
    lines: list = []
    lines.append("# LATENCY_RESULTS: Portable Onboard-Latency Harness (Phase 6, Track B)")
    lines.append("")
    lines.append(
        "_Generated by `uav-tda latency` (`uav_tda/latency.py`) from a live run of the "
        "FULL production inference path -- standardize -> Rips -> slice/truncate -> "
        "Wasserstein-2 -- per sampled decision, on this machine. Every number below is "
        "machine-measured from that run; this report makes no claims of its own beyond "
        "restating and interpreting them._"
    )
    lines.append("")
    if render_note:
        lines.append(render_note)
    lines.append("")

    # --- 1. Machine header ---------------------------------------------------
    lines.append("## 1. Machine header")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| :--- | :--- |")
    for key in ("platform", "machine", "hardware_arch", "rosetta_translated", "cpu_brand",
                "physical_cores", "logical_cores", "python_version", "gudhi_version",
                "hostname", "timestamp_utc"):
        lines.append(f"| {key} | {info[key]} |")
    lines.append("")

    machine_field, hw_field = info["machine"], info["hardware_arch"]
    translated = bool(info.get("rosetta_translated", False))
    if hw_field.lower() in ("arm64", "aarch64") and translated:
        arch_disclosure = (
            f"**this Python process's `platform.machine()` reports `{machine_field}`, "
            f"NOT ARM64** -- this anaconda Python {info['python_version']} build is an "
            f"{machine_field} binary running under Apple's Rosetta 2 TRANSLATION LAYER "
            f"on genuinely ARM64 hardware. This is confirmed via a KERNEL-level hardware "
            f"query, `sysctl -n hw.optional.arm64` (`hardware_arch` = `{hw_field}` above) "
            "-- NOT via a plain `uname -m` subprocess call, which is itself translated by "
            f"Rosetta when launched from a translated process and would self-defeatingly "
            f"ALSO report `{machine_field}`, same as `platform.machine()`. "
            "`sysctl -n sysctl.proc_translated` (`rosetta_translated` above) "
            "independently confirms this process itself is the one being translated, and "
            f"`cpu_brand` names the actual silicon as \"{info['cpu_brand']}\" -- "
            "together, coherent, non-self-refuting evidence. The measurements below "
            "therefore ran under x86_64-on-ARM64 EMULATION, not natively -- an "
            "ADDITIONAL disclosure beyond the plan's original ARM-ISA-but-workstation-"
            "class framing: a native-arm64 Python build on this same hardware could "
            "measure differently (this report does not assume faster or slower either "
            "way), and a companion board's Python (typically a native ARM64 build with "
            "no translation layer) will not carry this same emulation overhead."
        )
    elif hw_field.lower() in ("arm64", "aarch64"):
        arch_disclosure = (
            f"this machine's `platform.machine()` reports `{machine_field}` (ARM64) "
            "directly -- the SAME instruction-set family as deployment-class companion "
            "boards (NVIDIA Jetson, Raspberry Pi); `rosetta_translated` = False confirms "
            "this process is running natively, with no translation layer involved."
        )
    else:
        arch_disclosure = (
            f"`hardware_arch` (the kernel-level `sysctl -n hw.optional.arm64` check on "
            "macOS, or the `/proc/cpuinfo`/`platform.machine()` fallback elsewhere) "
            f"reports `{hw_field}` -- a native, non-ARM host with no Rosetta-style "
            f"translation layer involved (`rosetta_translated` = {translated})."
        )

    cpu_brand = info["cpu_brand"]
    if cpu_brand.strip() == "Apple M1 Pro":
        machine_narrative = (
            "this is an Apple M1 Pro **workstation**: unified memory, thermal envelope, "
            "core count, and clock behavior are all substantially above a Jetson Nano/Orin "
            "or Raspberry Pi 4/5 running on battery/USB power in an airframe."
        )
    else:
        machine_narrative = (
            f"this is a workstation-class host (`cpu_brand` = \"{cpu_brand}\"): its power "
            "budget, thermal envelope, core count, and clock behavior are presumptively "
            "substantially above a Jetson Nano/Orin or Raspberry Pi 4/5 running on "
            "battery/USB power in an airframe, though this has NOT been verified against "
            "any specific companion-class board's datasheet -- a report generated on a "
            "different host should confirm this assumption explicitly rather than inherit "
            "it from a different machine's framing."
        )

    lines.append(
        f"**ARM-ISA-but-workstation-class disclosure**: {arch_disclosure} Regardless of "
        f"the process architecture, {machine_narrative} The numbers below are therefore a "
        "LOWER BOUND on real onboard latency, not a companion-hardware measurement -- "
        "disclosed per the user's 2026-09-22 decision to run the characterization on this "
        "machine (workstation-class, ARM-ISA-disclosed) and ship a one-command-portable "
        "harness (§6) for the eventual companion-class run, rather than block this report "
        "on acquiring that hardware."
    )
    lines.append("")

    # --- 2. Per-flow arm -------------------------------------------------------
    lines.append("## 2. Per-flow arm")
    lines.append("")
    lines.append(
        f"n={n} sampled test flows, the paper's DEPLOYED probe-style config "
        f"(`SPARSE_RIPS_EPSILON={config.SPARSE_RIPS_EPSILON}`, "
        f"`PROBE_TOP_K={config.PROBE_TOP_K}`, `PROBE_DELTA={config.PROBE_DELTA}` -- NOT "
        "the Track-A exact-campaign config). Stage 1 (`scaler transform`) is timed on "
        "the genuine pre-scaler feature vector, recovered via the StandardScaler's exact "
        "affine inverse from the stored standardized test row (see module docstring); "
        "stages 2-4 mirror `uav-tda tda`'s per-flow Rips construction + "
        "`probe.run_probe`'s slice/truncate/W2 scoring exactly."
    )
    lines.append("")
    pf_summary = summary_df[summary_df["mode"] == "per_flow"]
    lines.append("| Stage | mean (s) | median (s) | p95 (s) | n |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    stage_order = ["scaler_s", "rips_s", "slice_s", "w2_s", "total_s"]
    ordered_cols = [f"{m}_{s}" for m in config.MANIFOLDS for s in stage_order] + ["total_s"]
    for col in ordered_cols:
        row = pf_summary[pf_summary["stage"] == col]
        if len(row):
            lines.append(_fmt_stat_row(row.iloc[0]))
    lines.append("")

    # --- 3. Windowed arm ---------------------------------------------------
    lines.append("## 3. Windowed arm")
    lines.append("")
    lines.append(
        f"n={n} sampled test-split windows per W (FlowID order, the production "
        "\"ordered\" configuration), `config.WINDOWED_SPARSE` per manifold. The "
        "val-Normal medoid baseline is built ONCE per W, OUTSIDE the timed per-window "
        "loop; its cost is `baseline_setup_s` (a FIXED, one-time cost, reported "
        "separately -- NOT folded into the per-window `total_s`, matching the "
        "matched-compute-frontier convention already used in `paper/WINDOWED_RESULTS.md`)."
    )
    lines.append("")
    for w in sorted(windowed_frames):
        df_w = windowed_frames[w]
        w_summary = summary_df[(summary_df["mode"] == "windowed") & (summary_df["w"] == w)]
        setup_s = float(df_w["baseline_setup_s"].iloc[0]) if len(df_w) else float("nan")
        lines.append(f"### W={w} (baseline_setup_s = {setup_s:.4f}s, fixed one-time cost)")
        lines.append("")
        lines.append("| Stage | mean (s) | median (s) | p95 (s) | n |")
        lines.append("| :--- | ---: | ---: | ---: | ---: |")
        w_ordered_cols = [f"{m}_{s}" for m in config.MANIFOLDS for s in ("rips_s", "w2_s", "total_s")] + ["total_s"]
        for col in w_ordered_cols:
            row = w_summary[w_summary["stage"] == col]
            if len(row):
                lines.append(_fmt_stat_row(row.iloc[0]))
        lines.append("")

    # --- 4. Paper claim check --------------------------------------------------
    lines.append("## 4. Paper's \"1-3 s per flow\" claim vs measured")
    lines.append("")
    lines.append(f"Published claim (`CLAUDE/PROJECT_BRIEF.md`): \"{_PAPER_CLAIM_TEXT}\"")
    lines.append("")
    total_row = pf_summary[pf_summary["stage"] == "total_s"].iloc[0]
    mean_total, median_total, p95_total = (
        float(total_row["mean_s"]), float(total_row["median_s"]), float(total_row["p95_s"]),
    )

    def _band_verdict(x: float) -> str:
        if _PAPER_CLAIM_LO <= x <= _PAPER_CLAIM_HI:
            return "within the claimed 1-3s band"
        if x < _PAPER_CLAIM_LO:
            return f"BELOW the claimed 1-3s band ({x:.4f}s < {_PAPER_CLAIM_LO}s)"
        return f"ABOVE the claimed 1-3s band ({x:.4f}s > {_PAPER_CLAIM_HI}s)"

    lines.append(
        f"Measured on this machine, **overall per-flow decision** (all 3 manifolds "
        f"summed -- `total_s`): mean = **{mean_total:.4f}s**, median = "
        f"**{median_total:.4f}s**, p95 = **{p95_total:.4f}s** -- "
        f"{_band_verdict(mean_total)}."
    )
    lines.append("")
    lines.append(
        "Per-manifold (the paper's laptop-measured claim did not specify whether it "
        "meant one manifold's Wasserstein computation or the full 3-manifold decision "
        "-- both readings are reported here for an unambiguous comparison):"
    )
    lines.append("")
    lines.append("| Manifold | mean total_s | median total_s | p95 total_s | vs 1-3s band |")
    lines.append("| :--- | ---: | ---: | ---: | :--- |")
    for m in config.MANIFOLDS:
        row = pf_summary[pf_summary["stage"] == f"{m}_total_s"].iloc[0]
        mean_m = float(row["mean_s"])
        lines.append(
            f"| {m} | {mean_m:.4f} | {float(row['median_s']):.4f} | "
            f"{float(row['p95_s']):.4f} | {_band_verdict(mean_m)} |"
        )
    lines.append("")

    # --- 5. Windowed marginal-vs-setup + fastest mode --------------------------
    lines.append("## 5. Matched-compute framing: fastest available decision mode")
    lines.append("")
    fastest_row = pf_summary[pf_summary["stage"] == "total_s"].iloc[0]
    fastest_label, fastest_mean = "per_flow", float(fastest_row["mean_s"])
    for w in sorted(windowed_frames):
        w_row = summary_df[
            (summary_df["mode"] == "windowed") & (summary_df["w"] == w)
            & (summary_df["stage"] == "total_s")
        ]
        if len(w_row) and float(w_row.iloc[0]["mean_s"]) < fastest_mean:
            fastest_label, fastest_mean = f"windowed W={w}", float(w_row.iloc[0]["mean_s"])
    lines.append(
        f"Comparing the per-flow arm's mean `total_s` against each windowed W's mean "
        "MARGINAL `total_s` (per-window Rips+W2, net of the one-time "
        f"`baseline_setup_s`): the cheapest measured per-decision cost is "
        f"**{fastest_label}** at **{fastest_mean:.4f}s/decision**. A windowed decision "
        "amortizes its Rips-complex construction over W flows at once, so its marginal "
        "per-window cost is not directly a per-FLOW cost -- see "
        "`paper/WINDOWED_RESULTS.md`'s matched-compute frontier "
        "(`compute_frontier.csv`) for the AUC-vs-marginal-cost tradeoff at each W."
    )
    lines.append("")

    # --- 6. Companion-hardware one-command instructions -------------------------
    lines.append("## 6. Companion-hardware (Jetson/RPi) one-command instructions")
    lines.append("")
    lines.append(
        "This harness is designed to run unmodified on a companion-class embedded "
        "board, ARM-ISA and otherwise, given the following artifacts copied from this "
        "workstation's `outputs/` and `results/` trees (small: CSVs, `.npy`/`.json` "
        "metadata, scaler pickles -- no persistence-diagram pickles):"
    )
    lines.append("")
    for a in _COMPANION_ARTIFACTS:
        lines.append(f"- `{a}`")
    lines.append("")
    lines.append(_EXCLUDED_ARTIFACTS)
    lines.append("")
    lines.append("Command (on the companion box, from a checkout of this repo with the "
                  "artifacts above staged under its own `outputs/`/`results/` trees):")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -e .")
    lines.append("uav-tda latency")
    lines.append("```")
    lines.append("")
    lines.append(
        "This writes `results/tables/rebuild/latency_{hostname}.csv` (summary) + "
        "per-mode detail CSVs + `paper/LATENCY_RESULTS.md`, keyed to that box's own "
        "hostname -- runs on this workstation and a future companion-box run never "
        "overwrite each other's output files."
    )
    lines.append("")

    # --- 7. Pending sign-off -----------------------------------------------------
    lines.append("## 7. Pending sign-off")
    lines.append("")
    lines.append(
        "These are workstation-class, ARM-ISA-disclosed latency measurements -- "
        "candidates for the manuscript's §V onboard-latency characterization, NOT a "
        "substitute for a real companion-hardware run. Author sign-off required before "
        "use, per this project's standing report-not-loosen convention: every number "
        "above is machine-generated from a live `uav-tda latency` run and traces to "
        f"`results/tables/rebuild/latency_{info['hostname']}.csv` and its per-mode "
        "detail CSVs."
    )
    lines.append("")

    report_text = "\n".join(lines)
    paper_dir = ws.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    report_path = paper_dir / "LATENCY_RESULTS.md"
    report_path.write_text(report_text)
    return report_path


def run_latency(ws: Workspace, n: int = 30) -> Path:
    """Full latency harness: the per-flow arm + every W in `config.WINDOW_SIZES`.

    Writes `results/tables/rebuild/latency_{hostname}.csv` (mean/median/p95
    summary per stage per mode) + one detail CSV per mode
    (`latency_{hostname}_per_flow.csv`, `latency_{hostname}_w{W}.csv`), each
    with a provenance sidecar, and `paper/LATENCY_RESULTS.md`. Returns the
    report path.
    """
    ws.ensure()
    info = machine_info()
    hostname = info["hostname"]

    rebuild_dir = ws.tables_dir / "rebuild"
    rebuild_dir.mkdir(parents=True, exist_ok=True)

    per_flow_df = time_per_flow_decision(ws, n=n)
    per_flow_path = rebuild_dir / f"latency_{hostname}_per_flow.csv"
    per_flow_df.to_csv(per_flow_path, index=False)
    write_provenance(per_flow_path, {"mode": "per_flow", "n": n})

    windowed_frames: "dict[int, pd.DataFrame]" = {}
    for w in config.WINDOW_SIZES:
        df_w = time_windowed_decision(ws, w, n=n)
        windowed_frames[w] = df_w
        w_path = rebuild_dir / f"latency_{hostname}_w{w}.csv"
        df_w.to_csv(w_path, index=False)
        write_provenance(w_path, {"mode": f"windowed_w{w}", "n": n, "w": w})

    summary_df = _build_summary_table(per_flow_df, windowed_frames)
    summary_path = rebuild_dir / f"latency_{hostname}.csv"
    summary_df.to_csv(summary_path, index=False)
    write_provenance(summary_path, {"n": n, "hostname": hostname, "machine_info": info})

    report_path = _write_latency_report(ws, info, per_flow_df, windowed_frames, summary_df, n)
    log.info("latency harness complete: %s, %s", summary_path, report_path)
    return report_path
