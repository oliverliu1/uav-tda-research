"""Does hera's Wasserstein-2 terminate on this machine's per-flow comparisons?

hera's auction stops once its RELATIVE error is small enough. Bit-identical
diagrams return 0 immediately, but for diagrams that differ only by rounding
(observed at 2**-54, about 5.6e-17) the true distance is almost 0 and it can
loop forever. Many test flows leave a manifold's top bars unchanged, so on the machine
that built the baselines those comparisons are bit-identical; with baselines
built on another machine they can differ in the last bit and hang (first seen
on a Raspberry Pi 5 using baselines persisted on the Mac, 2026-11-23).

For the same sampled test flows as `uav-tda latency` (rng seed 0), this runs
every per-flow (manifold, dimension) comparison against the committed and the
machine-built baselines, in the probe configuration (top-K, delta=0.2) and the
full-test one (full diagrams, delta<=0.01), each in a forked child that is
killed after `TIMEOUT_S`. "timeout" means exactly that: not finished in time.
"""
from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, exact, latency, probe
from .latency import machine_info
from .provenance import write_provenance
from .workspace import Workspace

# Comparisons normally take ~0.02 s (probe) or up to ~2 s (exact) on the Mac.
TIMEOUT_S = {"probe": 5.0, "exact": 30.0}
# Above exact equality, below the smallest distance seen to terminate (1e-6 relative).
NEAR_GAP = 1e-7


def diagram_gap(a: np.ndarray, b: np.ndarray) -> "float | None":
    """Largest coordinate difference between the sorted diagrams; None if sizes differ."""
    if a.shape != b.shape:
        return None
    if not len(a):
        return 0.0
    sa, sb = a[np.lexsort(a.T)], b[np.lexsort(b.T)]
    return float(np.abs(sa - sb).max())


def _child(conn, fn, a, b, delta) -> None:
    conn.send(float(fn(a, b, order=2.0, internal_p=2.0, delta=delta)))
    conn.close()


def timed_w2(a, b, delta: float, timeout_s: float, w2_fn=None) -> dict:
    """One W2 call in a forked child: status ok / timeout / error, value, seconds."""
    if w2_fn is None:
        from gudhi.hera import wasserstein_distance as w2_fn  # noqa: PLC0415

    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    t0 = time.perf_counter()
    p = ctx.Process(target=_child, args=(send, w2_fn, a, b, delta))
    p.start()
    send.close()  # so a crashed child reads as EOF, not as a timeout
    status, value = "timeout", float("nan")
    try:
        if recv.poll(timeout_s):
            value = recv.recv()
            status = "ok"
    except EOFError:
        status = "error"
    finally:
        if p.is_alive() and status == "timeout":
            p.kill()
        p.join()
        recv.close()
    return {"status": status, "w2": value, "seconds": time.perf_counter() - t0}


def run_hang_check(ws: Workspace, n: int = 30, rng_seed: int = 0) -> pd.DataFrame:
    """One row per (baseline, config, flow, manifold, dim) comparison."""
    from . import tda as tda_module

    max_edge = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    clouds = tda_module.compute_reference_clouds(
        ws, np.load(ws.outputs_dir / "reference_indices.npy"))
    scalers = latency._load_scalers(ws)
    baselines = {}
    if latency.baseline_source(ws) == "committed":
        baselines["committed"] = latency._load_baselines_for_harness(ws)
    baselines["fresh"] = latency._load_baselines_for_harness(ws, fresh=True)
    configs = {"probe": (config.PROBE_TOP_K, config.PROBE_DELTA),
               "exact": (None, exact.PRIMARY_DELTA)}

    n_test = len(pd.read_csv(ws.outputs_dir / "labels_test.csv"))
    n_sample = min(n, n_test)
    positions = np.sort(np.random.default_rng(rng_seed).choice(
        n_test, size=n_sample, replace=False))
    raw = latency._raw_manifold_vectors(ws, positions, scalers)

    rows = []
    for i, pos in enumerate(positions):
        for m in config.MANIFOLDS:
            hd = config.MAX_HOM_DIM[m]
            scaled = scalers[m].transform(raw[m].iloc[i:i + 1])[0]
            diag = tda_module._persistence_for_point(
                scaled, clouds[m], max_edge[m], hd + 1, config.SPARSE_RIPS_EPSILON.get(m))
            for dim in range(hd + 1):
                flow = probe._slice_dim(diag, dim, max_edge[m])
                for cfg, (top_k, delta) in configs.items():
                    a = latency._truncate(flow, top_k)
                    for name, base in baselines.items():
                        b = latency._truncate(base[m][dim], top_k)
                        res = timed_w2(a, b, delta=delta, timeout_s=TIMEOUT_S[cfg])
                        rows.append({
                            "baseline": name, "config": cfg, "row_idx": int(pos),
                            "manifold": m, "dim": dim, "n_flow_bars": len(a),
                            "n_baseline_bars": len(b), "gap": diagram_gap(a, b),
                            "timeout_s": TIMEOUT_S[cfg], **res,
                        })
        print(f"[hang-check] flow {i + 1}/{n_sample}", flush=True)
    df = pd.DataFrame(rows)
    df["gap"] = df["gap"].astype(float)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per (baseline, config)."""
    out = []
    for (name, cfg), g in df.groupby(["baseline", "config"], sort=True):
        stuck = g[g["status"] == "timeout"]
        out.append({
            "baseline": name, "config": cfg, "n_comparisons": len(g),
            "n_equal": int((g["gap"] == 0.0).sum()),
            "n_near": int(((g["gap"] > 0.0) & (g["gap"] < NEAR_GAP)).sum()),
            "n_timeout": len(stuck), "n_error": int((g["status"] == "error").sum()),
            "n_flows": g["row_idx"].nunique(), "n_flows_timeout": stuck["row_idx"].nunique(),
            "max_ok_s": float(g.loc[g["status"] == "ok", "seconds"].max()),
        })
    return pd.DataFrame(out)


def write_hang_check(ws: Workspace, df: pd.DataFrame, n: int) -> "tuple[Path, Path]":
    """`results/tables/rebuild/hang_check_{host}{,_summary}.csv` with sidecars."""
    info = machine_info()
    rebuild = ws.tables_dir / "rebuild"
    rebuild.mkdir(parents=True, exist_ok=True)
    params = {"n": n, "timeout_s": TIMEOUT_S, "near_gap": NEAR_GAP, "machine_info": info}
    detail = rebuild / f"hang_check_{info['hostname']}.csv"
    summary = rebuild / f"hang_check_{info['hostname']}_summary.csv"
    df.to_csv(detail, index=False)
    summarize(df).to_csv(summary, index=False)
    for p in (detail, summary):
        write_provenance(p, params)
    return detail, summary
