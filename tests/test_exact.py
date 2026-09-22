"""Tests for uav_tda.exact (Task 1: sharded resumable exact-W2 runner).

All tests are fast/synthetic: no real persistence diagrams, no real
sparse-Rips baseline computation (`probe._load_baseline_barcodes` and
`probe._w2_subprocess_target` are monkeypatched where real computation
would be slow or nondeterministic). Nothing here writes to the repo's real
`outputs/` or `results/` trees -- every test uses `tmp_path`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uav_tda import exact, probe
from uav_tda.workspace import Workspace


# ---------------------------------------------------------------------------
# exact_w2_flow
# ---------------------------------------------------------------------------


def test_exact_w2_flow_finite_no_timeout():
    # dim-0 only diagram (max_hom_dim=0); tiny so the real exact hera call
    # returns almost instantly and no timeout path is exercised.
    diagram = np.array([
        [0.0, 0.1, 0.5],
        [0.0, 0.2, 0.6],
    ])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    total, n_timeouts, approx_flag = exact.exact_w2_flow(
        diagram, baselines_m, max_edge=1.0, max_hom_dim=0, timeout_s=10.0,
    )
    assert np.isfinite(total)
    assert total >= 0.0
    assert n_timeouts == 0
    assert approx_flag is False


def _hanging_target(queue, d1, d2, order, internal_p, delta) -> None:  # noqa: ARG001
    time.sleep(5.0)


def test_exact_w2_flow_timeout_path(monkeypatch):
    monkeypatch.setattr(probe, "_w2_subprocess_target", _hanging_target)

    diagram = np.array([
        [0.0, 0.1, 0.5],
        [0.0, 0.2, 0.6],
    ])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    total, n_timeouts, approx_flag = exact.exact_w2_flow(
        diagram, baselines_m, max_edge=1.0, max_hom_dim=0, timeout_s=0.3,
    )
    assert total == 0.0
    assert n_timeouts >= 1
    assert approx_flag is True


# ---------------------------------------------------------------------------
# baseline persist / load / ensure
# ---------------------------------------------------------------------------


def _synthetic_baselines() -> dict:
    return {
        "c2": {
            0: np.array([[0.1, 0.4], [0.2, 0.5]]),
            1: np.array([[0.3, 0.6]]),
            2: np.empty((0, 2)),
        },
        "network": {
            0: np.array([[0.05, 0.9]]),
            1: np.array([[0.15, 0.25], [0.35, 0.45]]),
            2: np.array([[0.5, 0.55]]),
        },
        "physical": {
            0: np.array([[0.02, 0.03]]),
            1: np.empty((0, 2)),
        },
    }


def _make_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.at(tmp_path)
    ws.ensure()
    (ws.outputs_dir / "max_edge_lengths.json").write_text(
        json.dumps({"c2": 1.0, "network": 1.0, "physical": 1.0})
    )
    return ws


def test_persist_and_load_baselines_roundtrip(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    exact_dir = ws.tables_dir / "rebuild" / "exact"
    synthetic = _synthetic_baselines()
    monkeypatch.setattr(probe, "_load_baseline_barcodes", lambda max_edge_lengths: synthetic)

    returned = exact.persist_baselines(ws, exact_dir)
    for m, per_dim in synthetic.items():
        for dim, arr in per_dim.items():
            assert np.array_equal(returned[m][dim], arr)

    assert (exact_dir / "baselines_manifest.json").exists()

    loaded = exact.load_baselines(exact_dir)
    for m, per_dim in synthetic.items():
        for dim, arr in per_dim.items():
            assert np.array_equal(loaded[m][dim], arr)
            # npy roundtrip: exact equality on the saved .npy file too.
            fname = json.loads((exact_dir / "baselines_manifest.json").read_text())
            saved = np.load(exact_dir / fname["manifolds"][m][str(dim)]["file"])
            assert np.array_equal(saved, arr)


def test_ensure_baselines_does_not_recompute_on_second_call(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    exact_dir = ws.tables_dir / "rebuild" / "exact"
    synthetic = _synthetic_baselines()

    calls = {"n": 0}

    def _counting_loader(max_edge_lengths):
        calls["n"] += 1
        return synthetic

    monkeypatch.setattr(probe, "_load_baseline_barcodes", _counting_loader)

    first = exact.ensure_baselines(ws, exact_dir)
    assert calls["n"] == 1
    for m, per_dim in synthetic.items():
        for dim, arr in per_dim.items():
            assert np.array_equal(first[m][dim], arr)

    second = exact.ensure_baselines(ws, exact_dir)
    assert calls["n"] == 1, "ensure_baselines must LOAD on resume, never recompute"
    for m, per_dim in synthetic.items():
        for dim, arr in per_dim.items():
            assert np.array_equal(second[m][dim], arr)


# ---------------------------------------------------------------------------
# manifest resume in run_exact_campaign
# ---------------------------------------------------------------------------


def test_run_exact_campaign_skips_completed_shards(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    exact_dir = ws.tables_dir / "rebuild" / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)

    # val: 2 Normal-Traffic flows (out of 3); test: 4 flows.
    pd.DataFrame({"label": ["Normal Traffic", "Normal Traffic", "Blackhole Attack"]}).to_csv(
        ws.outputs_dir / "labels_val.csv", index=False
    )
    pd.DataFrame({
        "label": ["Normal Traffic", "Sybil Attack", "Wormhole Attack", "Flooding Attack"],
    }).to_csv(ws.outputs_dir / "labels_test.csv", index=False)

    # Pre-mark the (only) val shard as complete.
    manifest = {"shards": {"val_00000": {"split": "val", "start": 0, "size": 2, "n_rows": 2}}}
    (exact_dir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(exact, "ensure_baselines", lambda ws, exact_dir: {"dummy": True})

    calls = []

    def _fake_run_shard(ws, exact_dir, split, start, size, baselines, class_filter=None, n_jobs=-1):
        calls.append((split, start, size, class_filter))
        return exact_dir / f"shard_{split}_{start:05d}.csv"

    monkeypatch.setattr(exact, "run_shard", _fake_run_shard)

    exact.run_exact_campaign(ws, n_jobs=1, shard_size=2)

    # val_00000 was already complete -> must NOT be re-run.
    assert ("val", 0, 2, "Normal Traffic") not in calls
    # test has 4 flows, shard_size=2 -> two shards, both new -> both run.
    assert ("test", 0, 2, None) in calls
    assert ("test", 2, 2, None) in calls
    assert len(calls) == 2
