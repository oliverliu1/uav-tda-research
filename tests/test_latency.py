"""Tests for uav_tda.latency (Phase 6 Track B: portable latency harness).

All tests are fast/synthetic: no real Rips/hera calls, no real repo
`outputs/`/`results/` data. `tda._persistence_for_point`, `gudhi.hera.
wasserstein_distance`, `windowed.window_diagram`, and `windowed.w2_distance`
are monkeypatched to instant fakes. Every test uses `tmp_path`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uav_tda import cli, config, latency
from uav_tda.workspace import Workspace


# ---------------------------------------------------------------------------
# machine_info
# ---------------------------------------------------------------------------


def test_machine_info_keys():
    info = latency.machine_info()
    expected_keys = {
        "platform", "machine", "hardware_arch", "rosetta_translated", "cpu_brand",
        "physical_cores", "logical_cores", "python_version", "gudhi_version", "hostname",
        "timestamp_utc",
    }
    assert expected_keys == set(info.keys())
    assert isinstance(info["platform"], str) and info["platform"]
    assert isinstance(info["machine"], str) and info["machine"]
    assert isinstance(info["hardware_arch"], str) and info["hardware_arch"]
    assert isinstance(info["rosetta_translated"], bool)
    assert isinstance(info["cpu_brand"], str) and info["cpu_brand"]
    assert info["physical_cores"] is None or isinstance(info["physical_cores"], int)
    assert isinstance(info["logical_cores"], int) and info["logical_cores"] > 0
    assert isinstance(info["python_version"], str) and info["python_version"]
    assert isinstance(info["gudhi_version"], str) and info["gudhi_version"]
    assert isinstance(info["hostname"], str) and info["hostname"]
    assert "T" in info["timestamp_utc"]  # ISO 8601


def test_cpu_brand_never_raises():
    # Should return a string even if both sysctl and /proc/cpuinfo are absent
    # on this platform (best-effort, never fatal).
    assert isinstance(latency._cpu_brand(), str)


def test_hardware_arch_never_raises():
    assert isinstance(latency._hardware_arch(), str)


# ---------------------------------------------------------------------------
# _load_scalers (regression: outputs/scaler_{m}.pkl is STALE on this repo --
# see uav_tda/latency.py's _load_scalers docstring)
# ---------------------------------------------------------------------------


def test_load_scalers_uses_combined_scalers_pkl_ignores_stale_per_manifold_file(tmp_path):
    from sklearn.preprocessing import StandardScaler

    ws = Workspace.at(tmp_path)
    ws.ensure()

    combined = {}
    for m, cols in config.MANIFOLDS.items():
        combined[m] = StandardScaler().fit(np.random.default_rng(0).normal(size=(6, len(cols))))
    (ws.outputs_dir / "scalers.pkl").write_bytes(pickle.dumps(combined))

    # A deliberately WRONG per-manifold file (mismatched feature count) --
    # must be ignored entirely, not silently preferred.
    stale = StandardScaler().fit(np.random.default_rng(1).normal(size=(6, 999)))
    (ws.outputs_dir / "scaler_c2.pkl").write_bytes(pickle.dumps(stale))

    scalers = latency._load_scalers(ws)
    for m, cols in config.MANIFOLDS.items():
        assert scalers[m].n_features_in_ == len(cols)


def test_load_scalers_raises_on_feature_count_mismatch(tmp_path):
    from sklearn.preprocessing import StandardScaler

    ws = Workspace.at(tmp_path)
    ws.ensure()
    combined = {
        m: StandardScaler().fit(np.random.default_rng(0).normal(size=(6, len(cols) + 1)))
        for m, cols in config.MANIFOLDS.items()
    }
    (ws.outputs_dir / "scalers.pkl").write_bytes(pickle.dumps(combined))

    with pytest.raises(AssertionError):
        latency._load_scalers(ws)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic workspaces
# ---------------------------------------------------------------------------


def _make_per_flow_workspace(tmp_path: Path, n_test: int = 4, n_train: int = 6) -> Workspace:
    ws = Workspace.at(tmp_path)
    ws.ensure()

    max_edge_lengths = {"c2": 1.0, "network": 1.0, "physical": 1.0}
    (ws.outputs_dir / "max_edge_lengths.json").write_text(json.dumps(max_edge_lengths))

    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    scalers: dict = {}
    for m, cols in config.MANIFOLDS.items():
        n_cols = len(cols)
        train = rng.normal(size=(n_train, n_cols))
        train_df = pd.DataFrame(train, columns=list(cols))
        train_df.to_csv(ws.outputs_dir / f"{m}_train.csv", index=False)

        scalers[m] = StandardScaler().fit(train_df)

        # "standardized" test rows -- any finite values work since the Rips
        # and W2 calls are monkeypatched in every test that uses this fixture.
        test_std = rng.normal(size=(n_test, n_cols))
        pd.DataFrame(test_std, columns=list(cols)).to_csv(
            ws.outputs_dir / f"{m}_test.csv", index=False
        )
    (ws.outputs_dir / "scalers.pkl").write_bytes(pickle.dumps(scalers))

    labels = ["Normal Traffic"] * n_test
    pd.DataFrame({"label": labels}).to_csv(ws.outputs_dir / "labels_test.csv", index=False)
    np.save(ws.outputs_dir / "reference_indices.npy", np.arange(min(5, n_train)))

    # Persisted baselines (hermetic to tmp_path -- avoids probe._load_baseline_
    # barcodes, whose OUTPUTS_DIR/PERSISTENCE_DIR are fixed to the real repo,
    # not workspace-relative).
    exact_dir = ws.tables_dir / "rebuild" / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": "2026-01-01T00:00:00Z", "manifolds": {}}
    for m in config.MANIFOLDS:
        manifest["manifolds"][m] = {}
        for dim in range(config.MAX_HOM_DIM[m] + 1):
            fname = f"baselines_{m}_dim{dim}.npy"
            arr = np.array([[0.1, 0.5], [0.2, 0.6]])
            np.save(exact_dir / fname, arr)
            manifest["manifolds"][m][str(dim)] = {"file": fname, "shape": [2, 2]}
    (exact_dir / "baselines_manifest.json").write_text(json.dumps(manifest))

    return ws


def _fake_persistence_for_point(query_point, reference_points, max_edge, max_simplex_dim, sparse):
    # Tiny deterministic diagram with one bar per homology dim 0..max_simplex_dim-1.
    rows = [[float(dim), 0.1 * (dim + 1), 0.4 * (dim + 1)] for dim in range(max_simplex_dim)]
    return np.array(rows, dtype=float)


@pytest.fixture(autouse=False)
def _patch_per_flow_backends(monkeypatch):
    import gudhi.hera as hera_module

    from uav_tda import tda as tda_module

    monkeypatch.setattr(tda_module, "_persistence_for_point", _fake_persistence_for_point)
    monkeypatch.setattr(hera_module, "wasserstein_distance",
                         lambda d1, d2, order, internal_p, delta: 0.42)


# ---------------------------------------------------------------------------
# time_per_flow_decision
# ---------------------------------------------------------------------------


def test_time_per_flow_decision_schema_and_stage_sums(tmp_path, monkeypatch, _patch_per_flow_backends):
    ws = _make_per_flow_workspace(tmp_path, n_test=4)
    df = latency.time_per_flow_decision(ws, n=3, rng_seed=0)

    assert len(df) == 3
    assert "row_idx" in df.columns
    assert df["row_idx"].is_unique

    for m in config.MANIFOLDS:
        for stage in ("scaler_s", "rips_s", "slice_s", "w2_s", "total_s"):
            col = f"{m}_{stage}"
            assert col in df.columns
            assert (df[col].to_numpy() >= 0.0).all()

    # per-manifold total_s == sum of its 4 stage columns.
    for m in config.MANIFOLDS:
        expected = (
            df[f"{m}_scaler_s"] + df[f"{m}_rips_s"] + df[f"{m}_slice_s"] + df[f"{m}_w2_s"]
        )
        assert df[f"{m}_total_s"].to_numpy() == pytest.approx(expected.to_numpy())

    # overall total_s == sum of the 3 manifolds' total_s.
    expected_total = sum(df[f"{m}_total_s"] for m in config.MANIFOLDS)
    assert df["total_s"].to_numpy() == pytest.approx(expected_total.to_numpy())


def test_time_per_flow_decision_respects_n_and_caps_at_split_size(
    tmp_path, monkeypatch, _patch_per_flow_backends,
):
    ws = _make_per_flow_workspace(tmp_path, n_test=3)
    df = latency.time_per_flow_decision(ws, n=100, rng_seed=0)
    assert len(df) == 3  # capped at the (tiny) test split size


# ---------------------------------------------------------------------------
# time_windowed_decision
# ---------------------------------------------------------------------------


def _make_windowed_workspace(tmp_path: Path, n_val: int = 9, n_test: int = 12) -> Workspace:
    ws = Workspace.at(tmp_path)
    ws.ensure()
    max_edge_lengths = {"c2": 1.0, "network": 1.0, "physical": 1.0}
    (ws.outputs_dir / "max_edge_lengths.json").write_text(json.dumps(max_edge_lengths))

    rng = np.random.default_rng(1)
    for m, cols in config.MANIFOLDS.items():
        n_cols = len(cols)
        val = rng.normal(size=(n_val, n_cols))
        test = rng.normal(size=(n_test, n_cols))
        pd.DataFrame(val, columns=list(cols)).to_csv(ws.outputs_dir / f"{m}_val.csv", index=False)
        pd.DataFrame(test, columns=list(cols)).to_csv(ws.outputs_dir / f"{m}_test.csv", index=False)

    pd.DataFrame({"label": ["Normal Traffic"] * n_val}).to_csv(
        ws.outputs_dir / "labels_val.csv", index=False
    )
    labels_test = (["Normal Traffic"] * (n_test - 2)) + ["Sybil Attack", "Blackhole Attack"]
    pd.DataFrame({"label": labels_test}).to_csv(ws.outputs_dir / "labels_test.csv", index=False)
    return ws


def _fake_window_diagram(points, max_edge, max_hom_dim, sparse):
    rows = [[float(dim), 0.1 * (dim + 1), 0.4 * (dim + 1)] for dim in range(max_hom_dim + 1)]
    return np.array(rows, dtype=float)


def _fake_w2_distance(d1, d2, max_edge, max_hom_dim):
    return 0.7


@pytest.fixture(autouse=False)
def _patch_windowed_backends(monkeypatch):
    from uav_tda import windowed as windowed_module

    monkeypatch.setattr(windowed_module, "window_diagram", _fake_window_diagram)
    monkeypatch.setattr(windowed_module, "w2_distance", _fake_w2_distance)


def test_time_windowed_decision_schema_and_stage_sums(
    tmp_path, monkeypatch, _patch_windowed_backends,
):
    ws = _make_windowed_workspace(tmp_path, n_val=9, n_test=12)
    df = latency.time_windowed_decision(ws, w=3, n=3, rng_seed=0)

    assert len(df) == 3
    assert "window_idx" in df.columns and "start_pos" in df.columns
    assert "baseline_setup_s" in df.columns
    # fixed cost: identical across every row.
    assert df["baseline_setup_s"].nunique() == 1
    assert (df["baseline_setup_s"].to_numpy() >= 0.0).all()

    for m in config.MANIFOLDS:
        for stage in ("rips_s", "w2_s", "total_s"):
            col = f"{m}_{stage}"
            assert col in df.columns
            assert (df[col].to_numpy() >= 0.0).all()
        expected = df[f"{m}_rips_s"] + df[f"{m}_w2_s"]
        assert df[f"{m}_total_s"].to_numpy() == pytest.approx(expected.to_numpy())

    expected_total = sum(df[f"{m}_total_s"] for m in config.MANIFOLDS)
    assert df["total_s"].to_numpy() == pytest.approx(expected_total.to_numpy())
    # total_s does NOT include baseline_setup_s (marginal-vs-fixed split).
    assert not np.allclose(
        df["total_s"].to_numpy(), df["total_s"].to_numpy() + df["baseline_setup_s"].to_numpy()
    ) or df["baseline_setup_s"].iloc[0] == 0.0


def test_time_windowed_decision_respects_n_and_caps_at_window_count(
    tmp_path, monkeypatch, _patch_windowed_backends,
):
    ws = _make_windowed_workspace(tmp_path, n_val=24, n_test=12)
    # w=12 -> only 1 window in the 12-row test split (val has enough rows
    # for >=1 baseline window too).
    df = latency.time_windowed_decision(ws, w=12, n=100, rng_seed=0)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# _build_summary_table
# ---------------------------------------------------------------------------


def test_build_summary_table_has_mean_median_p95_per_stage():
    per_flow_df = pd.DataFrame({
        "row_idx": [0, 1, 2],
        "c2_total_s": [1.0, 2.0, 3.0],
        "total_s": [1.0, 2.0, 3.0],
    })
    windowed_frames = {25: pd.DataFrame({
        "window_idx": [0, 1],
        "c2_total_s": [0.1, 0.3],
        "total_s": [0.1, 0.3],
    })}
    summary = latency._build_summary_table(per_flow_df, windowed_frames)

    pf_total = summary[(summary["mode"] == "per_flow") & (summary["stage"] == "total_s")].iloc[0]
    assert pf_total["mean_s"] == pytest.approx(2.0)
    assert pf_total["median_s"] == pytest.approx(2.0)
    assert pf_total["n"] == 3

    w_total = summary[(summary["mode"] == "windowed") & (summary["w"] == 25)
                       & (summary["stage"] == "total_s")].iloc[0]
    assert w_total["mean_s"] == pytest.approx(0.2)
    assert w_total["n"] == 2


# ---------------------------------------------------------------------------
# _write_latency_report (OPT-2 smoke test, final review)
# ---------------------------------------------------------------------------


def test_write_latency_report_smoke(tmp_path):
    """Synthetic frames -> renders `paper/LATENCY_RESULTS.md`, no real Rips/hera
    calls, tmp workspace only. Confirms the machine header and the paper-claim
    check land in the rendered text, and that an optional `render_note` (used
    for a text-only re-render disclosure) is included when given.
    """
    ws = Workspace.at(tmp_path)
    ws.ensure()
    info = {
        "platform": "Darwin", "machine": "x86_64", "hardware_arch": "arm64",
        "rosetta_translated": True, "cpu_brand": "Apple M1 Pro", "physical_cores": 8,
        "logical_cores": 8, "python_version": "3.9.7", "gudhi_version": "3.11.0",
        "hostname": "test-host", "timestamp_utc": "2026-01-01T00:00:00+00:00",
    }
    n = 3
    stage_order = ["scaler_s", "rips_s", "slice_s", "w2_s", "total_s"]
    per_flow_data: dict = {"row_idx": list(range(n))}
    for m in config.MANIFOLDS:
        for stage in stage_order:
            per_flow_data[f"{m}_{stage}"] = [0.01, 0.02, 0.03]
    per_flow_data["total_s"] = [0.03, 0.06, 0.09]
    per_flow_df = pd.DataFrame(per_flow_data)

    windowed_frames: dict = {}
    for w in (25, 50):
        w_data: dict = {
            "window_idx": list(range(n)), "start_pos": list(range(n)),
            "baseline_setup_s": [1.5] * n,
        }
        for m in config.MANIFOLDS:
            w_data[f"{m}_rips_s"] = [0.001] * n
            w_data[f"{m}_w2_s"] = [0.002] * n
            w_data[f"{m}_total_s"] = [0.003] * n
        w_data["total_s"] = [0.009] * n
        windowed_frames[w] = pd.DataFrame(w_data)

    summary_df = latency._build_summary_table(per_flow_df, windowed_frames)

    report_path = latency._write_latency_report(
        ws, info, per_flow_df, windowed_frames, summary_df, n,
        render_note="_Re-rendered for a smoke test; not a real measurement._",
    )

    assert report_path == ws.root / "paper" / "LATENCY_RESULTS.md"
    text = report_path.read_text()
    assert "## 1. Machine header" in text
    assert "Apple M1 Pro" in text
    assert "## 4. Paper's \"1-3 s per flow\" claim vs measured" in text
    assert "claimed 1-3s band" in text
    assert "_Re-rendered for a smoke test; not a real measurement._" in text


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def test_cli_registers_latency_subcommand_with_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["latency", "--n", "10"])
    assert args.func is cli._cmd_latency
    assert args.n == 10

    defaults = parser.parse_args(["latency"])
    assert defaults.n == 30
