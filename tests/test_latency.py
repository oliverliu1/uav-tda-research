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


_STAGES = ["scaler_s", "rips_s", "slice_s", "w2_s", "total_s"]


def _write_host(ws, host, scale=1.0, machine="x86_64", hw="arm64", rosetta=True,
                brand="Apple M1 Pro", exact=False, baseline_source=None, extra=None):
    """Synthetic committed CSVs + sidecars for one host, the way `run_latency` writes them."""
    from uav_tda.provenance import write_provenance

    rebuild = ws.tables_dir / "rebuild"
    rebuild.mkdir(parents=True, exist_ok=True)
    data = {"row_idx": [0, 1, 2]}
    for m in config.MANIFOLDS:
        for stage in _STAGES:
            data[f"{m}_{stage}"] = [0.01 * scale, 0.02 * scale, 0.03 * scale]
    data["total_s"] = [1.0 * scale, 2.0 * scale, 3.0 * scale]
    pf = pd.DataFrame(data)
    wins = {}
    for w in (25, 50):
        wd = {"window_idx": [0, 1, 2], "start_pos": [0, 1, 2], "baseline_setup_s": [1.5] * 3}
        for m in config.MANIFOLDS:
            for st in ("rips_s", "w2_s", "total_s"):
                wd[f"{m}_{st}"] = [0.001 * scale] * 3
        wd["total_s"] = [0.009 * scale] * 3
        wins[w] = pd.DataFrame(wd)
        wins[w].to_csv(rebuild / f"latency_{host}_w{w}.csv", index=False)
    info = {"platform": "X", "machine": machine, "hardware_arch": hw,
            "rosetta_translated": rosetta, "cpu_brand": brand, "physical_cores": 4,
            "logical_cores": 4, "python_version": "3.9", "gudhi_version": "3.11.0",
            "hostname": host, "timestamp_utc": "2026-01-01T00:00:00+00:00"}
    params = {"n": 3, "hostname": host, "machine_info": info, **(extra or {})}
    if baseline_source:
        params["baseline_source"] = baseline_source
    summary = latency._build_summary_table(pf, wins)
    path = rebuild / f"latency_{host}.csv"
    summary.to_csv(path, index=False)
    write_provenance(path, params)
    if exact:
        ex = pf.copy()
        ex["total_s"] = ex["total_s"] * 1.5
        epath = rebuild / f"latency_{host}_per_flow_exact.csv"
        ex.to_csv(epath, index=False)
        write_provenance(epath, {"mode": "per_flow_exact", "n": 3,
                                 "baseline_source": "committed", "top_k": None, "delta": 0.01})
    return summary


def test_write_latency_report_single_workstation(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    _write_host(ws, "test-host", extra={"rerender_reason": "synthetic re-render note"})
    path = latency.write_latency_report(ws)
    assert path == ws.root / "paper" / "LATENCY_RESULTS.md"
    text = path.read_text()
    assert "Apple M1 Pro" in text and "Rosetta" in text
    assert "claimed 1-3s band" in text
    assert "synthetic re-render note" in text
    assert "0.9082 configuration" not in text


def test_report_covers_both_hosts_with_role_specific_wording(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LATENCY_PRIMARY_HOST", "mac-host")
    monkeypatch.setattr(config, "LATENCY_HOSTS", {
        "mac-host": {"role": "workstation", "label": "Mac"},
        "pi-host": {"role": "companion", "label": "Raspberry Pi 5"}})
    ws = Workspace.at(tmp_path)
    ws.ensure()
    _write_host(ws, "pi-host", scale=2.0, machine="aarch64", hw="arm64", rosetta=False,
                brand="unknown", exact=True, baseline_source="fresh")
    _write_host(ws, "mac-host", exact=True)
    text = latency.write_latency_report(ws).read_text()
    # primary host first, then the companion
    assert text.index("Host: Mac") < text.index("Host: Raspberry Pi 5")
    pi = text[text.index("Host: Raspberry Pi 5"):text.index("## Companion-hardware")]
    assert "workstation" not in pi and "LOWER BOUND" not in pi
    assert "companion-class" in pi
    mac = text[text.index("Host: Mac"):text.index("Host: Raspberry Pi 5")]
    assert "LOWER BOUND" in mac
    # cross-host ratio 2.0 on per-flow, exact arm on both hosts
    assert "2.00" in text
    assert text.count("0.9082 configuration") >= 2
    assert "Baseline source: **fresh**" in pi


def test_load_latency_hosts_takes_the_exact_arm_from_its_detail_file(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    _write_host(ws, "h", exact=True)
    (host,) = latency.load_latency_hosts(ws)
    ex = host["exact"]
    assert float(ex.loc[ex["stage"] == "total_s", "mean_s"].iloc[0]) == pytest.approx(3.0)
    assert host["exact_meta"]["baseline_source"] == "committed"


def test_exact_only_writes_just_the_exact_arm(tmp_path, monkeypatch):
    ws = _make_per_flow_workspace(tmp_path)
    calls = _stub_arms(monkeypatch)
    host = latency.machine_info()["hostname"]
    _write_host(ws, host)
    rebuild = ws.tables_dir / "rebuild"
    before = {p.name: p.read_bytes() for p in rebuild.glob("latency_*")}
    report = latency.run_latency(ws, n=2, exact_only=True, watchdog_s=0)
    assert len(calls) == 1 and calls[0]["top_k"] is None
    after = {p.name: p.read_bytes() for p in rebuild.glob("latency_*")}
    for name, data in before.items():
        assert after[name] == data, name
    assert set(after) - set(before) == {
        f"latency_{host}_per_flow_exact.csv",
        f"latency_{host}_per_flow_exact.csv.provenance.json"}
    side = json.loads((rebuild / f"latency_{host}_per_flow_exact.csv.provenance.json").read_text())
    assert side["params"]["machine_info"]["hostname"] == host
    assert "0.9082 configuration" in report.read_text()


def test_exact_only_refuses_without_an_existing_summary(tmp_path, monkeypatch):
    ws = _make_per_flow_workspace(tmp_path)
    _stub_arms(monkeypatch)
    with pytest.raises(FileNotFoundError, match="summary"):
        latency.run_latency(ws, n=2, exact_only=True, watchdog_s=0)


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


# ---------------------------------------------------------------------------
# Pi harness: fresh baselines, exact arm, progress + watchdog
# ---------------------------------------------------------------------------


def test_fresh_baselines_are_computed_from_the_workspace_reference_cloud(tmp_path, monkeypatch):
    from uav_tda import tda as tda_module

    ws = _make_per_flow_workspace(tmp_path)
    seen = {}

    def spy(query_point, reference_points, max_edge, max_simplex_dim, sparse):
        seen.setdefault("clouds", []).append(np.vstack([reference_points, query_point]))
        seen.setdefault("sparse", []).append(sparse)
        return _fake_persistence_for_point(
            query_point, reference_points, max_edge, max_simplex_dim, sparse)

    monkeypatch.setattr(tda_module, "_persistence_for_point", spy)
    committed = latency._load_baselines_for_harness(ws)
    assert "clouds" not in seen  # committed baselines are loaded, not computed
    fresh = latency._load_baselines_for_harness(ws, fresh=True)

    idx = np.load(ws.outputs_dir / "reference_indices.npy")
    for m, cloud, sparse in zip(config.MANIFOLDS, seen["clouds"], seen["sparse"]):
        expected = pd.read_csv(ws.outputs_dir / f"{m}_train.csv").to_numpy()[idx]
        np.testing.assert_array_equal(cloud, expected)  # exactly the reference cloud, in order
        assert sparse == config.SPARSE_RIPS_EPSILON.get(m)
        assert set(fresh[m]) == set(range(config.MAX_HOM_DIM[m] + 1))
    np.testing.assert_allclose(fresh["c2"][1], [[0.2, 0.8]])
    assert committed["c2"][1].shape == (2, 2)


def test_missing_committed_baselines_fall_back_to_this_workspace(tmp_path, _patch_per_flow_backends):
    ws = _make_per_flow_workspace(tmp_path)
    (ws.tables_dir / "rebuild" / "exact" / "baselines_manifest.json").unlink()
    assert latency.baseline_source(ws) == "fresh"
    assert latency._load_baselines_for_harness(ws)["c2"][0].shape == (1, 2)


def test_per_flow_exact_config_uses_full_diagrams_and_tight_delta(tmp_path, monkeypatch):
    import gudhi.hera as hera_module

    from uav_tda import exact
    from uav_tda import tda as tda_module

    ws = _make_per_flow_workspace(tmp_path)
    n_bars = config.PROBE_TOP_K + 7

    def many_bars(query_point, reference_points, max_edge, max_simplex_dim, sparse):
        births = np.linspace(0.0, 0.1, n_bars)
        return np.column_stack([np.zeros(n_bars), births, births + np.linspace(0.1, 0.5, n_bars)])

    calls = []
    monkeypatch.setattr(tda_module, "_persistence_for_point", many_bars)
    monkeypatch.setattr(
        hera_module, "wasserstein_distance",
        lambda d1, d2, order, internal_p, delta: calls.append((len(d1), delta)) or 0.1)

    latency.time_per_flow_decision(ws, n=1)
    assert max(c[0] for c in calls) == config.PROBE_TOP_K
    assert {c[1] for c in calls} == {config.PROBE_DELTA}

    calls.clear()
    latency.time_per_flow_decision(ws, n=1, top_k=None, delta=exact.PRIMARY_DELTA)
    assert max(c[0] for c in calls) == n_bars
    assert {c[1] for c in calls} == {exact.PRIMARY_DELTA}


def _stub_arms(monkeypatch):
    calls = []

    def fake_per_flow(ws, n=30, rng_seed=0, **kw):
        calls.append(kw)
        data = {"row_idx": [0, 1]}
        for m in config.MANIFOLDS:
            for s in ("scaler_s", "rips_s", "slice_s", "w2_s", "total_s"):
                data[f"{m}_{s}"] = [0.1, 0.2]
        data["total_s"] = [0.3, 0.6]
        return pd.DataFrame(data)

    def fake_windowed(ws, w, n=30, rng_seed=0, **kw):
        data = {"window_idx": [0, 1], "start_pos": [0, w], "baseline_setup_s": [1.0, 1.0]}
        for m in config.MANIFOLDS:
            for s in ("rips_s", "w2_s", "total_s"):
                data[f"{m}_{s}"] = [0.01, 0.02]
        data["total_s"] = [0.03, 0.06]
        return pd.DataFrame(data)

    monkeypatch.setattr(latency, "time_per_flow_decision", fake_per_flow)
    monkeypatch.setattr(latency, "time_windowed_decision", fake_windowed)
    return calls


def test_run_latency_default_has_no_exact_arm(tmp_path, monkeypatch):
    ws = _make_per_flow_workspace(tmp_path)
    calls = _stub_arms(monkeypatch)
    latency.run_latency(ws, n=2, watchdog_s=0)
    host = latency.machine_info()["hostname"]
    rebuild = ws.tables_dir / "rebuild"
    assert len(calls) == 1
    assert not (rebuild / f"latency_{host}_per_flow_exact.csv").exists()
    summary = pd.read_csv(rebuild / f"latency_{host}.csv")
    assert set(summary["mode"]) == {"per_flow", "windowed"}
    side = json.loads((rebuild / f"latency_{host}.csv.provenance.json").read_text())
    assert side["params"]["baseline_source"] == "committed"


def test_run_latency_exact_arm_and_fresh_baselines_are_recorded(
    tmp_path, monkeypatch, _patch_per_flow_backends,
):
    from uav_tda import exact

    ws = _make_per_flow_workspace(tmp_path)
    calls = _stub_arms(monkeypatch)
    report = latency.run_latency(
        ws, n=2, fresh_baselines=True, exact_arm=True, watchdog_s=0)
    host = latency.machine_info()["hostname"]
    rebuild = ws.tables_dir / "rebuild"
    # one baseline realization shared by both per-flow arms
    assert len(calls) == 2
    assert calls[0]["baselines"] is calls[1]["baselines"]
    assert calls[1]["top_k"] is None and calls[1]["delta"] == exact.PRIMARY_DELTA

    summary = pd.read_csv(rebuild / f"latency_{host}.csv")
    ex = summary[(summary["mode"] == "per_flow_exact") & (summary["stage"] == "total_s")]
    assert len(ex) == 1 and ex.iloc[0]["n"] == 2
    detail = rebuild / f"latency_{host}_per_flow_exact.csv"
    side = json.loads(Path(str(detail) + ".provenance.json").read_text())
    assert side["params"]["baseline_source"] == "fresh"
    assert side["params"]["delta"] == exact.PRIMARY_DELTA and side["params"]["top_k"] is None

    text = report.read_text()
    assert "0.9082 configuration" in text
    assert "Baseline source: **fresh**" in text


def test_progress_lines_name_each_stage_on_stderr(tmp_path, monkeypatch, capsys):
    ws = _make_per_flow_workspace(tmp_path)
    _stub_arms(monkeypatch)
    latency.run_latency(ws, n=2, watchdog_s=0)
    err = capsys.readouterr().err
    for needle in ("baselines", "per_flow", "W=25", "W=200"):
        assert needle in err


def test_per_flow_progress_names_each_flow_outside_the_timed_stages(
    tmp_path, capsys, _patch_per_flow_backends,
):
    ws = _make_per_flow_workspace(tmp_path, n_test=3)
    seen = []
    latency.time_per_flow_decision(ws, n=3, progress=seen.append)
    assert len(seen) == 3
    assert seen[0].startswith("per_flow 1/3: flow ")


def test_watchdog_kills_a_stuck_step_and_names_it():
    import subprocess
    import sys

    code = (
        "import time\n"
        "from uav_tda.latency import Watchdog\n"
        "w = Watchdog(1)\n"
        "w.step('stuck step here')\n"
        "time.sleep(60)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50)
    assert res.returncode != 0
    assert "stuck step here" in res.stderr
    assert "Timeout" in res.stderr and "line 5" in res.stderr


def test_watchdog_rearms_on_each_step_and_cancels():
    import subprocess
    import sys

    code = (
        "import time\n"
        "from uav_tda.latency import Watchdog\n"
        "w = Watchdog(2)\n"
        "for i in range(3):\n"
        "    w.step(f'step {i}')\n"
        "    time.sleep(1)\n"
        "w.cancel()\n"
        "time.sleep(2.5)\n"
        "print('survived')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50)
    assert res.returncode == 0, res.stderr
    assert "survived" in res.stdout


def test_cli_latency_passes_new_flags(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(latency, "run_latency",
                        lambda ws, **kw: got.update(kw) or tmp_path / "x.md")
    parser = cli.build_parser()
    args = parser.parse_args(["latency", "--root", str(tmp_path), "--n", "4",
                              "--fresh-baselines", "--exact-arm", "--watchdog", "60"])
    assert args.func(args) == 0
    assert got == {"n": 4, "fresh_baselines": True, "exact_arm": True, "watchdog_s": 60.0,
                   "exact_only": False}
    d = parser.parse_args(["latency"])
    assert (d.fresh_baselines, d.exact_arm, d.watchdog, d.exact_only) == (False, False, 900.0, False)
    assert parser.parse_args(["latency-report"]).func is cli._cmd_latency_report
