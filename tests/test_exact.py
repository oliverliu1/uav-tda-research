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

from uav_tda import cli, config, exact, metrics, probe
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


def test_exact_w2_flow_uses_primary_then_retry_delta(monkeypatch):
    """Campaign definition (2026-09-22 revision): first attempt at
    `exact.PRIMARY_DELTA` (0.01, hera's own default tolerance -- delta=0.0
    true-exact was found intractable at 120s on real diagrams); on timeout,
    retry once at `exact.RETRY_DELTA` (0.05), NOT delta=0.0.
    """
    calls = []

    def _fake_w2_with_timeout(d1, d2, order, internal_p, delta, timeout_sec):  # noqa: ARG001
        calls.append(delta)
        return float("nan") if delta == exact.PRIMARY_DELTA else 0.42

    monkeypatch.setattr(probe, "_w2_with_timeout", _fake_w2_with_timeout)

    diagram = np.array([[0.0, 0.1, 0.5]])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    total, n_timeouts, approx_flag = exact.exact_w2_flow(
        diagram, baselines_m, max_edge=1.0, max_hom_dim=0, timeout_s=10.0,
    )
    assert calls == [exact.PRIMARY_DELTA, exact.RETRY_DELTA]
    assert 0.0 not in calls  # never attempts delta=0.0 (true exact) -- intractable, dropped
    assert total == pytest.approx(0.42)
    assert n_timeouts == 0
    assert approx_flag is True


def test_direct_w2_flow_finite_real_call():
    """The campaign path: no fork/timeout wrapper, real (unmocked) hera call."""
    diagram = np.array([
        [0.0, 0.1, 0.5],
        [0.0, 0.2, 0.6],
    ])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    total, n_timeouts, approx_flag = exact.direct_w2_flow(
        diagram, baselines_m, max_edge=1.0, max_hom_dim=0,
    )
    assert np.isfinite(total)
    assert total >= 0.0
    assert n_timeouts == 0
    assert approx_flag is False


def test_direct_w2_flow_uses_given_delta(monkeypatch):
    calls = []

    def _fake_wdist(d1, d2, order, internal_p, delta):  # noqa: ARG001
        calls.append(delta)
        return 2.71

    import gudhi.hera as hera_module
    monkeypatch.setattr(hera_module, "wasserstein_distance", _fake_wdist)

    diagram = np.array([[0.0, 0.1, 0.5]])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    total, n_timeouts, approx_flag = exact.direct_w2_flow(
        diagram, baselines_m, max_edge=1.0, max_hom_dim=0, delta=0.01,
    )
    assert calls == [0.01]
    assert total == pytest.approx(2.71)
    assert n_timeouts == 0
    assert approx_flag is False


def test_direct_w2_flow_catches_python_exception_counts_as_timeout(monkeypatch, caplog):
    """Isolation-probe diagnostic (2026-09-22, third intervention): a
    Python-level exception from hera is caught (does NOT crash the worker),
    logged with row_idx/manifold/dim context, and treated exactly like a
    timeout for row-schema purposes -- distinguishing catchable Python
    errors from uncatchable native crashes (which would still kill the
    process outright, unaffected by this try/except).
    """
    def _raising_wdist(d1, d2, order, internal_p, delta):  # noqa: ARG001
        raise RuntimeError("boom")

    import gudhi.hera as hera_module
    monkeypatch.setattr(hera_module, "wasserstein_distance", _raising_wdist)

    diagram = np.array([[0.0, 0.1, 0.5]])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    with caplog.at_level("ERROR", logger="uav_tda.exact"):
        total, n_timeouts, approx_flag = exact.direct_w2_flow(
            diagram, baselines_m, max_edge=1.0, max_hom_dim=0, row_idx=42, manifold="c2",
        )
    assert total == 0.0
    assert n_timeouts == 1
    assert approx_flag is True
    assert any("row_idx=42" in r.message and "manifold=c2" in r.message for r in caplog.records)


def test_free_memory_mb_returns_float_or_none():
    """Best-effort telemetry helper (2026-09-22, third intervention): must
    never raise, and returns either a float or None (environment-dependent
    -- vm_stat on macOS, psutil fallback elsewhere, None if both fail).
    """
    result = exact._free_memory_mb()
    assert result is None or isinstance(result, float)


def test_run_shard_chunks_and_combines_rows_correctly(tmp_path, monkeypatch):
    """Low-memory chunked profile (2026-09-22, third intervention):
    `run_shard` with `chunk_size` smaller than the shard size must still
    produce exactly one correctly-combined row per flow, per manifold,
    regardless of how flows were split into chunks.
    """
    import pickle

    ws = _make_workspace(tmp_path)
    n = 5
    pd.DataFrame({"label": ["Normal Traffic"] * n}).to_csv(
        ws.outputs_dir / "labels_test.csv", index=False
    )
    ws.persistence_dir.mkdir(parents=True, exist_ok=True)
    for m in config.MANIFOLDS:
        diagrams = [np.array([[0.0, 0.1, 0.5]]) for _ in range(n)]
        (ws.persistence_dir / f"{m}_test.pkl").write_bytes(pickle.dumps(diagrams))

    baselines = {m: {k: np.array([[0.15, 0.55]]) for k in range(config.MAX_HOM_DIM[m] + 1)}
                 for m in config.MANIFOLDS}

    def _fake_direct(diagram, baselines_m, max_edge, max_hom_dim, delta=exact.PRIMARY_DELTA,  # noqa: ARG001
                      row_idx=None, manifold=None):
        # deterministic, identifiable per (row_idx, manifold)
        return float(row_idx) * 10.0 + (0.1 if manifold == "network" else 0.2 if manifold == "physical" else 0.0), 0, False

    monkeypatch.setattr(exact, "direct_w2_flow", _fake_direct)

    exact_dir = ws.tables_dir / "rebuild" / "exact"
    shard_path = exact.run_shard(ws, exact_dir, "test", 0, n, baselines, n_jobs=1, chunk_size=2)

    df = pd.read_csv(shard_path)
    assert list(df["row_idx"]) == list(range(n))
    for i in range(n):
        row = df[df["row_idx"] == i].iloc[0]
        assert row["W2_c2"] == pytest.approx(i * 10.0)
        assert row["W2_network"] == pytest.approx(i * 10.0 + 0.1)
        assert row["W2_physical"] == pytest.approx(i * 10.0 + 0.2)
        assert row["n_timeouts"] == 0
        assert row["approx_flag"] == False  # noqa: E712


def test_run_shard_calls_direct_w2_flow_not_exact_w2_flow(tmp_path, monkeypatch):
    """Regression guard: the campaign path (`run_shard`) must use the
    unwrapped in-process `direct_w2_flow`, not `exact_w2_flow`'s
    fork-timeout machinery (unstable when nested inside joblib/loky
    workers -- see module docstring, 2026-09-22 intervention).
    """
    import pickle

    ws = _make_workspace(tmp_path)
    pd.DataFrame({"label": ["Normal Traffic", "Sybil Attack"]}).to_csv(
        ws.outputs_dir / "labels_test.csv", index=False
    )
    ws.persistence_dir.mkdir(parents=True, exist_ok=True)
    for m in config.MANIFOLDS:
        diagrams = [np.array([[0.0, 0.1, 0.5]]), np.array([[0.0, 0.1, 0.5]])]
        (ws.persistence_dir / f"{m}_test.pkl").write_bytes(pickle.dumps(diagrams))

    baselines = {m: {k: np.array([[0.15, 0.55]]) for k in range(config.MAX_HOM_DIM[m] + 1)}
                 for m in config.MANIFOLDS}

    direct_calls = {"n": 0}
    exact_calls = {"n": 0}

    def _fake_direct(diagram, baselines_m, max_edge, max_hom_dim, delta=exact.PRIMARY_DELTA,  # noqa: ARG001
                      row_idx=None, manifold=None):  # noqa: ARG001
        direct_calls["n"] += 1
        return 1.0, 0, False

    def _fake_exact(*args, **kwargs):  # noqa: ARG001
        exact_calls["n"] += 1
        return 1.0, 0, False

    monkeypatch.setattr(exact, "direct_w2_flow", _fake_direct)
    monkeypatch.setattr(exact, "exact_w2_flow", _fake_exact)

    exact_dir = ws.tables_dir / "rebuild" / "exact"
    exact.run_shard(ws, exact_dir, "test", 0, 2, baselines, n_jobs=1)

    assert direct_calls["n"] == 2 * len(config.MANIFOLDS)
    assert exact_calls["n"] == 0


def test_exact_w2_flow_primary_delta_success_no_retry(monkeypatch):
    calls = []

    def _fake_w2_with_timeout(d1, d2, order, internal_p, delta, timeout_sec):  # noqa: ARG001
        calls.append(delta)
        return 1.23

    monkeypatch.setattr(probe, "_w2_with_timeout", _fake_w2_with_timeout)

    diagram = np.array([[0.0, 0.1, 0.5]])
    baselines_m = {0: np.array([[0.15, 0.55]])}
    total, n_timeouts, approx_flag = exact.exact_w2_flow(
        diagram, baselines_m, max_edge=1.0, max_hom_dim=0, timeout_s=10.0,
    )
    assert calls == [exact.PRIMARY_DELTA]
    assert total == pytest.approx(1.23)
    assert n_timeouts == 0
    assert approx_flag is False


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


# ---------------------------------------------------------------------------
# assemble_distances (Task 2)
# ---------------------------------------------------------------------------


def _write_shard(exact_dir: Path, split: str, start: int, rows: "list[dict]") -> Path:
    exact_dir.mkdir(parents=True, exist_ok=True)
    path = exact_dir / f"shard_{split}_{start:05d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _shard_row(row_idx: int, label: str, c2: float, network: float, physical: float) -> dict:
    return {
        "row_idx": row_idx, "label": label,
        "W2_c2": c2, "W2_network": network, "W2_physical": physical,
        "n_timeouts": 0, "approx_flag": False,
    }


def test_assemble_distances_order_count_and_subset_sums(tmp_path):
    ws = _make_workspace(tmp_path)
    exact_dir = ws.tables_dir / "rebuild" / "exact"

    # Two out-of-order test shards plus one unrelated val shard, to check
    # start-order concatenation and split filtering.
    _write_shard(exact_dir, "test", 3, [
        _shard_row(3, "Sybil Attack", 1.0, 5.0, 0.5),
        _shard_row(4, "Normal Traffic", 0.2, 0.3, 0.1),
        _shard_row(5, "Wormhole Attack", 0.4, 0.6, 4.0),
    ])
    _write_shard(exact_dir, "test", 0, [
        _shard_row(0, "Normal Traffic", 0.1, 0.2, 0.3),
        _shard_row(1, "Blackhole Attack", 0.9, 0.8, 3.0),
    ])
    _write_shard(exact_dir, "val", 0, [
        _shard_row(0, "Normal Traffic", 0.05, 0.06, 0.07),
    ])

    manifest = {
        "shards": {
            "test_00003": {"split": "test", "start": 3, "size": 3, "n_rows": 3,
                            "path": "shard_test_00003.csv"},
            "test_00000": {"split": "test", "start": 0, "size": 2, "n_rows": 2,
                            "path": "shard_test_00000.csv"},
            "val_00000": {"split": "val", "start": 0, "size": 1, "n_rows": 1,
                          "path": "shard_val_00000.csv"},
        }
    }
    (exact_dir / "manifest.json").write_text(json.dumps(manifest))

    df = exact.assemble_distances(exact_dir, "test")

    assert list(df["row_idx"]) == [0, 1, 3, 4, 5]
    assert len(df) == 5

    for subset, manifolds in metrics.MANIFOLD_SUBSETS.items():
        expected = sum(df[f"W2_{m}"] for m in manifolds)
        assert np.allclose(df[f"W2_{subset}"].to_numpy(), expected.to_numpy())


def test_assemble_distances_raises_on_row_count_mismatch(tmp_path):
    ws = _make_workspace(tmp_path)
    exact_dir = ws.tables_dir / "rebuild" / "exact"

    _write_shard(exact_dir, "test", 0, [
        _shard_row(0, "Normal Traffic", 0.1, 0.2, 0.3),
    ])
    manifest = {
        "shards": {
            # manifest claims 2 rows but the shard file only has 1.
            "test_00000": {"split": "test", "start": 0, "size": 2, "n_rows": 2,
                            "path": "shard_test_00000.csv"},
        }
    }
    (exact_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="row count mismatch"):
        exact.assemble_distances(exact_dir, "test")


def test_assemble_distances_raises_when_no_shards_recorded(tmp_path):
    ws = _make_workspace(tmp_path)
    exact_dir = ws.tables_dir / "rebuild" / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    (exact_dir / "manifest.json").write_text(json.dumps({"shards": {}}))

    with pytest.raises(ValueError, match="no completed shards"):
        exact.assemble_distances(exact_dir, "test")


# ---------------------------------------------------------------------------
# build_exact_tables / write_exact_tables (Task 2)
# ---------------------------------------------------------------------------

_ATTACK_DOMINANT_MANIFOLD = {
    "Blackhole Attack": "physical",
    "Wormhole Attack": "physical",
    "Sybil Attack": "network",
    "Flooding Attack": "network",
}


def _add_subset_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for subset, manifolds in metrics.MANIFOLD_SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)
    return df


def _make_synthetic_val_test(flagged_test_idx: "set[int]" = frozenset()):
    """Synthetic val (Normal-only) + test frames shaped like `assemble_distances`
    output: label + per-manifold W2_* + the 7 subset-sum columns +
    n_timeouts/approx_flag. Attack rows are boosted in one "dominant"
    manifold per `_ATTACK_DOMINANT_MANIFOLD` so per-attack dominance is
    checkable deterministically.
    """
    rng = np.random.default_rng(0)

    def _manifold_values(dominant: "str | None") -> dict:
        vals = {m: float(rng.uniform(0.0, 1.0)) for m in ("c2", "network", "physical")}
        if dominant is not None:
            vals[dominant] += 5.0
        return vals

    val_rows = []
    for i in range(8):
        row = {"row_idx": i, "label": "Normal Traffic", "n_timeouts": 0, "approx_flag": False}
        row.update({f"W2_{m}": v for m, v in _manifold_values(None).items()})
        val_rows.append(row)
    val_df = _add_subset_cols(pd.DataFrame(val_rows))

    classes = ["Normal Traffic", *_ATTACK_DOMINANT_MANIFOLD.keys()]
    test_rows = []
    idx = 0
    for label in classes:
        dominant = _ATTACK_DOMINANT_MANIFOLD.get(label)
        for _ in range(6):
            row = {"row_idx": idx, "label": label, "n_timeouts": 0,
                   "approx_flag": idx in flagged_test_idx}
            row.update({f"W2_{m}": v for m, v in _manifold_values(dominant).items()})
            test_rows.append(row)
            idx += 1
    test_df = _add_subset_cols(pd.DataFrame(test_rows))
    return val_df, test_df


def _write_fake_probe_table(ws: Workspace) -> None:
    rows = []
    for subset in metrics.MANIFOLD_SUBSETS:
        for scoring in ("raw", "znorm"):
            rows.append({
                "subset": subset, "scoring": scoring,
                "mean": 0.75, "std": 0.02, "ci_lo": 0.70, "ci_hi": 0.80, "n_seeds": 10,
            })
    probe_dir = ws.tables_dir / "rebuild"
    probe_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(probe_dir / "binary_auc.csv", index=False)


def _patch_assemble(monkeypatch, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def _fake(exact_dir, split):  # noqa: ARG001
        return val_df.copy() if split == "val" else test_df.copy()
    monkeypatch.setattr(exact, "assemble_distances", _fake)


def test_build_exact_tables_shapes_ci_and_dominant_flags(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    val_df, test_df = _make_synthetic_val_test()
    _patch_assemble(monkeypatch, val_df, test_df)
    _write_fake_probe_table(ws)

    tables = exact.build_exact_tables(ws, B=20, bootstrap_seed=0)

    assert set(tables) == {"exact_binary_auc", "exact_per_attack_auc", "exact_vs_probe"}

    binary = tables["exact_binary_auc"]
    assert len(binary) == 14  # 7 subsets x {raw, znorm}
    assert set(binary["subset"]) == set(metrics.MANIFOLD_SUBSETS)
    assert set(binary["scoring"]) == {"raw", "znorm"}
    assert (binary["ci_lo"] <= binary["auc"]).all()
    assert (binary["auc"] <= binary["ci_hi"]).all()
    assert (binary["n_flows"] == len(test_df)).all()
    assert (binary["n_approx_flagged"] == 0).all()

    per_attack = tables["exact_per_attack_auc"]
    assert len(per_attack) == 4 * 3
    assert set(per_attack["attack_class"]) == set(_ATTACK_DOMINANT_MANIFOLD)
    assert (per_attack["ci_lo"] <= per_attack["auc"]).all()
    assert (per_attack["auc"] <= per_attack["ci_hi"]).all()
    for attack, expected_dominant in _ATTACK_DOMINANT_MANIFOLD.items():
        rows = per_attack[per_attack["attack_class"] == attack]
        assert rows["dominant"].sum() == 1, "exactly one dominant manifold per attack"
        dominant_row = rows[rows["dominant"]]
        assert dominant_row["manifold"].iloc[0] == expected_dominant

    vs_probe = tables["exact_vs_probe"]
    expected_cols = {"subset", "scoring", "exact_auc", "probe10_mean", "probe10_std",
                      "published3_mean", "published3_std", "delta_exact_minus_probe"}
    assert set(vs_probe.columns) == expected_cols
    assert len(vs_probe) == 14

    raw_rows = vs_probe[vs_probe["scoring"] == "raw"].set_index("subset")
    for subset, (mean, std) in exact.PUBLISHED3_RAW.items():
        assert raw_rows.loc[subset, "published3_mean"] == pytest.approx(mean)
        assert raw_rows.loc[subset, "published3_std"] == pytest.approx(std)

    znorm_rows = vs_probe[vs_probe["scoring"] == "znorm"]
    assert znorm_rows["published3_mean"].isna().all()
    assert znorm_rows["published3_std"].isna().all()

    assert np.allclose(vs_probe["probe10_mean"].to_numpy(), 0.75)
    assert np.allclose(
        vs_probe["delta_exact_minus_probe"].to_numpy(),
        (vs_probe["exact_auc"] - vs_probe["probe10_mean"]).to_numpy(),
    )


def test_build_exact_tables_excl_flagged_only_when_flags_present(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    _write_fake_probe_table(ws)

    # No approx-flagged flows -> no excl-flagged table.
    val_df, test_df = _make_synthetic_val_test()
    _patch_assemble(monkeypatch, val_df, test_df)
    tables = exact.build_exact_tables(ws, B=10, bootstrap_seed=0)
    assert "exact_binary_auc_excl_flagged" not in tables

    # Two flagged test flows -> excl-flagged table present, computed on
    # fewer flows.
    val_df2, test_df2 = _make_synthetic_val_test(flagged_test_idx={0, 5})
    _patch_assemble(monkeypatch, val_df2, test_df2)
    tables2 = exact.build_exact_tables(ws, B=10, bootstrap_seed=0)
    assert "exact_binary_auc_excl_flagged" in tables2
    excl = tables2["exact_binary_auc_excl_flagged"]
    assert (excl["n_flows"] == len(test_df2) - 2).all()


def test_write_exact_tables_writes_csvs_and_provenance(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    _write_fake_probe_table(ws)
    val_df, test_df = _make_synthetic_val_test(flagged_test_idx={0})
    _patch_assemble(monkeypatch, val_df, test_df)

    tables = exact.build_exact_tables(ws, B=10, bootstrap_seed=0)
    paths = exact.write_exact_tables(ws, tables)

    exact_dir = ws.tables_dir / "rebuild" / "exact"
    assert paths["exact_binary_auc"] == exact_dir / "exact_binary_auc.csv"
    for name, path in paths.items():
        assert path.exists()
        assert path.with_name(path.name + ".provenance.json").exists()
    assert "exact_binary_auc_excl_flagged" in paths


# ---------------------------------------------------------------------------
# CLI registration (Task 2)
# ---------------------------------------------------------------------------


def test_cli_registers_exact_subcommand_with_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["exact", "--n-jobs", "3", "--shard-size", "250"])
    assert args.func is cli._cmd_exact
    assert args.n_jobs == 3
    assert args.shard_size == 250

    defaults = parser.parse_args(["exact"])
    assert defaults.n_jobs == -1
    assert defaults.shard_size == 500


def test_cli_registers_exact_report_subcommand_with_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["exact-report", "--bootstrap", "500"])
    assert args.func is cli._cmd_exact_report
    assert args.bootstrap == 500

    defaults = parser.parse_args(["exact-report"])
    assert defaults.bootstrap == 2000


# ---------------------------------------------------------------------------
# insensitivity_check / write_insensitivity_check (Task 3, controller ruling
# 2026-09-22)
# ---------------------------------------------------------------------------


def _make_workspace_with_test_split(tmp_path: Path, n_normal: int = 3, n_attack: int = 3) -> Workspace:
    import pickle

    ws = _make_workspace(tmp_path)
    labels = ["Normal Traffic"] * n_normal + ["Sybil Attack"] * n_attack
    pd.DataFrame({"label": labels}).to_csv(ws.outputs_dir / "labels_test.csv", index=False)

    ws.persistence_dir.mkdir(parents=True, exist_ok=True)
    n = n_normal + n_attack
    for m in config.MANIFOLDS:
        # Diagram content only needs to identify its row index (i encoded in
        # [0, 0]) -- `_w2_sum_at_delta` is monkeypatched in these tests, so
        # the real slicing/hera path is never exercised.
        diagrams = [np.array([[float(i), 0.0, 0.0]]) for i in range(n)]
        (ws.persistence_dir / f"{m}_test.pkl").write_bytes(pickle.dumps(diagrams))
    return ws


def test_insensitivity_check_shapes_and_perfect_separation(tmp_path, monkeypatch):
    n_normal, n_attack = 3, 3
    ws = _make_workspace_with_test_split(tmp_path, n_normal, n_attack)
    exact_dir = ws.tables_dir / "rebuild" / "exact"

    monkeypatch.setattr(exact, "ensure_baselines", lambda ws, exact_dir: {
        m: {} for m in config.MANIFOLDS
    })

    def _fake_w2_sum_at_delta(diagram, baselines_m, max_edge, max_hom_dim, delta, timeout_s):  # noqa: ARG001
        i = int(diagram[0, 0])
        base = 10.0 if i >= n_normal else 1.0  # attack rows score strictly higher
        bump = 0.0 if delta == 0.01 else 0.5  # delta=0.05 differs but doesn't flip rank order
        return base + bump

    monkeypatch.setattr(exact, "_w2_sum_at_delta", _fake_w2_sum_at_delta)

    df = exact.insensitivity_check(ws, exact_dir, n=n_normal + n_attack, rng_seed=0, n_jobs=1)

    assert set(df.columns) == {"subset", "auc_delta01", "auc_delta05", "delta_auc", "n_flows"}
    assert len(df) == len(metrics.MANIFOLD_SUBSETS)
    assert set(df["subset"]) == set(metrics.MANIFOLD_SUBSETS)
    assert (df["n_flows"] == n_normal + n_attack).all()
    assert np.allclose(df["auc_delta01"].to_numpy(), 1.0)
    assert np.allclose(df["auc_delta05"].to_numpy(), 1.0)
    assert np.allclose(df["delta_auc"].to_numpy(), 0.0)


def test_insensitivity_check_respects_n_subsample(tmp_path, monkeypatch):
    ws = _make_workspace_with_test_split(tmp_path, n_normal=5, n_attack=5)
    exact_dir = ws.tables_dir / "rebuild" / "exact"

    monkeypatch.setattr(exact, "ensure_baselines", lambda ws, exact_dir: {
        m: {} for m in config.MANIFOLDS
    })
    monkeypatch.setattr(
        exact, "_w2_sum_at_delta",
        lambda diagram, baselines_m, max_edge, max_hom_dim, delta, timeout_s: float(diagram[0, 0]),  # noqa: ARG005
    )

    df = exact.insensitivity_check(ws, exact_dir, n=4, rng_seed=1, n_jobs=1)
    assert (df["n_flows"] == 4).all()


def test_write_insensitivity_check_writes_csv_and_provenance(tmp_path):
    ws = _make_workspace(tmp_path)
    df = pd.DataFrame([
        {"subset": "all_three", "auc_delta01": 0.9, "auc_delta05": 0.899,
         "delta_auc": -0.001, "n_flows": 500},
    ])
    path = exact.write_insensitivity_check(ws, df)
    assert path == ws.tables_dir / "rebuild" / "exact" / "insensitivity_check.csv"
    assert path.exists()
    assert path.with_name(path.name + ".provenance.json").exists()
