"""Tests for uav_tda.hangcheck (hera non-termination on near-identical diagrams).

Fast/synthetic: Rips is faked and the W2 call is injected; only tiny forked
children run. Every test uses `tmp_path`.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from uav_tda import cli, config, hangcheck
from tests.test_latency import _make_per_flow_workspace


def test_diagram_gap_is_order_free_and_none_for_different_sizes():
    a = np.array([[0.0, 0.5], [0.1, 0.9]])
    assert hangcheck.diagram_gap(a, a[::-1]) == 0.0
    b = a.copy()
    b[0, 1] += 2 ** -54 * 2
    assert 0 < hangcheck.diagram_gap(a, b) < 1e-15
    assert hangcheck.diagram_gap(a, a[:1]) is None
    assert hangcheck.diagram_gap(np.empty((0, 2)), np.empty((0, 2))) == 0.0


def test_timed_w2_reports_ok_timeout_and_error():
    a = np.zeros((1, 2))

    def hang(*args, **kw):
        time.sleep(60)

    def boom(*args, **kw):
        raise RuntimeError("no")

    ok = hangcheck.timed_w2(a, a, delta=0.2, timeout_s=10, w2_fn=lambda *x, **k: 1.5)
    assert (ok["status"], ok["w2"]) == ("ok", 1.5)
    t0 = time.perf_counter()
    stuck = hangcheck.timed_w2(a, a, delta=0.2, timeout_s=0.5, w2_fn=hang)
    assert stuck["status"] == "timeout" and np.isnan(stuck["w2"])
    assert time.perf_counter() - t0 < 10
    assert hangcheck.timed_w2(a, a, delta=0.2, timeout_s=10, w2_fn=boom)["status"] == "error"


def _fake_persistence(query_point, reference_points, max_edge, max_simplex_dim, sparse):
    return np.array([[float(d), 0.1 * (d + 1), 0.4 * (d + 1)] for d in range(max_simplex_dim)])


@pytest.fixture
def _fast(monkeypatch):
    from uav_tda import tda as tda_module

    monkeypatch.setattr(tda_module, "_persistence_for_point", _fake_persistence)

    def fake(a, b, delta, timeout_s, w2_fn=None):
        stuck = len(a) == len(b) and delta == config.PROBE_DELTA
        return {"status": "timeout" if stuck else "ok",
                "w2": np.nan if stuck else 0.1, "seconds": 0.0}

    monkeypatch.setattr(hangcheck, "timed_w2", fake)


def test_run_hang_check_covers_both_baselines_and_configs(tmp_path, _fast):
    ws = _make_per_flow_workspace(tmp_path, n_test=4)
    df = hangcheck.run_hang_check(ws, n=2)
    n_dims = sum(h + 1 for h in config.MAX_HOM_DIM.values())
    assert len(df) == 2 * 2 * 2 * n_dims  # flows x baselines x configs x dims
    assert set(df["baseline"]) == {"committed", "fresh"}
    assert set(df["config"]) == {"probe", "exact"}
    # fresh baselines equal the (fake) flow diagrams exactly; committed ones do not match in size
    fresh = df[(df["baseline"] == "fresh") & (df["config"] == "probe")]
    assert (fresh["gap"] == 0.0).all() and (fresh["status"] == "timeout").all()
    assert set(df["timeout_s"]) == set(hangcheck.TIMEOUT_S.values())


def test_summary_counts_and_flows_affected(tmp_path, _fast):
    ws = _make_per_flow_workspace(tmp_path, n_test=4)
    s = hangcheck.summarize(hangcheck.run_hang_check(ws, n=2))
    row = s[(s["baseline"] == "fresh") & (s["config"] == "probe")].iloc[0]
    assert row["n_comparisons"] == row["n_timeout"] == row["n_equal"]
    assert row["n_flows"] == row["n_flows_timeout"] == 2
    assert row["n_near"] == 0 and row["n_error"] == 0


def test_skips_committed_when_absent(tmp_path, _fast):
    ws = _make_per_flow_workspace(tmp_path, n_test=4)
    (ws.tables_dir / "rebuild" / "exact" / "baselines_manifest.json").unlink()
    assert set(hangcheck.run_hang_check(ws, n=1)["baseline"]) == {"fresh"}


def test_write_outputs_with_sidecars_keyed_by_host(tmp_path, _fast):
    ws = _make_per_flow_workspace(tmp_path, n_test=4)
    detail, summary = hangcheck.write_hang_check(ws, hangcheck.run_hang_check(ws, n=1), n=1)
    host = hangcheck.machine_info()["hostname"]
    assert detail.name == f"hang_check_{host}.csv"
    assert summary.name == f"hang_check_{host}_summary.csv"
    for p in (detail, summary):
        side = json.loads(p.with_name(p.name + ".provenance.json").read_text())
        assert side["params"]["n"] == 1
        assert side["params"]["machine_info"]["hostname"] == host
    assert len(pd.read_csv(summary)) == 4
    assert not summary.name.startswith("latency_")  # never mistaken for a latency summary


def test_cli_registers_hang_check():
    args = cli.build_parser().parse_args(["hang-check", "--n", "5"])
    assert args.func is cli._cmd_hang_check and args.n == 5
