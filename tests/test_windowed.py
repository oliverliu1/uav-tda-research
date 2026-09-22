import numpy as np
import pandas as pd
import pytest

from uav_tda import cli, config, windowed
from uav_tda.metrics import MANIFOLD_SUBSETS
from uav_tda.workspace import Workspace


def test_config_constants():
    assert config.WINDOW_SIZES == (25, 50, 100, 200)
    assert config.WINDOWED_REPEATS == 10
    assert config.WINDOWED_SHUFFLE_SEEDS == tuple(range(10))


def test_windowed_sparse_covers_all_manifolds():
    # Fixed by the Task-1 benchmark gate (see comment in config.py).
    assert set(config.WINDOWED_SPARSE) == set(config.MANIFOLDS)
    for eps in config.WINDOWED_SPARSE.values():
        assert eps is None or isinstance(eps, float)


def test_make_windows_drops_trailing_partial():
    wins = windowed.make_windows(105, 25)
    assert len(wins) == 4
    assert all(len(w) == 25 for w in wins)
    assert wins[0].tolist() == list(range(25))
    assert wins[3].tolist() == list(range(75, 100))  # rows 100-104 dropped


def test_make_windows_respects_permutation():
    order = np.arange(50)[::-1]  # reversed
    wins = windowed.make_windows(50, 10, order=order)
    assert wins[0].tolist() == list(range(49, 39, -1))
    # every row appears at most once across windows
    flat = np.concatenate(wins)
    assert len(np.unique(flat)) == len(flat)


def test_window_diagram_shape_and_determinism_exact():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 5))
    d1 = windowed.window_diagram(pts, max_edge=0.8, max_hom_dim=1, sparse=None)
    d2 = windowed.window_diagram(pts, max_edge=0.8, max_hom_dim=1, sparse=None)
    assert d1.ndim == 2 and d1.shape[1] == 3
    assert np.array_equal(d1, d2)  # exact Rips: bit-identical, no sort needed
    assert set(np.unique(d1[:, 0])) <= {0.0, 1.0}


# --- Task 2: w2_distance, baseline_medoid_diagram, labeling helpers --------

def _diag(rows):
    """Build an (n, 3) [dim, birth, death] diagram from a list of (dim, b, d)."""
    return np.array(rows, dtype=float) if rows else np.empty((0, 3), dtype=float)


def test_w2_distance_zero_on_identical():
    d = _diag([(0, 0.0, 0.5), (0, 0.1, 0.3), (1, 0.2, 0.6)])
    assert windowed.w2_distance(d, d, max_edge=1.0, max_hom_dim=1) == pytest.approx(0.0, abs=1e-9)


def test_w2_distance_symmetry():
    d1 = _diag([(0, 0.0, 0.5), (0, 0.1, 0.3), (1, 0.2, 0.6)])
    d2 = _diag([(0, 0.0, 0.4), (0, 0.15, 0.35), (1, 0.25, 0.55)])
    fwd = windowed.w2_distance(d1, d2, max_edge=1.0, max_hom_dim=1)
    rev = windowed.w2_distance(d2, d1, max_edge=1.0, max_hom_dim=1)
    assert fwd == pytest.approx(rev, abs=1e-9)
    assert fwd > 0.0


def test_baseline_medoid_diagram_picks_similar_pair():
    similar_a = _diag([(0, 0.0, 0.5), (0, 0.1, 0.3)])
    similar_b = _diag([(0, 0.0, 0.52), (0, 0.11, 0.31)])
    outlier = _diag([(0, 0.0, 5.0), (0, 3.0, 9.0)])
    idx, medoid = windowed.baseline_medoid_diagram(
        [similar_a, similar_b, outlier], max_edge=10.0, max_hom_dim=0,
    )
    assert idx in (0, 1)
    assert np.array_equal(medoid, [similar_a, similar_b][idx])


def test_window_majority_label_normal_majority():
    labels = ["Normal Traffic"] * 6 + ["Sybil Attack"] * 4
    assert windowed.window_majority_label(labels) == "Normal Traffic"
    assert windowed.window_attack_frac(labels) == pytest.approx(0.4)


def test_window_majority_label_attack_majority():
    labels = ["Normal Traffic"] * 4 + ["Sybil Attack"] * 6
    assert windowed.window_majority_label(labels) == "Sybil Attack"
    assert windowed.window_attack_frac(labels) == pytest.approx(0.6)


@pytest.mark.slow
def test_run_windowed_smoke_w100():
    from uav_tda.workspace import Workspace

    ws = Workspace.default()
    window_df, stats, timing = windowed.run_windowed(ws, w=100)

    assert len(window_df) == 18326 // 100 == 183

    score_cols = [c for c in window_df.columns if c.startswith(("W2_", "Z2_"))]
    assert len(score_cols) == 20  # 3 manifolds + 7 subsets, raw + Z2
    for c in score_cols:
        assert np.isfinite(window_df[c].to_numpy()).all()

    assert set(stats.keys()) == {"c2", "network", "physical"}
    for m, (mean, std) in stats.items():
        assert std > 0

    assert timing["total_s"] > 0
    assert timing["n_windows"] == len(window_df)

    attack_heavy = window_df[window_df["attack_frac"] > 0.9]
    pure_normal = window_df[window_df["attack_frac"] == 0.0]
    assert len(attack_heavy) > 0 and len(pure_normal) > 0
    assert attack_heavy["Z2_all_three"].mean() > pure_normal["Z2_all_three"].mean()


# --- Task 3: campaign artifacts, tables, CLI --------------------------------

_ATTACKS = ("Blackhole Attack", "Wormhole Attack", "Sybil Attack", "Flooding Attack")


def _synthetic_window_df(n_windows, w, seed):
    """Synthetic window_df with known separation: attack-majority windows
    score higher (base=1.0 + noise) than normal-majority windows (base=0.0
    + noise) on every manifold, independently per manifold."""
    rng = np.random.default_rng(seed)
    rows = []
    attack_counter = 0
    for i in range(n_windows):
        is_attack = i % 2 == 1
        if is_attack:
            majority_label = _ATTACKS[attack_counter % len(_ATTACKS)]
            attack_counter += 1
            attack_frac = float(rng.uniform(0.55, 1.0))
        else:
            majority_label = "Normal Traffic"
            attack_frac = 0.0 if i % 4 == 0 else float(rng.uniform(0.0, 0.45))
        row = {
            "window_idx": i, "start_pos": i * w,
            "majority_label": majority_label, "attack_frac": attack_frac,
        }
        base = 1.0 if is_attack else 0.0
        for m in config.MANIFOLDS:
            row[f"W2_{m}"] = base + 0.1 * rng.normal()
        rows.append(row)
    df = pd.DataFrame(rows)
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)
    # Identity "znorm" (already-centered synthetic scores) keeps the fixture simple.
    for m in config.MANIFOLDS:
        df[f"Z2_{m}"] = df[f"W2_{m}"]
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        df[f"Z2_{subset}"] = sum(df[f"Z2_{m}"] for m in manifolds)
    return df


def _write_synthetic_run(ws, w, arm, k, n_windows=40, seed=0,
                          total_s=10.0, baseline_s=2.0, n_windows_timing=None):
    df = _synthetic_window_df(n_windows, w, seed=seed)
    stats = {m: (0.0, 1.0) for m in config.MANIFOLDS}
    n_windows_timing = n_windows_timing or n_windows
    timing = {
        "total_s": total_s, "n_windows": n_windows_timing,
        "per_window_s": total_s / n_windows_timing,
        "rips_s": total_s * 0.5, "w2_s": total_s * 0.3, "baseline_s": baseline_s,
    }
    windowed.write_run_artifacts(ws, w, arm, k, df, stats, timing)
    return df, stats, timing


def test_missing_runs_full_grid_when_empty(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    missing = windowed.missing_runs(ws)
    assert len(missing) == 80
    assert (25, "ordered", 0) in missing
    assert (200, "shuffled", 9) in missing


def test_missing_runs_shrinks_with_fakes_present(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    _write_synthetic_run(ws, 25, "ordered", 0)
    _write_synthetic_run(ws, 25, "shuffled", 3)
    missing = windowed.missing_runs(ws)
    assert len(missing) == 78
    assert (25, "ordered", 0) not in missing
    assert (25, "shuffled", 3) not in missing


def test_write_load_run_roundtrip(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    df, stats, timing = _write_synthetic_run(ws, 50, "ordered", 1, n_windows=20, seed=1)
    loaded_df, loaded_stats, loaded_timing = windowed.load_run(ws, 50, "ordered", 1)

    pd.testing.assert_frame_equal(df, loaded_df)
    assert loaded_stats == stats
    assert loaded_timing == timing

    run_dir = ws.tables_dir / "rebuild" / "windowed"
    run_csv = run_dir / "run_w50_ordered1.csv"
    meta_json = run_dir / "run_w50_ordered1_meta.json"
    assert run_csv.exists() and meta_json.exists()
    assert run_csv.with_name(run_csv.name + ".provenance.json").exists()
    assert meta_json.with_name(meta_json.name + ".provenance.json").exists()


def test_build_detection_table_shape_and_ci(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    _write_synthetic_run(ws, 25, "ordered", 0, n_windows=40, seed=0)
    _write_synthetic_run(ws, 25, "ordered", 1, n_windows=40, seed=1)

    table = windowed.build_detection_table(ws, B=200, bootstrap_seed=0)

    assert len(table) == 14  # 7 subsets x 2 scorings, one (w, arm) group
    assert set(table["subset"]) == set(MANIFOLD_SUBSETS)
    assert set(table["scoring"]) == {"raw", "znorm"}
    assert (table["n_runs"] == 2).all()
    assert (table["ci_lo"] <= table["mean"]).all()
    assert (table["mean"] <= table["ci_hi"]).all()


def test_build_windowed_attribution_dominant_unique(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    _write_synthetic_run(ws, 25, "ordered", 0, n_windows=80, seed=2)
    _write_synthetic_run(ws, 25, "ordered", 1, n_windows=80, seed=3)
    # A shuffled-arm run must be excluded from attribution entirely.
    _write_synthetic_run(ws, 25, "shuffled", 0, n_windows=80, seed=4)

    table = windowed.build_windowed_attribution_table(ws, B=200, bootstrap_seed=0)

    assert len(table) == len(_ATTACKS) * len(config.MANIFOLDS)
    for (_w, _attack), sub in table.groupby(["w", "attack_class"]):
        assert sub["dominant"].sum() == 1


def test_contamination_bins_partition_and_bin0_pure_normal(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    df, _stats, _timing = _write_synthetic_run(ws, 25, "ordered", 0, n_windows=80, seed=4)

    table = windowed.build_contamination_table(ws)
    agg = table[table["majority_class"] == "all"]
    w25 = agg[agg["w"] == 25]

    assert w25["n_windows"].sum() == 80

    zero_windows = df[df["attack_frac"] == 0.0]
    assert (zero_windows["majority_label"] == "Normal Traffic").all()
    bin0 = w25[w25["bin"] == "0"].iloc[0]
    assert bin0["n_windows"] == len(zero_windows)


def test_build_frontier_table_five_rows_and_marginal_cost(tmp_path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    for w in config.WINDOW_SIZES:
        _write_synthetic_run(ws, w, "ordered", 0, n_windows=30, seed=w,
                              total_s=100.0, baseline_s=10.0, n_windows_timing=30)

    rebuild = ws.tables_dir / "rebuild"
    rebuild.mkdir(parents=True, exist_ok=True)
    binary_auc_csv = rebuild / "binary_auc.csv"
    pd.DataFrame([{
        "subset": "all_three", "scoring": "znorm",
        "mean": 0.85, "std": 0.02, "ci_lo": 0.80, "ci_hi": 0.90, "n_seeds": 10,
    }]).to_csv(binary_auc_csv, index=False)

    table = windowed.build_frontier_table(
        ws, B=200, bootstrap_seed=0, binary_auc_csv=binary_auc_csv,
        time_probe_fn=lambda: 0.0123,
    )

    assert len(table) == 5
    assert set(table["row"]) == {"25", "50", "100", "200", "per_flow"}

    windowed_rows = table[table["row"] != "per_flow"]
    expected_marginal = (100.0 - 10.0) / 30
    assert (windowed_rows["marginal_s"].apply(lambda x: x == pytest.approx(expected_marginal))).all()
    assert np.allclose(windowed_rows["baseline_s"].to_numpy(), 10.0)

    pf_row = table[table["row"] == "per_flow"].iloc[0]
    assert pf_row["marginal_s"] == pytest.approx(0.0123)
    assert pf_row["auc_mean"] == pytest.approx(0.85)
    assert np.isnan(pf_row["baseline_s"])


def test_windowed_cli_registration_and_flags():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    assert "windowed" in sub.choices
    assert "windowed-report" in sub.choices

    p = sub.choices["windowed"]
    opts = {s for a in p._actions for s in a.option_strings}
    assert {"--w", "--order-seed", "--repeat"} <= opts

    pr = sub.choices["windowed-report"]
    opts_r = {s for a in pr._actions for s in a.option_strings}
    assert "--bootstrap" in opts_r

    args = parser.parse_args(["windowed", "--w", "25"])
    assert args.func == cli._cmd_windowed
    assert args.order_seed is None
    assert args.repeat == 0

    args2 = parser.parse_args(["windowed-report"])
    assert args2.func == cli._cmd_windowed_report
    assert args2.bootstrap == 2000


def test_windowed_report_cmd_runs_without_real_campaign(tmp_path, monkeypatch):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    for w in config.WINDOW_SIZES:
        _write_synthetic_run(ws, w, "ordered", 0, n_windows=20, seed=w)

    rebuild = ws.tables_dir / "rebuild"
    pd.DataFrame([{
        "subset": "all_three", "scoring": "znorm",
        "mean": 0.8, "std": 0.01, "ci_lo": 0.7, "ci_hi": 0.9, "n_seeds": 10,
    }]).to_csv(rebuild / "binary_auc.csv", index=False)

    monkeypatch.setattr(windowed, "run_missing_campaign", lambda *a, **k: None)
    monkeypatch.setattr(windowed, "_time_per_flow_probe", lambda *a, **k: 0.01)

    args = cli.build_parser().parse_args(
        ["windowed-report", "--root", str(tmp_path), "--bootstrap", "50"])
    rc = cli._cmd_windowed_report(args)
    assert rc == 0

    d = rebuild / "windowed"
    for name in ("windowed_detection.csv", "windowed_attribution.csv",
                 "contamination_curve.csv", "compute_frontier.csv"):
        assert (d / name).exists()
        assert (d / (name + ".provenance.json")).exists()
