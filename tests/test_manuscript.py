import json

import numpy as np
import pandas as pd
import pytest

from uav_tda import cli, config, manuscript


def test_manuscript_seeds_constant():
    assert config.MANUSCRIPT_SEEDS == (42, 7, 123, 0, 1, 2, 3, 4, 5, 6)
    assert config.MANUSCRIPT_SEEDS[:3] == config.PROBE_SEEDS


def test_stratified_resample_preserves_class_counts():
    labels = np.array(["A"] * 10 + ["B"] * 5)
    rng = np.random.default_rng(0)
    idx = manuscript.stratified_resample_indices(labels, rng)
    assert len(idx) == 15
    resampled = labels[idx]
    assert (resampled == "A").sum() == 10
    assert (resampled == "B").sum() == 5


def test_bootstrap_ci_contains_truth_and_is_deterministic():
    rng = np.random.default_rng(1)
    per_seed = []
    for _ in range(3):
        y = np.array([0] * 200 + [1] * 200)
        scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.2, 1, 200)])
        per_seed.append((y, scores))
    lo, hi = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=0)
    assert 0.5 < lo < hi < 1.0
    # true AUC for N(0,1) vs N(1.2,1) is Phi(1.2/sqrt(2)) ~= 0.802
    assert lo < 0.802 < hi
    lo2, hi2 = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=0)
    assert (lo, hi) == (lo2, hi2)
    lo3, hi3 = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=1)
    assert (lo, hi) != (lo3, hi3)


def test_bootstrap_never_drops_a_class():
    # tiny minority class: unstratified resampling would frequently lose it
    y = np.array([0] * 98 + [1] * 2)
    scores = np.arange(100, dtype=float)
    lo, hi = manuscript.bootstrap_mean_auc_ci([(y, scores)], B=200, bootstrap_seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)  # no ValueError from single-class replicate


# --- Ten-seed campaign orchestration -----------------------------------------

ATTACKS = ("Blackhole Attack", "Wormhole Attack", "Sybil Attack", "Flooding Attack")
CLASSES = ("Normal Traffic",) + ATTACKS
MANIFOLDS = ("c2", "network", "physical")
SUBSETS = {
    "c2_only": ("c2",),
    "network_only": ("network",),
    "physical_only": ("physical",),
    "c2_network": ("c2", "network"),
    "c2_physical": ("c2", "physical"),
    "network_physical": ("network", "physical"),
    "all_three": ("c2", "network", "physical"),
}


def _make_seed_csv(path, seed, n_per_class=10):
    """Tiny synthetic probe-distances CSV: attacks score higher than Normal
    on every manifold (known separation), with per-seed jitter so seeds differ."""
    rng = np.random.default_rng(seed)
    rows = []
    for idx, cls in enumerate(CLASSES):
        base = 0.0 if cls == "Normal Traffic" else 1.0 + idx  # widen separation per attack
        for i in range(n_per_class):
            row = {"test_idx": len(rows), "label": cls}
            for m in MANIFOLDS:
                row[f"W2_{m}"] = base + 0.1 * rng.normal()
            rows.append(row)
    df = pd.DataFrame(rows)
    for subset, manifolds in SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)
    df.to_csv(path, index=False)
    return df


def _make_stats_json(path, seed):
    rng = np.random.default_rng(seed + 1000)
    stats = {m: {"mean": float(rng.normal(0, 0.01)), "std": 1.0} for m in MANIFOLDS}
    payload = {
        "stats": stats,
        "timeout_counts": {f"n_timeouts_{m}": 0 for m in MANIFOLDS},
        "params": {"seed": seed, "per_class": 10, "top_k": 50, "delta": 0.2, "w2_timeout": 30.0},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return stats


def _write_two_seed_fixture(tmp_path):
    rebuild = tmp_path / "rebuild"
    rebuild.mkdir(parents=True)
    seeds = (11, 22)
    for seed in seeds:
        _make_seed_csv(rebuild / f"probe_distances_seed{seed}.csv", seed)
        _make_stats_json(rebuild / f"znorm_stats_seed{seed}.json", seed)
    return rebuild, seeds


def test_missing_seeds_detects_absent_artifacts(tmp_path):
    rebuild, seeds = _write_two_seed_fixture(tmp_path)
    assert manuscript.missing_seeds(seeds, rebuild) == []
    assert manuscript.missing_seeds(seeds + (99,), rebuild) == [99]

    # Remove one artifact from an otherwise-complete seed -> still missing.
    (rebuild / f"znorm_stats_seed{seeds[0]}.json").unlink()
    assert manuscript.missing_seeds(seeds, rebuild) == [seeds[0]]


def test_load_seed_frames_roundtrip(tmp_path):
    rebuild, seeds = _write_two_seed_fixture(tmp_path)
    seed_frames = manuscript.load_seed_frames(seeds, rebuild)
    assert set(seed_frames) == set(seeds)
    for seed in seeds:
        df, stats = seed_frames[seed]
        assert isinstance(df, pd.DataFrame)
        assert set(df["label"].unique()) == set(CLASSES)
        assert set(stats) == set(MANIFOLDS)
        for m in MANIFOLDS:
            mean, std = stats[m]
            assert isinstance(mean, float) and isinstance(std, float)


def test_build_binary_auc_table_shape_and_ci_sanity(tmp_path):
    rebuild, seeds = _write_two_seed_fixture(tmp_path)
    seed_frames = manuscript.load_seed_frames(seeds, rebuild)
    table = manuscript.build_binary_auc_table(seed_frames, B=200, bootstrap_seed=0)

    assert len(table) == 14  # 7 subsets x 2 scorings
    assert set(table["subset"]) == set(SUBSETS)
    assert set(table["scoring"]) == {"raw", "znorm"}
    assert list(table.columns) == ["subset", "scoring", "mean", "std", "ci_lo", "ci_hi", "n_seeds"]
    assert (table["n_seeds"] == 2).all()
    assert (table["ci_lo"] <= table["mean"]).all()
    assert (table["mean"] <= table["ci_hi"]).all()
    # Strong synthetic separation -> AUC should be high for every subset/scoring.
    assert (table["mean"] > 0.9).all()


def test_build_attribution_table_shape_and_dominance(tmp_path):
    rebuild, seeds = _write_two_seed_fixture(tmp_path)
    seed_frames = manuscript.load_seed_frames(seeds, rebuild)
    table = manuscript.build_attribution_table(seed_frames, B=200, bootstrap_seed=0)

    assert len(table) == 12  # 4 attacks x 3 manifolds
    assert set(table["attack_class"]) == set(ATTACKS)
    assert set(table["manifold"]) == set(MANIFOLDS)
    assert list(table.columns) == [
        "attack_class", "manifold", "mean", "std", "ci_lo", "ci_hi", "dominant"]
    for attack in ATTACKS:
        sub = table[table["attack_class"] == attack]
        assert sub["dominant"].sum() == 1
        assert sub.loc[sub["dominant"], "mean"].iloc[0] == sub["mean"].max()


def test_manuscript_report_subcommand_registered_with_flags():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    assert "manuscript-report" in sub.choices
    p = sub.choices["manuscript-report"]
    opts = {s for a in p._actions for s in a.option_strings}
    assert {"--seeds", "--bootstrap", "--w2-timeout"} <= opts


def test_manuscript_report_defaults():
    args = cli.build_parser().parse_args(["manuscript-report"])
    assert args.seeds == ",".join(str(s) for s in config.MANUSCRIPT_SEEDS)
    assert args.bootstrap == 2000
    assert args.w2_timeout == 30.0
    assert args.func == cli._cmd_manuscript_report


def test_manuscript_report_cmd_runs_without_real_probe(tmp_path, monkeypatch):
    rebuild, seeds = _write_two_seed_fixture(tmp_path)
    monkeypatch.setattr(cli, "TABLES_DIR", tmp_path)
    monkeypatch.setattr(manuscript, "run_missing_seeds", lambda *a, **k: None)

    args = cli.build_parser().parse_args([
        "manuscript-report", "--seeds", "11,22", "--bootstrap", "50"])
    rc = cli._cmd_manuscript_report(args)
    assert rc == 0
    assert (rebuild / "binary_auc.csv").exists()
    assert (rebuild / "manifold_attribution.csv").exists()
