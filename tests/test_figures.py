"""Phase-8 Task 2: manuscript figure generators (synthetic frames + committed CSVs only)."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from uav_tda import figures
from uav_tda.metrics import ATTACK_CLASSES, NORMAL
from uav_tda.paths import TABLES_DIR

REBUILD = TABLES_DIR / "rebuild"

_SINGLE_PT = 3.5 * 72
_DOUBLE_PT = 7.16 * 72


def _sidecar(path):
    return json.loads(path.with_name(path.name + ".provenance.json").read_text())


def _media_width_pt(path):
    m = re.search(rb"/MediaBox \[\s*0 0 ([0-9.]+) ([0-9.]+)\s*\]", path.read_bytes())
    assert m, "no MediaBox in PDF"
    return float(m.group(1))


def _assert_pdf(path, width_pt):
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes().startswith(b"%PDF")
    assert abs(_media_width_pt(path) - width_pt) < 0.5
    data = path.read_bytes()
    assert b"/Type3" not in data, "fonts must be embedded as TrueType (pdf.fonttype 42)"
    assert b"CreationDate" not in data
    side = _sidecar(path)
    assert side["artifact"] == path.name
    assert "params" in side


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------


def _frontier_df():
    return pd.DataFrame({
        "row": ["per_flow", "25", "50", "100", "200"],
        "w": [np.nan, 25.0, 50.0, 100.0, 200.0],
        "auc_mean": [0.89, 0.87, 0.90, 0.94, 0.91],
        "auc_ci_lo": [0.88, 0.86, 0.89, 0.93, 0.89],
        "auc_ci_hi": [0.90, 0.88, 0.91, 0.95, 0.93],
        "cost_s": [3.5, 0.0075, 0.027, 0.12, 0.60],
        "cost_p95_s": [3.9, 0.0096, 0.039, 0.21, 1.30],
        "campaign_marginal_s": [np.nan, 0.009, 0.03, 0.14, 0.68],
        "w2_only_per_flow_s": [0.06, np.nan, np.nan, np.nan, np.nan],
        "baseline_setup_s": [np.nan, 78.0, 64.0, 57.0, 49.0],
        "flows_per_decision": [1, 25, 50, 100, 200],
        "cost_per_flow_s": [3.5, 0.0003, 0.0005, 0.0012, 0.003],
        "speedup_vs_per_flow": [1.0, 466.0, 130.0, 29.0, 5.8],
        "cost_n": [30] * 5,
    })


def _detection_df():
    rows = []
    for w in (25, 50, 100, 200):
        for arm, base in (("ordered", 0.88), ("shuffled", 0.52)):
            for scoring in ("raw", "znorm"):
                rows.append({"w": w, "arm": arm, "subset": "all_three", "scoring": scoring,
                             "mean": base, "std": 0.01, "ci_lo": base - 0.01,
                             "ci_hi": base + 0.01, "n_runs": 10})
            rows.append({"w": w, "arm": arm, "subset": "c2_only", "scoring": "znorm",
                         "mean": 0.6, "std": 0.0, "ci_lo": 0.59, "ci_hi": 0.61, "n_runs": 10})
    return pd.DataFrame(rows)


_BINS = (("0", 0.0, 0.0), ("(0.00,0.25]", 0.0, 0.25), ("(0.25,0.50]", 0.25, 0.5),
         ("(0.50,0.75]", 0.5, 0.75), ("(0.75,1.00]", 0.75, 1.0))


def _contamination_df():
    rows = []
    for w in (25, 50, 100, 200):
        for i, (b, lo, hi) in enumerate(_BINS):
            rows.append({"w": w, "arm": "ordered", "bin": b, "bin_lo": lo, "bin_hi": hi,
                         "majority_class": "all", "n_windows": 10 * (i + 1) * 7,
                         "mean_raw_all_three": 1.0, "mean_znorm_all_three": 0.5,
                         "threshold": 1.5, "detection_rate": 0.1 + 0.2 * i})
            rows.append({"w": w, "arm": "ordered", "bin": b, "bin_lo": lo, "bin_hi": hi,
                         "majority_class": "Normal Traffic", "n_windows": 999,
                         "mean_raw_all_three": 1.0, "mean_znorm_all_three": 0.5,
                         "threshold": 1.5, "detection_rate": 0.99})
            rows.append({"w": w, "arm": "shuffled", "bin": b, "bin_lo": lo, "bin_hi": hi,
                         "majority_class": "all", "n_windows": 3,
                         "mean_raw_all_three": 1.0, "mean_znorm_all_three": 0.5,
                         "threshold": np.nan, "detection_rate": np.nan})
    return pd.DataFrame(rows)


def _latency_df():
    rows = []
    for m, rips in (("c2", 3.0), ("network", 0.05), ("physical", 0.4)):
        for stage, v in (("scaler_s", 0.001), ("rips_s", rips), ("slice_s", 0.0002),
                         ("w2_s", 0.02)):
            rows.append(("per_flow", np.nan, f"{m}_{stage}", v))
        rows.append(("per_flow", np.nan, f"{m}_total_s", rips + 0.0212))
    rows.append(("per_flow", np.nan, "total_s", 3.52))
    for w, t in ((25, 0.0075), (50, 0.027), (100, 0.12), (200, 0.6)):
        rows.append(("windowed", float(w), "baseline_setup_s", 60.0))
        rows.append(("windowed", float(w), "c2_total_s", t / 3))
        rows.append(("windowed", float(w), "total_s", t))
    df = pd.DataFrame(rows, columns=["mode", "w", "stage", "mean_s"])
    df["median_s"] = df["mean_s"]
    df["p95_s"] = df["mean_s"] * 1.3
    df["n"] = 30
    return df


def _exact_per_attack_df():
    rng = np.random.default_rng(0)
    rows = []
    for a in ATTACK_CLASSES:
        aucs = rng.uniform(0.2, 0.9, 3)
        for m, auc in zip(("c2", "network", "physical"), aucs):
            rows.append({"attack_class": a, "manifold": m, "auc": auc, "ci_lo": auc - 0.01,
                         "ci_hi": auc + 0.01, "dominant": bool(auc == aucs.max())})
    return pd.DataFrame(rows)


def _windowed_attribution_df():
    rng = np.random.default_rng(1)
    rows = []
    for w in (25, 50, 100, 200):
        for a in ATTACK_CLASSES:
            aucs = rng.uniform(0.2, 0.95, 3)
            for m, auc in zip(("c2", "network", "physical"), aucs):
                rows.append({"w": w, "attack_class": a, "manifold": m, "mean": auc,
                             "std": 0.0, "ci_lo": auc - 0.01, "ci_hi": auc + 0.01,
                             "dominant": bool(auc == aucs.max()), "n_runs": 10})
    return pd.DataFrame(rows)


def _roc_frames():
    rng = np.random.default_rng(2)

    def frame(n_normal, n_attack):
        labels = [NORMAL] * n_normal + [ATTACK_CLASSES[i % 4] for i in range(n_attack)]
        shift = np.r_[np.zeros(n_normal), np.ones(n_attack)]
        df = pd.DataFrame({"label": labels})
        for j, m in enumerate(("c2", "network", "physical")):
            df[f"W2_{m}"] = np.abs(rng.normal(0.1 * (j + 1), 0.05, len(df)) + 0.05 * shift)
        return df

    val = frame(200, 0)
    test = frame(300, 300)
    op = pd.DataFrame([{"subset": "all_three", "scoring": "znorm", "tau": 1.0,
                        "val_fpr": 0.05, "fpr": 0.06, "tpr": 0.7},
                       {"subset": "all_three", "scoring": "raw", "tau": 0.5,
                        "val_fpr": 0.05, "fpr": 0.2, "tpr": 0.2}])
    return val, test, op


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def test_fixed_colour_maps_are_okabe_ito():
    okabe_ito = {"#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
                 "#D55E00", "#CC79A7"}
    assert set(figures.MANIFOLD_COLORS) == {"c2", "network", "physical"}
    assert {"ordered", "shuffled", "per_flow"} <= set(figures.ARM_COLORS)
    for c in list(figures.MANIFOLD_COLORS.values()) + [
            figures.ARM_COLORS[k] for k in ("ordered", "shuffled", "per_flow")]:
        assert c.upper() in okabe_ito
    assert len(set(figures.MANIFOLD_COLORS.values())) == 3
    assert len({figures.ARM_COLORS[k] for k in ("ordered", "shuffled", "per_flow")}) == 3


# ---------------------------------------------------------------------------
# Each figure on synthetic frames
# ---------------------------------------------------------------------------


def test_fig_frontier_synthetic(tmp_path):
    out = figures.fig_frontier(_frontier_df(), 0.908, tmp_path / "fig_frontier.pdf")
    assert out == tmp_path / "fig_frontier.pdf"
    _assert_pdf(out, _SINGLE_PT)
    assert _sidecar(out)["params"]["exact_ref_auc"] == 0.908


def test_fig_frontier_is_byte_stable(tmp_path):
    a = figures.fig_frontier(_frontier_df(), 0.908, tmp_path / "a.pdf")
    b = figures.fig_frontier(_frontier_df(), 0.908, tmp_path / "b.pdf")
    assert a.read_bytes() == b.read_bytes()


def test_fig_detection_vs_w_synthetic(tmp_path):
    out = figures.fig_detection_vs_w(_detection_df(), 0.893, 0.908, tmp_path / "det.pdf")
    _assert_pdf(out, _SINGLE_PT)
    params = _sidecar(out)["params"]
    assert params["per_flow_auc"] == 0.893 and params["exact_ref_auc"] == 0.908


def test_fig_contamination_synthetic_unique_n(tmp_path):
    out = figures.fig_contamination(_contamination_df(), tmp_path / "contam.pdf")
    _assert_pdf(out, _DOUBLE_PT)
    unique_n = _sidecar(out)["params"]["unique_n"]
    # ordered arm, majority_class == "all": pooled n_windows / 10 repeats.
    assert unique_n["25"] == [7, 14, 21, 28, 35]
    assert set(unique_n) == {"25", "50", "100", "200"}


def test_fig_contamination_rejects_non_divisible_pooled_counts(tmp_path):
    df = _contamination_df()
    sel = (df["arm"] == "ordered") & (df["majority_class"] == "all")
    df.loc[df[sel].index[0], "n_windows"] = 71
    with pytest.raises(ValueError, match="not divisible"):
        figures.fig_contamination(df, tmp_path / "contam.pdf")


def test_fig_latency_synthetic(tmp_path):
    out = figures.fig_latency(_latency_df(), tmp_path / "lat.pdf")
    _assert_pdf(out, _SINGLE_PT)
    assert _sidecar(out)["params"]["per_flow_total_s"] == 3.52


def test_fig_attribution_synthetic(tmp_path):
    out = figures.fig_attribution(_exact_per_attack_df(), _windowed_attribution_df(),
                                  tmp_path / "attr.pdf")
    _assert_pdf(out, _DOUBLE_PT)
    params = _sidecar(out)["params"]
    assert params["columns"][:3] == ["per_flow:c2", "per_flow:network", "per_flow:physical"]
    assert len(params["columns"]) == 15


def test_fig_roc_synthetic(tmp_path):
    val, test, op = _roc_frames()
    out = figures.fig_roc(val, test, op, tmp_path / "roc.pdf")
    _assert_pdf(out, _SINGLE_PT)
    params = _sidecar(out)["params"]
    assert set(params["auc"]) == {"all_three", "network_physical", "c2_only",
                                  "network_only", "physical_only"}
    assert params["operating_point"] == {"fpr": 0.06, "tpr": 0.7}


# ---------------------------------------------------------------------------
# Committed CSVs (CI-safe; never outputs/)
# ---------------------------------------------------------------------------


def test_fig_roc_legend_auc_matches_exact_binary_auc(tmp_path):
    from uav_tda import artifacts
    from uav_tda.exact import assemble_distances

    exact_dir = REBUILD / "exact"
    val = assemble_distances(exact_dir, "val")
    test = assemble_distances(exact_dir, "test")
    op = artifacts.operating_point_table()
    out = figures.fig_roc(val, test, op, tmp_path / "roc.pdf")
    params = _sidecar(out)["params"]

    ref = pd.read_csv(exact_dir / "exact_binary_auc.csv")
    ref = ref[ref["scoring"] == "znorm"].set_index("subset")["auc"]
    for subset, auc in params["auc"].items():
        assert round(auc, 4) == round(float(ref[subset]), 4), subset
        assert f"{auc:.3f}" in params["legend"][subset]


def test_build_all_figures_on_committed_rebuild(tmp_path):
    paths = figures.build_all_figures(REBUILD, tmp_path / "manuscript")
    assert set(paths) == {"frontier", "detection_vs_w", "contamination", "latency",
                          "attribution", "roc"}
    for name, p in paths.items():
        assert p.parent == tmp_path / "manuscript"
        width = _DOUBLE_PT if name in ("contamination", "attribution") else _SINGLE_PT
        _assert_pdf(p, width)
    assert _sidecar(paths["frontier"])["params"]["exact_ref_auc"] == pytest.approx(
        0.908231183138054)


# ---------------------------------------------------------------------------
# Fix round 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["two", "zero"])
def test_fig_attribution_requires_exactly_one_dominant_per_block(tmp_path, which):
    exact = _exact_per_attack_df()
    rows = exact.index[exact["attack_class"] == ATTACK_CLASSES[0]]
    if which == "two":
        exact.loc[rows, "dominant"] = [True, True, False]
    else:
        exact.loc[rows, "dominant"] = False
    with pytest.raises(ValueError, match="exactly one dominant"):
        figures.fig_attribution(exact, _windowed_attribution_df(), tmp_path / "attr.pdf")


def test_fig_roc_legend_is_outside_axes(tmp_path):
    val, test, op = _roc_frames()
    out = figures.fig_roc(val, test, op, tmp_path / "roc.pdf")
    assert _sidecar(out)["params"]["legend_placement"] == "below axes"


def test_fig_latency_legend_names_tiny_stages_with_magnitude(tmp_path):
    out = figures.fig_latency(_latency_df(), tmp_path / "lat.pdf")
    labels = _sidecar(out)["params"]["legend_labels"]
    assert "scaler" not in labels and "slice" not in labels
    merged = [lab for lab in labels if lab.startswith("scaler + slice")]
    # synthetic: scaler 1 ms + slice 0.2 ms per manifold
    assert merged == ["scaler + slice (≤1.2 ms)"]


def test_fig_latency_tiny_stage_bound_rounds_up(tmp_path):
    lat = _latency_df()
    lat.loc[lat["stage"].str.endswith("_scaler_s"), "mean_s"] = 0.00084  # 0.84+0.2 = 1.04 ms
    out = figures.fig_latency(lat, tmp_path / "lat.pdf")
    labels = _sidecar(out)["params"]["legend_labels"]
    assert "scaler + slice (≤1.1 ms)" in labels


def test_save_closes_figure_when_savefig_fails(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    num = fig.number

    def boom(*a, **k):
        raise RuntimeError("savefig failed")

    monkeypatch.setattr(fig, "savefig", boom)
    with pytest.raises(RuntimeError):
        figures._save(fig, tmp_path / "x.pdf", {})
    assert not plt.fignum_exists(num)


# ---------------------------------------------------------------------------
# Final fix wave: derived run counts
# ---------------------------------------------------------------------------


def test_fig_frontier_and_detection_label_derive_seed_count(tmp_path):
    out = figures.fig_frontier(_frontier_df(), 0.908, tmp_path / "f.pdf", per_flow_n_runs=7)
    assert _sidecar(out)["params"]["per_flow_label"] == "per-flow (7-seed probe)"
    out = figures.fig_detection_vs_w(_detection_df(), 0.893, 0.908, tmp_path / "d.pdf",
                                     per_flow_n_runs=7)
    assert _sidecar(out)["params"]["per_flow_label"] == "per-flow (7-seed probe)"


def test_fig_attribution_windowed_value_derives_run_count(tmp_path):
    wa = _windowed_attribution_df()
    wa["n_runs"] = 4
    out = figures.fig_attribution(_exact_per_attack_df(), wa, tmp_path / "a.pdf")
    assert _sidecar(out)["params"]["windowed_value"] == "mean (4 runs)"


def test_fig_contamination_accepts_per_w_repeats(tmp_path):
    df = _contamination_df()
    sel = (df["arm"] == "ordered") & (df["majority_class"] == "all") & (df["w"] == 25)
    df.loc[sel, "n_windows"] = df.loc[sel, "n_windows"] // 10 * 5
    reps = {25: 5, 50: 10, 100: 10, 200: 10}
    out = figures.fig_contamination(df, tmp_path / "c.pdf", n_repeats=reps)
    params = _sidecar(out)["params"]
    assert params["unique_n"]["25"] == [7, 14, 21, 28, 35]
    assert params["n_repeats"] == {"25": 5, "50": 10, "100": 10, "200": 10}


def test_build_all_figures_passes_derived_counts(tmp_path, monkeypatch):
    from uav_tda import artifacts

    seen = {}
    real = figures.fig_contamination

    def spy(*a, **kw):
        seen["n_repeats"] = kw.get("n_repeats")
        return real(*a, **kw)

    monkeypatch.setattr(figures, "fig_contamination", spy)
    paths = figures.build_all_figures(REBUILD, tmp_path / "m")
    uw = artifacts.unique_window_counts(REBUILD)
    assert seen["n_repeats"] == int(uw["n_repeats"].iloc[0])
    cf = pd.read_csv(REBUILD / "windowed" / "compute_frontier.csv", dtype={"row": str})
    n = int(cf.loc[cf["row"] == "per_flow", "n_runs"].iloc[0])
    for name in ("frontier", "detection_vs_w"):
        assert _sidecar(paths[name])["params"]["per_flow_label"] == f"per-flow ({n}-seed probe)"
