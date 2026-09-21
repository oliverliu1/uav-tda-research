"""Equivalence tests for uav_tda.unsupervised vs pipeline.py SECTION 10.

WHY NOT a plain staged-input CSV diff at 1e-9 (the Task-6 pattern):
cmd_unsupervised RECOMPUTES the baseline barcodes via sparse Rips
(SPARSE_RIPS_EPSILON = 0.5 for c2/network) inside the phase — they are not a
staged artifact — and gudhi sparse Rips at 500-point scale is nondeterministic
EVEN BETWEEN CONSECUTIVE CALLS IN ONE PROCESS (observed 2026-09-21: two
identical in-process compute_baseline_barcodes calls gave c2 H0 max coordinate
delta 4.2e-3 and c2 H2 bar counts 37 vs 39; the Task-4 in-process probe's
50-point cloud is stable, but that stability does not extrapolate). The
monolith phase itself is therefore not run-to-run reproducible for c2/network,
and everything downstream of those baselines cannot be bit-reproduced by
construction. Observed oracle-vs-package deltas (debug, staged-identical flow
diagrams): c2_distance max_abs_delta=3.56e-2, network_distance
max_abs_delta=1.99e-4, physical_distance max_abs_delta=0.0 (bit-exact —
physical uses exact Rips).

Test structure (no tolerance loosening — 1e-9 kept wherever determinism holds):
  A. In-process monolith-vs-package equivalence of compute_baseline_barcodes
     under SPARSE_RIPS_EPSILON=None on both sides (deterministic exact Rips,
     120-point reference subset for tractability) — exact. The sparse kwarg
     branch is two verbatim lines; its runtime behaviour is pinned by B1's
     full-phase physical exactness.
  A2. In-process monolith-vs-package equivalence of _wasserstein_for_flow on
     sampled real flows against a SHARED baseline dict — exact (hera W2 is
     deterministic given identical inputs).
  B. Full package run, then:
     B1. physical (exact Rips, cross-process deterministic): distance column,
         threshold, and per-class-AUC rows vs the oracle at 1e-9.
     B2. In-process downstream replay: feed OUR distances through the ORACLE
         monolith's compute_thresholds / flag_distances / derive_inference_rule
         / apply_inference_rule / fraction_unmapped / compute_per_class_auc /
         compute_overall_metrics and require they reproduce OUR artifacts at
         1e-9 — proving every downstream ported function equivalent on real data.
     B3. The three figures exist and are nonempty.
"""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from tests import monolith_harness as mh
from uav_tda import config
from uav_tda.workspace import Workspace

SPLITS = ("train", "val", "test")


def _import_monolith(oracle_ws):
    """Import the oracle workspace's pipeline.py as a module (its REPO_ROOT
    resolves to the oracle workspace, so its loaders read the oracle's own
    outputs/ — the byte-identical source of our staged copies)."""
    spec = importlib.util.spec_from_file_location(
        "monolith_oracle_pipeline", oracle_ws / "pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sorted_rows(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    return arr[np.lexsort(arr.T[::-1])]


def _distances_from_frame(df, split):
    sub = df[df["split"] == split]
    return {m: sub[f"{m}_distance"].to_numpy() for m in config.MANIFOLDS}


@pytest.mark.slow
def test_unsupervised_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "unsupervised")
    mono = _import_monolith(oracle)
    from uav_tda import unsupervised as up
    from uav_tda.tda import load_labels_for_split

    ws = Workspace.at(tmp_path)
    ws.ensure()
    mh.stage_oracle(oracle, ws, include=("outputs",))

    max_edge = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    ref_idx = np.load(ws.outputs_dir / "reference_indices.npy")

    # ---- A: in-process baseline-barcode port equivalence (exact, sparse=None) ----
    # Sparse Rips is nondeterministic per-call (see module docstring), so exact
    # function-level equivalence is proven under exact Rips on a reference
    # subset; both sides are patched identically.
    none_sparse = {m: None for m in config.MANIFOLDS}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "SPARSE_RIPS_EPSILON", none_sparse)
        mp.setattr(mono, "SPARSE_RIPS_EPSILON", none_sparse)
        sub_idx = ref_idx[:120]
        base_pkg = up.compute_baseline_barcodes(ws, max_edge, sub_idx)
        base_mono = mono.compute_baseline_barcodes(max_edge, sub_idx)
        for m in config.MANIFOLDS:
            assert base_pkg[m].keys() == base_mono[m].keys(), m
            for k in base_pkg[m]:
                assert np.array_equal(
                    _sorted_rows(base_pkg[m][k]), _sorted_rows(base_mono[m][k])
                ), f"baseline barcode mismatch: {m} H{k}"

    # ---- A2: _wasserstein_for_flow port equivalence on sampled flows (exact) ----
    # Both sides get the SAME baseline dict, so hera-exact W2 must agree exactly.
    base_shared = up.compute_baseline_barcodes(ws, max_edge, ref_idx)
    rng = np.random.default_rng(0)
    for m in config.MANIFOLDS:
        diagrams = up.load_diagrams_pkl(ws, m, "test")
        for i in rng.choice(len(diagrams), size=20, replace=False):
            d_pkg = up._wasserstein_for_flow(
                diagrams[i], base_shared[m], max_edge[m], config.MAX_HOM_DIM[m])
            d_mono = mono._wasserstein_for_flow(
                diagrams[i], base_shared[m], max_edge[m], config.MAX_HOM_DIM[m])
            assert d_pkg == d_mono, (m, i, d_pkg, d_mono)

    # ---- B: full package phase run ----
    up.run_unsupervised(ws, debug=True)

    oracle_tables = oracle / "results" / "tables"
    ours = pd.read_csv(ws.tables_dir / "unsupervised_distances.csv")
    theirs = pd.read_csv(oracle_tables / "unsupervised_distances.csv")
    assert list(ours.columns) == list(theirs.columns)
    assert len(ours) == len(theirs)
    pd.testing.assert_series_equal(ours["split"], theirs["split"])
    pd.testing.assert_series_equal(ours["label"], theirs["label"])

    # B1: physical is exact-Rips — cross-process deterministic at 1e-9.
    np.testing.assert_allclose(
        ours["physical_distance"], theirs["physical_distance"], rtol=1e-9, atol=1e-12)
    thr_ours = json.loads((ws.outputs_dir / "thresholds.json").read_text())
    thr_oracle = json.loads((oracle / "outputs" / "thresholds.json").read_text())
    assert abs(thr_ours["physical"] - thr_oracle["physical"]) <= 1e-9 * abs(thr_oracle["physical"])
    auc_ours = pd.read_csv(ws.tables_dir / "unsupervised_per_class_auc.csv")
    auc_oracle = pd.read_csv(oracle_tables / "unsupervised_per_class_auc.csv")
    phys_ours = auc_ours[auc_ours["manifold"] == "physical"].reset_index(drop=True)
    phys_oracle = auc_oracle[auc_oracle["manifold"] == "physical"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        phys_ours, phys_oracle, check_exact=False, rtol=1e-9, atol=1e-12)
    # Non-physical AUC rows exist with matching (manifold, attack_class) keys.
    pd.testing.assert_frame_equal(
        auc_ours[["manifold", "attack_class"]], auc_oracle[["manifold", "attack_class"]])

    # B2: in-process downstream replay — OUR distances through the MONOLITH's
    # downstream functions must reproduce OUR artifacts at 1e-9.
    dist_by_split = {s: _distances_from_frame(ours, s) for s in SPLITS}
    labels_by_split = {s: load_labels_for_split(ws, s) for s in SPLITS}

    thr_mono = mono.compute_thresholds(dist_by_split["val"], labels_by_split["val"])
    for m in config.MANIFOLDS:
        assert abs(thr_mono[m] - thr_ours[m]) <= 1e-9 * abs(thr_mono[m]), m

    flags_mono = {s: mono.flag_distances(dist_by_split[s], thr_mono) for s in SPLITS}
    for s in SPLITS:
        sub = ours[ours["split"] == s]
        for m in config.MANIFOLDS:
            assert np.array_equal(
                flags_mono[s][m].astype(int), sub[f"{m}_flag"].to_numpy()), (s, m)

    rule_mono = mono.derive_inference_rule(flags_mono["train"], labels_by_split["train"])
    rule_json = json.loads((ws.outputs_dir / "inference_rule.json").read_text())
    assert {cls: list(pat) for cls, pat in rule_mono.items()} == rule_json
    rule_csv = pd.read_csv(ws.tables_dir / "inference_rule.csv")
    rule_from_csv = {
        r["class"]: (int(r["c2_flag"]), int(r["network_flag"]), int(r["physical_flag"]))
        for _, r in rule_csv.iterrows()
    }
    assert rule_from_csv == rule_mono

    for s in SPLITS:
        pred_mono = mono.apply_inference_rule(rule_mono, flags_mono[s])
        assert pred_mono == ours[ours["split"] == s]["predicted_class"].tolist(), s

    auc_mono = mono.compute_per_class_auc(dist_by_split["test"], labels_by_split["test"])
    pd.testing.assert_frame_equal(
        auc_mono, auc_ours, check_exact=False, rtol=1e-9, atol=1e-12)

    overall_mono = mono.compute_overall_metrics(
        dist_by_split["test"],
        mono.apply_inference_rule(rule_mono, flags_mono["test"]),
        labels_by_split["test"],
    )
    overall_mono["unmapped_pattern_fraction_test"] = mono.fraction_unmapped(
        rule_mono, flags_mono["test"])
    overall_ours = pd.read_csv(ws.tables_dir / "unsupervised_overall_metrics.csv")
    assert list(overall_ours.columns) == list(overall_mono.keys())
    for key, val in overall_mono.items():
        got = float(overall_ours[key].iloc[0])
        assert abs(got - float(val)) <= 1e-9 * max(abs(float(val)), 1e-12), (key, got, val)

    # B3: figures exist and are nonempty (oracle produced the same three).
    for name in (
        "unsupervised_distance_distributions.png",
        "unsupervised_roc_curves.png",
        "unsupervised_pattern_heatmap.png",
    ):
        fig = ws.figures_dir / name
        assert fig.is_file() and fig.stat().st_size > 0, name
        assert (oracle / "results" / "figures" / name).is_file(), name
