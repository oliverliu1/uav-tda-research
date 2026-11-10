"""Phase-8 Task 1: manuscript artifacts derived from committed rebuild CSVs."""
import json
import math

import pandas as pd
import pytest

from uav_tda import artifacts
from uav_tda.provenance import run_git

# ---------------------------------------------------------------------------
# Step 1: like-for-like full-path compute frontier
# ---------------------------------------------------------------------------

_FRONTIER_CSV = (
    "row,w,auc_mean,auc_std,ci_lo,ci_hi,marginal_s,baseline_s,n_runs\n"
    "25,25.0,0.87,0.0001,0.86,0.88,0.009,76.0,10\n"
    "per_flow,,0.89,0.008,0.88,0.90,0.06,,10\n"
)

_LATENCY_SUMMARY_CSV = (
    "mode,w,stage,mean_s,median_s,p95_s,n\n"
    "per_flow,,c2_total_s,1.0,1.0,1.5,30\n"
    "per_flow,,total_s,4.0,3.9,5.0,30\n"
    "windowed,25.0,baseline_setup_s,78.0,78.0,78.0,30\n"
    "windowed,25.0,c2_total_s,0.002,0.002,0.003,30\n"
    "windowed,25.0,total_s,0.008,0.0075,0.01,30\n"
)


def _synthetic_rebuild(tmp_path):
    rebuild = tmp_path / "rebuild"
    (rebuild / "windowed").mkdir(parents=True)
    (rebuild / "windowed" / "compute_frontier.csv").write_text(_FRONTIER_CSV)
    (rebuild / "latency_somehost.csv").write_text(_LATENCY_SUMMARY_CSV)
    # Per-mode detail files must be ignored (wrong header).
    (rebuild / "latency_somehost_per_flow.csv").write_text("row_idx,total_s\n0,99.0\n")
    (rebuild / "latency_somehost_w25.csv").write_text(
        "window_idx,start_pos,baseline_setup_s,total_s\n0,0,1.0,99.0\n"
    )
    return rebuild


def test_full_path_frontier_synthetic_exact_values(tmp_path):
    df = artifacts.build_full_path_frontier(_synthetic_rebuild(tmp_path))
    assert list(df.columns) == [
        "row", "w", "auc_mean", "auc_ci_lo", "auc_ci_hi", "cost_s", "cost_p95_s",
        "campaign_marginal_s", "w2_only_per_flow_s", "baseline_setup_s",
        "flows_per_decision", "cost_per_flow_s", "speedup_vs_per_flow", "cost_n",
    ]
    assert list(df["row"]) == ["per_flow", "25"]
    pf, w25 = df.iloc[0], df.iloc[1]

    assert math.isnan(pf["w"])
    assert pf["auc_mean"] == 0.89 and pf["auc_ci_lo"] == 0.88 and pf["auc_ci_hi"] == 0.90
    assert pf["cost_s"] == 4.0 and pf["cost_p95_s"] == 5.0
    assert math.isnan(pf["campaign_marginal_s"])
    assert pf["w2_only_per_flow_s"] == 0.06
    assert math.isnan(pf["baseline_setup_s"])
    assert pf["flows_per_decision"] == 1
    assert pf["cost_per_flow_s"] == 4.0
    assert pf["speedup_vs_per_flow"] == 1.0
    assert pf["cost_n"] == 30

    assert w25["w"] == 25
    assert w25["auc_mean"] == 0.87 and w25["auc_ci_lo"] == 0.86 and w25["auc_ci_hi"] == 0.88
    assert w25["cost_s"] == 0.008 and w25["cost_p95_s"] == 0.01
    assert w25["campaign_marginal_s"] == 0.009
    assert math.isnan(w25["w2_only_per_flow_s"])
    assert w25["baseline_setup_s"] == 78.0
    assert w25["flows_per_decision"] == 25
    assert w25["cost_per_flow_s"] == pytest.approx(0.008 / 25)
    assert w25["speedup_vs_per_flow"] == pytest.approx(500.0)


def test_full_path_frontier_rejects_ambiguous_latency_summary(tmp_path):
    rebuild = _synthetic_rebuild(tmp_path)
    (rebuild / "latency_otherhost.csv").write_text(_LATENCY_SUMMARY_CSV)
    with pytest.raises(ValueError, match="latency summary"):
        artifacts.build_full_path_frontier(rebuild)


def test_full_path_frontier_on_committed_csvs():
    df = artifacts.build_full_path_frontier()
    assert list(df["row"]) == ["per_flow", "25", "50", "100", "200"]
    pf = df[df["row"] == "per_flow"].iloc[0]
    w25 = df[df["row"] == "25"].iloc[0]
    assert pf["cost_s"] == 3.5762238905333343
    assert pf["auc_mean"] == pytest.approx(0.8930, abs=5e-5)
    assert pf["w2_only_per_flow_s"] == pytest.approx(0.0602, abs=5e-5)
    assert w25["cost_s"] == 0.007486383133332641
    assert abs(w25["speedup_vs_per_flow"] - 477.7) < 0.5


# ---------------------------------------------------------------------------
# Step 2: full-path frontier pgfplots emitter
# ---------------------------------------------------------------------------


def test_emit_full_path_frontier_pgfplots_golden_string():
    # Scrambled order (W row first) proves the emitter puts per_flow first.
    df = pd.DataFrame([
        {"row": "25", "w": 25.0, "auc_mean": 0.869, "auc_ci_lo": 0.861,
         "auc_ci_hi": 0.877, "cost_s": 0.00749},
        {"row": "per_flow", "w": float("nan"), "auc_mean": 0.893, "auc_ci_lo": 0.885,
         "auc_ci_hi": 0.900, "cost_s": 3.58},
    ])
    expected = (
        "% source: windowed/compute_frontier.csv + latency_host.csv; "
        "x = full-path seconds per decision (latency harness, n=30)\n"
        "\\addplot coordinates {\n"
        "    (3.58, 0.893) += (0, 0.007) -= (0, 0.008)\n"
        "    (0.00749, 0.869) += (0, 0.008) -= (0, 0.008)\n"
        "};\n"
        r"\node[font=\small, anchor=south] at (axis cs:3.58,0.900) {per-flow};" "\n"
        r"\node[font=\small, anchor=south] at (axis cs:0.00749,0.877) {W=25};"
    )
    got = artifacts.emit_full_path_frontier_pgfplots(
        df, source="windowed/compute_frontier.csv + latency_host.csv", n=30)
    assert got == expected


def test_emit_full_path_frontier_pgfplots_derives_n_from_frame():
    df = pd.DataFrame([
        {"row": "per_flow", "w": float("nan"), "auc_mean": 0.893, "auc_ci_lo": 0.885,
         "auc_ci_hi": 0.900, "cost_s": 3.58, "cost_n": 17},
    ])
    first = artifacts.emit_full_path_frontier_pgfplots(df).splitlines()[0]
    assert first.endswith("(latency harness, n=17)")


# ---------------------------------------------------------------------------
# Step 3: unique (non-pooled) window counts
# ---------------------------------------------------------------------------


def test_unique_window_counts_committed_w200():
    df = artifacts.unique_window_counts().set_index("w")
    assert list(df.index) == [25, 50, 100, 200]
    r = df.loc[200]
    assert r["n_windows"] == 91
    assert r["n_blackhole_majority"] == 19
    assert r["n_pure_normal"] == 0
    assert r["n_repeats"] == 10
    assert r["pooled_n_windows"] == 910
    assert r["pooled_n_blackhole_majority"] == 190
    # Audit I-3: W=50 bin-0 "n=60" is 6 unique windows.
    assert df.loc[50, "n_pure_normal"] == 6
    assert df.loc[50, "pooled_n_pure_normal"] == 60


# ---------------------------------------------------------------------------
# Step 4: val-Normal Q95 operating points on the exact campaign
# ---------------------------------------------------------------------------


def test_operating_point_table_committed_exact():
    df = artifacts.operating_point_table()
    assert len(df) == 14
    assert set(df["scoring"]) == {"raw", "znorm"}
    row = df[(df["subset"] == "all_three") & (df["scoring"] == "znorm")].iloc[0]
    assert row["fpr"] == pytest.approx(0.056, abs=0.002)
    assert row["tpr"] == pytest.approx(0.570, abs=0.002)
    assert row["tpr_sybil"] == pytest.approx(0.91, abs=0.005)
    for col in ("tau", "tpr_blackhole", "tpr_flooding", "tpr_wormhole",
                "n_val_normal", "n_test_normal", "n_test_attack"):
        assert col in df.columns
    # Q95 of val-Normal => val-Normal exceedance is ~5% by construction.
    assert (df["val_fpr"] - 0.05).abs().max() < 0.002


# ---------------------------------------------------------------------------
# Step 5: persist eps_max JSON
# ---------------------------------------------------------------------------


def test_persist_eps_max_copies_with_sidecar(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    payload = {"c2": 1.5, "network": 0.7, "physical": 2.25}
    (outputs / "max_edge_lengths.json").write_text(json.dumps(payload))
    rebuild = tmp_path / "rebuild"

    dest = artifacts.persist_eps_max(outputs, rebuild)

    assert dest == rebuild / "artifacts" / "max_edge_lengths.json"
    assert json.loads(dest.read_text()) == payload
    side = json.loads((dest.parent / "max_edge_lengths.json.provenance.json").read_text())
    assert side["params"]["source"] == "outputs/max_edge_lengths.json"


def test_persist_eps_max_missing_source_warns(tmp_path, caplog):
    rebuild = tmp_path / "rebuild"
    with caplog.at_level("WARNING"):
        assert artifacts.persist_eps_max(tmp_path / "nope", rebuild) is None
    assert "max_edge_lengths.json" in caplog.text
    assert not (rebuild / "artifacts").exists()


# ---------------------------------------------------------------------------
# Step 6: backfill provenance sidecars for pre-existing artifacts
# ---------------------------------------------------------------------------


def test_backfill_sidecars_writes_missing_never_overwrites(tmp_path):
    rebuild = tmp_path / "rebuild"
    (rebuild / "paper_snippets").mkdir(parents=True)
    (rebuild / "exact").mkdir()
    for rel in artifacts.BACKFILL_TARGETS:
        (rebuild / rel).write_text("x")
    existing = rebuild / "znorm_summary.csv.provenance.json"
    existing.write_text('{"keep": true}')
    data_before = {rel: (rebuild / rel).read_text() for rel in artifacts.BACKFILL_TARGETS}

    written = artifacts.backfill_sidecars(rebuild, introduced_in_fn=lambda rel: f"sha-{rel}")

    assert existing.read_text() == '{"keep": true}'
    assert existing not in written
    assert len(written) == len(artifacts.BACKFILL_TARGETS) - 1
    side = json.loads(
        (rebuild / "exact" / "manifest.json.provenance.json").read_text())
    assert side["params"] == {"backfilled": True, "introduced_in": "sha-exact/manifest.json"}
    assert {rel: (rebuild / rel).read_text() for rel in artifacts.BACKFILL_TARGETS} == data_before
    # Idempotent: a second call writes nothing.
    assert artifacts.backfill_sidecars(rebuild, introduced_in_fn=lambda rel: "x") == []


def test_backfill_targets_match_brief():
    assert set(artifacts.BACKFILL_TARGETS) == {
        "znorm_summary.csv", "znorm_per_attack.csv",
        "paper_snippets/attribution_table_rows.tex",
        "paper_snippets/binary_auc_pgfplots.tex",
        "paper_snippets/windowed_frontier_pgfplots.tex",
        "exact/manifest.json",
    }


def _is_shallow_clone() -> bool:
    try:
        return run_git(["rev-parse", "--is-shallow-repository"]) == "true"
    except Exception:
        return True


@pytest.mark.skipif(_is_shallow_clone(),
                    reason="shallow clone (or git unreachable): --diff-filter=A history unavailable")
def test_default_introduced_in_resolves_committed_sha():
    sha = artifacts.git_introduced_in("results/tables/rebuild/exact/manifest.json")
    assert sha == "61131b8f83b85c6a505a303ff3abc3673f4c7bd7"


# ---------------------------------------------------------------------------
# Step 7: exact-campaign LaTeX snippets
# ---------------------------------------------------------------------------


def _pa(attack, manifold, auc, lo, hi, dominant):
    return {"attack_class": attack, "manifold": manifold, "auc": auc,
            "ci_lo": lo, "ci_hi": hi, "dominant": dominant}


def test_emit_exact_attribution_rows_golden_string():
    rows = [
        _pa("Blackhole Attack", "c2", 0.438, 0.428, 0.449, False),
        _pa("Blackhole Attack", "network", 0.361, 0.354, 0.369, False),
        _pa("Blackhole Attack", "physical", 0.792, 0.786, 0.797, True),
        _pa("Wormhole Attack", "physical", 0.728, 0.721, 0.736, True),
        _pa("Wormhole Attack", "network", 0.296, 0.289, 0.304, False),
        _pa("Wormhole Attack", "c2", 0.560, 0.550, 0.569, False),
        _pa("Sybil Attack", "network", 0.888, 0.883, 0.892, True),
        _pa("Sybil Attack", "c2", 0.679, 0.671, 0.687, False),
        _pa("Sybil Attack", "physical", 0.186, 0.179, 0.193, False),
        _pa("Flooding Attack", "physical", 0.419, 0.410, 0.428, False),
        _pa("Flooding Attack", "c2", 0.647, 0.637, 0.657, False),
        _pa("Flooding Attack", "network", 0.787, 0.780, 0.793, True),
    ]
    expected = (
        r"Sybil & $0.68\ [0.67, 0.69]$ & $\mathbf{0.89\ [0.88, 0.89]}$ & $0.19\ [0.18, 0.19]$ \\" "\n"
        r"Flooding & $0.65\ [0.64, 0.66]$ & $\mathbf{0.79\ [0.78, 0.79]}$ & $0.42\ [0.41, 0.43]$ \\" "\n"
        r"Blackhole & $0.44\ [0.43, 0.45]$ & $0.36\ [0.35, 0.37]$ & $\mathbf{0.79\ [0.79, 0.80]}$ \\" "\n"
        r"Wormhole & $0.56\ [0.55, 0.57]$ & $0.30\ [0.29, 0.30]$ & $\mathbf{0.73\ [0.72, 0.74]}$ \\"
    )
    assert artifacts.emit_exact_attribution_rows(pd.DataFrame(rows)) == expected


def _eb(subset, scoring, auc, lo, hi):
    return {"subset": subset, "scoring": scoring, "auc": auc, "ci_lo": lo, "ci_hi": hi,
            "n_flows": 100, "n_approx_flagged": 0}


def test_emit_exact_binary_pgfplots_golden_string():
    rows = [
        _eb("all_three", "raw", 0.500, 0.400, 0.600),
        _eb("all_three", "znorm", 0.908, 0.903, 0.913),
        _eb("network_physical", "znorm", 0.874, 0.867, 0.880),
        _eb("c2_physical", "znorm", 0.797, 0.788, 0.805),
        _eb("c2_network", "znorm", 0.842, 0.836, 0.848),
        _eb("physical_only", "znorm", 0.658, 0.649, 0.668),
        _eb("network_only", "znorm", 0.753, 0.747, 0.760),
        _eb("c2_only", "znorm", 0.784, 0.776, 0.793),
    ]
    node = r"\node[font=\small, anchor=south, fill=white, inner sep=1pt] at "
    expected = (
        "coordinates {\n"
        "    (C2, 0.784) += (0, 0.009) -= (0, 0.008)\n"
        "    (Network, 0.753) += (0, 0.007) -= (0, 0.006)\n"
        "    (Physical, 0.658) += (0, 0.010) -= (0, 0.009)\n"
        "    (C2+N, 0.842) += (0, 0.006) -= (0, 0.006)\n"
        "    (C2+P, 0.797) += (0, 0.008) -= (0, 0.009)\n"
        "    (N+P, 0.874) += (0, 0.006) -= (0, 0.007)\n"
        "    (All three, 0.908) += (0, 0.005) -= (0, 0.005)\n"
        "};\n"
        + node + "(axis cs:C2,0.826) {0.78};\n"
        + node + "(axis cs:Network,0.793) {0.75};\n"
        + node + "(axis cs:Physical,0.701) {0.66};\n"
        + node + "(axis cs:C2+N,0.881) {0.84};\n"
        + node + "(axis cs:C2+P,0.838) {0.80};\n"
        + node + "(axis cs:N+P,0.913) {0.87};\n"
        + node + "(axis cs:All three,0.946) {0.91};"
    )
    assert artifacts.emit_exact_binary_pgfplots(pd.DataFrame(rows), scoring="znorm") == expected


# ---------------------------------------------------------------------------
# Final fix wave
# ---------------------------------------------------------------------------


def test_find_latency_summary_selects_named_host(tmp_path):
    rebuild = _synthetic_rebuild(tmp_path)
    (rebuild / "latency_otherhost.csv").write_text(_LATENCY_SUMMARY_CSV)
    assert artifacts.find_latency_summary(rebuild, host="otherhost") == \
        rebuild / "latency_otherhost.csv"
    assert artifacts.find_latency_summary(rebuild, host="somehost") == \
        rebuild / "latency_somehost.csv"
    with pytest.raises(ValueError, match="nohost"):
        artifacts.find_latency_summary(rebuild, host="nohost")


def test_find_latency_summary_ambiguous_error_names_candidates(tmp_path):
    rebuild = _synthetic_rebuild(tmp_path)
    (rebuild / "latency_otherhost.csv").write_text(_LATENCY_SUMMARY_CSV)
    with pytest.raises(ValueError) as exc:
        artifacts.find_latency_summary(rebuild)
    msg = str(exc.value)
    assert "latency_otherhost.csv" in msg and "latency_somehost.csv" in msg
    assert "--latency-host" in msg
    # Single summary: unchanged behaviour without a host.
    (rebuild / "latency_otherhost.csv").unlink()
    assert artifacts.find_latency_summary(rebuild) == rebuild / "latency_somehost.csv"


def test_persist_eps_max_keeps_differing_committed_copy(tmp_path, caplog):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "max_edge_lengths.json").write_text(json.dumps({"c2": 9.0}))
    rebuild = tmp_path / "rebuild"
    dest = rebuild / "artifacts" / "max_edge_lengths.json"
    dest.parent.mkdir(parents=True)
    committed = json.dumps({"c2": 1.0})
    dest.write_text(committed)

    with caplog.at_level("WARNING"):
        out = artifacts.persist_eps_max(outputs, rebuild)

    assert out == dest
    assert dest.read_text() == committed
    assert "differs" in caplog.text


def _wa(w, attack, manifold, mean, lo, hi, dominant):
    return {"w": w, "attack_class": attack, "manifold": manifold, "mean": mean, "std": 0.0,
            "ci_lo": lo, "ci_hi": hi, "dominant": dominant, "n_runs": 10}


def test_unresolved_attributions_synthetic_reports_only_overlapping_blocks():
    wa = pd.DataFrame([
        # W=25: C2 dominant (flag), runner-up Physical overlaps -> unresolved.
        _wa(25, "Sybil Attack", "c2", 0.80, 0.75, 0.85, True),
        _wa(25, "Sybil Attack", "network", 0.40, 0.35, 0.45, False),
        _wa(25, "Sybil Attack", "physical", 0.78, 0.73, 0.83, False),
        # W=50: separated -> resolved.
        _wa(50, "Sybil Attack", "c2", 0.90, 0.88, 0.92, True),
        _wa(50, "Sybil Attack", "network", 0.40, 0.35, 0.45, False),
        _wa(50, "Sybil Attack", "physical", 0.70, 0.68, 0.72, False),
    ])
    ex = pd.DataFrame([
        _pa("Sybil Attack", "c2", 0.6, 0.59, 0.61, False),
        _pa("Sybil Attack", "network", 0.9, 0.89, 0.91, True),
        _pa("Sybil Attack", "physical", 0.2, 0.19, 0.21, False),
    ])
    out = artifacts.unresolved_attributions(wa, ex)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["attack_class"] == "Sybil Attack" and r["block"] == "W=25"
    assert r["dominant"] == "c2" and r["runner_up"] == "physical"
    assert r["dominant_auc"] == 0.80 and r["runner_up_auc"] == 0.78
    assert r["difference"] == pytest.approx(0.02)

    # Per-flow overlap is scanned too.
    ex.loc[ex["manifold"] == "c2", ["auc", "ci_lo", "ci_hi"]] = [0.88, 0.87, 0.895]
    out = artifacts.unresolved_attributions(wa, ex)
    assert list(out["block"]) == ["per-flow", "W=25"]


def test_unresolved_attributions_committed_data():
    rebuild = artifacts.DEFAULT_REBUILD_DIR
    out = artifacts.unresolved_attributions(
        pd.read_csv(rebuild / "windowed" / "windowed_attribution.csv"),
        pd.read_csv(rebuild / "exact" / "exact_per_attack_auc.csv"))
    got = {(r["attack_class"], r["block"], r["dominant"], r["runner_up"])
           for _, r in out.iterrows()}
    assert got == {
        ("Wormhole Attack", "W=50", "network", "physical"),
        ("Flooding Attack", "W=50", "c2", "network"),
        ("Flooding Attack", "W=100", "c2", "network"),
        ("Flooding Attack", "W=200", "network", "c2"),
    }
