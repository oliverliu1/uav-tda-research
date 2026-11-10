"""Phase-8 Task 3: `uav-tda artifacts` subcommand + MANUSCRIPT_ARTIFACTS report."""
import hashlib
import json
import shutil

import pytest

from uav_tda import artifacts, cli
from uav_tda.paths import TABLES_DIR

REBUILD = TABLES_DIR / "rebuild"


@pytest.fixture(scope="module")
def frames():
    return artifacts.load_report_frames(REBUILD)


def _inventory():
    return [("results/tables/rebuild/artifacts/full_path_frontier.csv",
             "windowed/compute_frontier.csv + latency summary")]


def test_report_has_all_sections_and_footer(frames):
    text = artifacts.render_artifacts_report(frames, _inventory())
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert heading in text
    assert "pending author sign-off" in text.lower()
    assert "results/tables/rebuild/artifacts/full_path_frontier.csv" in text


def test_report_supersedes_old_frontier_ratio_with_full_path_speedup(frames):
    text = artifacts.render_artifacts_report(frames, _inventory())
    fr = frames["frontier"]
    w25 = fr[fr["w"] == 25].iloc[0]
    pf = fr[fr["row"] == "per_flow"].iloc[0]
    old = pf["w2_only_per_flow_s"] / w25["campaign_marginal_s"]
    assert f"{old:.1f}×" == "6.6×"  # the historical figure, derived not typed
    assert "6.6×" in text
    assert f"{w25['speedup_vs_per_flow']:.0f}×" in text
    assert "superseded" in text


def test_report_numbers_track_the_frames(frames):
    base = artifacts.render_artifacts_report(frames, _inventory())
    tweaked = {k: v.copy() for k, v in frames.items()}
    fr = tweaked["frontier"]
    fr.loc[fr["w"] == 25, "cost_s"] = fr.loc[fr["w"] == 25, "cost_s"] * 2
    fr["speedup_vs_per_flow"] = fr["cost_s"].iloc[0] / fr["cost_s"]
    new = fr.loc[fr["w"] == 25, "speedup_vs_per_flow"].iloc[0]
    text = artifacts.render_artifacts_report(tweaked, _inventory())
    assert text != base
    assert f"{new:.0f}×" in text


def test_report_caveats(frames):
    text = artifacts.render_artifacts_report(frames, _inventory())
    # Wormhole W=50 Network vs Physical is a tie with overlapping CIs.
    assert "Wormhole" in text and "overlap" in text
    # Per-flow attribution block is one full-test estimate; windowed blocks are run means.
    assert "single full-test" in text
    n_runs = int(frames["windowed_attribution"]["n_runs"].iloc[0])
    assert f"{n_runs}-run mean" in text
    # Thin contamination bins: smallest non-zero unique n is reported with its W/bin.
    thin = artifacts.smallest_nonzero_contamination_bin(frames["contamination"],
                                                         frames["unique_windows"])
    assert f"n={thin['unique_n']}" in text and f"W={thin['w']}" in text


def test_report_operating_points_measured(frames):
    text = artifacts.render_artifacts_report(frames, _inventory())
    op = frames["operating_points"]
    r = op[(op["subset"] == "all_three") & (op["scoring"] == "znorm")].iloc[0]
    assert f"{r['fpr']:.4f}" in text and f"{r['tpr']:.4f}" in text
    np_rows = op[op["subset"] == "network_physical"]
    assert len(np_rows) == 2
    for _, r in np_rows.iterrows():
        assert f"{r['tau']:.4f}" in text


def test_report_unique_vs_pooled(frames):
    text = artifacts.render_artifacts_report(frames, _inventory())
    for _, r in frames["unique_windows"].iterrows():
        assert f"| {int(r['w'])} | {int(r['n_windows'])} |" in text
        assert str(int(r["pooled_n_windows"])) in text


def _snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_cli_artifacts_end_to_end_is_idempotent(tmp_path):
    root = tmp_path / "ws"
    # Exclude the committed generated outputs (their sidecars pin a git_sha that
    # legitimately differs from HEAD); the test starts from the pre-Phase-8 inputs.
    shutil.copytree(REBUILD, root / "results" / "tables" / "rebuild",
                    ignore=shutil.ignore_patterns(
                        "artifacts", "full_path_frontier_pgfplots.tex*",
                        "exact_attribution_rows.tex*", "exact_binary_auc_pgfplots.tex*"))
    (root / "outputs").mkdir(parents=True)
    (root / "outputs" / "max_edge_lengths.json").write_text(
        json.dumps({"c2": 0.5, "network": 0.1, "physical": 0.3}))
    before = _snapshot(root)

    assert cli.main(["artifacts", "--root", str(root)]) == 0
    first = _snapshot(root)

    # Pre-existing files are never modified.
    for rel, digest in before.items():
        assert first[rel] == digest, rel

    rb = "results/tables/rebuild/"
    expected = [
        rb + "artifacts/full_path_frontier.csv",
        rb + "artifacts/unique_window_counts.csv",
        rb + "artifacts/operating_points.csv",
        rb + "artifacts/max_edge_lengths.json",
        rb + "paper_snippets/full_path_frontier_pgfplots.tex",
        rb + "paper_snippets/exact_attribution_rows.tex",
        rb + "paper_snippets/exact_binary_auc_pgfplots.tex",
        "results/figures/manuscript/fig_frontier.pdf",
        "results/figures/manuscript/fig_roc.pdf",
        "paper/MANUSCRIPT_ARTIFACTS.md",
    ]
    for rel in expected:
        assert rel in first, rel
        if not rel.endswith(".md"):
            assert rel + ".provenance.json" in first, rel
    for rel in first:
        if rel.endswith(".pdf"):
            assert (root / rel).read_bytes()[:4] == b"%PDF"

    assert cli.main(["artifacts", "--root", str(root)]) == 0
    assert _snapshot(root) == first
