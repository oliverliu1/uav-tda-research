"""Phase-8 manuscript artifacts derived purely from committed rebuild CSVs.

No new science and no recompute: every function here reads CSV/JSON files
already committed under ``results/tables/rebuild/`` (or, for
`persist_eps_max`, one file from ``outputs/``) and returns DataFrames or
LaTeX strings. Writing the committed artifacts is the CLI's job (Phase-8
Task 3); the only writers here are the provenance helpers
`persist_eps_max` / `backfill_sidecars`.

Frontier ruling (Phase-8 global constraints): the per-flow AUC is the
Phase-4 10-seed probe value carried in ``windowed/compute_frontier.csv``;
every row's COST is the latency harness's full-path ``total_s`` mean
(single instrument, n=30). The campaign's own windowed marginal and the
old W2-only per-flow 0.0602 s are kept as labelled secondary columns,
never as the cost.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .config import WINDOW_SIZES
from .exact import assemble_distances
from .manuscript import _ATTRIBUTION_COLUMNS, _ATTRIBUTION_ROW_ORDER, _BINARY_SUBSET_ORDER, _NODE_Y_OFFSET
from .metrics import ATTACK_CLASSES, MANIFOLD_SUBSETS, NORMAL
from .paths import OUTPUTS_DIR, REPO_ROOT, TABLES_DIR
from .provenance import write_provenance

_log = logging.getLogger("uav_tda.artifacts")

DEFAULT_REBUILD_DIR = TABLES_DIR / "rebuild"
ARTIFACTS_SUBDIR = "artifacts"

_PER_FLOW_ROW = "per_flow"
_LATENCY_SUMMARY_HEADER = ("mode", "w", "stage")


def _rebuild(rebuild_dir) -> Path:
    return Path(rebuild_dir) if rebuild_dir is not None else DEFAULT_REBUILD_DIR


# ---------------------------------------------------------------------------
# Step 1-2: like-for-like full-path compute frontier
# ---------------------------------------------------------------------------


def find_latency_summary(rebuild_dir=None) -> Path:
    """The single ``latency_*.csv`` whose header starts ``mode,w,stage``.

    Per-mode detail files (``latency_<host>_{per_flow,w25,...}.csv``) carry
    per-sample columns instead and are skipped. Raises ``ValueError`` unless
    exactly one summary is present.
    """
    rebuild = _rebuild(rebuild_dir)
    found = []
    for p in sorted(rebuild.glob("latency_*.csv")):
        with open(p) as fh:
            header = tuple(fh.readline().strip().split(","))
        if header[:3] == _LATENCY_SUMMARY_HEADER:
            found.append(p)
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one latency summary CSV (header 'mode,w,stage,...') "
            f"in {rebuild}, found {[p.name for p in found]}"
        )
    return found[0]


def _latency_stage(lat: pd.DataFrame, mode: str, w, stage: str) -> pd.Series:
    sel = (lat["mode"] == mode) & (lat["stage"] == stage)
    sel &= lat["w"].isna() if w is None else (lat["w"] == float(w))
    hit = lat[sel]
    if len(hit) != 1:
        raise ValueError(f"latency summary: expected one row for mode={mode} w={w} "
                         f"stage={stage}, found {len(hit)}")
    return hit.iloc[0]


def build_full_path_frontier(rebuild_dir=None) -> pd.DataFrame:
    """Like-for-like frontier: AUC vs full-path latency-harness cost per decision.

    One row for ``per_flow`` then one per windowed W (ascending). AUC
    mean/CI come verbatim from ``windowed/compute_frontier.csv``;
    ``cost_s``/``cost_p95_s`` are the latency summary's ``total_s``
    mean/p95 for that mode/W. ``campaign_marginal_s`` (windowed only) and
    ``w2_only_per_flow_s`` (per-flow only) are the frontier CSV's
    ``marginal_s`` kept as labelled secondary columns.
    """
    rebuild = _rebuild(rebuild_dir)
    frontier = pd.read_csv(rebuild / "windowed" / "compute_frontier.csv",
                           dtype={"row": str}, float_precision="round_trip")
    lat = pd.read_csv(find_latency_summary(rebuild), float_precision="round_trip")

    per_flow = frontier[frontier["row"] == _PER_FLOW_ROW]
    windowed = frontier[frontier["row"] != _PER_FLOW_ROW].sort_values("w")
    if len(per_flow) != 1:
        raise ValueError("compute_frontier.csv must have exactly one per_flow row")

    rows = []
    for _, fr in pd.concat([per_flow, windowed]).iterrows():
        is_pf = fr["row"] == _PER_FLOW_ROW
        mode, w = ("per_flow", None) if is_pf else ("windowed", fr["w"])
        total = _latency_stage(lat, mode, w, "total_s")
        flows = 1 if is_pf else int(fr["w"])
        rows.append({
            "row": fr["row"],
            "w": np.nan if is_pf else float(fr["w"]),
            "auc_mean": fr["auc_mean"],
            "auc_ci_lo": fr["ci_lo"],
            "auc_ci_hi": fr["ci_hi"],
            "cost_s": float(total["mean_s"]),
            "cost_p95_s": float(total["p95_s"]),
            "campaign_marginal_s": np.nan if is_pf else fr["marginal_s"],
            "w2_only_per_flow_s": fr["marginal_s"] if is_pf else np.nan,
            "baseline_setup_s": (
                np.nan if is_pf
                else float(_latency_stage(lat, mode, w, "baseline_setup_s")["mean_s"])
            ),
            "flows_per_decision": flows,
            "cost_per_flow_s": float(total["mean_s"]) / flows,
        })
    df = pd.DataFrame(rows)
    df["speedup_vs_per_flow"] = df["cost_s"].iloc[0] / df["cost_s"]
    return df


DEFAULT_FRONTIER_SOURCE = "windowed/compute_frontier.csv + latency_<host>.csv"


def emit_full_path_frontier_pgfplots(df: pd.DataFrame, source: str = DEFAULT_FRONTIER_SOURCE) -> str:
    """Render `build_full_path_frontier` as a pgfplots ``\\addplot`` block.

    Same shape as `windowed.emit_frontier_pgfplots` (per_flow first, then W
    ascending; one ``\\node`` label per point in the same order) but
    x = ``cost_s`` and asymmetric bootstrap-CI error bars in pgfplots
    ``explicit`` syntax: ``(x, y) += (0, ci_hi-y) -= (0, y-ci_lo)``. Labels
    sit at the top of the CI bar (y = ``auc_ci_hi``). A leading ``%``
    comment names the source CSVs and the x-axis meaning.
    """
    per_flow = df[df["row"] == _PER_FLOW_ROW]
    windowed = df[df["row"] != _PER_FLOW_ROW].sort_values("w")
    ordered = pd.concat([per_flow, windowed], ignore_index=True)

    lines = [f"% source: {source}; x = full-path seconds per decision (latency harness, n=30)",
             r"\addplot coordinates {"]
    for _, r in ordered.iterrows():
        lines.append(
            f"    ({r['cost_s']:.3g}, {r['auc_mean']:.3f}) "
            f"+= (0, {r['auc_ci_hi'] - r['auc_mean']:.3f}) "
            f"-= (0, {r['auc_mean'] - r['auc_ci_lo']:.3f})"
        )
    lines.append("};")
    for _, r in ordered.iterrows():
        label = "per-flow" if r["row"] == _PER_FLOW_ROW else f"W={int(r['w'])}"
        lines.append(
            r"\node[font=\small, anchor=south] at "
            f"(axis cs:{r['cost_s']:.3g},{r['auc_ci_hi']:.3f}) {{{label}}};"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3: unique (non-pooled) window counts -- audit I-3
# ---------------------------------------------------------------------------

_BLACKHOLE = "Blackhole Attack"


def unique_window_counts(rebuild_dir=None, window_sizes=WINDOW_SIZES) -> pd.DataFrame:
    """Per W: unique window counts (repeat 0) and the pooled-over-repeats counts.

    The ordered arm's C2/Physical diagrams are bit-identical across repeats
    and window membership/labels are identical by construction, so
    ``run_w{W}_ordered0.csv`` IS the unique window population. The
    ``pooled_*`` columns sum over every ``run_w{W}_ordered*.csv`` present --
    the xN-inflated figures `paper/WINDOWED_RESULTS.md` quoted.
    """
    wdir = _rebuild(rebuild_dir) / "windowed"
    rows = []
    for w in window_sizes:
        runs = sorted(wdir.glob(f"run_w{w}_ordered[0-9]*.csv"))
        runs = [p for p in runs if p.stem[len(f"run_w{w}_ordered"):].isdigit()]
        if not runs:
            raise ValueError(f"no ordered runs for W={w} in {wdir}")

        def _counts(frame: pd.DataFrame) -> tuple:
            return (len(frame), int((frame["attack_frac"] == 0).sum()),
                    int((frame["majority_label"] == _BLACKHOLE).sum()))

        unique = _counts(pd.read_csv(wdir / f"run_w{w}_ordered0.csv",
                                     usecols=["majority_label", "attack_frac"]))
        pooled = np.sum([_counts(pd.read_csv(p, usecols=["majority_label", "attack_frac"]))
                         for p in runs], axis=0)
        rows.append({
            "w": int(w),
            "n_windows": unique[0],
            "n_pure_normal": unique[1],
            "n_blackhole_majority": unique[2],
            "n_repeats": len(runs),
            "pooled_n_windows": int(pooled[0]),
            "pooled_n_pure_normal": int(pooled[1]),
            "pooled_n_blackhole_majority": int(pooled[2]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: val-Normal Q95 operating points on the exact campaign
# ---------------------------------------------------------------------------

OPERATING_QUANTILE = 0.95


def _tpr_col(attack: str) -> str:
    return "tpr_" + attack.replace(" Attack", "").lower()


def operating_point_table(rebuild_dir=None) -> pd.DataFrame:
    """7 subsets x {raw, znorm}: tau = Q95 of val-Normal score, then test rates.

    Val/test frames come from `exact.assemble_distances` (manifest-ordered
    exact shards); z-normalization uses `metrics.znorm_stats_from_val` /
    `metrics.apply_znorm` on the same val frame. A flow is flagged when its
    subset score is strictly greater than tau. ``val_fpr`` is the
    val-Normal exceedance (~0.05 by construction, a sanity column).
    """
    exact_dir = _rebuild(rebuild_dir) / "exact"
    val_raw = assemble_distances(exact_dir, "val")
    test_raw = assemble_distances(exact_dir, "test")
    stats = metrics.znorm_stats_from_val(val_raw)
    frames = {
        "raw": (val_raw, test_raw),
        "znorm": (metrics.apply_znorm(val_raw, stats), metrics.apply_znorm(test_raw, stats)),
    }

    rows = []
    for subset in MANIFOLD_SUBSETS:
        for scoring, (val, test) in frames.items():
            col = f"W2_{subset}"
            val_normal = val.loc[val["label"] == NORMAL, col].to_numpy(dtype=float)
            tau = float(np.quantile(val_normal, OPERATING_QUANTILE))
            score = test[col].to_numpy(dtype=float)
            label = test["label"].to_numpy()
            flagged = score > tau
            is_normal = label == NORMAL
            row = {
                "subset": subset, "scoring": scoring, "tau": tau,
                "val_fpr": float(np.mean(val_normal > tau)),
                "fpr": float(flagged[is_normal].mean()),
                "tpr": float(flagged[~is_normal].mean()),
            }
            for attack in ATTACK_CLASSES:
                row[_tpr_col(attack)] = float(flagged[label == attack].mean())
            row.update({"n_val_normal": int(len(val_normal)),
                        "n_test_normal": int(is_normal.sum()),
                        "n_test_attack": int((~is_normal).sum())})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5-6: provenance hygiene
# ---------------------------------------------------------------------------


def persist_eps_max(outputs_dir=None, rebuild_dir=None):
    """Copy ``outputs/max_edge_lengths.json`` into ``rebuild/artifacts/`` + sidecar.

    Returns the destination path, or ``None`` (with a logged warning) when
    the source is absent -- ``outputs/`` is gitignored, so CI never has it.
    """
    src = Path(outputs_dir if outputs_dir is not None else OUTPUTS_DIR) / "max_edge_lengths.json"
    if not src.exists():
        _log.warning("persist_eps_max: %s not found; skipping", src)
        return None
    dest_dir = _rebuild(rebuild_dir) / ARTIFACTS_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "max_edge_lengths.json"
    shutil.copyfile(src, dest)
    write_provenance(dest, {"source": "outputs/max_edge_lengths.json"})
    return dest


# Pre-existing rebuild artifacts committed without a provenance sidecar.
BACKFILL_TARGETS = (
    "znorm_summary.csv",
    "znorm_per_attack.csv",
    "paper_snippets/attribution_table_rows.tex",
    "paper_snippets/binary_auc_pgfplots.tex",
    "paper_snippets/windowed_frontier_pgfplots.tex",
    "exact/manifest.json",
)


# Under Rosetta (x86_64 Python on Apple silicon) the /usr/bin/git xcrun shim
# can fail to load an arm64-only libxcrun; fall back to the Command Line
# Tools git binary directly.
_GIT_CANDIDATES = ("git", "/Library/Developer/CommandLineTools/usr/bin/git")


def git_introduced_in(repo_rel_path: str) -> str:
    """SHA of the commit that added ``repo_rel_path`` (``"unknown"`` if none)."""
    for git in _GIT_CANDIDATES:
        try:
            out = subprocess.check_output(
                [git, "log", "--diff-filter=A", "--format=%H", "--", repo_rel_path],
                cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
            ).decode().split()
        except Exception:
            continue
        # git log is newest-first; the original addition is the oldest entry.
        return out[-1] if out else "unknown"
    return "unknown"


def backfill_sidecars(rebuild_dir=None, introduced_in_fn=None) -> list:
    """Write a backfilled sidecar for each `BACKFILL_TARGETS` file lacking one.

    Never overwrites an existing sidecar and never touches the data file.
    ``introduced_in_fn(rel)`` (``rel`` relative to ``rebuild_dir``) supplies
    the adding commit; default `git_introduced_in` on the repo-relative path.
    Returns the list of sidecar paths written.
    """
    rebuild = _rebuild(rebuild_dir)
    if introduced_in_fn is None:
        def introduced_in_fn(rel):
            try:
                repo_rel = (rebuild / rel).resolve().relative_to(REPO_ROOT)
            except ValueError:
                return "unknown"
            return git_introduced_in(str(repo_rel))

    written = []
    for rel in BACKFILL_TARGETS:
        target = rebuild / rel
        if not target.exists():
            _log.warning("backfill_sidecars: %s not found; skipping", target)
            continue
        side = target.with_name(target.name + ".provenance.json")
        if side.exists():
            continue
        written.append(write_provenance(
            target, {"backfilled": True, "introduced_in": introduced_in_fn(rel)}))
    return written


# ---------------------------------------------------------------------------
# Step 7: exact-campaign LaTeX snippets
# ---------------------------------------------------------------------------


def emit_exact_attribution_rows(df: pd.DataFrame) -> str:
    """Render ``exact/exact_per_attack_auc.csv`` as Table tab:per_attack_auc rows.

    Same row/column order and line syntax as
    `manuscript.emit_attribution_rows`, but each cell is the single
    full-test AUC with its bootstrap CI: ``${auc:.2f}\\ [{lo:.2f}, {hi:.2f}]$``,
    the dominant manifold's cell wrapped in ``\\mathbf{...}``.
    """
    lines = []
    for attack in _ATTRIBUTION_ROW_ORDER:
        sub = df[df["attack_class"] == attack].set_index("manifold")
        cells = []
        for manifold_key, _header in _ATTRIBUTION_COLUMNS:
            r = sub.loc[manifold_key]
            cell = f"{r['auc']:.2f}\\ [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
            if bool(r["dominant"]):
                cell = f"\\mathbf{{{cell}}}"
            cells.append(f"${cell}$")
        lines.append(f"{attack.replace(' Attack', '')} & {' & '.join(cells)} \\\\")
    return "\n".join(lines)


def emit_exact_binary_pgfplots(df: pd.DataFrame, scoring: str = "znorm") -> str:
    """Render ``exact/exact_binary_auc.csv`` as a Fig. fig:binary_auc block.

    Same symbolic x coords and node style as
    `manuscript.emit_binary_pgfplots`, but asymmetric bootstrap-CI bars
    ``({label}, {auc}) += (0, ci_hi-auc) -= (0, auc-ci_lo)`` and value
    labels at y = ci_hi + `manuscript._NODE_Y_OFFSET`.
    """
    sub = df[df["scoring"] == scoring].set_index("subset")
    coord_lines = ["coordinates {"]
    node_lines = []
    for subset_key, label in _BINARY_SUBSET_ORDER:
        r = sub.loc[subset_key]
        coord_lines.append(
            f"    ({label}, {r['auc']:.3f}) += (0, {r['ci_hi'] - r['auc']:.3f}) "
            f"-= (0, {r['auc'] - r['ci_lo']:.3f})"
        )
        node_lines.append(
            r"\node[font=\small, anchor=south, fill=white, inner sep=1pt] at "
            f"(axis cs:{label},{r['ci_hi'] + _NODE_Y_OFFSET:.3f}) {{{r['auc']:.2f}}};"
        )
    coord_lines.append("};")
    return "\n".join(coord_lines + node_lines)
