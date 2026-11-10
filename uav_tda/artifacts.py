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
(single instrument, n=30 in the committed run). The campaign's own windowed marginal and the
old W2-only per-flow 0.0602 s are kept as labelled secondary columns,
never as the cost.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .config import WINDOW_SIZES
from .exact import assemble_distances
from .manuscript import _ATTRIBUTION_COLUMNS, _ATTRIBUTION_ROW_ORDER, _BINARY_SUBSET_ORDER, _NODE_Y_OFFSET
from .metrics import ATTACK_CLASSES, MANIFOLD_SUBSETS, NORMAL
from .paths import OUTPUTS_DIR, REPO_ROOT, TABLES_DIR
from .provenance import run_git, write_provenance

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


def find_latency_summary(rebuild_dir=None, host=None) -> Path:
    """The latency summary CSV: a ``latency_*.csv`` whose header starts ``mode,w,stage``.

    Per-mode detail files (``latency_<host>_{per_flow,w25,...}.csv``) carry
    per-sample columns instead and are skipped. With ``host`` the summary
    must be ``latency_<host>.csv``. Without it, exactly one summary must be
    present; several raise ``ValueError`` naming them (pass ``host`` /
    ``uav-tda artifacts --latency-host`` to choose).
    """
    rebuild = _rebuild(rebuild_dir)
    found = []
    for p in sorted(rebuild.glob("latency_*.csv")):
        with open(p) as fh:
            header = tuple(fh.readline().strip().split(","))
        if header[:3] == _LATENCY_SUMMARY_HEADER:
            found.append(p)
    names = [p.name for p in found]
    if host is not None:
        hit = [p for p in found if p.name == f"latency_{host}.csv"]
        if not hit:
            raise ValueError(f"no latency summary CSV latency_{host}.csv (header "
                             f"'mode,w,stage,...') in {rebuild}; summaries present: {names}")
        return hit[0]
    if len(found) != 1:
        hint = (" -- several hosts present; choose one with host=... "
                "(CLI: --latency-host <host>)" if found else "")
        raise ValueError(
            f"expected exactly one latency summary CSV (header 'mode,w,stage,...') "
            f"in {rebuild}, found {names}{hint}"
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


def build_full_path_frontier(rebuild_dir=None, latency_host=None) -> pd.DataFrame:
    """Like-for-like frontier: AUC vs full-path latency-harness cost per decision.

    One row for ``per_flow`` then one per windowed W (ascending). AUC
    mean/CI come verbatim from ``windowed/compute_frontier.csv``;
    ``cost_s``/``cost_p95_s`` are the latency summary's ``total_s``
    mean/p95 for that mode/W. ``campaign_marginal_s`` (windowed only) and
    ``w2_only_per_flow_s`` (per-flow only) are the frontier CSV's
    ``marginal_s`` kept as labelled secondary columns. ``cost_n`` is the
    latency summary's sample count ``n`` for that row's ``total_s``.
    ``latency_host`` selects ``latency_<host>.csv`` (see `find_latency_summary`).
    """
    rebuild = _rebuild(rebuild_dir)
    frontier = pd.read_csv(rebuild / "windowed" / "compute_frontier.csv",
                           dtype={"row": str}, float_precision="round_trip")
    lat = pd.read_csv(find_latency_summary(rebuild, host=latency_host),
                      float_precision="round_trip")

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
            "_cost_n": int(total["n"]),
        })
    df = pd.DataFrame(rows)
    df["speedup_vs_per_flow"] = df["cost_s"].iloc[0] / df["cost_s"]
    df["cost_n"] = df.pop("_cost_n")
    return df


DEFAULT_FRONTIER_SOURCE = "windowed/compute_frontier.csv + latency_<host>.csv"


def emit_full_path_frontier_pgfplots(df: pd.DataFrame, source: str = DEFAULT_FRONTIER_SOURCE,
                                     n=None) -> str:
    """Render `build_full_path_frontier` as a pgfplots ``\\addplot`` block.

    Same shape as `windowed.emit_frontier_pgfplots` (per_flow first, then W
    ascending; one ``\\node`` label per point in the same order) but
    x = ``cost_s`` and asymmetric bootstrap-CI error bars in pgfplots
    ``explicit`` syntax: ``(x, y) += (0, ci_hi-y) -= (0, y-ci_lo)``. Labels
    sit at the top of the CI bar (y = ``auc_ci_hi``). A leading ``%``
    comment names the source CSVs, the x-axis meaning, and the harness
    sample count ``n`` (default: the per-flow row's ``cost_n`` column).
    """
    per_flow = df[df["row"] == _PER_FLOW_ROW]
    windowed = df[df["row"] != _PER_FLOW_ROW].sort_values("w")
    ordered = pd.concat([per_flow, windowed], ignore_index=True)

    if n is None:
        n = int(per_flow["cost_n"].iloc[0])
    lines = [f"% source: {source}; x = full-path seconds per decision (latency harness, n={n})",
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
    A committed destination whose bytes differ from the source is never
    overwritten: a warning is logged and the committed copy is kept.
    """
    src = Path(outputs_dir if outputs_dir is not None else OUTPUTS_DIR) / "max_edge_lengths.json"
    if not src.exists():
        _log.warning("persist_eps_max: %s not found; skipping", src)
        return None
    dest_dir = _rebuild(rebuild_dir) / ARTIFACTS_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "max_edge_lengths.json"
    if dest.exists() and dest.read_bytes() != src.read_bytes():
        _log.warning("persist_eps_max: %s differs from %s; keeping the committed copy "
                     "(not overwritten)", dest, src)
    elif not dest.exists():
        shutil.copyfile(src, dest)
    write_provenance(dest, {"source": "outputs/max_edge_lengths.json"}, skip_if_unchanged=True)
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


def git_introduced_in(repo_rel_path: str) -> str:
    """SHA of the commit that added ``repo_rel_path`` (``"unknown"`` if none).

    Uses the shared Rosetta-safe `provenance.run_git`. On a shallow clone
    the answer is the graft commit, not the true introducing commit.
    """
    try:
        out = run_git(["log", "--diff-filter=A", "--format=%H", "--", repo_rel_path],
                      cwd=REPO_ROOT).split()
    except Exception:
        return "unknown"
    # git log is newest-first; the original addition is the oldest entry.
    return out[-1] if out else "unknown"


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


# ---------------------------------------------------------------------------
# Task 3: generation driver + machine-generated MANUSCRIPT_ARTIFACTS report
# ---------------------------------------------------------------------------

REPORT_NAME = "MANUSCRIPT_ARTIFACTS.md"
FULL_PATH_FRONTIER_SNIPPET = "full_path_frontier_pgfplots.tex"
EXACT_ATTRIBUTION_SNIPPET = "exact_attribution_rows.tex"
EXACT_BINARY_SNIPPET = "exact_binary_auc_pgfplots.tex"
_MANIFOLD_NAMES = {"c2": "C2", "network": "Network", "physical": "Physical"}
PER_FLOW_BLOCK = "per-flow"
_UNRESOLVED_COLUMNS = ["attack_class", "block", "w", "dominant", "dominant_auc",
                       "dominant_ci_lo", "dominant_ci_hi", "runner_up", "runner_up_auc",
                       "runner_up_ci_lo", "runner_up_ci_hi", "difference"]


def unresolved_attributions(windowed_attribution: pd.DataFrame,
                            exact_per_attack: pd.DataFrame) -> pd.DataFrame:
    """Every attribution cell whose dominance is not CI-resolved.

    Scans the per-flow exact block (``exact_per_attack_auc.csv``, value
    ``auc``) and every windowed W block (``windowed_attribution.csv``, value
    ``mean``) for every attack. The dominant manifold is the row flagged
    ``dominant`` (exactly one per block/attack, else ``ValueError``); the
    runner-up is the highest-valued other manifold. A cell is unresolved
    when the dominant CI low <= the runner-up CI high (the CIs overlap).
    ``difference`` = dominant value - runner-up value. Rows: per-flow
    first, then W ascending; attacks in table row order.
    """
    blocks = [(PER_FLOW_BLOCK, np.nan, exact_per_attack, "auc")]
    for w in sorted(int(w) for w in windowed_attribution["w"].unique()):
        blocks.append((f"W={w}", float(w),
                       windowed_attribution[windowed_attribution["w"] == w], "mean"))
    rows = []
    for block, w, df, val in blocks:
        for attack in _ATTRIBUTION_ROW_ORDER:
            sub = df[df["attack_class"] == attack]
            if sub.empty:
                continue
            dom = sub[sub["dominant"].astype(bool)]
            if len(dom) != 1:
                raise ValueError(f"attribution block {block}, {attack}: expected exactly one "
                                 f"dominant manifold, found {len(dom)}")
            d = dom.iloc[0]
            r = sub[~sub["dominant"].astype(bool)].sort_values(val, ascending=False).iloc[0]
            if d["ci_lo"] <= r["ci_hi"]:
                rows.append([attack, block, w, d["manifold"], float(d[val]), float(d["ci_lo"]),
                             float(d["ci_hi"]), r["manifold"], float(r[val]), float(r["ci_lo"]),
                             float(r["ci_hi"]), float(d[val] - r[val])])
    return pd.DataFrame(rows, columns=_UNRESOLVED_COLUMNS)


def _n_attribution_cells(windowed_attribution: pd.DataFrame,
                         exact_per_attack: pd.DataFrame) -> int:
    return (exact_per_attack["attack_class"].nunique()
            + len(windowed_attribution[["w", "attack_class"]].drop_duplicates()))


def load_report_frames(rebuild_dir=None, latency_host=None) -> dict:
    """Every frame `render_artifacts_report` formats numbers from (no writes)."""
    rebuild = _rebuild(rebuild_dir)
    return {
        "frontier": build_full_path_frontier(rebuild, latency_host=latency_host),
        "compute_frontier": pd.read_csv(rebuild / "windowed" / "compute_frontier.csv",
                                        dtype={"row": str}),
        "operating_points": operating_point_table(rebuild),
        "unique_windows": unique_window_counts(rebuild),
        "windowed_attribution": pd.read_csv(rebuild / "windowed" / "windowed_attribution.csv"),
        "exact_per_attack": pd.read_csv(rebuild / "exact" / "exact_per_attack_auc.csv"),
        "exact_binary": pd.read_csv(rebuild / "exact" / "exact_binary_auc.csv"),
        "contamination": pd.read_csv(rebuild / "windowed" / "contamination_curve.csv",
                                     dtype={"bin": str}),
    }


def smallest_nonzero_contamination_bin(contamination: pd.DataFrame,
                                       unique_windows: pd.DataFrame) -> dict:
    """Thinnest non-empty ordered-arm (all-classes) contamination bin, in UNIQUE windows.

    ``contamination_curve.csv`` pools ``n_windows`` over the ordered repeats;
    dividing by that W's ``n_repeats`` gives the unique count (see
    `figures.fig_contamination`). Also returns the number of empty bins.
    """
    sel = contamination[(contamination["arm"] == "ordered")
                        & (contamination["majority_class"] == "all")].copy()
    reps = unique_windows.set_index("w")["n_repeats"]
    sel["unique_n"] = [int(n) // int(reps.loc[int(w)]) for w, n in zip(sel["w"], sel["n_windows"])]
    nonzero = sel[sel["unique_n"] > 0].sort_values(["unique_n", "w"])
    r = nonzero.iloc[0]
    return {"w": int(r["w"]), "bin": str(r["bin"]), "unique_n": int(r["unique_n"]),
            "detection_rate": float(r["detection_rate"]),
            "n_empty_bins": int((sel["unique_n"] == 0).sum()),
            "empty_bins": [f"W={int(w)} bin {b}" for w, b in
                           zip(sel.loc[sel["unique_n"] == 0, "w"],
                               sel.loc[sel["unique_n"] == 0, "bin"])]}


def _ci(lo, hi) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def _md_table(header, rows) -> list:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---:" if i else "---" for i in range(len(header))) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def render_artifacts_report(frames: dict, inventory) -> str:
    """Render `paper/MANUSCRIPT_ARTIFACTS.md`; every number is formatted from ``frames``.

    ``inventory`` is a list of ``(repo-relative path, source description)``.
    """
    fr = frames["frontier"]
    op = frames["operating_points"]
    uw = frames["unique_windows"]
    wa = frames["windowed_attribution"]
    ex = frames["exact_binary"]

    pf = fr[fr["row"] == _PER_FLOW_ROW].iloc[0]
    win = fr[fr["row"] != _PER_FLOW_ROW].sort_values("w")
    w_min = win.iloc[0]
    old_ratio = pf["w2_only_per_flow_s"] / w_min["campaign_marginal_s"]
    exact_ref = ex[(ex["subset"] == "all_three") & (ex["scoring"] == "znorm")].iloc[0]
    canon = op[(op["subset"] == "all_three") & (op["scoring"] == "znorm")].iloc[0]
    n_runs = int(wa["n_runs"].iloc[0])
    cf = frames["compute_frontier"]
    pf_seeds = int(cf.loc[cf["row"] == _PER_FLOW_ROW, "n_runs"].iloc[0])

    unresolved = unresolved_attributions(wa, frames["exact_per_attack"])
    n_cells = _n_attribution_cells(wa, frames["exact_per_attack"])
    if unresolved.empty:
        tie_lines = [
            f"- *Attribution dominance is CI-resolved:* No attribution cell (of {n_cells} "
            f"attack × block cells, per-flow and every windowed W) has the dominant manifold's "
            f"CI low at or below the runner-up's CI high."]
    else:
        k = len(unresolved)
        tie_lines = [
            f"- *Unresolved attributions:* in {k} attribution cell{'s' if k != 1 else ''} "
            f"(of {n_cells} attack × block cells) the dominant manifold's CI low is at or below "
            f"the runner-up's CI high (the CIs overlap), so the dominance marked in the "
            f"attribution figure should not be read as a resolved attribution there:"]
        for _, u in unresolved.iterrows():
            tie_lines.append(
                f"  - {u['attack_class'].replace(' Attack', '')} {u['block']}: dominant "
                f"{_MANIFOLD_NAMES[u['dominant']]} {u['dominant_auc']:.4f} "
                f"{_ci(u['dominant_ci_lo'], u['dominant_ci_hi'])} vs runner-up "
                f"{_MANIFOLD_NAMES[u['runner_up']]} {u['runner_up_auc']:.4f} "
                f"{_ci(u['runner_up_ci_lo'], u['runner_up_ci_hi'])} "
                f"(difference {u['difference']:.4f})")
    setup_lo, setup_hi = win["baseline_setup_s"].min(), win["baseline_setup_s"].max()
    pure_normal = ", ".join(f"W={int(r['w'])}: {int(r['n_pure_normal'])}"
                            for _, r in uw.sort_values("w").iterrows())
    thin = smallest_nonzero_contamination_bin(frames["contamination"], uw)
    uw_min = uw.sort_values("w").iloc[0]

    L = [
        "# MANUSCRIPT_ARTIFACTS: Phase-8 manuscript figures, tables and snippets",
        "",
        "_Generated by `uav-tda artifacts` (`uav_tda/artifacts.py`, `uav_tda/figures.py`) "
        "purely from CSVs already committed under `results/tables/rebuild/`; no recompute, "
        "no new science. Every number below is formatted from those frames at generation "
        "time; this report makes no claims of its own beyond restating them._",
        "",
        "## 1. What this supersedes",
        "",
        f"- **Compute frontier.** The old frontier's \"{old_ratio:.1f}×\" "
        f"(`paper/WINDOWED_RESULTS.md` §3: W2-only per-flow marginal "
        f"{pf['w2_only_per_flow_s']:.4f} s ÷ W={int(w_min['w'])} campaign marginal "
        f"{w_min['campaign_marginal_s']:.6f} s) is **superseded**: the per-flow figure excluded "
        f"Rips construction. On the like-for-like full-path latency harness (n={int(pf['cost_n'])}), "
        f"per-flow costs {pf['cost_s']:.4f} s per decision and W={int(w_min['w'])} costs "
        f"{w_min['cost_s']:.6f} s per decision, a **{w_min['speedup_vs_per_flow']:.0f}×** "
        f"per-decision speedup (one decision covers {int(w_min['flows_per_decision'])} flows). "
        f"Per-flow AUC stays the Phase-4 {pf_seeds}-seed probe value {pf['auc_mean']:.4f}.",
        f"- **Window counts.** `paper/WINDOWED_RESULTS.md` quoted window counts pooled over "
        f"{int(uw_min['n_repeats'])} ordered repeats (e.g. W={int(uw_min['w'])}: "
        f"{int(uw_min['pooled_n_windows'])} windows, {int(uw_min['pooled_n_blackhole_majority'])} "
        f"Blackhole-majority); the repeats share window membership, so the unique population "
        f"is {int(uw_min['n_windows'])} windows / {int(uw_min['n_blackhole_majority'])} "
        f"Blackhole-majority (§4).",
        f"- **FPR claim now measured.** At the val-Normal Q{OPERATING_QUANTILE * 100:.0f} "
        f"threshold on the high-precision (δ≤0.01) full-test campaign, canonical z-norm "
        f"all-three scoring gives test FPR {canon['fpr']:.4f} and TPR {canon['tpr']:.4f} "
        f"(val exceedance {canon['val_fpr']:.4f}; §3).",
        "",
        "**Caption-level caveats.**",
        "",
        *tie_lines,
        f"- *Windowed cost scope:* the latency harness's windowed `total_s` covers "
        f"Rips + W2 only; it excludes the scaler/slice stages and the one-time "
        f"`baseline_setup_s` ({setup_lo:.2f}–{setup_hi:.2f} s across W), whereas the per-flow "
        f"`total_s` includes scaler + slice. The W={int(w_min['w'])} "
        f"{w_min['speedup_vs_per_flow']:.0f}× speedup is therefore an upper-end figure.",
        "- *AUC units differ between arms:* per-flow AUC is flow-level, windowed AUC is "
        "window-level (majority label). The frontier and detection-vs-W figures put both on "
        "one axis, so cross-arm AUC comparisons are indicative, not like-for-like (see "
        "\"Unit caveat (M2)\" in `paper/WINDOWED_RESULTS.md`).",
        f"- *Two different thresholds:* the contamination figure's detection rate uses the "
        f"empirical near-normal Q95 threshold computed per (W, arm) in "
        f"`windowed/contamination_curve.csv`, not the val-Normal "
        f"Q{OPERATING_QUANTILE * 100:.0f} threshold of §3. That reference population is tiny "
        f"at large W (unique pure-Normal windows {pure_normal}; §4).",
        f"- *Attribution figure blocks are not like-for-like:* the per-flow block is a "
        f"single full-test estimate (`exact/exact_per_attack_auc.csv`), whereas each windowed "
        f"block is a {n_runs}-run mean (`windowed/windowed_attribution.csv`).",
        f"- *Thin contamination bins:* the smallest non-empty ordered-arm bin is "
        f"n={thin['unique_n']} unique windows (W={thin['w']}, bin {thin['bin']}, detection rate "
        f"{thin['detection_rate']:.4f}); {thin['n_empty_bins']} bin(s) are empty "
        f"({', '.join(thin['empty_bins']) if thin['empty_bins'] else 'none'}). "
        f"Rates in thin bins are low-power.",
        "",
        "## 2. Full-path compute frontier",
        "",
        f"Source: `windowed/compute_frontier.csv` (AUC, bootstrap CI) + latency summary "
        f"`total_s` (cost). Horizontal reference: full-test high-precision all-three z-norm "
        f"AUC {exact_ref['auc']:.4f} {_ci(exact_ref['ci_lo'], exact_ref['ci_hi'])} "
        f"(not a frontier point).",
        "",
    ]
    rows = []
    for _, r in pd.concat([fr[fr["row"] == _PER_FLOW_ROW], win]).iterrows():
        is_pf = r["row"] == _PER_FLOW_ROW
        rows.append([
            "per-flow" if is_pf else f"W={int(r['w'])}",
            f"{r['auc_mean']:.4f}", _ci(r["auc_ci_lo"], r["auc_ci_hi"]),
            f"{r['cost_s']:.6f}", f"{r['cost_p95_s']:.6f}", f"{r['cost_per_flow_s']:.3e}",
            f"{r['speedup_vs_per_flow']:.1f}×",
            "–" if is_pf else f"{r['campaign_marginal_s']:.6f}",
            f"{r['w2_only_per_flow_s']:.4f}" if is_pf else "–",
            "–" if is_pf else f"{r['baseline_setup_s']:.2f}",
        ])
    L += _md_table(["row", "AUC", "95% CI", "cost s/decision (mean)", "cost p95 s",
                    "cost s/flow", "speedup vs per-flow", "campaign marginal s",
                    "W2-only per-flow s (old)", "baseline setup s"], rows)
    L += [
        "",
        "## 3. Operating points (val-Normal Q95 threshold, full test)",
        "",
        f"Threshold τ = Q{OPERATING_QUANTILE * 100:.0f} of the val-Normal subset score; a test "
        f"flow is flagged when its score > τ. n val-Normal = {int(canon['n_val_normal'])}, "
        f"n test Normal = {int(canon['n_test_normal'])}, n test attack = "
        f"{int(canon['n_test_attack'])}. Full 7-subset table: "
        f"`results/tables/rebuild/artifacts/operating_points.csv`.",
        "",
    ]
    tpr_cols = [_tpr_col(a) for a in ATTACK_CLASSES]
    rows = []
    for subset in ("all_three", "network_physical"):
        for scoring in ("raw", "znorm"):
            r = op[(op["subset"] == subset) & (op["scoring"] == scoring)].iloc[0]
            rows.append([subset, scoring, f"{r['tau']:.4f}", f"{r['val_fpr']:.4f}",
                         f"{r['fpr']:.4f}", f"{r['tpr']:.4f}"]
                        + [f"{r[c]:.4f}" for c in tpr_cols])
    L += _md_table(["subset", "scoring", "τ", "val FPR", "test FPR", "test TPR"]
                   + [f"TPR {a.replace(' Attack', '')}" for a in ATTACK_CLASSES], rows)
    L += [
        "",
        "## 4. Unique vs pooled window counts (ordered arm)",
        "",
        "Unique = `run_w{W}_ordered0.csv` (repeats share window membership and labels); "
        "pooled = summed over every ordered repeat (the figures previously quoted).",
        "",
    ]
    rows = [[int(r["w"]), int(r["n_windows"]), int(r["n_pure_normal"]),
             int(r["n_blackhole_majority"]), int(r["n_repeats"]), int(r["pooled_n_windows"]),
             int(r["pooled_n_pure_normal"]), int(r["pooled_n_blackhole_majority"])]
            for _, r in uw.sort_values("w").iterrows()]
    L += _md_table(["W", "unique windows", "unique pure-Normal", "unique Blackhole-majority",
                    "repeats", "pooled windows", "pooled pure-Normal",
                    "pooled Blackhole-majority"], rows)
    L += ["", "## 5. Artifact inventory", ""]
    L += _md_table(["artifact", "derived from"], [[f"`{p}`", s] for p, s in inventory])
    L += [
        "",
        "Every generated artifact above (all but this report and the backfilled sidecars) "
        "has a `.provenance.json` sidecar recording git SHA, params, library versions and a "
        "content SHA-256 (`artifact_sha256`); such a sidecar is only rewritten when one of "
        "those changes. Backfilled sidecars sit next to pre-existing files and record "
        "`backfilled: true` plus the introducing commit (`introduced_in`) instead of a "
        "content hash; the data files themselves are untouched.",
        "",
        "## 6. Pending author sign-off",
        "",
        "Every number in this report and in the artifacts it lists is machine-generated from "
        "committed CSVs and is **pending author sign-off** before use in the manuscript.",
        "",
    ]
    return "\n".join(L)


def write_artifacts_report(paper_dir, frames: dict, inventory) -> Path:
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    out = paper_dir / REPORT_NAME
    out.write_text(render_artifacts_report(frames, inventory))
    return out


def _write_text_artifact(path: Path, text: str, params: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    write_provenance(path, params, skip_if_unchanged=True)
    return path


def generate_artifacts(root=None, latency_host=None) -> dict:
    """Write every Phase-8 artifact under workspace ``root`` (default: repo root).

    Tables -> ``results/tables/rebuild/artifacts/``; snippets ->
    ``results/tables/rebuild/paper_snippets/`` (new filenames only); figures ->
    ``results/figures/manuscript/``; report -> ``paper/MANUSCRIPT_ARTIFACTS.md``.
    Idempotent: data files are deterministic and sidecars are only rewritten
    when their params, content hash or git SHA change. ``latency_host``
    selects ``latency_<host>.csv`` when several latency summaries exist.
    Returns ``{"report": Path, "inventory": [(rel, source), ...],
    "backfilled": [sidecar Path written this call, ...]}``.
    """
    from . import figures  # local import: figures imports this module

    root = Path(root) if root is not None else REPO_ROOT
    rebuild = root / "results" / "tables" / "rebuild"
    art = rebuild / ARTIFACTS_SUBDIR
    snippets = rebuild / "paper_snippets"
    art.mkdir(parents=True, exist_ok=True)
    latency_name = find_latency_summary(rebuild, host=latency_host).name
    frontier_src = f"windowed/compute_frontier.csv + {latency_name}"

    frames = load_report_frames(rebuild, latency_host=latency_host)
    inventory = []

    def rel(p: Path) -> str:
        return str(Path(p).relative_to(root))

    def table(name, df, source, builder):
        path = art / name
        df.to_csv(path, index=False)
        write_provenance(path, {"source": source, "builder": builder}, skip_if_unchanged=True)
        inventory.append((rel(path), source))

    table("full_path_frontier.csv", frames["frontier"], frontier_src,
          "artifacts.build_full_path_frontier")
    table("unique_window_counts.csv", frames["unique_windows"],
          "windowed/run_w{W}_ordered*.csv", "artifacts.unique_window_counts")
    table("operating_points.csv", frames["operating_points"],
          "exact/shard_{val,test}_*.csv (exact.assemble_distances)",
          "artifacts.operating_point_table")

    eps = persist_eps_max(outputs_dir=root / "outputs", rebuild_dir=rebuild)
    if eps is not None:
        inventory.append((rel(eps), "outputs/max_edge_lengths.json (copied)"))

    backfilled = backfill_sidecars(rebuild)
    for side in sorted(rebuild.rglob("*.provenance.json")):
        try:
            meta = json.loads(side.read_text())
        except ValueError:
            continue
        if isinstance(meta.get("params"), dict) and meta["params"].get("backfilled"):
            inventory.append((rel(side), "backfilled sidecar (introduced_in "
                              f"{meta['params'].get('introduced_in', 'unknown')[:12]})"))

    snippet_specs = [
        (FULL_PATH_FRONTIER_SNIPPET,
         emit_full_path_frontier_pgfplots(frames["frontier"], source=frontier_src),
         frontier_src, "artifacts.emit_full_path_frontier_pgfplots"),
        (EXACT_ATTRIBUTION_SNIPPET, emit_exact_attribution_rows(frames["exact_per_attack"]),
         "exact/exact_per_attack_auc.csv", "artifacts.emit_exact_attribution_rows"),
        (EXACT_BINARY_SNIPPET, emit_exact_binary_pgfplots(frames["exact_binary"], "znorm"),
         "exact/exact_binary_auc.csv (scoring=znorm)", "artifacts.emit_exact_binary_pgfplots"),
    ]
    for name, text, source, builder in snippet_specs:
        path = _write_text_artifact(snippets / name, text + "\n",
                                    {"source": source, "builder": builder})
        inventory.append((rel(path), source))

    fig_paths = figures.build_all_figures(rebuild, root / "results" / "figures" / "manuscript",
                                          latency_host=latency_host)
    for name, path in fig_paths.items():
        meta = json.loads(path.with_name(path.name + ".provenance.json").read_text())
        inventory.append((rel(path), str(meta["params"].get("source", ""))))

    report = write_artifacts_report(root / "paper", frames, inventory)
    return {"report": report, "inventory": inventory, "backfilled": backfilled}
