"""Phase-8 manuscript figures, derived purely from committed rebuild CSVs.

Every ``fig_*`` function takes DataFrames (never paths) plus an output path,
writes a vector PDF and a ``.provenance.json`` sidecar, and returns the PDF
path. `build_all_figures` loads the committed CSVs and calls them all.

Style (Phase-8 global constraints): Agg backend; single-column 3.5 in wide,
double-column 7.16 in where noted; 8 pt base font; Okabe-Ito palette with
fixed manifold and arm colours shared by every figure; no titles inside the
PDF (captions live in LaTeX). PDFs are saved without a CreationDate so
reruns are byte-stable.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from . import artifacts, metrics  # noqa: E402
from .exact import assemble_distances  # noqa: E402
from .manuscript import _ATTRIBUTION_ROW_ORDER  # noqa: E402
from .metrics import NORMAL  # noqa: E402
from .paths import FIGURES_DIR  # noqa: E402
from .provenance import write_provenance  # noqa: E402

SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.16
BASE_FONT_PT = 8

# Okabe-Ito (Okabe & Ito 2008), colour-blind safe.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
MANIFOLD_COLORS = {
    "c2": OKABE_ITO["blue"],
    "network": OKABE_ITO["orange"],
    "physical": OKABE_ITO["green"],
}
MANIFOLD_LABELS = {"c2": "C2", "network": "Network", "physical": "Physical"}
ARM_COLORS = {
    "ordered": OKABE_ITO["vermillion"],
    "shuffled": OKABE_ITO["sky_blue"],
    "per_flow": OKABE_ITO["purple"],
    "exact": "#555555",
}
# ROC subsets: single manifolds reuse the manifold colours.
SUBSET_STYLE = {
    "all_three": ("All three", OKABE_ITO["black"], "-"),
    "network_physical": ("N+P", OKABE_ITO["purple"], "--"),
    "c2_only": ("C2", MANIFOLD_COLORS["c2"], ":"),
    "network_only": ("Network", MANIFOLD_COLORS["network"], ":"),
    "physical_only": ("Physical", MANIFOLD_COLORS["physical"], ":"),
}
STAGES = (("scaler_s", "scaler", "#E0E0E0"), ("rips_s", "Rips", "#7F7F7F"),
          ("slice_s", "slice", OKABE_ITO["yellow"]), ("w2_s", "W2", "#1A1A1A"))
EXACT_REF_LABEL = "per-flow, full test (δ≤0.01)"

_MANIFOLDS = ("c2", "network", "physical")
_PDF_METADATA = {"CreationDate": None}

_RC = {
    "font.size": BASE_FONT_PT,
    "axes.labelsize": BASE_FONT_PT,
    "axes.titlesize": BASE_FONT_PT,
    "xtick.labelsize": BASE_FONT_PT - 1,
    "ytick.labelsize": BASE_FONT_PT - 1,
    "legend.fontsize": BASE_FONT_PT - 1,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType, not Type 3
    "font.family": "DejaVu Sans",
}


@contextlib.contextmanager
def _style():
    with matplotlib.rc_context(_RC):
        yield


def _save(fig, out_path, params: dict) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # rcParams are read at save time too (pdf.fonttype), so re-enter the style.
    with _style():
        fig.savefig(out, format="pdf", metadata=_PDF_METADATA)
    plt.close(fig)
    write_provenance(out, params)
    return out


def _w_label(w) -> str:
    return f"W={int(w)}"


# ---------------------------------------------------------------------------
# Step 1: frontier
# ---------------------------------------------------------------------------


def fig_frontier(frontier: pd.DataFrame, exact_ref_auc: float, out_path,
                 source: str = artifacts.DEFAULT_FRONTIER_SOURCE) -> Path:
    """Log-x full-path cost per decision vs znorm all-three AUC (bootstrap CIs).

    ``frontier`` is `artifacts.build_full_path_frontier` output; the
    per-flow row gets a distinct marker. ``exact_ref_auc`` (the exact
    full-test all-three znorm AUC) is a dashed horizontal reference, not a
    frontier point.
    """
    pf = frontier[frontier["row"] == "per_flow"]
    win = frontier[frontier["row"] != "per_flow"].sort_values("w")
    with _style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.4), constrained_layout=True)
        ax.axhline(exact_ref_auc, color=ARM_COLORS["exact"], ls="--", lw=0.8, zorder=1,
                   label=EXACT_REF_LABEL)

        def _err(df):
            return np.vstack([df["auc_mean"] - df["auc_ci_lo"], df["auc_ci_hi"] - df["auc_mean"]])

        ax.errorbar(win["cost_s"], win["auc_mean"], yerr=_err(win), fmt="o-",
                    color=ARM_COLORS["ordered"], capsize=2, lw=0.8, elinewidth=0.8,
                    label="windowed (ordered)", zorder=3)
        ax.errorbar(pf["cost_s"], pf["auc_mean"], yerr=_err(pf), fmt="s",
                    color=ARM_COLORS["per_flow"], capsize=2, elinewidth=0.8, ms=5,
                    label="per-flow (10-seed probe)", zorder=3)
        for _, r in win.iterrows():
            ax.annotate(_w_label(r["w"]), (r["cost_s"], r["auc_ci_hi"]),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=BASE_FONT_PT - 1)
        ax.set_xscale("log")
        ax.set_xlabel("Full-path cost per decision (s)")
        ax.set_ylabel("AUC (z-norm, all three)")
        lo = min(frontier["auc_ci_lo"].min(), exact_ref_auc)
        hi = max(frontier["auc_ci_hi"].max(), exact_ref_auc)
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - 2.6 * pad, hi + 1.6 * pad)
        ax.set_xlim(frontier["cost_s"].min() / 2.5, frontier["cost_s"].max() * 2.5)
        ax.legend(loc="lower right")
    return _save(fig, out_path, {
        "figure": "frontier", "source": source,
        "exact_ref_source": "exact/exact_binary_auc.csv (all_three, znorm)",
        "exact_ref_auc": float(exact_ref_auc),
        "rows": [str(r) for r in frontier["row"]],
        "x": "cost_s (latency-harness full-path total_s mean)",
    })


# ---------------------------------------------------------------------------
# Step 2: detection vs W
# ---------------------------------------------------------------------------


def fig_detection_vs_w(detection: pd.DataFrame, per_flow_auc: float, exact_ref_auc: float,
                       out_path, source: str = "windowed/windowed_detection.csv") -> Path:
    """Znorm all-three AUC vs W (log2 x), ordered vs shuffled with CI bands."""
    sel = detection[(detection["subset"] == "all_three") & (detection["scoring"] == "znorm")]
    ws = sorted(int(w) for w in sel["w"].unique())
    with _style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.4), constrained_layout=True)
        for arm in ("ordered", "shuffled"):
            d = sel[sel["arm"] == arm].sort_values("w")
            c = ARM_COLORS[arm]
            ax.fill_between(d["w"], d["ci_lo"], d["ci_hi"], color=c, alpha=0.25, lw=0)
            ax.plot(d["w"], d["mean"], "o-", color=c, label=f"windowed, {arm}")
        ax.axhline(per_flow_auc, color=ARM_COLORS["per_flow"], ls="-.", lw=0.8,
                   label="per-flow (10-seed probe)")
        ax.axhline(exact_ref_auc, color=ARM_COLORS["exact"], ls="--", lw=0.8,
                   label=EXACT_REF_LABEL)
        ax.axhline(0.5, color="#BBBBBB", ls=":", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ws)
        ax.set_xticklabels([str(w) for w in ws])
        ax.minorticks_off()
        ax.set_xlim(ws[0] / 1.25, ws[-1] * 1.25)
        ax.set_xlabel("Window size W (flows)")
        ax.set_ylabel("AUC (z-norm, all three)")
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.575))
    return _save(fig, out_path, {
        "figure": "detection_vs_w", "source": source,
        "per_flow_source": "windowed/compute_frontier.csv (per_flow auc_mean)",
        "exact_ref_source": "exact/exact_binary_auc.csv (all_three, znorm)",
        "per_flow_auc": float(per_flow_auc), "exact_ref_auc": float(exact_ref_auc),
        "w": ws,
    })


# ---------------------------------------------------------------------------
# Step 3: contamination
# ---------------------------------------------------------------------------


def fig_contamination(contamination: pd.DataFrame, out_path, n_repeats: int = 10,
                      source: str = "windowed/contamination_curve.csv") -> Path:
    """Detection rate vs attack-fraction bin, one panel per W (ordered arm, all classes).

    ``n_windows`` in the CSV is pooled over ``n_repeats`` ordered repeats
    whose window membership is identical, so the annotated unique n per bin
    is ``n_windows / n_repeats`` (must divide exactly).
    """
    sel = contamination[(contamination["arm"] == "ordered")
                        & (contamination["majority_class"] == "all")].copy()
    if (sel["n_windows"] % n_repeats != 0).any():
        bad = sel.loc[sel["n_windows"] % n_repeats != 0, ["w", "bin", "n_windows"]]
        raise ValueError(f"pooled n_windows not divisible by n_repeats={n_repeats}:\n{bad}")
    sel["mid"] = (sel["bin_lo"] + sel["bin_hi"]) / 2
    sel["unique_n"] = (sel["n_windows"] // n_repeats).astype(int)
    ws = sorted(int(w) for w in sel["w"].unique())

    unique_n = {}
    with _style():
        fig, axes = plt.subplots(1, len(ws), figsize=(DOUBLE_COL_IN, 1.9), sharey=True,
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, w in zip(axes, ws):
            d = sel[sel["w"] == w].sort_values("mid")
            unique_n[str(w)] = [int(n) for n in d["unique_n"]]
            ax.plot(d["mid"], d["detection_rate"], "o-", color=ARM_COLORS["ordered"])
            ax.axhline(0.0, color="#BBBBBB", lw=0.5, zorder=0)
            for _, r in d.iterrows():
                ax.text(r["mid"], -0.2, f"{r['unique_n']}", ha="center", va="center",
                        fontsize=BASE_FONT_PT - 2, color="#444444")
            ax.text(-0.13, -0.2, "n", ha="right", va="center", style="italic",
                    fontsize=BASE_FONT_PT - 2, color="#444444")
            ax.text(0.03, 0.97, _w_label(w), transform=ax.transAxes, ha="left", va="top")
            ax.set_xlim(-0.12, 1.0)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
            ax.set_ylim(-0.3, 1.08)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axes[0].set_ylabel("Detection rate")
        fig.supxlabel("Attack fraction of window (bin midpoint); n = unique windows per bin",
                      fontsize=BASE_FONT_PT)
    return _save(fig, out_path, {
        "figure": "contamination", "source": source, "arm": "ordered",
        "majority_class": "all", "n_repeats": int(n_repeats),
        "unique_n_rule": "pooled n_windows / n_repeats", "unique_n": unique_n,
    })


# ---------------------------------------------------------------------------
# Step 4: latency
# ---------------------------------------------------------------------------


def _stage_mean(latency: pd.DataFrame, mode: str, w, stage: str) -> float:
    return float(artifacts._latency_stage(latency, mode, w, stage)["mean_s"])


def fig_latency(latency: pd.DataFrame, out_path, source: str = "latency_<host>.csv") -> Path:
    """Per-flow stage breakdown per manifold (top) and windowed total per W (bottom)."""
    stage_means = {m: [_stage_mean(latency, "per_flow", None, f"{m}_{s}") for s, _, _ in STAGES]
                   for m in _MANIFOLDS}
    pf_total = _stage_mean(latency, "per_flow", None, "total_s")
    win = latency[(latency["mode"] == "windowed") & (latency["stage"] == "total_s")]
    win = win.sort_values("w")
    ws = [int(w) for w in win["w"]]

    with _style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_IN, 3.2),
                                       gridspec_kw={"height_ratios": [1, 1.1]},
                                       constrained_layout=True)
        ys = np.arange(len(_MANIFOLDS))[::-1]
        left = np.zeros(len(_MANIFOLDS))
        for k, (_, label, color) in enumerate(STAGES):
            vals = np.array([stage_means[m][k] for m in _MANIFOLDS])
            ax1.barh(ys, vals, left=left, color=color, edgecolor="black", lw=0.3,
                     height=0.6, label=label)
            left += vals
        rips_k = [s for s, _, _ in STAGES].index("rips_s")
        for y, tot, m in zip(ys, left, _MANIFOLDS):
            share = stage_means[m][rips_k] / tot
            ax1.text(tot + 0.02 * left.max(), y, f"{tot:.3g} s (Rips {100 * share:.0f}%)",
                     va="center", fontsize=BASE_FONT_PT - 1)
        ax1.set_yticks(ys)
        ax1.set_yticklabels([MANIFOLD_LABELS[m] for m in _MANIFOLDS])
        ax1.set_xlim(0, left.max() * 1.45)
        ax1.set_xlabel("Per-flow latency per stage (s, mean)")
        ax1.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.0, columnspacing=1.0)

        y2 = np.arange(len(ws))[::-1]
        ax2.plot(win["mean_s"], y2, "o", color=ARM_COLORS["ordered"],
                 label="windowed (per decision)")
        for y, v in zip(y2, win["mean_s"]):
            ax2.annotate(f"{v:.3g} s", (v, y), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=BASE_FONT_PT - 1)
        ax2.axvline(pf_total, color=ARM_COLORS["per_flow"], ls="-.", lw=0.8)
        ax2.text(pf_total, 1.0, f" per-flow total {pf_total:.3g} s",
                 transform=ax2.get_xaxis_transform(), ha="right", va="bottom",
                 fontsize=BASE_FONT_PT - 1, color=ARM_COLORS["per_flow"])
        ax2.set_xscale("log")
        ax2.set_yticks(y2)
        ax2.set_yticklabels([_w_label(w) for w in ws])
        ax2.set_ylim(-0.6, len(ws) - 0.4)
        ax2.set_xlim(win["mean_s"].min() / 3, pf_total * 3)
        ax2.set_xlabel("Full-path latency per decision (s, mean, log)")
    return _save(fig, out_path, {
        "figure": "latency", "source": source,
        "per_flow_stage_means_s": {m: dict(zip([s for s, _, _ in STAGES], v))
                                   for m, v in stage_means.items()},
        "per_flow_total_s": pf_total,
        "windowed_total_s": {str(w): float(v) for w, v in zip(ws, win["mean_s"])},
    })


# ---------------------------------------------------------------------------
# Step 5: attribution heatmap
# ---------------------------------------------------------------------------


def fig_attribution(exact_per_attack: pd.DataFrame, windowed_attribution: pd.DataFrame,
                    out_path,
                    source: str = "exact/exact_per_attack_auc.csv + "
                                  "windowed/windowed_attribution.csv") -> Path:
    """Per-attack one-vs-rest AUC heatmap: per-flow full test, then W=25..200.

    Diverging colour map centred at 0.5 so anti-correlated (AUC<0.5) cells
    read as such; the dominant manifold per (attack, block) is outlined.
    """
    ws = sorted(int(w) for w in windowed_attribution["w"].unique())
    blocks = [("per_flow", exact_per_attack.rename(columns={"auc": "value"}))]
    for w in ws:
        d = windowed_attribution[windowed_attribution["w"] == w]
        blocks.append((str(w), d.rename(columns={"mean": "value"})))

    rows = list(_ATTRIBUTION_ROW_ORDER)
    ncol = 3 * len(blocks)
    grid = np.full((len(rows), ncol), np.nan)
    dominant = np.zeros_like(grid, dtype=bool)
    columns = []
    for b, (name, d) in enumerate(blocks):
        idx = d.set_index(["attack_class", "manifold"])
        for j, m in enumerate(_MANIFOLDS):
            columns.append(f"{name}:{m}")
            for i, a in enumerate(rows):
                grid[i, 3 * b + j] = float(idx.loc[(a, m), "value"])
                dominant[i, 3 * b + j] = bool(idx.loc[(a, m), "dominant"])

    with _style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 2.1), constrained_layout=True)
        cmap = plt.get_cmap("PuOr_r")
        norm = Normalize(vmin=0.0, vmax=1.0)
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=BASE_FONT_PT - 1.5,
                        color="white" if abs(v - 0.5) > 0.3 else "black")
                if dominant[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec="black", lw=1.4, zorder=3))
        for b in range(1, len(blocks)):
            ax.axvline(3 * b - 0.5, color="white", lw=3)
        ax.set_xticks(range(ncol))
        ax.set_xticklabels(["C2", "N", "P"] * len(blocks))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([a.replace(" Attack", "") for a in rows])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        top = ax.secondary_xaxis("top")
        top.set_xticks([3 * b + 1 for b in range(len(blocks))])
        top.set_xticklabels(["Per-flow (full test)"] + [_w_label(w) for w in ws])
        top.tick_params(length=0)
        top.spines["top"].set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("One-vs-rest AUC")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.outline.set_linewidth(0.4)
    return _save(fig, out_path, {
        "figure": "attribution", "source": source, "rows": rows, "columns": columns,
        "per_flow_value": "auc", "windowed_value": "mean (10 runs)",
        "colour_map": "PuOr_r, Normalize(0, 1) (centred at 0.5)",
    })


# ---------------------------------------------------------------------------
# Step 6: ROC
# ---------------------------------------------------------------------------


def fig_roc(val: pd.DataFrame, test: pd.DataFrame, operating_points: pd.DataFrame, out_path,
            source: str = "exact/shard_{val,test}_*.csv (via exact.assemble_distances)") -> Path:
    """Full-test ROC curves (z-norm) for all three, N+P and each single manifold.

    ``val``/``test`` are `exact.assemble_distances` frames; z-normalisation
    reuses `metrics.znorm_stats_from_val` / `metrics.apply_znorm`. The
    all_three/znorm row of ``operating_points`` (`artifacts.operating_point_table`)
    is marked as the val-Normal Q95 operating point.
    """
    stats = metrics.znorm_stats_from_val(val)
    scored = metrics.apply_znorm(test, stats)
    y = (scored["label"] != NORMAL).astype(int).to_numpy()
    op = operating_points[(operating_points["subset"] == "all_three")
                          & (operating_points["scoring"] == "znorm")]
    if len(op) != 1:
        raise ValueError("operating_points must have exactly one all_three/znorm row")
    op_fpr, op_tpr = float(op["fpr"].iloc[0]), float(op["tpr"].iloc[0])

    aucs, legend = {}, {}
    with _style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 3.0), constrained_layout=True)
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.6, ls="-")
        for subset in reversed(list(SUBSET_STYLE)):
            label, color, ls = SUBSET_STYLE[subset]
            score = scored[f"W2_{subset}"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, score)
            auc = float(roc_auc_score(y, score))
            aucs[subset] = auc
            legend[subset] = f"{label} (AUC {auc:.3f})"
            ax.plot(fpr, tpr, color=color, ls=ls,
                    lw=1.3 if subset == "all_three" else 0.9, label=legend[subset])
        ax.plot([op_fpr], [op_tpr], "o", mfc="white", mec="black", mew=1.0, ms=5, zorder=5,
                label=f"val-Normal Q95 (FPR {op_fpr:.3f}, TPR {op_tpr:.3f})")
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(legend[s]) for s in SUBSET_STYLE] + [len(labels) - 1]
        leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                        loc="lower right", frameon=True, fancybox=False, framealpha=0.92,
                        edgecolor="#BBBBBB", borderpad=0.4, handlelength=1.8)
        leg.get_frame().set_linewidth(0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
    return _save(fig, out_path, {
        "figure": "roc", "source": source, "scoring": "znorm",
        "auc": aucs, "legend": legend,
        "operating_point": {"fpr": op_fpr, "tpr": op_tpr},
        "operating_point_source": "artifacts.operating_point_table (all_three, znorm)",
        "n_test": int(len(scored)),
    })


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FIGURE_FILES = {
    "frontier": "fig_frontier.pdf",
    "detection_vs_w": "fig_detection_vs_w.pdf",
    "contamination": "fig_contamination.pdf",
    "latency": "fig_latency.pdf",
    "attribution": "fig_attribution.pdf",
    "roc": "fig_roc.pdf",
}


def build_all_figures(rebuild_dir=None, out_dir=None) -> dict:
    """Load the committed rebuild CSVs and render every manuscript figure.

    Returns ``{name: pdf_path}``. ``out_dir`` defaults to
    ``results/figures/manuscript/``.
    """
    rebuild = artifacts._rebuild(rebuild_dir)
    out = Path(out_dir) if out_dir is not None else FIGURES_DIR / "manuscript"
    exact_dir = rebuild / "exact"
    windowed_dir = rebuild / "windowed"

    exact_bin = pd.read_csv(exact_dir / "exact_binary_auc.csv")
    ref = exact_bin[(exact_bin["subset"] == "all_three") & (exact_bin["scoring"] == "znorm")]
    exact_ref_auc = float(ref["auc"].iloc[0])
    frontier = artifacts.build_full_path_frontier(rebuild)
    per_flow_auc = float(frontier.loc[frontier["row"] == "per_flow", "auc_mean"].iloc[0])
    latency_path = artifacts.find_latency_summary(rebuild)
    latency = pd.read_csv(latency_path)

    paths = {
        "frontier": fig_frontier(frontier, exact_ref_auc, out / FIGURE_FILES["frontier"]),
        "detection_vs_w": fig_detection_vs_w(
            pd.read_csv(windowed_dir / "windowed_detection.csv"), per_flow_auc,
            exact_ref_auc, out / FIGURE_FILES["detection_vs_w"]),
        "contamination": fig_contamination(
            pd.read_csv(windowed_dir / "contamination_curve.csv", dtype={"bin": str}),
            out / FIGURE_FILES["contamination"]),
        "latency": fig_latency(latency, out / FIGURE_FILES["latency"],
                               source=latency_path.name),
        "attribution": fig_attribution(
            pd.read_csv(exact_dir / "exact_per_attack_auc.csv"),
            pd.read_csv(windowed_dir / "windowed_attribution.csv"),
            out / FIGURE_FILES["attribution"]),
        "roc": fig_roc(assemble_distances(exact_dir, "val"),
                       assemble_distances(exact_dir, "test"),
                       artifacts.operating_point_table(rebuild),
                       out / FIGURE_FILES["roc"]),
    }
    return paths


__all__ = [
    "ARM_COLORS", "MANIFOLD_COLORS", "FIGURE_FILES", "build_all_figures",
    "fig_attribution", "fig_contamination", "fig_detection_vs_w", "fig_frontier",
    "fig_latency", "fig_roc",
]
