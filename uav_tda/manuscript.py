"""Manuscript-grade statistics: ten-seed cluster bootstrap AUC CIs.

Phase 4 adds a bootstrap-CI treatment of the across-seed mean AUC for the
manuscript's ten-seed evaluation (see config.MANUSCRIPT_SEEDS). Resampling
is stratified per class label so that a replicate never drops a minority
class and roc_auc_score never fails with a single-class ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import metrics
from .metrics import ATTACK_CLASSES, MANIFOLD_SUBSETS, NORMAL
from .paths import REPO_ROOT, TABLES_DIR
from .provenance import write_provenance


def stratified_resample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample indices with replacement, stratified per unique label.

    For each unique label, draws that class's count of indices (with
    replacement) from that class's positions in `labels`, then concatenates
    across classes. The returned indices index into the original rows.
    """
    labels = np.asarray(labels)
    idx_parts = []
    for label in np.unique(labels):
        class_positions = np.flatnonzero(labels == label)
        draw = rng.choice(class_positions, size=len(class_positions), replace=True)
        idx_parts.append(draw)
    return np.concatenate(idx_parts)


def bootstrap_mean_auc_ci(
    per_seed: list[tuple[np.ndarray, np.ndarray]],
    B: int = 2000,
    bootstrap_seed: int = 0,
    labels_per_seed: list[np.ndarray] | None = None,
) -> tuple[float, float]:
    """Cluster bootstrap CI for the across-seed mean AUC.

    `per_seed` is a list of (y_binary, scores) pairs, one per seed.
    `labels_per_seed[i]`, if given, carries the multiclass labels used for
    stratified resampling of seed i; when None, resampling is stratified on
    the binary labels `y_binary`.

    Each of B replicates: for every seed, stratified-resample that seed's
    rows, compute roc_auc_score on the resample, then average across seeds.
    Returns the 2.5th/97.5th percentiles of the B replicate means.
    """
    rng = np.random.default_rng(bootstrap_seed)
    replicate_means = np.empty(B, dtype=float)

    for b in range(B):
        seed_aucs = np.empty(len(per_seed), dtype=float)
        for i, (y_binary, scores) in enumerate(per_seed):
            strat_labels = labels_per_seed[i] if labels_per_seed is not None else y_binary
            idx = stratified_resample_indices(strat_labels, rng)
            seed_aucs[i] = roc_auc_score(y_binary[idx], scores[idx])
        replicate_means[b] = seed_aucs.mean()

    ci_lo, ci_hi = np.percentile(replicate_means, [2.5, 97.5])
    return float(ci_lo), float(ci_hi)


# --- Ten-seed campaign orchestration -----------------------------------------
# Reuses the Phase-3 coupled test+val Z-normalized probe (probe.run_probe_with_znorm)
# and its artifact-writing routine, shared here so both the `uav-tda probe --znorm`
# single-seed CLI path and this multi-seed campaign path write identical artifacts.

def _rebuild_paths(seed: int, rebuild_dir: Path | None = None) -> dict[str, Path]:
    rebuild = rebuild_dir if rebuild_dir is not None else (TABLES_DIR / "rebuild")
    return {
        "raw": rebuild / f"probe_distances_seed{seed}.csv",
        "znorm": rebuild / f"probe_distances_seed{seed}_znorm.csv",
        "val": rebuild / f"val_normal_distances_seed{seed}.csv",
        "stats": rebuild / f"znorm_stats_seed{seed}.json",
    }


def run_znorm_probe_and_write(
    seed: int, w2_timeout: float | None,
    per_class: int = 200, top_k: int = 50, delta: float = 0.2,
    rebuild_dir: Path | None = None,
) -> tuple[dict[str, Path], pd.DataFrame, dict]:
    """Run the coupled test+val znorm probe for one seed and write rebuild/ artifacts.

    Shared by the `uav-tda probe --znorm` single-seed CLI path and the
    ten-seed manuscript campaign (`run_missing_seeds`) so both write
    byte-identical artifact layouts. Returns (paths, test_df, stats) where
    `paths` is the dict of written file paths, `test_df` is the raw
    (un-normalized) test probe dataframe, and `stats` is
    `{manifold: (mean, std)}` from the coupled validation pass.
    """
    from . import probe  # local import: avoids a probe<->manuscript import cycle

    paths = _rebuild_paths(seed, rebuild_dir)
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)

    test_df, val_df, stats, timeout_counts = probe.run_probe_with_znorm(
        seed=seed, per_class=per_class, top_k=top_k, delta=delta,
        w2_timeout=w2_timeout,
    )
    znorm_df = metrics.apply_znorm(test_df, stats)

    params = {
        "seed": seed, "per_class": per_class, "top_k": top_k,
        "delta": delta, "w2_timeout": w2_timeout,
    }

    test_df.to_csv(paths["raw"], index=False)
    write_provenance(paths["raw"], params)

    znorm_df.to_csv(paths["znorm"], index=False)
    write_provenance(paths["znorm"], params)

    val_df.to_csv(paths["val"], index=False)
    write_provenance(paths["val"], params)

    stats_payload = {
        "stats": {m: {"mean": mean, "std": std} for m, (mean, std) in stats.items()},
        "timeout_counts": timeout_counts,
        "params": params,
    }
    paths["stats"].write_text(json.dumps(stats_payload, indent=2, sort_keys=True))
    write_provenance(paths["stats"], params)

    return paths, test_df, stats


def missing_seeds(seeds: tuple[int, ...], rebuild_dir: Path) -> list[int]:
    """Seeds among `seeds` lacking a probe-distances CSV or znorm-stats JSON."""
    missing = []
    for seed in seeds:
        raw_csv = rebuild_dir / f"probe_distances_seed{seed}.csv"
        stats_json = rebuild_dir / f"znorm_stats_seed{seed}.json"
        if not raw_csv.exists() or not stats_json.exists():
            missing.append(seed)
    return missing


def run_missing_seeds(seeds: tuple[int, ...], w2_timeout: float = 30.0) -> None:
    """Run the Phase-3 znorm probe for each seed in `seeds` lacking artifacts.

    Writes into the default `results/tables/rebuild/` location via
    `run_znorm_probe_and_write` (the same writer the `uav-tda probe --znorm`
    CLI path uses), so already-present seeds are left untouched.
    """
    rebuild_dir = TABLES_DIR / "rebuild"
    for seed in missing_seeds(seeds, rebuild_dir):
        run_znorm_probe_and_write(seed=seed, w2_timeout=w2_timeout, rebuild_dir=rebuild_dir)


def load_seed_frames(
    seeds: tuple[int, ...], rebuild_dir: Path,
) -> dict[int, tuple[pd.DataFrame, dict]]:
    """Load per-seed raw test probe df + znorm stats ({manifold: (mean, std)})."""
    out: dict[int, tuple[pd.DataFrame, dict]] = {}
    for seed in seeds:
        raw_csv = rebuild_dir / f"probe_distances_seed{seed}.csv"
        stats_json = rebuild_dir / f"znorm_stats_seed{seed}.json"
        df = pd.read_csv(raw_csv)
        payload = json.loads(stats_json.read_text())
        stats = {m: (v["mean"], v["std"]) for m, v in payload["stats"].items()}
        out[seed] = (df, stats)
    return out


def build_binary_auc_table(
    seed_frames: dict[int, tuple[pd.DataFrame, dict]],
    B: int = 2000, bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap-CI mean binary (Normal-vs-attack) AUC per subset x scoring.

    Rows: subset (the 7 `metrics.MANIFOLD_SUBSETS` keys) x scoring in
    {raw, znorm}. Mean/std are taken across per-seed AUCs (std ddof=1, the
    pandas default, matching Phase 3's `metrics.aggregate_over_seeds`). CI
    is `bootstrap_mean_auc_ci` over the per-seed (y_binary, scores) pairs,
    stratified per-seed-resample on the seed's multiclass labels; znorm rows
    use `metrics.apply_znorm(df, stats)` (that seed's own stats) to derive
    the subset score.
    """
    rows = []
    for subset in MANIFOLD_SUBSETS:
        for scoring in ("raw", "znorm"):
            per_seed = []
            labels_per_seed = []
            aucs = []
            for seed, (df, stats) in seed_frames.items():
                score_df = metrics.apply_znorm(df, stats) if scoring == "znorm" else df
                y = (df["label"] != NORMAL).astype(int).to_numpy()
                scores = score_df[f"W2_{subset}"].to_numpy()
                per_seed.append((y, scores))
                labels_per_seed.append(df["label"].to_numpy())
                aucs.append(float(roc_auc_score(y, scores)))

            aucs_arr = np.asarray(aucs, dtype=float)
            mean = float(aucs_arr.mean())
            std = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_mean_auc_ci(
                per_seed, B=B, bootstrap_seed=bootstrap_seed, labels_per_seed=labels_per_seed)
            rows.append({
                "subset": subset, "scoring": scoring,
                "mean": mean, "std": std,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n_seeds": len(seed_frames),
            })
    return pd.DataFrame(rows)


def build_attribution_table(
    seed_frames: dict[int, tuple[pd.DataFrame, dict]],
    B: int = 2000, bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap-CI mean one-vs-rest AUC per (attack class, manifold).

    Rows: attack x manifold, raw scores only (per-manifold W2 distance is
    unmodified by Z-normalization's affine per-manifold rescaling, so
    per-manifold AUC is identical raw vs. znorm — a monotone, strictly
    increasing transform of a single column never changes ROC AUC). `mean`
    is the across-seed mean of per-seed `metrics.per_attack_auc` values
    (std ddof=1); `ci_lo`/`ci_hi` via `bootstrap_mean_auc_ci` with
    y = (label == attack), stratified on each seed's multiclass labels.
    `dominant` is True for exactly the manifold with the largest `mean`
    AUC within each attack class.
    """
    manifolds = list(metrics.MANIFOLDS)
    per_attack_per_manifold: dict[tuple[str, str], list[float]] = {}
    ci_by_key: dict[tuple[str, str], tuple[float, float]] = {}

    for attack in ATTACK_CLASSES:
        for m in manifolds:
            per_seed = []
            labels_per_seed = []
            aucs = []
            for seed, (df, _stats) in seed_frames.items():
                y = (df["label"].to_numpy() == attack).astype(int)
                scores = df[f"W2_{m}"].to_numpy()
                per_seed.append((y, scores))
                labels_per_seed.append(df["label"].to_numpy())
                aucs.append(float(roc_auc_score(y, scores)))
            per_attack_per_manifold[(attack, m)] = aucs
            ci_by_key[(attack, m)] = bootstrap_mean_auc_ci(
                per_seed, B=B, bootstrap_seed=bootstrap_seed, labels_per_seed=labels_per_seed)

    rows = []
    for attack in ATTACK_CLASSES:
        means = {}
        for m in manifolds:
            aucs_arr = np.asarray(per_attack_per_manifold[(attack, m)], dtype=float)
            means[m] = float(aucs_arr.mean())
        dominant_manifold = max(means, key=means.get)
        for m in manifolds:
            aucs_arr = np.asarray(per_attack_per_manifold[(attack, m)], dtype=float)
            std = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
            ci_lo, ci_hi = ci_by_key[(attack, m)]
            rows.append({
                "attack_class": attack, "manifold": m,
                "mean": means[m], "std": std,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "dominant": m == dominant_manifold,
            })
    return pd.DataFrame(rows)


# --- LaTeX/pgfplots emitters --------------------------------------------------
# These re-derive the paper's Table tab:per_attack_auc and Fig. fig:binary_auc
# literal snippets from `build_attribution_table`/`build_binary_auc_table`
# output, so the manuscript numbers and the code that produced them never
# drift apart. See CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex for the
# hand-written originals these are meant to match/replace.

# Paper's row order for Table tab:per_attack_auc (not ATTACK_CLASSES' order).
_ATTRIBUTION_ROW_ORDER = ("Sybil Attack", "Flooding Attack", "Blackhole Attack", "Wormhole Attack")
# Paper's column order for Table tab:per_attack_auc / manifold key -> header.
_ATTRIBUTION_COLUMNS = (("c2", "C2"), ("network", "Network"), ("physical", "Physical"))

# Paper's symbolic x-coordinate order for Fig. fig:binary_auc, mapping each
# `metrics.MANIFOLD_SUBSETS` key to its plotted label.
_BINARY_SUBSET_ORDER = (
    ("c2_only", "C2"),
    ("network_only", "Network"),
    ("physical_only", "Physical"),
    ("c2_network", "C2+N"),
    ("c2_physical", "C2+P"),
    ("network_physical", "N+P"),
    ("all_three", "All three"),
)

# Vertical offset (axis units) from a bar's (mean + std) upper whisker to its
# manual pgfplots value-label node, matching the paper's Fig. fig:binary_auc
# hand-placed labels (see e.g. `node ... at (axis cs:C2,0.792) {0.75}` where
# mean=0.749, std=0.027: 0.749 + 0.027 + 0.016 != 0.792 exactly in the
# hand-authored original, so this is the emitter's own convention, inferred
# to sit just above the error bar; flagged as a candidate pending author
# sign-off, not a byte-for-byte reproduction of the hand-placed original).
_NODE_Y_OFFSET = 0.033


def emit_attribution_rows(attr_df: pd.DataFrame) -> str:
    """Render `build_attribution_table` output as Table tab:per_attack_auc rows.

    One line per attack (order: Sybil, Flooding, Blackhole, Wormhole),
    columns C2/Network/Physical in that order, each cell
    `${mean:.2f} \\pm {std:.2f}$`, the dominant manifold's cell wrapped in
    `\\mathbf{...}`, each line ending in ` \\\\`. Matches
    `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`'s
    `tab:per_attack_auc` row syntax exactly.
    """
    lines = []
    for attack in _ATTRIBUTION_ROW_ORDER:
        short_name = attack.replace(" Attack", "")
        sub = attr_df[attr_df["attack_class"] == attack].set_index("manifold")
        cells = []
        for manifold_key, _header in _ATTRIBUTION_COLUMNS:
            row = sub.loc[manifold_key]
            cell = f"{row['mean']:.2f} \\pm {row['std']:.2f}"
            if bool(row["dominant"]):
                cell = f"\\mathbf{{{cell}}}"
            cells.append(f"${cell}$")
        lines.append(f"{short_name} & {' & '.join(cells)} \\\\")
    return "\n".join(lines)


def emit_binary_pgfplots(bin_df: pd.DataFrame, scoring: str = "znorm") -> str:
    """Render `build_binary_auc_table` output as the Fig. fig:binary_auc pgfplots block.

    Filters `bin_df` to `scoring`, emits a `coordinates { ... };` block (one
    `({label}, {mean:.3f}) +- (0, {std:.3f})` per subset, in the paper's
    symbolic x-coordinate order C2, Network, Physical, C2+N, C2+P, N+P, All
    three), followed by the seven manual `\\node` value-label lines at
    y = mean + std + `_NODE_Y_OFFSET`, matching the paper's placement
    convention. See `_NODE_Y_OFFSET` for the sign-off caveat on that offset.
    """
    sub = bin_df[bin_df["scoring"] == scoring].set_index("subset")

    coord_lines = ["coordinates {"]
    for subset_key, label in _BINARY_SUBSET_ORDER:
        row = sub.loc[subset_key]
        coord_lines.append(f"    ({label}, {row['mean']:.3f}) +- (0, {row['std']:.3f})")
    coord_lines.append("};")

    node_lines = []
    for subset_key, label in _BINARY_SUBSET_ORDER:
        row = sub.loc[subset_key]
        y = row["mean"] + row["std"] + _NODE_Y_OFFSET
        node_lines.append(
            r"\node[font=\small, anchor=south, fill=white, inner sep=1pt] at "
            f"(axis cs:{label},{y:.3f}) {{{row['mean']:.2f}}};"
        )

    return "\n".join(coord_lines + node_lines)


# --- Manuscript-stats orchestrator -------------------------------------------
# Phase-3 (3-seed) comparison numbers, computed from build_binary_auc_table /
# build_attribution_table restricted to config.PROBE_SEEDS. Distinguished in
# MANUSCRIPT_STATS.md from the PUBLISHED 3-seed literals below, which are
# quoted (not recomputed) from paper/MULTI_SEED_VARIANCE.md.
PAPER_DIR = REPO_ROOT / "paper"
SNIPPETS_DIR = TABLES_DIR / "rebuild" / "paper_snippets"

# Published 3-seed literals, quoted verbatim from paper/MULTI_SEED_VARIANCE.md
# (Diagnostic C: seeds 42/7/123, test-Normal-substitute lineage, 5s w2_timeout
# for seeds 7/123) for side-by-side comparison ONLY. Not recomputed here.
PUBLISHED_3SEED_BINARY_AUC = {
    # subset -> (mean, std), raw scoring (Diagnostic C predates znorm scoring).
    "c2_only": (0.7488, 0.0269), "network_only": (0.7610, 0.0163),
    "physical_only": (0.6114, 0.0152), "c2_network": (0.8304, 0.0211),
    "c2_physical": (0.7542, 0.0339), "network_physical": (0.8594, 0.0043),
    "all_three": (0.8577, 0.0276),
}
PUBLISHED_3SEED_ATTRIBUTION = {
    # (attack, manifold) -> (mean, std), raw scoring.
    ("Blackhole Attack", "c2"): (0.4562, 0.0198),
    ("Blackhole Attack", "network"): (0.3285, 0.0172),
    ("Blackhole Attack", "physical"): (0.7880, 0.0162),
    ("Flooding Attack", "c2"): (0.6035, 0.0293),
    ("Flooding Attack", "network"): (0.7858, 0.0053),
    ("Flooding Attack", "physical"): (0.3815, 0.0090),
    ("Sybil Attack", "c2"): (0.6491, 0.0051),
    ("Sybil Attack", "network"): (0.8698, 0.0021),
    ("Sybil Attack", "physical"): (0.2031, 0.0114),
    ("Wormhole Attack", "c2"): (0.5400, 0.0270),
    ("Wormhole Attack", "network"): (0.2769, 0.0175),
    ("Wormhole Attack", "physical"): (0.7388, 0.0122),
}


def _fmt_mean_std_ci(mean: float, std: float, ci_lo: float, ci_hi: float) -> str:
    return f"{mean:.4f} ± {std:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]"


def _render_manuscript_stats(
    seeds: tuple[int, ...], B: int, w2_timeout: float,
    binary_df: pd.DataFrame, attribution_df: pd.DataFrame,
    binary_df_3seed: pd.DataFrame, attribution_df_3seed: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# MANUSCRIPT_STATS: Ten-Seed Bootstrap-CI Evaluation")
    lines.append("")
    lines.append(
        "_Generated by `uav-tda manuscript-report` (`uav_tda/manuscript.py`) "
        f"from `results/tables/rebuild/probe_distances_seed{{{','.join(str(s) for s in seeds)}}}.csv` "
        "and the matching `znorm_stats_seed*.json`, across "
        f"{len(seeds)} seeds ({', '.join(str(s) for s in seeds)})._"
    )
    lines.append("")

    # Section 1: header/config.
    lines.append("## 1. Configuration")
    lines.append("")
    lines.append(f"- Seeds ({len(seeds)}): {', '.join(str(s) for s in seeds)}")
    lines.append(f"- Bootstrap replicates: B = {B}")
    lines.append(f"- Wasserstein-2 per-call timeout: {w2_timeout}s")
    lines.append(
        "- CI method: cluster bootstrap over seeds, stratified per-seed "
        "resampling on multiclass label, 2.5th/97.5th percentile of the "
        "across-seed mean AUC over B replicates (`bootstrap_mean_auc_ci`)."
    )
    lines.append("")

    # Section 2: binary AUC raw+znorm with CIs.
    lines.append("## 2. Binary AUC (Normal vs any attack), raw + znorm, with bootstrap CIs")
    lines.append("")
    lines.append(f"Mean ± std across {len(seeds)} seeds, with 95% bootstrap CI on the mean.")
    lines.append("")
    lines.append("| Subset | Raw AUC (mean ± std [95% CI]) | Znorm AUC (mean ± std [95% CI]) |")
    lines.append("| :--- | ---: | ---: |")
    for subset in MANIFOLD_SUBSETS:
        raw_row = binary_df[(binary_df["subset"] == subset) & (binary_df["scoring"] == "raw")].iloc[0]
        z_row = binary_df[(binary_df["subset"] == subset) & (binary_df["scoring"] == "znorm")].iloc[0]
        lines.append(
            f"| {subset} "
            f"| {_fmt_mean_std_ci(raw_row['mean'], raw_row['std'], raw_row['ci_lo'], raw_row['ci_hi'])} "
            f"| {_fmt_mean_std_ci(z_row['mean'], z_row['std'], z_row['ci_lo'], z_row['ci_hi'])} |"
        )
    lines.append("")

    # Section 3: attribution table with CIs + dominance.
    lines.append("## 3. Per-attack manifold attribution, with bootstrap CIs")
    lines.append("")
    lines.append(
        "One-vs-rest AUC per (attack, manifold), raw scoring (per-manifold "
        "AUC is scoring-invariant; see `build_attribution_table`), with "
        "dominant (highest-mean) manifold marked."
    )
    lines.append("")
    lines.append("| Attack | Manifold | AUC (mean ± std [95% CI]) | Dominant? |")
    lines.append("| :--- | :--- | ---: | :---: |")
    for attack in _ATTRIBUTION_ROW_ORDER:
        for manifold_key, header in _ATTRIBUTION_COLUMNS:
            row = attribution_df[(attribution_df["attack_class"] == attack)
                                  & (attribution_df["manifold"] == manifold_key)].iloc[0]
            mark = "**yes**" if bool(row["dominant"]) else ""
            lines.append(
                f"| {attack} | {header} "
                f"| {_fmt_mean_std_ci(row['mean'], row['std'], row['ci_lo'], row['ci_hi'])} | {mark} |"
            )
    lines.append("")
    dominance_summary = {
        attack: attribution_df[(attribution_df["attack_class"] == attack)
                                & (attribution_df["dominant"])]["manifold"].iloc[0]
        for attack in _ATTRIBUTION_ROW_ORDER
    }
    lines.append(
        "Dominant-manifold summary (10-seed): "
        + "; ".join(f"{a.replace(' Attack', '')} -> {m}" for a, m in dominance_summary.items())
        + "."
    )
    lines.append("")

    # Section 4: comparison vs Phase-3 3-seed (recomputed) and vs published 3-seed (quoted).
    lines.append("## 4. Comparison: 10-seed vs Phase-3 3-seed (recomputed) vs published 3-seed (quoted)")
    lines.append("")
    lines.append(
        "\"Phase-3 3-seed (recomputed)\" restricts this same pipeline/code "
        "path to seeds 42, 7, 123 only (raw scoring), so it is directly "
        "comparable to the 10-seed column. \"Published 3-seed (quoted)\" is "
        "reproduced **as a literal**, not recomputed, from "
        "`paper/MULTI_SEED_VARIANCE.md` (Diagnostic C: seeds 42/7/123, "
        "test-Normal-substitute lineage, 5s w2_timeout for seeds 7/123) — "
        "the numbers currently in the published extended abstract."
    )
    lines.append("")
    lines.append("### 4a. Binary AUC (raw scoring)")
    lines.append("")
    lines.append("| Subset | 10-seed (mean ± std) | Phase-3 3-seed, recomputed (mean ± std) | Published 3-seed, quoted (mean ± std) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for subset in MANIFOLD_SUBSETS:
        ten_row = binary_df[(binary_df["subset"] == subset) & (binary_df["scoring"] == "raw")].iloc[0]
        three_row = binary_df_3seed[(binary_df_3seed["subset"] == subset)
                                     & (binary_df_3seed["scoring"] == "raw")].iloc[0]
        pub_mean, pub_std = PUBLISHED_3SEED_BINARY_AUC[subset]
        lines.append(
            f"| {subset} | {ten_row['mean']:.4f} ± {ten_row['std']:.4f} "
            f"| {three_row['mean']:.4f} ± {three_row['std']:.4f} "
            f"| {pub_mean:.4f} ± {pub_std:.4f} |"
        )
    lines.append("")
    lines.append("### 4b. Per-attack dominant-manifold AUC (raw scoring)")
    lines.append("")
    lines.append("| Attack | Manifold | 10-seed (mean ± std) | Phase-3 3-seed, recomputed (mean ± std) | Published 3-seed, quoted (mean ± std) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: |")
    for attack in _ATTRIBUTION_ROW_ORDER:
        for manifold_key, header in _ATTRIBUTION_COLUMNS:
            ten_row = attribution_df[(attribution_df["attack_class"] == attack)
                                      & (attribution_df["manifold"] == manifold_key)].iloc[0]
            three_row = attribution_df_3seed[(attribution_df_3seed["attack_class"] == attack)
                                              & (attribution_df_3seed["manifold"] == manifold_key)].iloc[0]
            pub_mean, pub_std = PUBLISHED_3SEED_ATTRIBUTION[(attack, manifold_key)]
            lines.append(
                f"| {attack} | {header} | {ten_row['mean']:.4f} ± {ten_row['std']:.4f} "
                f"| {three_row['mean']:.4f} ± {three_row['std']:.4f} "
                f"| {pub_mean:.4f} ± {pub_std:.4f} |"
            )
    lines.append("")

    # Section 5: .tex snippet usage note.
    lines.append("## 5. LaTeX snippet drop-in usage note")
    lines.append("")
    lines.append(
        "`results/tables/rebuild/paper_snippets/attribution_table_rows.tex` "
        "(from `emit_attribution_rows`) contains candidate replacement rows "
        "for Table `tab:per_attack_auc`'s tabular body in "
        "`CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`, and "
        "`results/tables/rebuild/paper_snippets/binary_auc_pgfplots.tex` "
        "(from `emit_binary_pgfplots`, znorm scoring) contains a candidate "
        "replacement `coordinates {...}` + `\\node` block for Fig. "
        "`fig:binary_auc`'s pgfplots axis. Both are generated fresh from the "
        "10-seed bootstrap-CI tables above at report-build time, so the "
        "manuscript text and the code that produced it cannot silently "
        "drift apart. **These are candidate snippets pending author "
        "sign-off** — the 10-seed numbers, the switch to znorm scoring, and "
        "the `_NODE_Y_OFFSET` label-placement convention are all authorial "
        "decisions this report does not make; do not paste them into "
        "`main.tex` without review."
    )
    lines.append("")

    return "\n".join(lines)


def build_manuscript_stats(
    seeds: tuple[int, ...] = None,
    B: int = 2000,
    w2_timeout: float = 30.0,
    bootstrap_seed: int = 0,
    rebuild_dir: Path | None = None,
) -> None:
    """Ensure `seeds`' artifacts, build tables, and write all Phase-4 outputs.

    Orchestrates: ensure seeds (`run_missing_seeds`) -> load frames -> build
    binary/attribution tables -> write `binary_auc.csv` /
    `manifold_attribution.csv` (+ provenance) -> write the two `.tex`
    snippets under `results/tables/rebuild/paper_snippets/` -> write
    `paper/MANUSCRIPT_STATS.md`. Also recomputes the Phase-3-comparable
    3-seed (42, 7, 123) tables via this same code path for the report's
    §4 comparison section. Never overwrites another seed's already-present
    rebuild/ artifacts (delegated to `run_missing_seeds`/`missing_seeds`).
    """
    from .config import MANUSCRIPT_SEEDS, PROBE_SEEDS

    if seeds is None:
        seeds = MANUSCRIPT_SEEDS
    rebuild = rebuild_dir if rebuild_dir is not None else (TABLES_DIR / "rebuild")
    snippets_dir = rebuild / "paper_snippets"
    # The report goes to the repo's paper/ tree whenever `rebuild` is the
    # canonical rebuild location (whether defaulted or passed explicitly, as
    # the CLI does); a redirected `rebuild_dir` (tests) instead gets a
    # sibling paper/ dir so tests never touch the real paper/ tree.
    paper_dir = PAPER_DIR if rebuild == (TABLES_DIR / "rebuild") else (rebuild.parent / "paper")

    run_missing_seeds(seeds, w2_timeout=w2_timeout)

    seed_frames = load_seed_frames(seeds, rebuild)
    binary_df = build_binary_auc_table(seed_frames, B=B, bootstrap_seed=bootstrap_seed)
    attribution_df = build_attribution_table(seed_frames, B=B, bootstrap_seed=bootstrap_seed)

    rebuild.mkdir(parents=True, exist_ok=True)
    binary_out = rebuild / "binary_auc.csv"
    attribution_out = rebuild / "manifold_attribution.csv"
    binary_df.to_csv(binary_out, index=False)
    write_provenance(binary_out, {"seeds": list(seeds), "bootstrap": B, "w2_timeout": w2_timeout})
    attribution_df.to_csv(attribution_out, index=False)
    write_provenance(attribution_out, {"seeds": list(seeds), "bootstrap": B, "w2_timeout": w2_timeout})

    # Phase-3-comparable 3-seed tables, recomputed through this same code
    # path (not quoted), restricted to seed_frames already loaded above.
    three_seed_frames = {s: seed_frames[s] for s in PROBE_SEEDS if s in seed_frames}
    binary_df_3seed = build_binary_auc_table(three_seed_frames, B=B, bootstrap_seed=bootstrap_seed)
    attribution_df_3seed = build_attribution_table(three_seed_frames, B=B, bootstrap_seed=bootstrap_seed)

    snippets_dir.mkdir(parents=True, exist_ok=True)
    attribution_tex = snippets_dir / "attribution_table_rows.tex"
    attribution_tex.write_text(emit_attribution_rows(attribution_df) + "\n")
    binary_tex = snippets_dir / "binary_auc_pgfplots.tex"
    binary_tex.write_text(emit_binary_pgfplots(binary_df, scoring="znorm") + "\n")

    report_text = _render_manuscript_stats(
        seeds, B, w2_timeout, binary_df, attribution_df, binary_df_3seed, attribution_df_3seed)
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "MANUSCRIPT_STATS.md").write_text(report_text)
