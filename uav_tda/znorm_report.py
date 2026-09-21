"""Build the three-seed raw-vs-znorm reanalysis report (paper §III.E / §V).

Reads the per-seed rebuild/ CSVs produced by ``uav-tda probe --znorm``
(running any missing seed at ``w2_timeout=30.0`` first), aggregates raw and
Z-normalized binary/per-attack AUCs across seeds (42, 7, 123), and writes:

- ``results/tables/rebuild/znorm_summary.csv`` — per subset x {raw, znorm}:
  mean/std of binary AUC over the 3 seeds.
- ``results/tables/rebuild/znorm_per_attack.csv`` — per (attack, manifold) x
  {raw, znorm}: mean/std of one-vs-rest AUC over the 3 seeds, plus the
  dominant (highest-mean) manifold under each scoring.
- ``paper/ZNORM_RESULTS.md`` — the five-section markdown report. Every number
  in the report tables is read back out of the two CSVs above at report-build
  time; nothing is hand-typed except the historical comparison literals
  (explicitly called out as such, sourced from paper/MULTI_SEED_VARIANCE.md).

This module does not touch pipeline.py, tools/, data/, outputs/, or any
existing paper/*.md file.
"""
from __future__ import annotations

import json

import pandas as pd

from . import metrics
from .paths import REPO_ROOT, TABLES_DIR

PAPER_DIR = REPO_ROOT / "paper"
REBUILD_DIR = TABLES_DIR / "rebuild"
SEEDS = (42, 7, 123)
W2_TIMEOUT = 30.0

# --- Historical literals, quoted (not recomputed) for side-by-side comparison
# in the report. Sourced from paper/MULTI_SEED_VARIANCE.md (Diagnostic C,
# test-Normal-substitute lineage, --w2-timeout=5s for seeds 7/123) and
# paper/C2_NORMALIZATION_DIAGNOSTIC.md (Diagnostic B, single-seed=42,
# test-Normal-substitute znorm). These are the paper's PUBLISHED numbers,
# reproduced here as literals for comparison only; they are not derived from
# any CSV this module reads.
HISTORICAL_BINARY_AUC_RAW_MEAN = {
    "c2_only": 0.7488, "network_only": 0.7610, "physical_only": 0.6114,
    "c2_network": 0.8304, "c2_physical": 0.7542, "network_physical": 0.8594,
    "all_three": 0.8577,
}
HISTORICAL_BINARY_AUC_RAW_STD = {
    "c2_only": 0.0269, "network_only": 0.0163, "physical_only": 0.0152,
    "c2_network": 0.0211, "c2_physical": 0.0339, "network_physical": 0.0043,
    "all_three": 0.0276,
}
HISTORICAL_ALL_THREE_ZNORM_SEED42 = 0.8983  # Diagnostic B, single-seed, test-Normal substitute
HISTORICAL_ALL_THREE_RAW_SEED42 = 0.8712  # Diagnostic B, single-seed, test-Normal substitute
HISTORICAL_SEED123_TIMEOUT_SEC = 5
HISTORICAL_SEED123_C2_TIMEOUTS = 139
HISTORICAL_SEED123_TOTAL_C2_CALLS = 3000  # 3 dims x 1000 candidate W2 calls (approx., per note)


def _rebuild_paths(seed: int) -> dict:
    return {
        "raw": REBUILD_DIR / f"probe_distances_seed{seed}.csv",
        "znorm": REBUILD_DIR / f"probe_distances_seed{seed}_znorm.csv",
        "val": REBUILD_DIR / f"val_normal_distances_seed{seed}.csv",
        "stats": REBUILD_DIR / f"znorm_stats_seed{seed}.json",
    }


def _seed_artifacts_present(seed: int) -> bool:
    return all(p.exists() for p in _rebuild_paths(seed).values())


def _run_missing_seed(seed: int) -> None:
    """Run the znorm probe for one seed via the CLI's own code path."""
    from . import cli  # noqa: PLC0415 (avoid import cycle at module load)

    cli._run_znorm_probe(seed=seed, w2_timeout=W2_TIMEOUT)


def _load_seed(seed: int) -> dict:
    paths = _rebuild_paths(seed)
    raw_df = pd.read_csv(paths["raw"])
    stats_payload = json.loads(paths["stats"].read_text())
    stats = {m: (v["mean"], v["std"]) for m, v in stats_payload["stats"].items()}
    return {
        "raw_df": raw_df,
        "stats": stats,
        "timeout_counts": stats_payload["timeout_counts"],
        "params": stats_payload["params"],
    }


def _binary_summary(per_seed: dict) -> pd.DataFrame:
    rows = []
    for seed, data in per_seed.items():
        raw_auc = metrics.binary_auc_by_subset(data["raw_df"])
        znorm_auc = metrics.binary_auc_by_subset_znorm(data["raw_df"], data["stats"])
        for subset in raw_auc:
            rows.append({"seed": seed, "subset": subset, "scoring": "raw", "auc": raw_auc[subset]})
            rows.append({"seed": seed, "subset": subset, "scoring": "znorm", "auc": znorm_auc[subset]})
    long_df = pd.DataFrame(rows)
    summary = (long_df.groupby(["subset", "scoring"])["auc"]
               .agg(["mean", "std"]).reset_index()
               .sort_values(["subset", "scoring"]))
    return summary


def _per_attack_summary(per_seed: dict) -> pd.DataFrame:
    rows = []
    for seed, data in per_seed.items():
        raw_pa = metrics.per_attack_auc(data["raw_df"])
        raw_pa["scoring"] = "raw"
        znorm_df = metrics.apply_znorm(data["raw_df"], data["stats"])
        z_pa = metrics.per_attack_auc(znorm_df)
        z_pa["scoring"] = "znorm"
        for pa in (raw_pa, z_pa):
            pa["seed"] = seed
            rows.append(pa)
    long_df = pd.concat(rows, ignore_index=True)
    summary = (long_df.groupby(["attack_class", "manifold", "scoring"])["auc"]
               .agg(["mean", "std"]).reset_index())

    dominant_rows = []
    for (attack, scoring), grp in summary.groupby(["attack_class", "scoring"]):
        top = grp.loc[grp["mean"].idxmax()]
        dominant_rows.append({
            "attack_class": attack, "scoring": scoring,
            "dominant_manifold": top["manifold"], "dominant_mean_auc": top["mean"],
        })
    dominant = pd.DataFrame(dominant_rows)
    summary = summary.merge(dominant, on=["attack_class", "scoring"], how="left")
    summary["is_dominant"] = summary["manifold"] == summary["dominant_manifold"]
    return summary.sort_values(["attack_class", "scoring", "manifold"])


def _timeout_table(per_seed: dict) -> pd.DataFrame:
    rows = []
    for seed, data in per_seed.items():
        tc = data["timeout_counts"]
        rows.append({
            "seed": seed,
            "w2_timeout_sec": data["params"].get("w2_timeout"),
            "n_timeouts_c2": tc.get("n_timeouts_c2", 0),
            "n_timeouts_network": tc.get("n_timeouts_network", 0),
            "n_timeouts_physical": tc.get("n_timeouts_physical", 0),
        })
    return pd.DataFrame(rows).sort_values("seed")


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _render_report(binary_summary: pd.DataFrame, attack_summary: pd.DataFrame,
                    timeout_df: pd.DataFrame, per_seed: dict) -> str:
    lines: list[str] = []
    lines.append("# ZNORM_RESULTS: Clean-Lineage Z-Normalized Scoring Reanalysis")
    lines.append("")
    lines.append(
        "_Generated by `uav-tda znorm-report` "
        f"(`uav_tda/znorm_report.py`) from `results/tables/rebuild/probe_distances_seed{{{','.join(str(s) for s in SEEDS)}}}.csv`, "
        "the matching `_znorm` and `val_normal_distances_*` CSVs, and "
        "`znorm_stats_seed*.json`, across seeds "
        f"{', '.join(str(s) for s in SEEDS)}._"
    )
    lines.append("")

    # Section 1: header / supersession note
    lines.append("## 1. Supersession note")
    lines.append("")
    lines.append(
        "This report **supersedes** "
        "[`paper/C2_NORMALIZATION_DIAGNOSTIC.md`](C2_NORMALIZATION_DIAGNOSTIC.md) "
        "(Diagnostic B) for the Z-normalized-scoring question. Diagnostic B "
        "substituted the 200 *test*-split Normal-Traffic flows present in "
        "`probe_distances.csv` as the empirical Normal reference for "
        "Z-normalization, because the original probe only computed W_2 on "
        "the test split (see that file's own Methodology note). This report "
        "instead uses `metrics.znorm_stats_from_val` computed on the full "
        "3,926-flow **validation**-split Normal-Traffic population, coupled "
        "to the test-split scoring pass via a single shared sparse-Rips "
        "baseline realization per seed (`probe.run_probe_with_znorm`, "
        "paper §III.E). It also extends the Hera Wasserstein-2 timeout from "
        "the 5s used for seeds 7/123 in "
        "[`paper/MULTI_SEED_VARIANCE.md`](MULTI_SEED_VARIANCE.md) "
        "(Diagnostic C) to 30s, to test whether the seed-123 C2 timeout "
        "spike was a compute-budget artifact. `paper/C2_NORMALIZATION_DIAGNOSTIC.md` "
        "and `paper/MULTI_SEED_VARIANCE.md` are preserved unmodified as the "
        "historical record; only the literal comparison numbers quoted in "
        "§4-5 below are drawn from them."
    )
    lines.append("")

    # Section 2: raw vs znorm binary AUC table
    lines.append("## 2. Raw vs Z-normalized binary AUC (3-seed mean ± std)")
    lines.append("")
    lines.append(
        "Normal-vs-any-attack AUC, all 7 manifold subsets, seeds "
        f"{', '.join(str(s) for s in SEEDS)} (val-Normal Z-normalization, "
        f"w2_timeout={W2_TIMEOUT}s)."
    )
    lines.append("")
    lines.append("| Subset | Raw AUC (mean ± std) | Znorm AUC (mean ± std) | Delta (znorm - raw) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for subset in metrics.MANIFOLD_SUBSETS:
        raw_row = binary_summary[(binary_summary["subset"] == subset)
                                  & (binary_summary["scoring"] == "raw")].iloc[0]
        z_row = binary_summary[(binary_summary["subset"] == subset)
                                & (binary_summary["scoring"] == "znorm")].iloc[0]
        delta = z_row["mean"] - raw_row["mean"]
        lines.append(
            f"| {subset} | {_fmt_mean_std(raw_row['mean'], raw_row['std'])} "
            f"| {_fmt_mean_std(z_row['mean'], z_row['std'])} | {delta:+.4f} |"
        )
    lines.append("")

    # Section 3: per-attack attribution under both scorings
    lines.append("## 3. Per-attack dominant-manifold attribution (both scorings)")
    lines.append("")
    lines.append(
        "One-vs-rest AUC per (attack, manifold), 3-seed mean ± std, with the "
        "dominant (highest-mean) manifold marked, under raw and znorm scoring."
    )
    lines.append("")
    lines.append("| Attack | Scoring | Manifold | AUC (mean ± std) | Dominant? |")
    lines.append("| :--- | :--- | :--- | ---: | :---: |")
    for _, row in attack_summary.iterrows():
        mark = "**yes**" if row["is_dominant"] else ""
        lines.append(
            f"| {row['attack_class']} | {row['scoring']} | {row['manifold']} "
            f"| {_fmt_mean_std(row['mean'], row['std'])} | {mark} |"
        )
    lines.append("")
    lines.append("Dominant-manifold summary (highest-mean-AUC manifold per attack, per scoring):")
    lines.append("")
    lines.append("| Attack | Raw-dominant | Znorm-dominant |")
    lines.append("| :--- | :--- | :--- |")
    for attack in sorted(attack_summary["attack_class"].unique()):
        raw_dom = attack_summary[(attack_summary["attack_class"] == attack)
                                  & (attack_summary["scoring"] == "raw")
                                  & (attack_summary["is_dominant"])]["manifold"].iloc[0]
        z_dom = attack_summary[(attack_summary["attack_class"] == attack)
                                & (attack_summary["scoring"] == "znorm")
                                & (attack_summary["is_dominant"])]["manifold"].iloc[0]
        lines.append(f"| {attack} | {raw_dom} | {z_dom} |")
    lines.append("")

    # Section 4: seed-123 timeout reanalysis
    lines.append("## 4. Seed-123 timeout reanalysis (30s vs historical 5s)")
    lines.append("")
    lines.append("W_2-call timeout counts per manifold, this reanalysis (30s timeout, val+test passes combined):")
    lines.append("")
    lines.append("| Seed | w2_timeout (s) | C2 timeouts | Network timeouts | Physical timeouts |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for _, row in timeout_df.iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['w2_timeout_sec']:.0f} | {int(row['n_timeouts_c2'])} "
            f"| {int(row['n_timeouts_network'])} | {int(row['n_timeouts_physical'])} |"
        )
    lines.append("")
    seed123_row = timeout_df[timeout_df["seed"] == 123].iloc[0]
    seed123_c2_this = per_seed[123]
    c2_auc_123_raw = metrics.binary_auc_by_subset(seed123_c2_this["raw_df"])["c2_only"]
    c2_auc_123_znorm = metrics.binary_auc_by_subset_znorm(
        seed123_c2_this["raw_df"], seed123_c2_this["stats"])["c2_only"]
    lines.append(
        f"At `--w2-timeout {W2_TIMEOUT:.0f}`, seed 123 recorded "
        f"**{int(seed123_row['n_timeouts_c2'])} C2 timeouts** "
        f"(network={int(seed123_row['n_timeouts_network'])}, "
        f"physical={int(seed123_row['n_timeouts_physical'])}), versus the "
        f"historical **{HISTORICAL_SEED123_C2_TIMEOUTS} C2 timeouts** reported in "
        f"`paper/MULTI_SEED_VARIANCE.md` (Diagnostic C) at a "
        f"{HISTORICAL_SEED123_TIMEOUT_SEC}s timeout on the old test-Normal-substitute "
        f"lineage (~{HISTORICAL_SEED123_C2_TIMEOUTS / HISTORICAL_SEED123_TOTAL_C2_CALLS:.1%} "
        "of candidate C2 W_2 calls). The resulting seed-123 c2_only binary AUC "
        f"under this reanalysis is **{c2_auc_123_raw:.4f}** (raw) / "
        f"**{c2_auc_123_znorm:.4f}** (znorm) on the clean val-Normal lineage."
    )
    lines.append("")

    # Section 5: numbers for the revisable abstract
    lines.append("## 5. Numbers available for the revisable abstract")
    lines.append("")
    lines.append(
        "The following are the refreshed, clean-lineage 3-seed numbers from "
        "this report, laid alongside the paper's currently **published** "
        "numbers (quoted as literals from `paper/MULTI_SEED_VARIANCE.md` "
        "Diagnostic C and `paper/C2_NORMALIZATION_DIAGNOSTIC.md` Diagnostic B "
        "— test-Normal-substitute lineage, 5s timeout for seeds 7/123, "
        "**not** recomputed here) for the authors to compare before revising "
        "the abstract. These are candidates only; author sign-off required "
        "before use in the manuscript."
    )
    lines.append("")
    lines.append("| Subset | Published raw AUC (mean ± std) | This report: raw AUC (mean ± std) | This report: znorm AUC (mean ± std) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for subset in metrics.MANIFOLD_SUBSETS:
        pub_mean = HISTORICAL_BINARY_AUC_RAW_MEAN[subset]
        pub_std = HISTORICAL_BINARY_AUC_RAW_STD[subset]
        raw_row = binary_summary[(binary_summary["subset"] == subset)
                                  & (binary_summary["scoring"] == "raw")].iloc[0]
        z_row = binary_summary[(binary_summary["subset"] == subset)
                                & (binary_summary["scoring"] == "znorm")].iloc[0]
        lines.append(
            f"| {subset} | {_fmt_mean_std(pub_mean, pub_std)} "
            f"| {_fmt_mean_std(raw_row['mean'], raw_row['std'])} "
            f"| {_fmt_mean_std(z_row['mean'], z_row['std'])} |"
        )
    lines.append("")
    lines.append(
        f"For reference, Diagnostic B's single-seed (seed=42), test-Normal-substitute "
        f"znorm all_three AUC was **{HISTORICAL_ALL_THREE_ZNORM_SEED42:.4f}** "
        f"(raw {HISTORICAL_ALL_THREE_RAW_SEED42:.4f}); this report's 3-seed "
        f"all_three znorm mean is **"
        f"{binary_summary[(binary_summary['subset'] == 'all_three') & (binary_summary['scoring'] == 'znorm')]['mean'].iloc[0]:.4f}"
        "** (val-Normal lineage)."
    )
    lines.append("")

    return "\n".join(lines)


def build_report() -> None:
    """Run any missing seeds, then write znorm_summary.csv, znorm_per_attack.csv, ZNORM_RESULTS.md."""
    for seed in SEEDS:
        if not _seed_artifacts_present(seed):
            _run_missing_seed(seed)

    per_seed = {seed: _load_seed(seed) for seed in SEEDS}

    binary_summary = _binary_summary(per_seed)
    attack_summary = _per_attack_summary(per_seed)
    timeout_df = _timeout_table(per_seed)

    REBUILD_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REBUILD_DIR / "znorm_summary.csv"
    binary_summary.to_csv(summary_path, index=False)

    per_attack_path = REBUILD_DIR / "znorm_per_attack.csv"
    attack_summary.to_csv(per_attack_path, index=False)

    report_text = _render_report(binary_summary, attack_summary, timeout_df, per_seed)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "ZNORM_RESULTS.md").write_text(report_text)

    print(f"wrote {summary_path}")
    print(f"wrote {per_attack_path}")
    print(f"wrote {PAPER_DIR / 'ZNORM_RESULTS.md'}")


if __name__ == "__main__":
    build_report()
