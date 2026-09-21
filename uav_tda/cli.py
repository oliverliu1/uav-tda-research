"""`uav-tda` command-line entry point (probe + pipeline-phase subcommands)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data, evaluate, features, metrics, probe, supervised, unsupervised
from . import tda as tda_module
from .config import PROBE_DELTA, PROBE_PER_CLASS, PROBE_TOP_K
from .paths import TABLES_DIR
from .provenance import write_provenance
from .workspace import Workspace

ZNORM_SEEDS = (42, 7, 123)


def _cmd_probe(args: argparse.Namespace) -> int:
    if args.znorm:
        return _run_znorm_probe(args.seed, args.w2_timeout)

    df = probe.run_probe(
        seed=args.seed, per_class=args.per_class, top_k=args.top_k,
        delta=args.delta, w2_timeout=args.w2_timeout,
    )
    # Fresh runs go to results/tables/rebuild/ — NEVER overwrite the published
    # results/tables/probe_distances*.csv oracles.
    out = Path(args.out) if args.out else (
        TABLES_DIR / "rebuild" / f"probe_distances_seed{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_provenance(out, {
        "seed": args.seed, "per_class": args.per_class, "top_k": args.top_k,
        "delta": args.delta, "w2_timeout": args.w2_timeout,
    })
    for subset, auc in metrics.binary_auc_by_subset(df).items():
        print(f"  {subset:20s} binary AUC = {auc:.4f}")
    print(f"wrote {out}")
    return 0


def _rebuild_paths(seed: int) -> dict[str, Path]:
    rebuild = TABLES_DIR / "rebuild"
    return {
        "raw": rebuild / f"probe_distances_seed{seed}.csv",
        "znorm": rebuild / f"probe_distances_seed{seed}_znorm.csv",
        "val": rebuild / f"val_normal_distances_seed{seed}.csv",
        "stats": rebuild / f"znorm_stats_seed{seed}.json",
    }


def _run_znorm_probe(seed: int, w2_timeout: float | None,
                      per_class: int = 200, top_k: int = 50,
                      delta: float = 0.2) -> int:
    """Run the coupled test+val znorm probe for one seed; write rebuild/ artifacts."""
    paths = _rebuild_paths(seed)
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

    raw_auc = metrics.binary_auc_by_subset(test_df)
    znorm_auc = metrics.binary_auc_by_subset_znorm(test_df, stats)
    for subset in raw_auc:
        print(f"  {subset:20s} raw AUC = {raw_auc[subset]:.4f}  "
              f"znorm AUC = {znorm_auc[subset]:.4f}")
    print(f"wrote {paths['raw']}, {paths['znorm']}, {paths['val']}, {paths['stats']}")
    return 0


def _cmd_znorm_report(args: argparse.Namespace) -> int:
    from . import znorm_report
    znorm_report.build_report()
    return 0


def _workspace_for(args: argparse.Namespace) -> Workspace:
    if getattr(args, "root", None):
        return Workspace.at(Path(args.root))
    return Workspace.default()


def _cmd_prep(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    data.run_prep(ws, debug=args.debug, seed=args.seed)
    return 0


def _cmd_tda(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    tda_module.run_tda(
        ws, manifold=args.manifold, split=args.split, seed=args.seed,
        debug=args.debug, n_jobs=args.n_jobs,
    )
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    features.run_features(ws, debug=args.debug)
    return 0


def _cmd_supervised(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    supervised.run_supervised(ws, debug=args.debug)
    return 0


def _cmd_unsupervised(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    unsupervised.run_unsupervised(ws, debug=args.debug, n_jobs=args.n_jobs)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    evaluate.run_evaluate(ws, debug=args.debug)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    ws = _workspace_for(args)
    ws.ensure()
    data.run_prep(ws, debug=args.debug, seed=args.seed)
    tda_module.run_tda(
        ws, manifold=args.manifold, split=args.split, seed=args.seed,
        debug=args.debug, n_jobs=args.n_jobs,
    )
    features.run_features(ws, debug=args.debug)
    supervised.run_supervised(ws, debug=args.debug)
    unsupervised.run_unsupervised(ws, debug=args.debug, n_jobs=args.n_jobs)
    evaluate.run_evaluate(ws, debug=args.debug)
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--debug", action="store_true")
    p.add_argument("--root", type=str, default=None,
                    help="Workspace root (default: repo root via Workspace.default()).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uav-tda")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="Run the unsupervised Wasserstein-2 probe.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-class", type=int, default=PROBE_PER_CLASS)
    p.add_argument("--top-k", type=int, default=PROBE_TOP_K)
    p.add_argument("--delta", type=float, default=PROBE_DELTA)
    p.add_argument("--w2-timeout", type=float, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--znorm", action="store_true",
                    help="Run the coupled test+val Z-normalized probe (paper Sec III.E) "
                         "instead of the raw sum-scored probe.")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("znorm-report", help="Build the 3-seed znorm-vs-raw reanalysis report.")
    p.set_defaults(func=_cmd_znorm_report)

    p = sub.add_parser("prep", help="Phase 2: data prep and splits.")
    _add_common_args(p)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_prep)

    p = sub.add_parser("tda", help="Phase 3: persistence diagrams per manifold/split.")
    _add_common_args(p)
    p.add_argument("--manifold", choices=["c2", "network", "physical", "all"], default="all")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.set_defaults(func=_cmd_tda)

    p = sub.add_parser("features", help="Phase 4: featurize persistence diagrams.")
    _add_common_args(p)
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("supervised", help="Phase 5: supervised classification.")
    _add_common_args(p)
    p.set_defaults(func=_cmd_supervised)

    p = sub.add_parser("unsupervised", help="Phase 6: unsupervised analysis.")
    _add_common_args(p)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.set_defaults(func=_cmd_unsupervised)

    p = sub.add_parser("evaluate", help="Phase 7: evaluation and ablations.")
    _add_common_args(p)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("all", help="Run all six phases in order.")
    _add_common_args(p)
    p.add_argument("--manifold", choices=["c2", "network", "physical", "all"], default="all")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
