"""`uav-tda` command-line entry point (probe subcommand)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import metrics, probe
from .config import PROBE_DELTA, PROBE_PER_CLASS, PROBE_TOP_K
from .paths import TABLES_DIR
from .provenance import write_provenance


def _cmd_probe(args: argparse.Namespace) -> int:
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
    p.set_defaults(func=_cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
