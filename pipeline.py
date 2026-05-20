"""Multi-manifold persistent-homology pipeline for UAV intrusion detection on UAVIDS-2025.

Single-file orchestration for the AIAA SciTech 2027 paper. Each phase is exposed as an
argparse subcommand. Run ``python pipeline.py --help`` to see the available commands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable


# ==== SECTION 1: PATHS ====

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "UAVIDS-2025.csv"

OUTPUTS_DIR = REPO_ROOT / "outputs"
PERSISTENCE_DIR = OUTPUTS_DIR / "persistence_diagrams"
TDA_FEATURES_DIR = OUTPUTS_DIR / "tda_features"

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

LOGS_DIR = REPO_ROOT / "logs"


# ==== SECTION 2: DATASET SCHEMA ====

EXPECTED_COLUMNS = (
    "FlowID", "FlowDuration/s", "SrcAddr", "SrcPort", "DstAddr", "DstPort",
    "Protocol", "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps", "MeanPacketSize",
    "PacketDropRate", "AverageHopCount", "label",
)
EXPECTED_CLASSES = (
    "Normal Traffic", "Blackhole Attack", "Wormhole Attack",
    "Sybil Attack", "Flooding Attack",
)
DROP_COLUMNS = ("FlowID", "Protocol")
LABEL_COLUMN = "label"

# Known port values in the dataset; encoded as one-hot binary columns.
KNOWN_PORTS = (9, 654)


# ==== SECTION 3: MANIFOLDS ====
# Three disjoint functional categories per the UAVIDS-2025 paper.
# Categorical columns (addresses, ports) appear here in their post-encoding names.

C2_FEATURES = (
    "SrcAddr_last_octet",
    "SrcPort_9", "SrcPort_654",
    "DstAddr_last_octet",
    "DstPort_9", "DstPort_654",
    "FlowDuration/s",
)
NETWORK_FEATURES = (
    "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanPacketSize",
)
PHYSICAL_FEATURES = (
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps",
    "PacketDropRate", "AverageHopCount",
)
MANIFOLDS = {
    "c2": C2_FEATURES,
    "network": NETWORK_FEATURES,
    "physical": PHYSICAL_FEATURES,
}


# ==== SECTION 4: PIPELINE CONFIG ====

# Stratified split fractions.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# TDA reference-set settings.
REFERENCE_CLOUD_SIZE = 500          # k-medoid sample of normal training flows
QUERY_NEIGHBORHOOD_SIZE = 50        # subset drawn per query for windowing
MAX_EDGE_PERCENTILE = 95            # percentile of pairwise distances within reference cloud

# Cap homology dimensions per manifold (Physical is 5D; H2 there is noise).
MAX_HOM_DIM = {"c2": 2, "network": 2, "physical": 1}

# Persistence images.
PI_RESOLUTION = (20, 20)

# Hyperparameter grids (Phase 5).
LR_C_GRID = (0.1, 1.0, 10.0)
RF_N_ESTIMATORS_GRID = (100, 300)
RF_MAX_DEPTH_GRID = (10, 20, None)
SVM_C_GRID = (0.1, 1.0, 10.0)
SVM_GAMMA_GRID = ("scale", "auto")
GRID_CV_FOLDS = 3

# Curated RF feature subset (Phase 5, item 14).
TOP_K_RF_IMPORTANCE = 30
TOP_K_MUTUAL_INFO = 30

# Reproducibility.
PRIMARY_SEED = 42
SEEDS = (42, 7, 2024)

# Unsupervised thresholding (Phase 6).
THRESHOLD_PERCENTILE = 95

# Debug mode loads only this many rows of the CSV.
DEBUG_SAMPLE_ROWS = 5000


# ==== SECTION 5: ENVIRONMENT SETUP ====

ALL_DIRS = (
    OUTPUTS_DIR, PERSISTENCE_DIR, TDA_FEATURES_DIR,
    RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR,
    LOGS_DIR,
)


def ensure_directories() -> None:
    """Create every output directory used by the pipeline."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure root logging to stdout and a timestamped logfile."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"pipeline_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    log = logging.getLogger("pipeline")
    log.info("logging to %s", log_path)
    return log


# ==== SECTION 6: SUBCOMMAND STUBS ====
# Each phase is implemented in its own section in later commits. For now the
# subcommands log a "not implemented yet" message so the CLI is exercisable.


def cmd_prep(args: argparse.Namespace) -> None:
    """Phase 2: load, split, scale, save per-manifold CSVs."""
    logging.getLogger("pipeline.prep").info(
        "prep: not implemented yet (debug=%s)", args.debug
    )


def cmd_tda(args: argparse.Namespace) -> None:
    """Phase 3: compute per-flow persistence diagrams."""
    logging.getLogger("pipeline.tda").info(
        "tda: not implemented yet (manifold=%s, split=%s, seed=%s, debug=%s)",
        args.manifold, args.split, args.seed, args.debug,
    )


def cmd_features(args: argparse.Namespace) -> None:
    """Phase 4: summary stats and persistence images from diagrams."""
    logging.getLogger("pipeline.features").info(
        "features: not implemented yet (debug=%s)", args.debug
    )


def cmd_supervised(args: argparse.Namespace) -> None:
    """Phase 5: classifier training, tuning, and evaluation."""
    logging.getLogger("pipeline.supervised").info(
        "supervised: not implemented yet (debug=%s)", args.debug
    )


def cmd_unsupervised(args: argparse.Namespace) -> None:
    """Phase 6: Wasserstein-distance anomaly detection."""
    logging.getLogger("pipeline.unsupervised").info(
        "unsupervised: not implemented yet (debug=%s)", args.debug
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Phase 7: ablations, final tables, and paper figures."""
    logging.getLogger("pipeline.evaluate").info(
        "evaluate: not implemented yet (debug=%s)", args.debug
    )


def cmd_all(args: argparse.Namespace) -> None:
    """Run every phase in sequence."""
    logging.getLogger("pipeline.all").info(
        "all: not implemented yet (debug=%s)", args.debug
    )


COMMAND_DISPATCH: dict[str, Callable[[argparse.Namespace], None]] = {
    "prep": cmd_prep,
    "tda": cmd_tda,
    "features": cmd_features,
    "supervised": cmd_supervised,
    "unsupervised": cmd_unsupervised,
    "evaluate": cmd_evaluate,
    "all": cmd_all,
}


# ==== SECTION 7: CLI ====


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands and shared flags."""
    # Shared flags attached to every subparser so `pipeline.py <cmd> --debug` works.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        help=f"Sample only the first {DEBUG_SAMPLE_ROWS} rows of the dataset.",
    )
    common.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Multi-manifold persistent-homology pipeline for UAVIDS-2025.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="phase")

    sub.add_parser(
        "prep", parents=[common],
        help="Phase 2: load, split, scale, and save per-manifold CSVs.",
    )

    tda_p = sub.add_parser(
        "tda", parents=[common],
        help="Phase 3: compute per-flow persistence diagrams.",
    )
    tda_p.add_argument(
        "--manifold",
        choices=("c2", "network", "physical", "all"),
        default="all",
        help="Which manifold(s) to compute persistence for.",
    )
    tda_p.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which split(s) to compute diagrams for.",
    )
    tda_p.add_argument(
        "--seed", type=int, default=PRIMARY_SEED,
        help="Seed for the reference-cloud sampler.",
    )

    sub.add_parser(
        "features", parents=[common],
        help="Phase 4: extract summary stats and persistence images.",
    )
    sub.add_parser(
        "supervised", parents=[common],
        help="Phase 5: train and evaluate supervised classifiers.",
    )
    sub.add_parser(
        "unsupervised", parents=[common],
        help="Phase 6: Wasserstein-based detection and attack-type inference.",
    )
    sub.add_parser(
        "evaluate", parents=[common],
        help="Phase 7: ablation, final tables, and paper figures.",
    )
    sub.add_parser(
        "all", parents=[common],
        help="Run every phase in order.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, set up environment, and dispatch the requested phase."""
    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_directories()
    setup_logging(args.verbose)

    COMMAND_DISPATCH[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
