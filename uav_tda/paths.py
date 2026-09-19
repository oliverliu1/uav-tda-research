"""Canonical filesystem locations for oracle and artifact files."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "UAVIDS-2025.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PERSISTENCE_DIR = OUTPUTS_DIR / "persistence_diagrams"
TABLES_DIR = REPO_ROOT / "results" / "tables"
FIGURES_DIR = REPO_ROOT / "results" / "figures"


def probe_distances_csv(seed: int) -> Path:
    """Surviving per-flow probe output for a given seed (42 has the unsuffixed name)."""
    name = "probe_distances.csv" if seed == 42 else f"probe_distances_seed{seed}.csv"
    return TABLES_DIR / name
