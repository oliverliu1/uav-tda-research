from pathlib import Path

import pytest

# tests/ lives directly under the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared portability guards (Phase-7 Task 2): data/UAVIDS-2025.csv is tracked
# in git and ships with every clone, but these guards stay as forward-hedges
# for clones/environments where it is absent -- e.g. if redistribution of the
# dataset is later removed (see data/README.md) -- and for outputs/, which is
# gitignored and populated only by running `uav-tda prep/tda/...` locally.
# Golden-master tests that only read the committed `results/tables/**`
# oracles are NOT guarded -- those files ARE in git and CI must exercise
# them.
#
# `requires_data` guards the one dataset file fast tests may symlink/read.
requires_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "UAVIDS-2025.csv").exists(),
    reason="local dataset (data/UAVIDS-2025.csv) not present",
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tables_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "tables"


@pytest.fixture(scope="session")
def monolith_cache(repo_root: Path) -> Path:
    cache_dir = repo_root / ".cache" / "monolith-debug"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
