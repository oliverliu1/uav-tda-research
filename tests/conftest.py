from pathlib import Path

import pytest

# tests/ lives directly under the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared portability guards (Phase-7 Task 2): a fresh clone has neither the
# gitignored dataset nor any outputs/ artifacts, so any FAST (not slow) test
# that touches either must self-skip rather than fail. Golden-master tests
# that only read the committed `results/tables/**` oracles are NOT guarded --
# those files ARE in git and CI must exercise them.
#
# `requires_data` guards the one dataset file fast tests may symlink/read.
requires_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "UAVIDS-2025.csv").exists(),
    reason="local dataset (data/UAVIDS-2025.csv, gitignored) not present",
)

# `requires_outputs` guards the specific outputs/ artifact a fast test reads
# (outputs/ is gitignored and populated only by running `uav-tda prep/tda/...`
# locally). Default artifact is outputs/max_edge_lengths.json (written by the
# `tda` phase); pass a different relative path for tests needing another file.
def _outputs_artifact_missing(*relpath: str) -> bool:
    parts = relpath or ("max_edge_lengths.json",)
    return not (REPO_ROOT / "outputs" / Path(*parts)).exists()


requires_outputs = pytest.mark.skipif(
    _outputs_artifact_missing(),
    reason="local outputs/max_edge_lengths.json (gitignored) not present",
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
