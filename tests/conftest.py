from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # tests/ lives directly under the repo root.
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tables_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "tables"
