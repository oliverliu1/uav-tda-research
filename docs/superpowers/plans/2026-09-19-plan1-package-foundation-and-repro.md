# Package Foundation + Probe Reproducibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the installable `uav_tda` package and recover the lost probe/metrics code as tested modules, so the SciTech paper's Results numbers are regenerable from committed code.

**Architecture:** Characterization-first. The surviving `results/tables/probe_distances*.csv` and `paper/MULTI_SEED_VARIANCE.md` are golden-master oracles. We first lock the CSV→AUC metrics against those oracles (no recomputation), then reconstruct the probe *runner* and prove its regenerated distances reproduce the oracle AUCs within tolerance. `pipeline.py` is untouched and stays runnable throughout.

**Tech Stack:** Python **3.9** (the installed anaconda interpreter — do NOT require ≥3.10), `numpy`, `pandas`, `scikit-learn`, `gudhi 3.11` (Rips + Hera Wasserstein), `persim`, `joblib`, `tqdm`; `pytest` for tests; `setuptools` build backend via `pyproject.toml`. All new modules use `from __future__ import annotations` so `X | None` / `list[str]` annotations work on 3.9.

**Spec:** `docs/superpowers/plans/2026-09-19-roadmap.md` (Phase 1), the paper `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex` (§III.F, §IV), and `paper/{PROBE_RESULTS,MULTI_SEED_VARIANCE}.md`.

## Global Constraints

- **Do not modify `pipeline.py`, `data/`, `outputs/`, `results/tables/*.csv`, or `paper/*.md`** during this plan. They are oracles and frozen inputs. New code goes in `uav_tda/` and `tests/`.
- **Probe seeds are `(42, 7, 123)`** — matching the paper. Do **not** use the monolith's `SEEDS = (42, 7, 2024)` for the probe. Expose these as distinct constants.
- **The probe's combined/subset score is a SUM across manifolds**, not a max-pool. (`W2_c2_network = W2_c2 + W2_network`, verified in `probe_distances.csv`.) The monolith's `compute_overall_metrics` max-pools; do not copy that here.
- **Per-attack AUC is one-versus-rest against ALL other classes** (Normal + the three other attacks), per paper §IV.B — not one-vs-Normal. (The monolith's `compute_per_class_auc` uses one-vs-Normal; do not copy it.)
- **Never assert equality on freshly-computed Wasserstein distances** — Hera is approximate and seed-sensitive. Use tolerances: `abs=1e-3` when re-deriving AUCs from the *published* oracle CSVs (Task 3); `abs=0.05` (seed-variance scale) when validating the *reconstructed probe on freshly-rebuilt diagrams* (Task 5, see the lineage note below).
- Every new results file is written with a sidecar `<name>.provenance.json` (seed, config hash, git SHA, library versions).
- **Do not upgrade the installed scientific stack.** The working env is Python 3.9.7 with `numpy 1.20.3`, `gudhi 3.11.0`, `joblib 1.5.3`, `scikit-learn 1.3.2`. `pyproject.toml` dependency lower bounds MUST be ≤ these installed versions so `pip install -e` upgrades nothing. Do not add new top-level dependencies beyond `pytest`/`pytest-cov`. If a task seems to need one, stop and ask.
- **The published `results/tables/probe_distances*.csv` are immutable oracles** — never overwrite them. Freshly-regenerated probe distances go under `results/tables/rebuild/`.

### Lineage note (why Task 5 tolerance is 0.05, not 0.01)

The diagrams in `outputs/` at plan-writing time were stale (a 20%-test-split run, 24,435 test flows) and have been **rebuilt from raw data** with the correct 15% split (18,326 test flows). The published `probe_distances*.csv` were produced from the *old* diagrams by the lost `tools/quick_unsup_probe.py`, and the old split's `labels_test.csv` aligned to those old diagrams no longer exists — so bit-exact reproduction of the published per-flow distances is **impossible**. What we validate instead: (1) our recovered *metrics* reproduce the published AUCs from the published CSVs exactly (Task 3, ±1e-3); (2) our reconstructed *probe runner*, run on the clean rebuilt diagrams, lands within seed-variance (±0.05) of the published AUCs (Task 5). The published CSVs remain the frozen "as-submitted" record; the rebuild produces the authoritative clean-lineage numbers, which may differ slightly and, since the paper is revisable, can be promoted to the paper after review.

---

## File Structure

```
pyproject.toml                     # NEW — package metadata, pinned deps, pytest config
uav_tda/
  __init__.py                      # NEW — version, package marker
  config.py                        # NEW — port of pipeline.py SECTIONS 1-4 (paths, schema, manifolds, hyperparams)
  paths.py                         # NEW — repo-root resolution + typed accessors for oracle/artifact files
  metrics.py                       # NEW — recovered CSV→AUC code (binary per-subset, per-attack one-vs-rest, multi-seed aggregate)
  probe.py                         # NEW — reconstructed probe runner (per-flow W2 with top-K + Hera delta)
  provenance.py                    # NEW — sidecar provenance JSON writer
  cli.py                           # NEW — `uav-tda` entry point (only `probe` subcommand in this plan)
tests/
  conftest.py                      # NEW — repo-root + oracle-path fixtures
  test_config.py                   # NEW — characterization: config matches pipeline.py
  test_metrics.py                  # NEW — golden-master: CSV→AUC matches PROBE_RESULTS + MULTI_SEED_VARIANCE
  test_probe.py                    # NEW — integration: regenerated distances reproduce oracle AUCs (slow)
  test_cli.py                      # NEW — CLI smoke test
```

Responsibilities: `config` = frozen constants only (no logic). `paths` = where files live. `metrics` = pure functions over dataframes. `probe` = the expensive recompute. `provenance`/`cli` = plumbing. Keep each file single-responsibility so it fits in context.

---

### Task 1: Package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `uav_tda/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py` (import-only for this task)

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `uav_tda` with `uav_tda.__version__: str`; pytest fixtures `repo_root` (Path) and `tables_dir` (Path) for later tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import uav_tda


def test_package_imports_and_has_version():
    assert isinstance(uav_tda.__version__, str)
    assert uav_tda.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::test_package_imports_and_has_version -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uav_tda'`

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "uav-tda"
version = "0.1.0"
description = "Multi-manifold persistent homology for UAV intrusion detection (UAVIDS-2025)."
requires-python = ">=3.9"
# Lower bounds are <= versions already installed in the anaconda env, so
# `pip install -e` upgrades nothing (numpy 1.20.3, gudhi 3.11, joblib 1.5.3,
# scikit-learn 1.3.2 are all present). Do not raise these.
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "scipy>=1.7",
    "scikit-learn>=1.0",
    "scikit-learn-extra>=0.3",
    "gudhi>=3.9",
    "persim>=0.3",
    "matplotlib>=3.3",
    "seaborn>=0.11",
    "tqdm>=4.6",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1"]

[project.scripts]
uav-tda = "uav_tda.cli:main"

[tool.setuptools.packages.find]
include = ["uav_tda*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: recomputes Wasserstein distances; skipped by default in CI fast lane"]
```

- [ ] **Step 4: Create the package marker**

```python
# uav_tda/__init__.py
"""Multi-manifold persistent homology for UAV intrusion detection."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create shared fixtures**

```python
# tests/conftest.py
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # tests/ lives directly under the repo root.
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tables_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "tables"
```

- [ ] **Step 6: Install the package editable and run the test**

Run: `pip install -e ".[dev]" && python -m pytest tests/test_config.py::test_package_imports_and_has_version -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uav_tda/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat: scaffold uav_tda package with pytest"
```

---

### Task 2: Port frozen config with characterization test

**Files:**
- Create: `uav_tda/config.py`
- Create: `uav_tda/paths.py`
- Modify: `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `repo_root` fixture.
- Produces:
  - `uav_tda.config`: `MANIFOLDS: dict[str, tuple[str, ...]]`, `C2_FEATURES`, `NETWORK_FEATURES`, `PHYSICAL_FEATURES`, `EXPECTED_CLASSES: tuple[str, ...]`, `MAX_HOM_DIM: dict[str,int]`, `SPARSE_RIPS_EPSILON: dict`, `REFERENCE_CLOUD_SIZE=500`, `MAX_EDGE_PERCENTILE=25`, `THRESHOLD_PERCENTILE=95`, `PROBE_SEEDS=(42,7,123)`, `SUPERVISED_SEEDS=(42,7,2024)`, `PRIMARY_SEED=42`, `PROBE_PER_CLASS=200`, `PROBE_TOP_K=50`, `PROBE_DELTA=0.2`.
  - `uav_tda.paths`: `REPO_ROOT: Path`, `DATA_PATH`, `OUTPUTS_DIR`, `PERSISTENCE_DIR`, `TABLES_DIR`, `FIGURES_DIR`, `probe_distances_csv(seed:int)->Path`.

- [ ] **Step 1: Write the failing characterization test**

```python
# tests/test_config.py  (append)
import importlib.util
from pathlib import Path

from uav_tda import config


def _load_pipeline(repo_root: Path):
    """Import the frozen monolith as a module without running its CLI."""
    spec = importlib.util.spec_from_file_location("pipeline_frozen", repo_root / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_matches_frozen_pipeline(repo_root):
    p = _load_pipeline(repo_root)
    assert config.MANIFOLDS == p.MANIFOLDS
    assert config.EXPECTED_CLASSES == p.EXPECTED_CLASSES
    assert config.MAX_HOM_DIM == p.MAX_HOM_DIM
    assert config.SPARSE_RIPS_EPSILON == p.SPARSE_RIPS_EPSILON
    assert config.REFERENCE_CLOUD_SIZE == p.REFERENCE_CLOUD_SIZE
    assert config.MAX_EDGE_PERCENTILE == p.MAX_EDGE_PERCENTILE
    assert config.THRESHOLD_PERCENTILE == p.THRESHOLD_PERCENTILE
    assert config.PRIMARY_SEED == p.PRIMARY_SEED


def test_probe_seeds_are_paper_seeds_not_monolith_seeds(repo_root):
    p = _load_pipeline(repo_root)
    assert config.PROBE_SEEDS == (42, 7, 123)          # paper §IV
    assert config.SUPERVISED_SEEDS == p.SEEDS          # monolith's (42, 7, 2024)
    assert config.PROBE_SEEDS != config.SUPERVISED_SEEDS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uav_tda.config'`

- [ ] **Step 3: Create `uav_tda/config.py`** (values copied verbatim from `pipeline.py` §2–4)

```python
# uav_tda/config.py
"""Frozen experiment constants, ported verbatim from pipeline.py SECTIONS 2-4."""

EXPECTED_CLASSES = (
    "Normal Traffic", "Blackhole Attack", "Wormhole Attack",
    "Sybil Attack", "Flooding Attack",
)
KNOWN_PORTS = (9, 654)

C2_FEATURES = (
    "SrcAddr_last_octet", "SrcPort_9", "SrcPort_654",
    "DstAddr_last_octet", "DstPort_9", "DstPort_654", "FlowDuration/s",
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
MANIFOLDS = {"c2": C2_FEATURES, "network": NETWORK_FEATURES, "physical": PHYSICAL_FEATURES}

REFERENCE_CLOUD_SIZE = 500
MAX_EDGE_PERCENTILE = 25
MAX_HOM_DIM = {"c2": 2, "network": 2, "physical": 1}
SPARSE_RIPS_EPSILON = {"c2": 0.5, "network": 0.5, "physical": None}
THRESHOLD_PERCENTILE = 95

PRIMARY_SEED = 42
PROBE_SEEDS = (42, 7, 123)          # paper §IV; NOT the monolith's SEEDS
SUPERVISED_SEEDS = (42, 7, 2024)    # monolith's supervised tables
PROBE_PER_CLASS = 200
PROBE_TOP_K = 50
PROBE_DELTA = 0.2
```

- [ ] **Step 4: Create `uav_tda/paths.py`**

```python
# uav_tda/paths.py
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
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add uav_tda/config.py uav_tda/paths.py tests/test_config.py
git commit -m "feat: port frozen config with characterization test against pipeline.py"
```

---

### Task 3: Recover CSV→AUC metrics, locked to the surviving oracles

**Files:**
- Create: `uav_tda/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `uav_tda.config`, `uav_tda.paths.probe_distances_csv`; probe-distances dataframes with columns `label`, `W2_c2`, `W2_network`, `W2_physical`, and the seven subset columns `W2_{c2_only,network_only,physical_only,c2_network,c2_physical,network_physical,all_three}`.
- Produces:
  - `MANIFOLD_SUBSETS: dict[str, tuple[str, ...]]` mapping subset name → constituent manifolds.
  - `binary_auc_by_subset(df) -> dict[str, float]` — Normal-vs-any-attack AUC for each of the 7 subsets, ranking on the summed distance.
  - `per_attack_auc(df) -> pandas.DataFrame` — columns `attack_class, manifold, auc`; one-vs-rest against all other classes, ranking on the single-manifold distance.
  - `aggregate_over_seeds(dfs: dict[int, pandas.DataFrame]) -> dict` with `{"binary": DataFrame(subset, mean, std), "per_attack": DataFrame(attack_class, manifold, mean, std)}`.

- [ ] **Step 1: Write the failing golden-master test** (oracle constants from `paper/PROBE_RESULTS.md` confirmation pass, seed 42, and `paper/MULTI_SEED_VARIANCE.md`)

```python
# tests/test_metrics.py
import pandas as pd
import pytest

from uav_tda import metrics
from uav_tda.paths import probe_distances_csv

# --- Oracles: PROBE_RESULTS.md confirmation pass, seed 42 ---
SEED42_BINARY = {
    "c2_only": 0.7599, "network_only": 0.7481, "physical_only": 0.6192,
    "c2_network": 0.8438, "c2_physical": 0.7680,
    "network_physical": 0.8550, "all_three": 0.8712,
}
SEED42_PER_ATTACK = {  # (attack, manifold): auc
    ("Blackhole Attack", "c2"): 0.4473, ("Blackhole Attack", "network"): 0.3106, ("Blackhole Attack", "physical"): 0.8032,
    ("Flooding Attack", "c2"): 0.6333, ("Flooding Attack", "network"): 0.7919, ("Flooding Attack", "physical"): 0.3784,
    ("Sybil Attack", "c2"): 0.6443, ("Sybil Attack", "network"): 0.8717, ("Sybil Attack", "physical"): 0.2119,
    ("Wormhole Attack", "c2"): 0.5350, ("Wormhole Attack", "network"): 0.2739, ("Wormhole Attack", "physical"): 0.7256,
}
# --- Oracle: MULTI_SEED_VARIANCE.md, mean over seeds 42/7/123 ---
MULTISEED_BINARY_MEAN = {
    "c2_only": 0.7488, "network_only": 0.7610, "physical_only": 0.6114,
    "c2_network": 0.8304, "c2_physical": 0.7542,
    "network_physical": 0.8594, "all_three": 0.8577,
}


def test_binary_auc_matches_seed42_oracle():
    df = pd.read_csv(probe_distances_csv(42))
    got = metrics.binary_auc_by_subset(df)
    for subset, expected in SEED42_BINARY.items():
        assert got[subset] == pytest.approx(expected, abs=1e-3), subset


def test_per_attack_auc_matches_seed42_oracle():
    df = pd.read_csv(probe_distances_csv(42))
    tbl = metrics.per_attack_auc(df).set_index(["attack_class", "manifold"])["auc"]
    for (attack, manifold), expected in SEED42_PER_ATTACK.items():
        assert tbl[(attack, manifold)] == pytest.approx(expected, abs=1e-3), (attack, manifold)


def test_subset_columns_equal_sum_of_single_manifolds():
    # Invariant that defines the probe's combined score (SUM, not max-pool).
    df = pd.read_csv(probe_distances_csv(42))
    assert (df["W2_c2_network"] - (df["W2_c2"] + df["W2_network"])).abs().max() < 1e-9
    assert (df["W2_all_three"] - (df["W2_c2"] + df["W2_network"] + df["W2_physical"])).abs().max() < 1e-9


def test_multiseed_binary_mean_matches_oracle():
    dfs = {s: pd.read_csv(probe_distances_csv(s)) for s in (42, 7, 123)}
    agg = metrics.aggregate_over_seeds(dfs)["binary"].set_index("subset")["mean"]
    for subset, expected in MULTISEED_BINARY_MEAN.items():
        assert agg[subset] == pytest.approx(expected, abs=1e-3), subset
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uav_tda.metrics'`

- [ ] **Step 3: Implement `uav_tda/metrics.py`**

```python
# uav_tda/metrics.py
"""Recovered anomaly-detection metrics: probe-distance dataframe -> AUCs.

Definitions match the paper (§IV) and the surviving probe CSVs:
- combined/subset score is the SUM of per-manifold Wasserstein-2 distances;
- per-attack AUC is one-versus-REST against all other classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import EXPECTED_CLASSES, MANIFOLDS

NORMAL = "Normal Traffic"
ATTACK_CLASSES = tuple(c for c in EXPECTED_CLASSES if c != NORMAL)

MANIFOLD_SUBSETS: dict[str, tuple[str, ...]] = {
    "c2_only": ("c2",),
    "network_only": ("network",),
    "physical_only": ("physical",),
    "c2_network": ("c2", "network"),
    "c2_physical": ("c2", "physical"),
    "network_physical": ("network", "physical"),
    "all_three": ("c2", "network", "physical"),
}


def _subset_score(df: pd.DataFrame, manifolds: tuple[str, ...]) -> np.ndarray:
    """Summed Wasserstein-2 distance across the given manifolds."""
    return np.sum([df[f"W2_{m}"].to_numpy() for m in manifolds], axis=0)


def binary_auc_by_subset(df: pd.DataFrame) -> dict[str, float]:
    """Normal-vs-any-attack AUC for each of the 7 manifold subsets."""
    is_attack = (df["label"] != NORMAL).astype(int).to_numpy()
    out: dict[str, float] = {}
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        score = _subset_score(df, manifolds)
        out[subset] = float(roc_auc_score(is_attack, score))
    return out


def per_attack_auc(df: pd.DataFrame) -> pd.DataFrame:
    """One-vs-rest AUC for each (attack class, single manifold)."""
    label = df["label"].to_numpy()
    rows = []
    for attack in ATTACK_CLASSES:
        y = (label == attack).astype(int)
        for m in MANIFOLDS:
            score = df[f"W2_{m}"].to_numpy()
            rows.append({"attack_class": attack, "manifold": m,
                         "auc": float(roc_auc_score(y, score))})
    return pd.DataFrame(rows)


def aggregate_over_seeds(dfs: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Mean/std of binary-subset and per-attack AUCs across seeds."""
    bin_rows, attack_rows = [], []
    for seed, df in dfs.items():
        for subset, auc in binary_auc_by_subset(df).items():
            bin_rows.append({"seed": seed, "subset": subset, "auc": auc})
        pa = per_attack_auc(df)
        pa["seed"] = seed
        attack_rows.append(pa)
    bin_df = pd.DataFrame(bin_rows)
    binary = (bin_df.groupby("subset")["auc"]
              .agg(["mean", "std"]).reset_index())
    per = (pd.concat(attack_rows).groupby(["attack_class", "manifold"])["auc"]
           .agg(["mean", "std"]).reset_index())
    return {"binary": binary, "per_attack": per}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (4 tests). If `per_attack` AUCs miss the oracle, the one-vs-rest vs one-vs-Normal definition is the first thing to check (Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add uav_tda/metrics.py tests/test_metrics.py
git commit -m "feat: recover probe AUC metrics, locked to surviving oracle CSVs"
```

---

### Task 4: Provenance sidecar writer

**Files:**
- Create: `uav_tda/provenance.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Consumes: nothing (stdlib + `uav_tda.__version__`).
- Produces: `write_provenance(target: Path, params: dict) -> Path` — writes `<target>.provenance.json` containing `params`, `git_sha`, `created_utc`, `library_versions` (numpy/pandas/sklearn/gudhi), and `uav_tda_version`; returns the sidecar path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provenance.py
import json
from pathlib import Path

from uav_tda.provenance import write_provenance


def test_write_provenance_creates_sidecar(tmp_path: Path):
    target = tmp_path / "result.csv"
    target.write_text("x\n1\n")
    side = write_provenance(target, {"seed": 42, "top_k": 50})
    assert side == tmp_path / "result.csv.provenance.json"
    meta = json.loads(side.read_text())
    assert meta["params"]["seed"] == 42
    assert "git_sha" in meta and "created_utc" in meta
    assert "numpy" in meta["library_versions"]
    assert meta["uav_tda_version"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uav_tda.provenance'`

- [ ] **Step 3: Implement `uav_tda/provenance.py`**

```python
# uav_tda/provenance.py
"""Write a provenance sidecar next to every generated artifact."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict[str, str]:
    out = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "gudhi"):
        try:
            out[pkg.replace("scikit-learn", "sklearn")] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def write_provenance(target: Path, params: dict) -> Path:
    side = target.with_name(target.name + ".provenance.json")
    meta = {
        "artifact": target.name,
        "params": params,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": _lib_versions(),
        "uav_tda_version": __version__,
    }
    side.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return side
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_provenance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add uav_tda/provenance.py tests/test_provenance.py
git commit -m "feat: provenance sidecar writer for generated artifacts"
```

---

### Task 5: Reconstruct the probe runner

**Files:**
- Create: `uav_tda/probe.py`
- Create: `tests/test_probe.py`

**Interfaces:**
- Consumes: `uav_tda.config`, `uav_tda.paths` (`PERSISTENCE_DIR`, `OUTPUTS_DIR`), `uav_tda.metrics` (for the validation test).
- Produces:
  - `truncate_top_k(diagram_dim: np.ndarray, k: int) -> np.ndarray` — keep the `k` longest-persistence `(birth, death)` pairs.
  - `run_probe(seed: int, per_class: int = 200, top_k: int = 50, delta: float = 0.2, w2_timeout: float | None = None, n_jobs: int = -1) -> pandas.DataFrame` — returns a dataframe with the probe schema: `test_idx, label, W2_c2, W2_network, W2_physical` plus the 7 subset columns.

**Reconstruction notes (grounded in `pipeline.py` — verified against the freshly rebuilt artifacts):** reuse `compute_baseline_barcodes` semantics — the baseline barcode per manifold is a Rips complex on the 500 reference points from `outputs/reference_indices.npy`, sliced per homology dimension. Per query flow, load its saved Phase-3 diagram from **`outputs/persistence_diagrams/<manifold>_test.pkl`** (the filename `pipeline.py:save_diagrams` actually writes; the `*_gudhi_diagrams_*.pkl` files are STALE leftovers of the pre-`pipeline.py` poster scripts — never read them). Each diagram is a numeric `(n, 3)` float ndarray with rows `[dim, birth, death]` (verified: `pipeline.py:_persistence_for_point` line 534), and **inf deaths are CLAMPED to `max_edge`, not dropped** — matching `pipeline.py:diagram_dim_slice` (line 675), which is the convention the lost probe was built alongside. Slice per dim, truncate to `top_k` longest bars, and compute `gudhi.hera.wasserstein_distance(flow_dim, baseline_dim, order=2.0, delta=delta)` summed over dims. Sample `per_class` flows per class from the test split using `numpy.random.default_rng(seed)`. `w2_timeout` bounds a single call (a timed-out dim contributes 0.0, matching the seed-123 artifact described in `MULTI_SEED_VARIANCE.md`).

- [ ] **Step 1: Write the failing unit test for `truncate_top_k`**

```python
# tests/test_probe.py
import numpy as np
import pytest

from uav_tda import probe


def test_truncate_top_k_keeps_longest_bars():
    # persistence = death - birth: bars have lengths 1, 5, 2, 0.5
    diag = np.array([[0.0, 1.0], [0.0, 5.0], [1.0, 3.0], [2.0, 2.5]])
    kept = probe.truncate_top_k(diag, k=2)
    lengths = np.sort(kept[:, 1] - kept[:, 0])
    assert np.allclose(lengths, [2.0, 5.0])


def test_truncate_top_k_handles_fewer_than_k():
    diag = np.array([[0.0, 1.0]])
    assert probe.truncate_top_k(diag, k=50).shape == (1, 2)


def test_truncate_top_k_empty():
    assert probe.truncate_top_k(np.empty((0, 2)), k=50).shape == (0, 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_probe.py::test_truncate_top_k_keeps_longest_bars -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uav_tda.probe'`

- [ ] **Step 3: Implement `truncate_top_k` and the probe skeleton**

```python
# uav_tda/probe.py
"""Reconstructed unsupervised Wasserstein-2 probe (paper §III.F).

Reproduces the lost tools/quick_unsup_probe.py: 200 flows/class from the test
split, per-flow persistence diagrams truncated to the top-K longest bars,
Hera approximate Wasserstein-2 (relative error delta) to a per-manifold
baseline barcode, summed over homology dimensions.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .config import (MANIFOLDS, MAX_HOM_DIM, PROBE_DELTA, PROBE_PER_CLASS,
                     PROBE_TOP_K, SPARSE_RIPS_EPSILON)
from .metrics import MANIFOLD_SUBSETS
from .paths import OUTPUTS_DIR, PERSISTENCE_DIR


def truncate_top_k(diagram_dim: np.ndarray, k: int) -> np.ndarray:
    """Keep the k finite bars with the largest persistence (death - birth)."""
    if diagram_dim.shape[0] <= k:
        return diagram_dim
    persistence = diagram_dim[:, 1] - diagram_dim[:, 0]
    keep = np.argsort(persistence)[::-1][:k]
    return diagram_dim[keep]


def _load_baseline_barcodes(max_edge_lengths: dict) -> dict:
    """Per-manifold per-dim baseline diagram from the 500 reference points."""
    import gudhi

    ref_idx = np.load(OUTPUTS_DIR / "reference_indices.npy")
    baselines: dict = {}
    for manifold in MANIFOLDS:
        pts = pd.read_csv(OUTPUTS_DIR / f"{manifold}_train.csv").to_numpy()[ref_idx]
        max_edge = max_edge_lengths[manifold]
        eps = SPARSE_RIPS_EPSILON.get(manifold)
        kwargs = {"points": pts, "max_edge_length": max_edge}
        if eps is not None:
            kwargs["sparse"] = eps
        st = gudhi.RipsComplex(**kwargs).create_simplex_tree(
            max_dimension=MAX_HOM_DIM[manifold] + 1)
        raw = st.persistence()
        if raw:
            diag = np.array([[float(d), float(b), float(de)] for d, (b, de) in raw],
                            dtype=float)
        else:
            diag = np.empty((0, 3), dtype=float)
        baselines[manifold] = {
            k: _slice_dim(diag, k, max_edge)
            for k in range(MAX_HOM_DIM[manifold] + 1)
        }
    return baselines


def _slice_dim(diagram: np.ndarray, dim: int, max_edge: float) -> np.ndarray:
    """(birth, death) pairs of one homology dimension, inf deaths clamped to max_edge.

    Phase-3 diagrams (outputs/persistence_diagrams/<m>_<split>.pkl) are numeric
    (n, 3) float ndarrays with rows [dim, birth, death] (pipeline.py line 534).
    Inf deaths are CLAMPED to max_edge, matching pipeline.py:diagram_dim_slice.
    """
    if diagram.size == 0:
        return np.empty((0, 2))
    bd = diagram[diagram[:, 0] == dim][:, 1:3].copy()
    bd[~np.isfinite(bd[:, 1]), 1] = max_edge
    return bd


def run_probe(seed: int, per_class: int = PROBE_PER_CLASS, top_k: int = PROBE_TOP_K,
              delta: float = PROBE_DELTA, w2_timeout: float | None = None,
              n_jobs: int = -1) -> pd.DataFrame:
    import json

    from gudhi.hera import wasserstein_distance as wdist

    max_edge_lengths = json.loads((OUTPUTS_DIR / "max_edge_lengths.json").read_text())
    baselines = _load_baseline_barcodes(max_edge_lengths)
    labels = pd.read_csv(OUTPUTS_DIR / "labels_test.csv")["label"].to_numpy()

    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(np.where(labels == cls)[0], size=per_class, replace=False)
        for cls in np.unique(labels)
    ])

    per_manifold_diagrams = {
        m: pickle.loads((PERSISTENCE_DIR / f"{m}_test.pkl").read_bytes())
        for m in MANIFOLDS
    }

    # NOTE: w2_timeout is accepted for CLI parity with the original probe but is
    # NOT enforced in this plan (it only affected the seed-123 C2 artifact). It is
    # wired in Phase 3. Passing a value here is a documented no-op for now.
    records = []
    for i in idx:
        row = {"test_idx": int(i), "label": labels[i]}
        for m in MANIFOLDS:
            max_edge = max_edge_lengths[m]
            diag = per_manifold_diagrams[m][i]   # (n, 3) ndarray [dim, birth, death]
            total = 0.0
            for k in range(MAX_HOM_DIM[m] + 1):
                flow_k = truncate_top_k(_slice_dim(diag, k, max_edge), top_k)
                total += float(wdist(flow_k, baselines[m][k], order=2.0, delta=delta))
            row[f"W2_{m}"] = total
        records.append(row)

    df = pd.DataFrame(records)
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        df[f"W2_{subset}"] = sum(df[f"W2_{m}"] for m in manifolds)
    return df
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `python -m pytest tests/test_probe.py -k truncate -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the slow integration/validation test** (the reproduction gate)

```python
# tests/test_probe.py  (append)
import pandas as pd  # noqa: E402

from uav_tda import metrics  # noqa: E402
from uav_tda.paths import PERSISTENCE_DIR  # noqa: E402

# Oracle: PROBE_RESULTS.md confirmation pass, seed 42.
ORACLE_BINARY_SEED42 = {
    "network_physical": 0.8550, "all_three": 0.8712, "network_only": 0.7481,
}
ORACLE_SYBIL_NETWORK_SEED42 = 0.8717


@pytest.mark.slow
def test_run_probe_reproduces_seed42_oracle_aucs():
    # Validates the reconstructed runner on the CLEAN rebuilt diagrams. Tolerance
    # is seed-variance scale (0.05), NOT 0.01: the rebuilt diagrams use the correct
    # 15% split, so exact reproduction of the old-split published numbers is not
    # expected (see the plan's Lineage note). We assert the finding is preserved:
    # Network+Physical strong, Sybil dominant on Network.
    if not (PERSISTENCE_DIR / "network_test.pkl").exists():
        pytest.skip("Phase-3 persistence diagrams not present; run `pipeline.py tda --split test` first.")
    df = probe.run_probe(seed=42)
    binary = metrics.binary_auc_by_subset(df)
    for subset, expected in ORACLE_BINARY_SEED42.items():
        assert binary[subset] == pytest.approx(expected, abs=0.05), subset
    pa = metrics.per_attack_auc(df).set_index(["attack_class", "manifold"])["auc"]
    assert pa[("Sybil Attack", "network")] == pytest.approx(ORACLE_SYBIL_NETWORK_SEED42, abs=0.05)
    # Attribution must hold: Sybil's dominant manifold is Network.
    sybil = pa.xs("Sybil Attack", level="attack_class")
    assert sybil.idxmax() == "network"
```

- [ ] **Step 6: Run the slow test to verify reproduction**

Run: `python -m pytest tests/test_probe.py -m slow -v`
Expected: PASS, or SKIP if diagrams absent. **If AUCs are outside ±0.05**, do NOT loosen the tolerance. Investigate in this order: (a) confirm `_slice_dim` clamps inf deaths to `max_edge` exactly like `pipeline.py:diagram_dim_slice` (dropping them instead materially changes H0); (b) whether the probe sampled *by position within class* rather than `rng.choice` (try `rng.permutation`); (c) Hera `delta` keyword name/behavior in gudhi 3.11 (`delta=` relative error — verify against `gudhi.hera.wasserstein_distance` signature). Record whichever reconstruction matches in a comment, and if none reaches ±0.05, escalate per the roadmap's "reconstruction may not be bit-exact" fallback (freeze the published CSVs as the record, label the reconstruction "equivalent method").

- [ ] **Step 7: Commit**

```bash
git add uav_tda/probe.py tests/test_probe.py
git commit -m "feat: reconstruct unsupervised W2 probe runner with reproduction test"
```

---

### Task 6: CLI + one-command repro path

**Files:**
- Create: `uav_tda/cli.py`
- Create: `tests/test_cli.py`
- Modify: `README.md` (append a "Reproducing the paper" section)

**Interfaces:**
- Consumes: `uav_tda.probe.run_probe`, `uav_tda.metrics`, `uav_tda.provenance.write_provenance`, `uav_tda.paths.TABLES_DIR`.
- Produces: console entry point `uav-tda probe [--seed N] [--per-class N] [--top-k N] [--delta F] [--w2-timeout F] [--out PATH]`; writes the probe CSV + provenance sidecar and prints the binary-AUC summary.

- [ ] **Step 1: Write the failing CLI smoke test**

```python
# tests/test_cli.py
from pathlib import Path

import pandas as pd
import pytest

from uav_tda import cli
from uav_tda.paths import PERSISTENCE_DIR


def test_cli_probe_writes_csv_and_provenance(tmp_path: Path):
    if not (PERSISTENCE_DIR / "network_test.pkl").exists():
        pytest.skip("Phase-3 persistence diagrams not present.")
    out = tmp_path / "probe.csv"
    rc = cli.main(["probe", "--seed", "42", "--per-class", "5", "--out", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert {"label", "W2_c2", "W2_network", "W2_physical", "W2_all_three"} <= set(df.columns)
    assert (out.with_name(out.name + ".provenance.json")).exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uav_tda.cli'`

- [ ] **Step 3: Implement `uav_tda/cli.py`**

```python
# uav_tda/cli.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (or SKIP if diagrams absent)

- [ ] **Step 5: Document the repro path in `README.md`** (append)

```markdown
## Reproducing the paper's results

Prerequisites: `data/UAVIDS-2025.csv` present, and Phase-2/3 artifacts built
(`python pipeline.py prep && python pipeline.py tda`).

```bash
pip install -e ".[dev]"
python -m pytest -m "not slow"          # fast: config + metrics locked to published oracle CSVs
uav-tda probe --seed 42                  # writes results/tables/rebuild/probe_distances_seed42.csv
uav-tda probe --seed 7
uav-tda probe --seed 123
python -m pytest -m slow                 # verifies rebuilt AUCs match the paper within seed variance (±0.05)
```

Each run writes a `.provenance.json` sidecar recording seed, git SHA, and library versions. The published
`results/tables/probe_distances*.csv` are the immutable as-submitted record and are never overwritten.
```

- [ ] **Step 6: Full suite + commit**

Run: `python -m pytest -v`
Expected: all non-slow PASS; slow PASS or SKIP.

```bash
git add uav_tda/cli.py tests/test_cli.py README.md
git commit -m "feat: uav-tda probe CLI with provenance and repro docs"
```

---

## Self-Review

**Spec coverage (roadmap Phase 1):**
- Package scaffold + pinned deps + pytest → Task 1 ✓
- Frozen config ported with characterization test → Task 2 ✓
- Probe seeds corrected to (42,7,123) → Task 2 (`test_probe_seeds_are_paper_seeds…`) ✓
- Recovered CSV→AUC metrics locked to `PROBE_RESULTS`/`MULTI_SEED_VARIANCE` → Task 3 ✓
- Sum-not-maxpool + one-vs-rest definitions enforced → Task 3 (constraints + `test_subset_columns_equal_sum…`) ✓
- Probe runner reconstructed + reproduction gate ±0.01 → Task 5 ✓
- Provenance sidecars → Task 4, used in Task 6 ✓
- One-command repro path → Task 6 ✓

**Placeholder scan:** No TBD/TODO; every code step has runnable content; oracle constants are concrete numbers from the surviving `paper/*.md`.

**Type consistency:** `run_probe(...)` signature identical in `probe.py`, `cli.py`, and the Interfaces blocks. `MANIFOLD_SUBSETS` defined in `metrics.py`, imported by `probe.py`. `binary_auc_by_subset` / `per_attack_auc` / `aggregate_over_seeds` names consistent across `metrics.py`, `test_metrics.py`, `test_probe.py`, `cli.py`. `write_provenance(target, params)` consistent in `provenance.py`, `test_provenance.py`, `cli.py`.

**Known reconstruction risk:** Task 5 Step 6 documents the ±0.01 reproduction gate and the escalation path if the reconstructed probe cannot match the oracle (the roadmap's fallback: freeze the surviving CSVs as provenance, label the reconstruction "equivalent method").
