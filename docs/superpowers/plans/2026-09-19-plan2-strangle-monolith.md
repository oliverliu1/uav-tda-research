# Strangle the Monolith — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port every `pipeline.py` phase (prep → tda → features → supervised → unsupervised → evaluate) into tested `uav_tda` modules with a `uav-tda` CLI mirroring the monolith, each port gated by an equivalence test against a fresh monolith run on identical inputs.

**Architecture:** Strangler-fig with a hermetic oracle. `pipeline.py` remains **byte-untouched** — it is both the paper's historical record and the equivalence oracle. A test harness copies `pipeline.py` into a temp workspace (with `data/` symlinked), runs monolith phases there in `--debug` mode, caches the results on disk keyed by the pipeline's content hash, and each ported module must produce equivalent artifacts from the same inputs. "Retirement" of the monolith is documentation-level: `uav-tda` becomes the maintained entry point; `pipeline.py` is declared frozen-oracle.

**Tech Stack:** Python 3.9 (anaconda; `from __future__ import annotations` in every module), numpy 1.20.3, pandas, scikit-learn 1.3.2, scikit-learn-extra (KMedoids), gudhi 3.11, persim, joblib, pytest.

**Spec:** `docs/superpowers/plans/2026-09-19-roadmap.md` (Phase 2). Source of truth for behavior: `pipeline.py` sections 6–11.

## Global Constraints

- **`pipeline.py` must remain byte-identical** through this entire plan. Same for `data/`, `paper/`, `CLAUDE/` (except the two doc files named in Task 9), and the published `results/tables/probe_distances*.csv`.
- **Never write into the repo's real `outputs/` or `results/` from any test.** All test I/O goes to pytest tmp dirs or the gitignored `.cache/monolith-debug/` harness cache. A background `pipeline.py tda --split train` rebuild may be running in the repo — repo-level `outputs/` is off-limits to tests for this reason too.
- **Equivalence definition:** package output vs monolith output *on identical inputs in hermetic workspaces*. Historical artifacts are NOT bit-exact oracles (empirical: clean-rebuild probe AUC 0.8433 vs May-lineage 0.8550 at seed 42). Comparisons: exact for ints/strings/indices/splits; `rtol=1e-9, atol=1e-12` for floats unless a task documents a looser bound with the reason.
- **Sparse-Rips nondeterminism ruling (Task 4 finding, controller-decided):** GUDHI 3.11 `RipsComplex(sparse=0.5)` is deterministic *within* a process but **nondeterministic across processes** (proven: identical inputs → 751 vs 753 bars in two fresh processes; exact Rips fully reproducible). Consequences, binding on Tasks 4–8: (a) diagram comparisons for `physical` (sparse=None) stay EXACT; for `c2`/`network` use the statistical comparator `assert_diagram_pkls_statistically_equal` (sampled per-dim bar-count / total-persistence / W2 gates); (b) every downstream phase test (T5–T8) stages the ORACLE's upstream artifacts into the package workspace and runs ONLY its own phase — never re-runs package tda — so nondeterminism cannot compound and downstream tolerances stay tight; (c) port fidelity for tda rests on the verbatim-port review + exact upstream artifacts + exact physical + in-process probe; the statistical test is a regression guard, and its docstring must say so.
- **Ported code is a port, not a rewrite.** Copy function bodies from the cited `pipeline.py` line ranges; permitted changes ONLY: (a) module-level imports instead of function-local, (b) paths through the `Workspace` object, (c) `logging.getLogger("uav_tda.<module>")`, (d) constants from `uav_tda.config`, (e) removal of `argparse` glue. Any behavioral change is a defect.
- Heavy equivalence tests are marked `@pytest.mark.slow`. The fast suite (`-m "not slow"`) must stay under ~1 minute.
- Python 3.9 compatibility: no `match`, no `X | Y` in runtime positions without `from __future__ import annotations`.
- Commit after every green task step, per-task messages as specified.

---

## File Structure

```
uav_tda/
  workspace.py        # NEW — Workspace dataclass (rebasing all pipeline paths under any root)
  data.py             # NEW — port of pipeline.py SECTION 6 (prep)
  tda.py              # NEW — port of SECTION 7 (reference cloud, max-edge, per-flow Rips)
  features.py         # NEW — port of SECTION 8 (summary stats + persistence images)
  supervised.py       # NEW — port of SECTION 9 (grid search, eval, curated RF)
  unsupervised.py     # NEW — port of SECTION 10 (W2 distances, thresholds, inference rule)
  evaluate.py         # NEW — port of SECTION 11 (ablations, final tables, figures)
  cli.py              # MODIFY — add prep/tda/features/supervised/unsupervised/evaluate/all subcommands
tests/
  monolith_harness.py # NEW — hermetic monolith workspace + phase runner + cached debug artifacts + comparators
  test_workspace.py   # NEW
  test_harness.py     # NEW
  test_data.py        # NEW (equivalence: prep)
  test_tda.py         # NEW (determinism probe + equivalence + real-artifact spot-check)
  test_features.py    # NEW
  test_supervised.py  # NEW
  test_unsupervised.py# NEW
  test_evaluate.py    # NEW
  test_cli_phases.py  # NEW (CLI wiring smoke tests)
.gitignore            # MODIFY — add .cache/
README.md / CLAUDE/CLAUDE.md / CLAUDE/PROJECT_BRIEF.md  # MODIFY in Task 9 only
```

Monolith line map (verified this session; re-verify with grep before porting): SECTION 6 prep = 187–426; SECTION 7 tda = 427–652; SECTION 8 features = 654–962; SECTION 9 supervised = 964–1377; SECTION 10 unsupervised = 1379–1830; SECTION 11 evaluate = 1832–2239.

---

### Task 1: `Workspace` — rebasable paths

**Files:**
- Create: `uav_tda/workspace.py`
- Test: `tests/test_workspace.py`

**Interfaces:**
- Produces: `Workspace` frozen dataclass with fields `root, data_csv, outputs_dir, persistence_dir, tda_features_dir, results_dir, tables_dir, figures_dir, models_dir, logs_dir` (all `Path`); classmethods `Workspace.default()` (built from `uav_tda.paths` constants) and `Workspace.at(root: Path)` (every dir rebased under `root`, `data_csv = root/"data"/"UAVIDS-2025.csv"`); method `ensure()` creating all dirs (not `data_csv`'s parent's file). All later tasks consume `Workspace`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workspace.py
from pathlib import Path

from uav_tda import paths
from uav_tda.workspace import Workspace


def test_default_matches_paths_constants():
    ws = Workspace.default()
    assert ws.root == paths.REPO_ROOT
    assert ws.data_csv == paths.DATA_PATH
    assert ws.outputs_dir == paths.OUTPUTS_DIR
    assert ws.persistence_dir == paths.PERSISTENCE_DIR
    assert ws.tables_dir == paths.TABLES_DIR
    assert ws.figures_dir == paths.FIGURES_DIR


def test_at_rebases_everything(tmp_path: Path):
    ws = Workspace.at(tmp_path)
    assert ws.root == tmp_path
    assert ws.outputs_dir == tmp_path / "outputs"
    assert ws.persistence_dir == tmp_path / "outputs" / "persistence_diagrams"
    assert ws.tda_features_dir == tmp_path / "outputs" / "tda_features"
    assert ws.tables_dir == tmp_path / "results" / "tables"
    assert ws.models_dir == tmp_path / "results" / "models"
    assert ws.logs_dir == tmp_path / "logs"
    assert ws.data_csv == tmp_path / "data" / "UAVIDS-2025.csv"


def test_ensure_creates_dirs(tmp_path: Path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    for d in (ws.outputs_dir, ws.persistence_dir, ws.tda_features_dir,
              ws.tables_dir, ws.figures_dir, ws.models_dir, ws.logs_dir):
        assert d.is_dir()
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_workspace.py -v` → `ModuleNotFoundError`
- [ ] **Step 3: Implement `uav_tda/workspace.py`**

```python
# uav_tda/workspace.py
"""Rebasable filesystem layout mirroring pipeline.py SECTION 1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Workspace:
    root: Path
    data_csv: Path
    outputs_dir: Path
    persistence_dir: Path
    tda_features_dir: Path
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    logs_dir: Path

    @classmethod
    def at(cls, root: Path) -> "Workspace":
        root = Path(root)
        outputs = root / "outputs"
        results = root / "results"
        return cls(
            root=root,
            data_csv=root / "data" / "UAVIDS-2025.csv",
            outputs_dir=outputs,
            persistence_dir=outputs / "persistence_diagrams",
            tda_features_dir=outputs / "tda_features",
            results_dir=results,
            tables_dir=results / "tables",
            figures_dir=results / "figures",
            models_dir=results / "models",
            logs_dir=root / "logs",
        )

    @classmethod
    def default(cls) -> "Workspace":
        return cls.at(paths.REPO_ROOT)

    def ensure(self) -> None:
        for d in (self.outputs_dir, self.persistence_dir, self.tda_features_dir,
                  self.results_dir, self.tables_dir, self.figures_dir,
                  self.models_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run to verify pass** — 3 tests PASS. (Note `Workspace.default()` equality with `paths` constants holds because `paths.OUTPUTS_DIR == REPO_ROOT/"outputs"` etc. — verify; if a constant differs structurally, match `at()` to the constants, not vice versa.)
- [ ] **Step 5: Commit** — `git add uav_tda/workspace.py tests/test_workspace.py && git commit -m "feat: Workspace dataclass with rebasable pipeline paths"`

---

### Task 2: Hermetic monolith harness with on-disk cache

**Files:**
- Create: `tests/monolith_harness.py`
- Modify: `.gitignore` (append `.cache/` under the existing comment style)
- Modify: `tests/conftest.py` (append fixture)
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `repo_root` fixture (Task 1 of Plan 1).
- Produces (all in `tests/monolith_harness.py`):
  - `pipeline_hash(repo_root) -> str` — first 12 hex of sha256 of `pipeline.py` bytes.
  - `make_monolith_workspace(dest: Path, repo_root: Path) -> Path` — creates `dest`, copies `pipeline.py` in, symlinks `repo_root/"data"` to `dest/"data"`; returns `dest`.
  - `run_monolith_phase(ws_root: Path, phase: str, *extra: str) -> None` — `subprocess.run([sys.executable, "pipeline.py", phase, "--debug", *extra], cwd=ws_root, check=True, capture_output=True)`; on failure raise with tail of stderr.
  - `ensure_debug_phase(cache_root: Path, repo_root: Path, phase: str) -> Path` — idempotent: workspace at `cache_root/<pipeline_hash>`; phase chain `prep → tda → features → supervised → unsupervised → evaluate`; runs (in order) every not-yet-done prerequisite up to and including `phase`, touching `.done-<phase>` markers; returns the workspace path.
  - Comparators: `assert_csvs_equal(a: Path, b: Path, float_rtol=1e-9)` (pandas `assert_frame_equal`, `check_exact=False, rtol=float_rtol, atol=1e-12`); `assert_diagram_pkls_equal(a: Path, b: Path)` (pickle lists, same length, per-element `np.allclose(..., equal_nan=True)` after sorting rows lexicographically — joblib order within a diagram is stable but sorting removes row-order sensitivity); `assert_json_equal(a: Path, b: Path, float_rtol=1e-9)`.
- Produces (in `tests/conftest.py`): session fixture `monolith_cache(repo_root) -> Path` returning `repo_root/".cache"/"monolith-debug"` (mkdir'd). Slow tests call `ensure_debug_phase(monolith_cache, repo_root, "<phase>")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_harness.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import monolith_harness as mh


def test_pipeline_hash_stable(repo_root):
    h1 = mh.pipeline_hash(repo_root)
    assert h1 == mh.pipeline_hash(repo_root)
    assert len(h1) == 12


def test_make_monolith_workspace(tmp_path, repo_root):
    ws = mh.make_monolith_workspace(tmp_path / "ws", repo_root)
    assert (ws / "pipeline.py").is_file()
    assert (ws / "data" / "UAVIDS-2025.csv").exists()  # via symlink


def test_assert_csvs_equal_detects_difference(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"x": [1.0, 2.0]}).to_csv(a, index=False)
    pd.DataFrame({"x": [1.0, 2.0]}).to_csv(b, index=False)
    mh.assert_csvs_equal(a, b)  # equal → no raise
    pd.DataFrame({"x": [1.0, 2.1]}).to_csv(b, index=False)
    with pytest.raises(AssertionError):
        mh.assert_csvs_equal(a, b)


@pytest.mark.slow
def test_ensure_debug_prep_builds_and_caches(monolith_cache, repo_root):
    ws = mh.ensure_debug_phase(monolith_cache, repo_root, "prep")
    assert (ws / ".done-prep").exists()
    assert (ws / "outputs" / "c2_train.csv").is_file()
    labels = pd.read_csv(ws / "outputs" / "labels_train.csv")
    assert len(labels) < 10000  # debug sample, not the full 85k split
    mtime = (ws / "outputs" / "c2_train.csv").stat().st_mtime
    ws2 = mh.ensure_debug_phase(monolith_cache, repo_root, "prep")  # cached: no re-run
    assert ws2 == ws
    assert (ws / "outputs" / "c2_train.csv").stat().st_mtime == mtime
```

- [ ] **Step 2: Run fast subset to verify failure** — `python3 -m pytest tests/test_harness.py -m "not slow" -v` → import error
- [ ] **Step 3: Implement `tests/monolith_harness.py`** per the Interfaces block above. `ensure_debug_phase` phase order: `("prep", "tda", "features", "supervised", "unsupervised", "evaluate")`; run each missing prerequisite via `run_monolith_phase`, then `touch` its marker.
- [ ] **Step 4: Fast tests pass** — `python3 -m pytest tests/test_harness.py -m "not slow" -v`
- [ ] **Step 5: Slow test passes (builds the cache once, ~15 s for prep)** — `python3 -m pytest tests/test_harness.py -m slow -v`
- [ ] **Step 6: Append `.cache/` to `.gitignore`; add the `monolith_cache` fixture to `tests/conftest.py`; commit** — `git add -A tests/ .gitignore && git commit -m "feat: hermetic monolith harness with content-hash debug cache"`

---

### Task 3: `data.py` — prep port + equivalence

**Files:**
- Create: `uav_tda/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Workspace`, `uav_tda.config`, harness.
- Produces: `run_prep(ws: Workspace, debug: bool = False, seed: int = config.PRIMARY_SEED) -> None` writing exactly the monolith's prep artifacts into `ws`: `{c2,network,physical,original_features}_{train,val,test}.csv`, `labels_{split}.csv`, `{train,val,test}_indices.npy`, `scaler_{manifold}.pkl`, `scalers.pkl`. Internal functions keep monolith names (`load_raw_dataset`, `encode_features`, `stratified_three_way_split`, `scale_manifold`, `validate_*`).

**Port source:** `pipeline.py` lines 187–426 (SECTION 6), including the debug-sample logic at 220–245. Permitted changes per Global Constraints only.

- [ ] **Step 1: Failing equivalence test**

```python
# tests/test_data.py
from pathlib import Path

import pytest

from tests import monolith_harness as mh
from uav_tda.workspace import Workspace

PREP_CSVS = [f"{m}_{s}.csv" for m in ("c2", "network", "physical", "original_features")
             for s in ("train", "val", "test")] + [f"labels_{s}.csv" for s in ("train", "val", "test")]


@pytest.mark.slow
def test_prep_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "prep")
    from uav_tda.data import run_prep
    ws = Workspace.at(tmp_path)
    (tmp_path / "data").symlink_to(repo_root / "data")
    ws.ensure()
    run_prep(ws, debug=True)
    for name in PREP_CSVS:
        mh.assert_csvs_equal(oracle / "outputs" / name, ws.outputs_dir / name)
    import numpy as np
    for name in ("train_indices.npy", "val_indices.npy", "test_indices.npy"):
        assert np.array_equal(np.load(oracle / "outputs" / name),
                              np.load(ws.outputs_dir / name))
```

- [ ] **Step 2: Verify it fails** (`ModuleNotFoundError: uav_tda.data`)
- [ ] **Step 3: Port SECTION 6 into `uav_tda/data.py`**, `run_prep` mirroring `cmd_prep` (lines 398–426) with `ws` paths.
- [ ] **Step 4: Slow test passes** — `python3 -m pytest tests/test_data.py -m slow -v`. If a CSV differs: diff the first divergent column; the usual culprits are column order (preserve monolith order exactly) and scaler dtype.
- [ ] **Step 5: Commit** — `"feat: port prep phase to uav_tda.data with monolith equivalence test"`

---

### Task 4: `tda.py` — persistence port + determinism probe + equivalence

**Files:**
- Create: `uav_tda/tda.py`
- Test: `tests/test_tda.py`

**Interfaces:**
- Consumes: `Workspace`, config, harness, `uav_tda.probe._slice_dim` (may import for spot-check).
- Produces: `run_tda(ws: Workspace, manifold: str = "all", split: str = "all", seed: int = config.PRIMARY_SEED, debug: bool = False, n_jobs: int = -1) -> None` writing `reference_indices.npy`, `max_edge_lengths.json`, `{m}_{s}.pkl/.npy`, `tda_summary.csv` into `ws`. Internal names per monolith: `sample_reference_indices`, `compute_reference_clouds`, `compute_max_edge_lengths`, `_persistence_for_point`, `compute_diagrams_for_split`, `save_diagrams`.

**Port source:** `pipeline.py` lines 427–652 (SECTION 7).

- [ ] **Step 1: Failing determinism-probe test (fast, in-process)**

```python
# tests/test_tda.py
import numpy as np
import pytest

from uav_tda import config


def test_persistence_for_point_deterministic_in_process():
    """Two identical calls must agree — sparse Rips determinism gate.

    If this FAILS, sparse Rips is nondeterministic: STOP, report to controller;
    diagram comparisons must switch to W2-tolerance comparisons plan-wide.
    """
    from uav_tda.tda import _persistence_for_point
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(50, 10))
    q = rng.normal(size=10)
    a = _persistence_for_point(q, ref, max_edge=0.5, max_simplex_dim=3, sparse=0.5)
    b = _persistence_for_point(q, ref, max_edge=0.5, max_simplex_dim=3, sparse=0.5)
    assert a.shape == b.shape
    assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))
```

- [ ] **Step 2: Verify fail → port SECTION 7 into `uav_tda/tda.py` → probe test passes.** If the probe test fails after a correct port, follow its docstring: STOP and return BLOCKED with the evidence.
- [ ] **Step 3: Add the statistical comparator to the harness, then the slow tests** (per the sparse-Rips ruling in Global Constraints).

First add to `tests/monolith_harness.py`:

```python
def _clamped_dim_slice(diagram: "np.ndarray", dim: int, max_edge: float) -> "np.ndarray":
    """(birth, death) rows of one H-dim; inf deaths clamped to max_edge (pipeline convention)."""
    import numpy as np
    d = np.asarray(diagram)
    if d.size == 0:
        return np.empty((0, 2))
    bd = d[d[:, 0] == dim][:, 1:3].copy()
    bd[~np.isfinite(bd[:, 1]), 1] = max_edge
    return bd


def assert_diagram_pkls_statistically_equal(
    a, b, max_hom_dim: int, max_edge: float,
    sample: int = 25, rel_tol: float = 0.05, w2_gate: float = None, seed: int = 0,
):
    """Sparse-Rips-tolerant diagram equivalence (regression guard, NOT primary
    proof of port fidelity — see the Task-4 ruling: gudhi sparse Rips is
    nondeterministic across processes, so bar-exact equality is unattainable
    for c2/network by construction).

    Checks: same diagram count; for `sample` seeded-random diagrams, per H-dim:
    bar-count delta <= max(2, rel_tol * count), total-persistence relative
    delta <= rel_tol, and exact-hera W2(a_i, b_i) <= w2_gate.
    w2_gate=None → CALIBRATION MODE: collect and print the max observed W2 per
    manifold instead of asserting; the caller then hard-codes 3x that value.
    """
    import pickle
    import numpy as np
    from pathlib import Path
    from gudhi.hera import wasserstein_distance
    A = pickle.loads(Path(a).read_bytes())
    B = pickle.loads(Path(b).read_bytes())
    assert len(A) == len(B), (a, len(A), len(B))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=min(sample, len(A)), replace=False)
    observed = 0.0
    for i in idx:
        for k in range(max_hom_dim + 1):
            xa = _clamped_dim_slice(A[i], k, max_edge)
            xb = _clamped_dim_slice(B[i], k, max_edge)
            ca, cb = len(xa), len(xb)
            assert abs(ca - cb) <= max(2, rel_tol * max(ca, cb, 1)), (a, i, k, ca, cb)
            ta = float((xa[:, 1] - xa[:, 0]).sum()) if ca else 0.0
            tb = float((xb[:, 1] - xb[:, 0]).sum()) if cb else 0.0
            assert abs(ta - tb) <= rel_tol * max(ta, tb, 1e-9), (a, i, k, ta, tb)
            w2 = float(wasserstein_distance(xa, xb, order=2.0, internal_p=2.0))
            observed = max(observed, w2)
            if w2_gate is not None:
                assert w2 <= w2_gate, (a, i, k, w2, w2_gate)
    return observed
```

Then the slow tests. **Calibration procedure (one-time, documented in the test):** first run the equivalence comparison with `w2_gate=None` to print per-manifold max observed W2; hard-code `W2_GATES = {"c2": <3x observed>, "network": <3x observed>}` into `tests/test_tda.py` with a comment recording the observed values and date; then the committed test asserts with those gates.

```python
# tests/test_tda.py (append)
from tests import monolith_harness as mh
from uav_tda.workspace import Workspace

# Calibrated per the Task-4 ruling: 3x max W2 observed between the monolith
# oracle and package tda runs on debug data (values recorded by the implementer
# during calibration; see task-4-report.md).
W2_GATES = {"c2": None, "network": None}  # replace None with calibrated floats


@pytest.mark.slow
def test_tda_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    import json
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "tda")
    from uav_tda.data import run_prep
    from uav_tda.tda import run_tda
    ws = Workspace.at(tmp_path)
    (tmp_path / "data").symlink_to(repo_root / "data")
    ws.ensure()
    run_prep(ws, debug=True)
    run_tda(ws, debug=True)
    mh.assert_json_equal(oracle / "outputs" / "max_edge_lengths.json",
                         ws.outputs_dir / "max_edge_lengths.json")
    assert np.array_equal(np.load(oracle / "outputs" / "reference_indices.npy"),
                          np.load(ws.outputs_dir / "reference_indices.npy"))
    max_edge = json.loads((ws.outputs_dir / "max_edge_lengths.json").read_text())
    for s in ("train", "val", "test"):
        # physical (exact Rips) is cross-process reproducible: exact comparison.
        mh.assert_diagram_pkls_equal(
            oracle / "outputs" / "persistence_diagrams" / f"physical_{s}.pkl",
            ws.persistence_dir / f"physical_{s}.pkl")
        # sparse manifolds: statistical comparator (see Task-4 ruling).
        for m in ("c2", "network"):
            mh.assert_diagram_pkls_statistically_equal(
                oracle / "outputs" / "persistence_diagrams" / f"{m}_{s}.pkl",
                ws.persistence_dir / f"{m}_{s}.pkl",
                max_hom_dim=config.MAX_HOM_DIM[m], max_edge=max_edge[m],
                sample=25, rel_tol=0.05, w2_gate=W2_GATES[m])


@pytest.mark.slow
def test_tda_spotcheck_against_production_artifacts(repo_root):
    """Recompute real test flows against the repo reference cloud. physical:
    exact match vs production pkl. c2/network: statistical gates only (sparse
    Rips is not bit-reproducible across processes — Task-4 ruling)."""
    import json
    import pickle

    import pandas as pd
    from gudhi.hera import wasserstein_distance
    from uav_tda.tda import _persistence_for_point
    out = repo_root / "outputs"
    ref_idx = np.load(out / "reference_indices.npy")
    max_edge = json.loads((out / "max_edge_lengths.json").read_text())
    for m in config.MANIFOLDS:
        ref = pd.read_csv(out / f"{m}_train.csv").to_numpy()[ref_idx]
        test_pts = pd.read_csv(out / f"{m}_test.csv").to_numpy()
        stored = pickle.loads((out / "persistence_diagrams" / f"{m}_test.pkl").read_bytes())
        for i in (0, 100, 5000):
            fresh = _persistence_for_point(
                test_pts[i], ref, max_edge[m],
                config.MAX_HOM_DIM[m] + 1, config.SPARSE_RIPS_EPSILON[m])
            if config.SPARSE_RIPS_EPSILON[m] is None:
                assert np.allclose(
                    fresh[np.lexsort(fresh.T[::-1])],
                    np.asarray(stored[i])[np.lexsort(np.asarray(stored[i]).T[::-1])]), (m, i)
            else:
                for k in range(config.MAX_HOM_DIM[m] + 1):
                    xa = mh._clamped_dim_slice(fresh, k, max_edge[m])
                    xb = mh._clamped_dim_slice(np.asarray(stored[i]), k, max_edge[m])
                    assert abs(len(xa) - len(xb)) <= max(2, 0.05 * max(len(xa), len(xb), 1)), (m, i, k)
                    w2 = float(wasserstein_distance(xa, xb, order=2.0, internal_p=2.0))
                    assert w2 <= W2_GATES[m], (m, i, k, w2)
```

- [ ] **Step 4: Calibrate, then run the slow tests.** The oracle debug tda cache already exists (`.done-tda` present — verify). Calibration: temporary run with `w2_gate=None` prints per-manifold max observed W2; hard-code `W2_GATES` at 3× those values (record observed values + date in the test comment AND the task report). Then `python3 -m pytest tests/test_tda.py -m slow -v` must PASS with the gates active. The spot-check reads repo `outputs/` files — read-only, allowed.
- [ ] **Step 5: Commit** — `"feat: port tda phase with determinism probe and statistical sparse-Rips equivalence"`

---

### Task 5: `features.py` — port + equivalence

**Files:** Create `uav_tda/features.py`; Test `tests/test_features.py`.

**Interfaces:** `run_features(ws: Workspace, debug: bool = False) -> None` writing `persistence_imagers.pkl` and `tda_features/{summary,images,combined}_{train,val,test}.csv` + `feature_names.json` (exact monolith file set — enumerate from `cmd_features`, lines 932–962, before porting). Port source: lines 654–962.

- [ ] **Step 0: Add the staging helper to `tests/monolith_harness.py`** (used by T5–T8; per the sparse-Rips ruling, downstream tests stage the ORACLE's upstream artifacts and never re-run package tda):

```python
def stage_oracle(oracle_ws, ws, include=("outputs",)):
    """Copy oracle workspace subtrees into a package Workspace root so a phase
    under test runs on IDENTICAL inputs to the monolith's own run (Task-4
    ruling: quarantines sparse-Rips cross-process nondeterminism)."""
    import shutil
    from pathlib import Path
    for sub in include:
        src = Path(oracle_ws) / sub
        dst = Path(ws.root) / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
```

- [ ] **Step 1: Failing equivalence test** — staged-input pattern: `oracle = ensure_debug_phase(..., "features")`; stage the oracle's `outputs/` into the tmp ws; run ONLY `run_features(ws, debug=True)`; compare every CSV the monolith's features phase wrote, bit-tight (`float_rtol=1e-9`) — deterministic given identical diagrams.

```python
# tests/test_features.py
import pytest

from tests import monolith_harness as mh
from uav_tda.workspace import Workspace


@pytest.mark.slow
def test_features_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "features")
    from uav_tda.features import run_features
    ws = Workspace.at(tmp_path)
    ws.ensure()
    mh.stage_oracle(oracle, ws, include=("outputs",))
    # remove the oracle's own features output so we prove OUR phase rebuilds it
    import shutil
    shutil.rmtree(ws.tda_features_dir, ignore_errors=True)
    (ws.outputs_dir / "persistence_imagers.pkl").unlink()
    run_features(ws, debug=True)
    oracle_csvs = sorted((oracle / "outputs" / "tda_features").glob("*.csv"))
    assert oracle_csvs, "oracle produced no feature CSVs — harness bug"
    for oc in oracle_csvs:
        mh.assert_csvs_equal(oc, ws.tda_features_dir / oc.name)
```

- [ ] **Step 2: fail → port → pass → commit** — `"feat: port features phase with staged-input monolith equivalence"`. PersistenceImager fitting is deterministic given identical diagrams; if CSVs mismatch, check dropped-all-zero-column bookkeeping (`determine_drops`, lines 898–913) first.

---

### Task 6: `supervised.py` — port + equivalence

**Files:** Create `uav_tda/supervised.py`; Test `tests/test_supervised.py`.

**Interfaces:** `run_supervised(ws: Workspace, debug: bool = False) -> None` writing `supervised_metrics.csv`, `supervised_summary.csv`, per-class/confusion CSVs, model pkls, figures — mirror `cmd_supervised` (lines 1323–1358). Port source: lines 964–1377.

- [ ] **Step 1: Failing equivalence test** — staged-input pattern (T5 Step 0's `stage_oracle`): oracle `"supervised"`; stage the oracle's `outputs/` (prep CSVs + tda_features) into tmp ws; run ONLY `run_supervised(ws, debug=True)`; compare `supervised_metrics.csv` and `supervised_summary.csv` with `float_rtol=1e-6` (seeded sklearn models; lbfgs/SVC-probability introduce tiny float noise across processes — 1e-6 is the documented bound; if it fails at 1e-6 report the max delta rather than loosening). Compare the confusion-matrix CSVs exactly (integers).
- [ ] **Step 2: fail → port → pass → commit** — `"feat: port supervised phase with staged-input monolith equivalence"`. Debug-mode grid search runs in minutes; the oracle build is the slow part (once).

---

### Task 7: `unsupervised.py` — port + equivalence

**Files:** Create `uav_tda/unsupervised.py`; Test `tests/test_unsupervised.py`.

**Interfaces:** `run_unsupervised(ws: Workspace, debug: bool = False, n_jobs: int = -1) -> None` writing `unsupervised_distances.csv`, `unsupervised_per_class_auc.csv`, `unsupervised_overall_metrics.csv`, `unsupervised_rule.json` (verify exact set from `save_unsupervised_tables` + `cmd_unsupervised`, lines 1735–1830) and the three figures. Port source: lines 1379–1830. NOTE: this phase's W2 is exact (no Hera delta) — deterministic; combined detection uses max-pool + flag patterns (the monolith's own semantics — port them faithfully; the probe's sum-semantics live in `uav_tda.probe` and are NOT to be merged here).

- [ ] **Step 1: Failing equivalence test** — staged-input pattern: oracle `"unsupervised"`; stage the oracle's `outputs/` into tmp ws (prep CSVs + persistence diagrams + reference/max-edge files); delete the oracle's own unsupervised tables from the staged ws results (stage only `outputs/`, so nothing to delete — the phase writes to `ws.tables_dir`/`ws.figures_dir` which start empty); run ONLY `run_unsupervised(ws, debug=True)`; `assert_csvs_equal` on the distance and AUC tables (`float_rtol=1e-9` — exact-hera W2 on identical diagrams is deterministic; if it fails, report max delta), exact on the rule JSON.
- [ ] **Step 2: fail → port → pass → commit** — `"feat: port unsupervised phase with staged-input monolith equivalence"`.

---

### Task 8: `evaluate.py` — port + equivalence

**Files:** Create `uav_tda/evaluate.py`; Test `tests/test_evaluate.py`.

**Interfaces:** `run_evaluate(ws: Workspace, debug: bool = False) -> None` writing `final_supervised.csv`, `final_unsupervised.csv`, ablation tables, and the three summary figures — mirror `cmd_evaluate` (lines 2182–2214). Port source: lines 1832–2239.

- [ ] **Step 1: Failing equivalence test** — staged-input pattern: oracle `"evaluate"`; stage the oracle's `outputs/` AND `results/` into tmp ws (`stage_oracle(oracle, ws, include=("outputs", "results"))` — evaluate reads earlier phases' tables), then delete from the staged `ws.tables_dir`/`ws.figures_dir` every artifact the evaluate phase itself writes (enumerate from `cmd_evaluate`, lines 2182–2214, before writing the test) so the test proves OUR phase regenerates them; run ONLY `run_evaluate(ws, debug=True)`; `assert_csvs_equal` on every `final_*.csv` and ablation CSV the oracle produced (`float_rtol=1e-6`, ablation retrains seeded RFs); assert the figure files exist and are nonempty (no pixel comparison).
- [ ] **Step 2: fail → port → pass → commit** — `"feat: port evaluate phase with staged-input monolith equivalence"`.

---

### Task 9: CLI phases, `all`, and monolith retirement docs

**Files:**
- Modify: `uav_tda/cli.py` (add subcommands)
- Modify: `README.md`, `CLAUDE/CLAUDE.md`, `CLAUDE/PROJECT_BRIEF.md` (decision log append only)
- Test: `tests/test_cli_phases.py`

**Interfaces:** `uav-tda {prep,tda,features,supervised,unsupervised,evaluate,all}` each accepting `--debug`, `--root PATH` (default: repo), plus phase-specific flags mirroring the monolith (`tda`: `--manifold/--split/--seed`); `all` chains the six phases in order. Each subcommand builds `Workspace.at(root)` (default `Workspace.default()`), calls the module `run_*`, returns 0.

- [ ] **Step 1: Failing CLI wiring test**

```python
# tests/test_cli_phases.py
from uav_tda import cli


def test_all_phase_subcommands_registered():
    parser = cli.build_parser()
    subactions = next(a for a in parser._actions if hasattr(a, "choices"))
    for cmd in ("probe", "prep", "tda", "features", "supervised",
                "unsupervised", "evaluate", "all"):
        assert cmd in subactions.choices, cmd


def test_prep_debug_smoke(tmp_path, repo_root=None):
    # wiring smoke only: --root at tmp with data symlink, --debug; asserts rc 0 and outputs exist
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    (tmp_path / "data").symlink_to(repo_root / "data")
    rc = cli.main(["prep", "--debug", "--root", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "outputs" / "labels_train.csv").is_file()
```

- [ ] **Step 2: fail → implement subcommands → pass.**
- [ ] **Step 3: Docs.** README: replace the pipeline-commands section's framing — `uav-tda` is the maintained CLI; `pipeline.py` retained verbatim as the frozen equivalence oracle and historical record of the paper's production run (do not delete the old command examples; mark them "legacy, equivalent"). `CLAUDE/CLAUDE.md`: update "Running the pipeline" to `uav-tda` with the same note, and update File discipline (`uav_tda/`, `tests/`, `docs/` are write zones; `pipeline.py` frozen-oracle). `CLAUDE/PROJECT_BRIEF.md`: append ONE decision-log entry dated with today's date: Phase-2 strangler completed; `uav_tda` package is the maintained implementation, equivalence-tested against `pipeline.py` in debug mode per artifact; `pipeline.py` remains frozen as oracle; §9 file discipline extended accordingly.
- [ ] **Step 4: Full suite** — `python3 -m pytest -m "not slow" -q` all green; then `python3 -m pytest -m slow -q` (cache warm from earlier tasks — should complete without rebuilding oracles).
- [ ] **Step 5: Commit** — `"feat: uav-tda phase subcommands + monolith retirement docs"`.

---

## Self-Review

- **Spec coverage** (roadmap Phase 2): extract data/tda/scoring modules ✓ (T3–T8); golden-master gates ✓ (per-task equivalence vs fresh monolith, rationale in Global Constraints); CLI mirroring subcommands ✓ (T9); retire monolith ✓ (T9 docs-level, pipeline.py kept as oracle — deliberate deviation from "shim or deleted", justified: it is the paper's historical record and the ongoing oracle).
- **Placeholder scan:** all test steps carry runnable code; port steps cite exact line ranges + exhaustive permitted-change list; no TBDs.
- **Type consistency:** `Workspace.at/default/ensure` consistent across T1 and all consumers; `run_prep(ws, debug)`, `run_tda(ws, manifold, split, seed, debug, n_jobs)`, `run_features(ws, debug)`, `run_supervised(ws, debug)`, `run_unsupervised(ws, debug, n_jobs)`, `run_evaluate(ws, debug)` consistent between task Interfaces and T9's CLI; harness names (`ensure_debug_phase`, `assert_csvs_equal`, `assert_diagram_pkls_equal`, `assert_json_equal`, `monolith_cache`) consistent T2→T3–T8.
- **Known risks, pre-ruled:** (a) sparse-Rips nondeterminism — T4's probe test is the tripwire with an explicit BLOCKED path; (b) sklearn float noise — T6/T8 use 1e-6 with a report-don't-loosen rule; (c) in-flight train rebuild — no test touches repo `outputs/` for writing; spot-check reads only completed test-split artifacts.
