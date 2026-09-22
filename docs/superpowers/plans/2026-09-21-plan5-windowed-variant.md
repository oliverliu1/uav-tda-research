# Time-Windowed Variant — Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the time-windowed multi-manifold persistence variant (window-only Rips, W2-medoid baseline, z-normalized scoring) and run the 4-window-size × (10 ordered + 10 shuffled) campaign that produces the manuscript's per-flow-vs-windowed compute frontier, contamination analysis, and attribution-survival tables.

**Architecture:** New single-responsibility module `uav_tda/windowed.py` reusing the established package machinery (metrics subset sums + z-norm helpers, manuscript bootstrap CIs, provenance sidecars, Phase-4 emitter/report patterns). One coupled invocation per run: val-normal windows → medoid baseline + z-stats → test windows → scores + metadata + timing. Campaign artifacts under `results/tables/rebuild/windowed/`; report `paper/WINDOWED_RESULTS.md`.

**Tech Stack:** Python 3.9 (`from __future__ import annotations`), numpy, pandas, gudhi (Rips + hera W2), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-21-windowed-variant-design.md` — the binding authority; every constraint below is copied from it.

## Global Constraints

- Windows: **W consecutive flows, FlowID order, non-overlapping, trailing partial window dropped**; shuffle control = same pipeline on a `np.random.default_rng(order_seed)` permutation of the test split.
- W grid: **{25, 50, 100, 200}**. Repetitions per W: **10 ordered repeats (repeat index 0–9) + 10 shuffle controls (order_seed 0–9)**.
- Per-window Rips: per-manifold on the window's own points; filtration cap = existing `outputs/max_edge_lengths.json`; hom dims = `config.MAX_HOM_DIM`. **Exact Rips preferred**; the Task-1 benchmark gate decides per manifold (budget: ≤ ~2 s/call at W=200); sparse ε=0.5 fallback per manifold only if over budget, recorded.
- Baseline per (manifold, W) = **W2-medoid** of val-Normal window diagrams (val Normal rows in their own order, chunked by W). Scores = per-dim W2 to baseline summed (hera, order=2, internal_p=2). **Z-normalization on the val-normal-window score distribution; same-run coupling** (baseline + z-stats + test scores in one invocation).
- Window ground truth: **majority label** (>0.5 attack fraction ⇒ positive); metadata keeps `attack_frac` + majority class. Contamination bins: `[0], (0,.25], (.25,.5], (.5,.75], (.75,1]`.
- CIs: `manuscript.bootstrap_mean_auc_ci`, B=2000, bootstrap_seed=0.
- Locked surfaces unchanged: golden-master metrics functions, probe default path, `pipeline.py`, `tools/`, published top-level `results/tables/*.csv`, pre-existing `paper/*.md`. New artifacts only under `results/tables/rebuild/windowed/`, `results/tables/rebuild/paper_snippets/`, `paper/WINDOWED_RESULTS.md`.
- All heavy runs detached (nohup+log+poll); slow tests `@pytest.mark.slow`; fast suite stays < ~1 min; report-not-loosen on any gate miss; Python 3.9 compatible; provenance sidecars on every artifact.

## File Structure

```
uav_tda/
  windowed.py      # NEW — everything windowed: windowing, diagrams, baseline, run_windowed, tables, emitter, report
  cli.py           # MODIFY: `windowed` + `windowed-report` subcommands
  config.py        # MODIFY (append): WINDOW_SIZES, WINDOWED_REPEATS, WINDOWED_SHUFFLE_SEEDS, CONTAMINATION_BINS
tests/
  test_windowed.py # NEW — all Phase-5 tests
results/tables/rebuild/windowed/   # OUTPUT
paper/WINDOWED_RESULTS.md          # OUTPUT
CLAUDE/PROJECT_BRIEF.md            # MODIFY in Task 4 only: append one decision-log entry
CLAUDE/CLAUDE.md                   # MODIFY in Task 4 only: add the two subcommands to the CLI list
```

---

### Task 1: Windowing core + exact-Rips benchmark gate

**Files:**
- Modify: `uav_tda/config.py` (append)
- Create: `uav_tda/windowed.py` (windowing + diagram section)
- Test: `tests/test_windowed.py`

**Interfaces (produced):**
- `config.WINDOW_SIZES = (25, 50, 100, 200)`; `config.WINDOWED_REPEATS = 10`; `config.WINDOWED_SHUFFLE_SEEDS = tuple(range(10))`; `config.CONTAMINATION_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)` (bin edges; bin 0 = exactly 0).
- `make_windows(n_rows: int, w: int, order: np.ndarray | None = None) -> list[np.ndarray]` — index arrays; identity order when None; trailing partial dropped.
- `window_diagram(points: np.ndarray, max_edge: float, max_hom_dim: int, sparse: float | None) -> np.ndarray` — one Rips; returns `(n,3)` `[dim, birth, death]` float array (same convention as `tda._persistence_for_point`, whose body it mirrors minus the query-point stacking).
- `benchmark_exact_rips(ws, w: int = 200, n_trials: int = 3) -> dict[str, dict]` — per manifold: median exact-call seconds on real test-split windows; used by Task-1 Step 5 to fix `WINDOWED_SPARSE: dict[str, float | None]` (module constant: None where exact is within budget).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_windowed.py
import numpy as np
import pytest

from uav_tda import config, windowed


def test_config_constants():
    assert config.WINDOW_SIZES == (25, 50, 100, 200)
    assert config.WINDOWED_REPEATS == 10
    assert config.WINDOWED_SHUFFLE_SEEDS == tuple(range(10))


def test_make_windows_drops_trailing_partial():
    wins = windowed.make_windows(105, 25)
    assert len(wins) == 4
    assert all(len(w) == 25 for w in wins)
    assert wins[0].tolist() == list(range(25))
    assert wins[3].tolist() == list(range(75, 100))  # rows 100-104 dropped


def test_make_windows_respects_permutation():
    order = np.arange(50)[::-1]  # reversed
    wins = windowed.make_windows(50, 10, order=order)
    assert wins[0].tolist() == list(range(49, 39, -1))
    # every row appears at most once across windows
    flat = np.concatenate(wins)
    assert len(np.unique(flat)) == len(flat)


def test_window_diagram_shape_and_determinism_exact():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 5))
    d1 = windowed.window_diagram(pts, max_edge=0.8, max_hom_dim=1, sparse=None)
    d2 = windowed.window_diagram(pts, max_edge=0.8, max_hom_dim=1, sparse=None)
    assert d1.ndim == 2 and d1.shape[1] == 3
    assert np.array_equal(d1, d2)  # exact Rips: bit-identical, no sort needed
    assert set(np.unique(d1[:, 0])) <= {0.0, 1.0}
```

- [ ] **Step 2: Run to verify fail** — `python3 -m pytest tests/test_windowed.py -v` → import error.
- [ ] **Step 3: Implement** — config appends; `make_windows` (`order = np.arange(n_rows) if order is None`; slice into `n_rows // w` chunks of `w`); `window_diagram` (mirror `tda._persistence_for_point` lines: `gudhi.RipsComplex(points=..., max_edge_length=...)` + optional `sparse=` kwarg, `create_simplex_tree(max_dimension=max_hom_dim + 1)`, `persistence()`, rows `[dim, birth, death]`, empty → `(0,3)`); `benchmark_exact_rips` (load `{m}_test.csv` per manifold via a `Workspace`, take the first `n_trials` windows of size `w`, time exact calls with `time.perf_counter`, return `{manifold: {"median_s": float, "w": w}}`).
- [ ] **Step 4: Tests pass; fast suite green.**
- [ ] **Step 5: RUN THE BENCHMARK GATE** — `python3 -c "from uav_tda.workspace import Workspace; from uav_tda.windowed import benchmark_exact_rips; print(benchmark_exact_rips(Workspace.default()))"`. Fix the module constant `WINDOWED_SPARSE = {"c2": ..., "network": ..., "physical": ...}` — `None` (exact) for every manifold with median ≤ 2.0 s, else `0.5`, with a comment recording the measured medians + date. Add a unit test asserting `WINDOWED_SPARSE` keys == the three manifolds.
- [ ] **Step 6: Commit** — `"feat: windowed core (make_windows, window_diagram) + exact-Rips benchmark gate"`

---

### Task 2: Baseline medoid + coupled run

**Files:**
- Modify: `uav_tda/windowed.py`
- Test: `tests/test_windowed.py` (append)

**Interfaces:**
- Consumes: Task 1's functions/constants; `metrics.MANIFOLD_SUBSETS`; artifacts `outputs/{m}_{split}.csv`, `outputs/labels_{split}.csv`, `outputs/max_edge_lengths.json`.
- Produces:
  - `w2_distance(d1: np.ndarray, d2: np.ndarray, max_edge: float, max_hom_dim: int) -> float` — per-dim clamped slices (reuse `probe._slice_dim`) + hera W2 (order=2, internal_p=2), summed over dims.
  - `baseline_medoid_diagram(diagrams: list[np.ndarray], max_edge: float, max_hom_dim: int) -> tuple[int, np.ndarray]` — pairwise `w2_distance` matrix; returns (medoid index, medoid diagram).
  - `run_windowed(ws, w: int, order_seed: int | None = None, repeat: int = 0) -> tuple[pd.DataFrame, dict, dict]` — the coupled pass. Returns `(window_df, stats, timing)`:
    - `window_df` columns: `window_idx, start_pos, majority_label, attack_frac, W2_c2, W2_network, W2_physical` + the 7 subset sums, plus `Z2_<...>` variants for all 10 score columns (z-scored manifolds + recomputed subset sums, matching `metrics.apply_znorm` semantics).
    - `stats` = `{manifold: (mean, std)}` from val-normal window scores (degenerate-σ guard as in `metrics.znorm_stats_from_val`).
    - `timing` = `{"total_s", "n_windows", "per_window_s", "rips_s", "w2_s", "baseline_s"}`.
    - Behavior: val-Normal rows (labels_val == Normal, original order) chunked by `w` → diagrams → per-manifold medoid baseline + val score distribution → z-stats; test split (permuted iff `order_seed is not None`) → windows → diagrams → scores. `majority_label` = modal label of the window (ties: attack wins over Normal, then alphabetical — document); `attack_frac` = fraction of non-Normal flows.

- [ ] **Step 1: Failing unit tests** — `w2_distance` symmetry + zero-on-identical (hand-built 2-bar diagrams); `baseline_medoid_diagram` picks the center of three hand-built diagrams (two similar + one outlier ⇒ medoid is one of the similar pair, assert index in {0,1}); majority/attack_frac labeling on a synthetic window of labels (6 Normal + 4 Sybil ⇒ majority Normal, frac 0.4; 4 Normal + 6 Sybil ⇒ majority Sybil, frac 0.6).
- [ ] **Step 2: fail → implement → pass.** Labeling helpers may be standalone functions (`window_majority_label(labels) -> str`, `window_attack_frac(labels) -> float`) so the unit tests need no filesystem.
- [ ] **Step 3: Slow smoke test** (append): `run_windowed(Workspace.default(), w=100)` on real data — assert `len(window_df) == 18326 // 100`; all score columns finite; `stats` has 3 manifolds with std > 0; timing dict populated with `total_s > 0`; windows overlapping the big attack blocks have higher mean `Z2_all_three` than pure-normal windows (weak sanity: mean score of `attack_frac > 0.9` windows > mean of `attack_frac == 0` windows). Mark `@pytest.mark.slow`. Expected runtime: minutes.
- [ ] **Step 4: fast suite + slow smoke green; commit** — `"feat: windowed coupled run (medoid baseline, z-stats, scores, timing)"`

---

### Task 3: Tables, contamination curve, frontier + CLI

**Files:**
- Modify: `uav_tda/windowed.py`, `uav_tda/cli.py`
- Test: `tests/test_windowed.py` (append)

**Interfaces:**
- Consumes: Task 2's `run_windowed` outputs written per run to `results/tables/rebuild/windowed/run_w{W}_{arm}{k}.csv` (+ stats/timing JSON + provenance), where arm ∈ {`ordered`, `shuffled`}, k = repeat index or order_seed.
- Produces:
  - `write_run_artifacts(ws, w, arm, k, window_df, stats, timing) -> None` and `missing_runs(ws) -> list[tuple]` over the full campaign grid (4 W × 2 arms × 10).
  - `build_windowed_tables(ws, B=2000, bootstrap_seed=0) -> dict[str, pd.DataFrame]` returning: `detection` (W × arm × subset × scoring: mean/std/ci_lo/ci_hi over the 10 runs, window-level majority-label AUC via `manuscript.bootstrap_mean_auc_ci` treating runs as seeds); `attribution` (W × attack × manifold, ordered arm only, one-vs-rest majority-label AUC, dominant flag per (W, attack)); `contamination` (W × bin: n_windows, mean raw/znorm all_three score, detection rate at the val-95th-percentile threshold, per majority-attack class where n ≥ 10); `frontier` (rows = 4 W + `per_flow`: znorm all_three AUC mean/std/CI + per-decision seconds; per-flow row from `results/tables/rebuild/binary_auc.csv` (Phase 4) + a timed 20-flow per-flow sample using `probe` internals, seconds recorded).
  - CLI: `uav-tda windowed --w W [--order-seed N] [--repeat K]` (single run + artifacts) and `uav-tda windowed-report [--bootstrap B]` (ensure campaign → tables → CSVs + provenance → snippet + report, wired fully in Task 4).
- [ ] **Step 1: Failing tests** — synthetic run CSVs (2 W values × 2 arms × 2 runs, tiny window counts with known label/score structure) in tmp ws: `missing_runs` grid math; `detection` has the right row count and `ci_lo <= mean <= ci_hi`; `attribution` exactly one dominant per (W, attack); `contamination` bins partition windows (n sums match) and bin 0 contains only pure-normal windows; `frontier` has 5 rows when a fake Phase-4 binary_auc.csv + fake timing are provided. CLI registration test (both subcommands, flags).
- [ ] **Step 2: fail → implement → pass; fast suite green; commit** — `"feat: windowed tables (detection, attribution, contamination, frontier) + CLI"`

---

### Task 4: Emitter, campaign, report, docs

**Files:**
- Modify: `uav_tda/windowed.py` (emitter + report generator), `uav_tda/cli.py` (finish `windowed-report`), `CLAUDE/PROJECT_BRIEF.md` (append ONE decision-log entry), `CLAUDE/CLAUDE.md` (add subcommands to the CLI list)
- Test: `tests/test_windowed.py` (append golden-string emitter test)
- Outputs (committed): 80 runs' artifacts, 4 table CSVs, `paper_snippets/windowed_frontier_pgfplots.tex`, `paper/WINDOWED_RESULTS.md`

- [ ] **Step 1: Golden-string emitter test** — `emit_frontier_pgfplots(frontier_df) -> str`: an `\addplot coordinates {...}` block of `({per_decision_s:.3g}, {mean:.3f}) +- (0, {std:.3f})` points ordered per-flow, W=25..200 (log-x-friendly), plus a `\node` label per point (`W=25` … `per-flow`); hand-write the expected literal for a 3-row synthetic frontier. fail → implement → pass.
- [ ] **Step 2: RUN THE CAMPAIGN (detached):** `uav-tda windowed-report --bootstrap 2000` — runs all 80 missing runs then builds everything. Estimated 2–6 h; artifacts land incrementally; restartable via `missing_runs`. Poll the log.
- [ ] **Step 3: SANITY GATES on results** (report-not-loosen; STOP-and-report if a science gate trips): pure-block attacks (Flooding/Sybil/Wormhole majority windows) detection AUC ≥ 0.85 at every W in the ordered arm; contamination bin 0 threshold-detection rate ≈ ≤ 0.10; per-decision seconds monotonically decreasing per window count (frontier x-axis sane); ordered-arm repeat determinism — if `WINDOWED_SPARSE` is all-exact, the 10 ordered repeats per W must be bit-identical (assert once, then report the determinism result); attribution survival — report at which W (if any) each attack's dominant manifold flips (a flip is a FINDING to report prominently, not an error).
- [ ] **Step 4: `paper/WINDOWED_RESULTS.md`** generated with: config header; detection table (W × arm × scoring); frontier table + snippet pointer; contamination analysis (the Blackhole mixed-window finding, quantified); shuffle-control comparison (ordered vs shuffled AUC deltas + interpretation per the spec); attribution-survival table; determinism note; "pending author sign-off" framing. Every number machine-generated.
- [ ] **Step 5: Docs** — decision-log entry (dated; Phase 5 complete; windowed variant per spec; headline frontier numbers; pending sign-off); CLAUDE.md CLI list updated.
- [ ] **Step 6: Fast suite; commit code + artifacts** — `"feat: windowed campaign, compute frontier, contamination analysis (WINDOWED_RESULTS)"`

---

## Self-Review

- **Spec coverage:** §2.1 windowing/shuffle → T1 (`make_windows(order)`) + T2 (`order_seed`) ✓; §2.2 window-only Rips + exact-gate → T1 ✓; §2.3 medoid baseline + z-norm same-run coupling → T2 ✓; §3.1 detection tables → T3 ✓; §3.2 contamination → T3/T4 ✓; §3.3 attribution survival → T3/T4 ✓; §3.4 frontier incl. per-flow arm + timing → T3 (builder) + T4 (emitter/campaign) ✓; §3.5 10+10 reps + bootstrap CIs → constants (T1) + campaign grid (T3) ✓; §4 architecture/CLI/artifacts → T1–T4 ✓; §5 tests incl. determinism + gates → T1/T2/T4 ✓; §6 campaign → T4 ✓.
- **Placeholder scan:** all test steps carry code or exact assertions; emitter format specified with golden-test requirement; tie-break rule for majority label stated. No TBDs.
- **Type consistency:** `make_windows`/`window_diagram`/`run_windowed` signatures consistent across tasks; `window_df` column names consistent T2→T3; run-artifact filename scheme consistent T3↔T4; `WINDOWED_SPARSE` defined T1, consumed T2.
- **Pre-ruled risks:** (a) benchmark gate may force sparse c2 — then ordered-repeat determinism degrades to statistical for c2 only; the T4 determinism gate applies only to all-exact configs, stated; (b) attribution flip at any W = STOP-and-report science finding; (c) medoid pool at W=200 is only 19 val windows — small but sufficient for a medoid; noted for the report's limitations line.
