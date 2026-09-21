# Ten Seeds + Bootstrap CIs + Paper-Emitting Tables — Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the probe evaluation from 3 to 10 seeds, attach bootstrap confidence intervals to every AUC, and emit the paper's table rows and pgfplots coordinates directly from data — delivering the §V "ten seeds with bootstrap confidence intervals" manuscript promise and eliminating paper↔code drift.

**Architecture:** New module `uav_tda/manuscript.py` consuming the Phase-3 per-seed rebuild artifacts (`results/tables/rebuild/probe_distances_seed{N}.csv` + `znorm_stats_seed{N}.json`). Seeds 42/7/123 are reused as-is (identical config: top-K 50, δ 0.2, w2_timeout 30); seven new seeds are produced by the existing `uav-tda probe --znorm --w2-timeout 30` path. Uncertainty is reported two ways: (i) across-seed mean ± std (the paper's existing convention), (ii) a **cluster bootstrap of the across-seed mean** — resample flows (stratified by class, with replacement) *within* each seed, recompute the per-seed AUC, average across seeds, repeat B times, take the 2.5/97.5 percentiles. LaTeX emitters generate the exact `main.tex` table-row and pgfplots-coordinate syntax.

**Tech Stack:** Python 3.9 (`from __future__ import annotations`), numpy, pandas, sklearn.metrics.roc_auc_score, pytest. No new dependencies.

**Spec:** roadmap Phase 4; paper §V ("The present three-seed evaluation will expand to ten seeds with bootstrap confidence intervals"); `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex` (Table `tab:per_attack_auc` row format, Fig. `fig:binary_auc` pgfplots coordinate format).

## Global Constraints

- **Seeds:** `MANUSCRIPT_SEEDS = (42, 7, 123, 0, 1, 2, 3, 4, 5, 6)` — the paper's three first, then seven sequential; add to `uav_tda/config.py` with a comment. Probe config for all runs: `per_class=200, top_k=50, delta=0.2, w2_timeout=30.0` (identical to Phase 3's three).
- **Reuse, don't recompute:** seeds 42/7/123 artifacts exist in `results/tables/rebuild/` — a seed is (re)run ONLY if its `probe_distances_seed{N}.csv` or `znorm_stats_seed{N}.json` is missing. Never overwrite an existing seed artifact.
- **Bootstrap determinism:** every bootstrap uses `np.random.default_rng(bootstrap_seed)` with `bootstrap_seed=0` default; B=2000 default. Same inputs + seed ⇒ identical CIs.
- **Stratified resampling:** within a seed, resample indices with replacement independently *per class label* (preserving each class's count), so no bootstrap replicate loses a class entirely (AUC would be undefined).
- Locked surfaces unchanged: golden-master `metrics` functions, `probe.py` default path, `pipeline.py`, `tools/`, published top-level `results/tables/*.csv`, pre-existing `paper/*.md`. New outputs: `results/tables/rebuild/binary_auc.csv`, `results/tables/rebuild/manifold_attribution.csv`, `results/tables/rebuild/paper_snippets/*.tex`, new `paper/MANUSCRIPT_STATS.md`.
- LaTeX emitters are pure functions string-tested against golden outputs — numbers formatted `{:.2f}` for table cells (matching the paper's 2-decimal style) and `{:.3f}` for pgfplots coordinates/errors (matching `(C2, 0.749) +- (0, 0.027)`).
- Runs >10 min detached (nohup+log+poll). The 7-seed campaign ≈ 2.5–3.5h.
- Python 3.9 compatible.

## File Structure

```
uav_tda/
  config.py        # MODIFY (append): MANUSCRIPT_SEEDS
  manuscript.py    # NEW — bootstrap machinery, table builders, LaTeX emitters, build_manuscript_stats()
  cli.py           # MODIFY: new `manuscript-report` subcommand
tests/
  test_manuscript.py       # NEW — bootstrap unit tests, table-builder tests on synthetic seeds, emitter golden-string tests
results/tables/rebuild/    # OUTPUT: binary_auc.csv, manifold_attribution.csv, paper_snippets/*.tex, 7 new seeds' artifacts
paper/MANUSCRIPT_STATS.md  # OUTPUT (committed)
CLAUDE/PROJECT_BRIEF.md    # MODIFY in Task 3 only: append one decision-log entry
```

---

### Task 1: Bootstrap machinery

**Files:**
- Modify: `uav_tda/config.py` (append `MANUSCRIPT_SEEDS`)
- Create: `uav_tda/manuscript.py` (bootstrap section)
- Test: `tests/test_manuscript.py`

**Interfaces (consumed by Tasks 2–3):**
- `config.MANUSCRIPT_SEEDS = (42, 7, 123, 0, 1, 2, 3, 4, 5, 6)`
- `stratified_resample_indices(labels: np.ndarray, rng) -> np.ndarray` — per-class with-replacement index resample, preserving class counts; output indices reference the original rows.
- `bootstrap_mean_auc_ci(per_seed: list[tuple[np.ndarray, np.ndarray]], B: int = 2000, bootstrap_seed: int = 0, labels_per_seed: list[np.ndarray] | None = None) -> tuple[float, float]` — cluster bootstrap: each element of `per_seed` is `(y_binary, scores)` for one seed (`labels_per_seed[i]` carries the multiclass labels used for stratification; when None, stratify on `y_binary`). Per replicate: for each seed, resample rows stratified, compute `roc_auc_score`, average across seeds; return `(ci_lo, ci_hi)` percentiles 2.5/97.5 of the B replicate means.

- [ ] **Step 1: Failing tests**

```python
# tests/test_manuscript.py
import numpy as np
import pytest

from uav_tda import config, manuscript


def test_manuscript_seeds_constant():
    assert config.MANUSCRIPT_SEEDS == (42, 7, 123, 0, 1, 2, 3, 4, 5, 6)
    assert config.MANUSCRIPT_SEEDS[:3] == config.PROBE_SEEDS


def test_stratified_resample_preserves_class_counts():
    labels = np.array(["A"] * 10 + ["B"] * 5)
    rng = np.random.default_rng(0)
    idx = manuscript.stratified_resample_indices(labels, rng)
    assert len(idx) == 15
    resampled = labels[idx]
    assert (resampled == "A").sum() == 10
    assert (resampled == "B").sum() == 5


def test_bootstrap_ci_contains_truth_and_is_deterministic():
    rng = np.random.default_rng(1)
    per_seed = []
    for _ in range(3):
        y = np.array([0] * 200 + [1] * 200)
        scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.2, 1, 200)])
        per_seed.append((y, scores))
    lo, hi = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=0)
    assert 0.5 < lo < hi < 1.0
    # true AUC for N(0,1) vs N(1.2,1) is Phi(1.2/sqrt(2)) ~= 0.802
    assert lo < 0.802 < hi
    lo2, hi2 = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=0)
    assert (lo, hi) == (lo2, hi2)
    lo3, hi3 = manuscript.bootstrap_mean_auc_ci(per_seed, B=500, bootstrap_seed=1)
    assert (lo, hi) != (lo3, hi3)


def test_bootstrap_never_drops_a_class():
    # tiny minority class: unstratified resampling would frequently lose it
    y = np.array([0] * 98 + [1] * 2)
    scores = np.arange(100, dtype=float)
    lo, hi = manuscript.bootstrap_mean_auc_ci([(y, scores)], B=200, bootstrap_seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)  # no ValueError from single-class replicate
```

- [ ] **Step 2: Verify fail → implement** (`MANUSCRIPT_SEEDS` appended to config with a comment naming the paper's three + seven sequential; `manuscript.py` bootstrap section with the two functions; stratification per unique label; percentile CI via `np.percentile(replicate_means, [2.5, 97.5])`).
- [ ] **Step 3: Tests pass; fast suite green; commit** — `"feat: cluster-bootstrap AUC CI machinery + MANUSCRIPT_SEEDS"`

---

### Task 2: Seed-campaign orchestration + stats tables

**Files:**
- Modify: `uav_tda/manuscript.py` (add)
- Modify: `uav_tda/cli.py` (add `manuscript-report` subcommand)
- Test: `tests/test_manuscript.py` (append)

**Interfaces:**
- `missing_seeds(seeds: tuple, rebuild_dir: Path) -> list[int]` — seeds lacking `probe_distances_seed{N}.csv` OR `znorm_stats_seed{N}.json`.
- `run_missing_seeds(seeds, w2_timeout=30.0) -> None` — for each missing seed, invoke the Phase-3 znorm probe path (import and call the same function `cli._run_znorm_probe` uses, or the underlying `probe.run_probe_with_znorm` + the same artifact-writing routine — REUSE the Phase-3 writer, do not duplicate CSV-writing logic; if the writer is CLI-internal, refactor it into `znorm_report.py`/`manuscript.py` as a shared function with the CLI delegating to it — a moved function, not a copy).
- `load_seed_frames(seeds, rebuild_dir) -> dict[int, tuple[pd.DataFrame, dict]]` — per seed: raw test df + znorm stats (from JSON).
- `build_binary_auc_table(seed_frames, B=2000, bootstrap_seed=0) -> pd.DataFrame` — rows: subset × scoring ∈ {raw, znorm}; columns: `subset, scoring, mean, std, ci_lo, ci_hi, n_seeds`. Mean/std across per-seed AUCs (std ddof=1, pandas convention, matching Phase 3); CI via `bootstrap_mean_auc_ci` with the subset score as `scores` (znorm rows use `metrics.apply_znorm(df, stats)` per seed), multiclass labels for stratification.
- `build_attribution_table(seed_frames, B=2000, bootstrap_seed=0) -> pd.DataFrame` — rows: attack × manifold; columns: `attack_class, manifold, mean, std, ci_lo, ci_hi, dominant` (dominant = manifold with max mean for that attack; one True per attack). One-vs-rest AUC per `metrics.per_attack_auc` semantics (raw scores only — znorm is monotone-invariant per manifold, note this in the docstring).
- CLI: `uav-tda manuscript-report [--seeds CSV] [--bootstrap B] [--w2-timeout SEC]` → ensures seeds, builds both tables, writes `results/tables/rebuild/binary_auc.csv` + `manifold_attribution.csv` (+ provenance sidecars), then Task 3's emitters + report (wired in T3; in T2 the subcommand may stop after the CSVs).

- [ ] **Step 1: Failing tests** — synthetic seed fixtures: write 2 fake seeds' `probe_distances_seed{N}.csv` (tiny, 10 rows/class, known score separations) + `znorm_stats_seed{N}.json` into `tmp_path`; assert `missing_seeds` identifies absent ones; `load_seed_frames` roundtrips; `build_binary_auc_table` returns 7 subsets × 2 scorings with `ci_lo <= mean <= ci_hi` and n_seeds=2; `build_attribution_table` returns 4 attacks × 3 manifolds with exactly one dominant per attack. CLI registration test: `manuscript-report` in subparsers with `--seeds/--bootstrap/--w2-timeout`.
- [ ] **Step 2: fail → implement → pass; fast suite green; commit** — `"feat: ten-seed campaign orchestration + bootstrap stats tables"`

---

### Task 3: LaTeX emitters, the 10-seed campaign, and the manuscript-stats report

**Files:**
- Modify: `uav_tda/manuscript.py` (emitters + `build_manuscript_stats()` orchestrator), `uav_tda/cli.py` (wire emitters/report into `manuscript-report`)
- Modify: `CLAUDE/PROJECT_BRIEF.md` (append ONE decision-log entry)
- Test: `tests/test_manuscript.py` (append golden-string emitter tests)
- Outputs (committed): 7 new seeds' rebuild artifacts; `binary_auc.csv`; `manifold_attribution.csv`; `paper_snippets/{attribution_table_rows.tex, binary_auc_pgfplots.tex}`; `paper/MANUSCRIPT_STATS.md`

**Emitter specs (exact formats, matching `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`):**
- `emit_attribution_rows(attr_df) -> str` — one line per attack, matching the paper's Table `tab:per_attack_auc` row syntax: `Sybil & $0.65 \pm 0.01$ & $\mathbf{0.87 \pm 0.00}$ & $0.20 \pm 0.01$ \\` — attack short-name (strip " Attack"), cells `{mean:.2f} \pm {std:.2f}`, dominant cell wrapped `\mathbf{...}`, column order C2, Network, Physical, rows ordered Sybil, Flooding, Blackhole, Wormhole.
- `emit_binary_pgfplots(bin_df, scoring="znorm") -> str` — the pgfplots `coordinates {...}` block matching Fig. `fig:binary_auc`: one `({label}, {mean:.3f}) +- (0, {std:.3f})` per subset with the paper's symbolic x coords `C2, Network, Physical, C2+N, C2+P, N+P, All three` (map from subset keys c2_only→C2, network_only→Network, physical_only→Physical, c2_network→C2+N, c2_physical→C2+P, network_physical→N+P, all_three→All three), plus the manual value-label `\node` lines (`{mean:.2f}` at y = mean+std+0.033, matching the paper's placement convention).
- `build_manuscript_stats(seeds, B, ...)` — orchestrates: ensure seeds → tables → CSVs → snippets → `paper/MANUSCRIPT_STATS.md` containing: (1) header (config, seeds, bootstrap spec); (2) binary AUC table raw+znorm with CIs (markdown, generated); (3) attribution table with CIs + dominance; (4) comparison vs the 3-seed Phase-3 numbers and vs the published 3-seed values (quote `paper/MULTI_SEED_VARIANCE.md` literals, labeled); (5) drop-in usage note for the .tex snippets ("pending author sign-off").

- [ ] **Step 1: Failing golden-string tests** — feed a hand-built 2-row `attr_df` / 7-row `bin_df` with known means/stds; assert `emit_attribution_rows` and `emit_binary_pgfplots` return EXACT expected strings (write the expected literals in the test; derive them by hand from the formats above).
- [ ] **Step 2: fail → implement emitters → golden tests pass.**
- [ ] **Step 3: RUN THE CAMPAIGN (detached):** `uav-tda manuscript-report --bootstrap 2000 --w2-timeout 30` — runs the 7 missing seeds (~2.5–3.5h; nohup+log+poll; verify each seed's artifacts land), then tables/snippets/report. Sanity-verify: `binary_auc.csv` has 14 rows with finite CIs and `ci_lo <= mean <= ci_hi` everywhere; attribution dominance matches the paper (Sybil/Flooding→Network, Blackhole/Wormhole→Physical) — if dominance FLIPS for any attack at 10 seeds, STOP and report (that's a science-level finding for the controller, not a bug to fix).
- [ ] **Step 4: Decision-log entry** (dated: Phase 4 complete; 10-seed + bootstrap-CI results in MANUSCRIPT_STATS.md pending author sign-off; .tex snippets generated from data).
- [ ] **Step 5: Fast suite; commit code + artifacts** — `"feat: ten-seed bootstrap-CI campaign + paper-emitting tables (MANUSCRIPT_STATS)"`

---

## Self-Review

- **Spec coverage:** §V ten seeds → MANUSCRIPT_SEEDS + campaign (T2/T3) ✓; bootstrap CIs on every AUC → cluster bootstrap in both tables (T1/T2) ✓; roadmap's `manifold_attribution.csv` + `binary_auc.csv` → T2 ✓; LaTeX/pgfplots emitters so paper and code never drift → T3 golden-tested emitters + snippets ✓; figures-from-data deliverable → pgfplots block ✓.
- **Placeholder scan:** T1 tests fully coded; T2 interfaces exact with a no-duplication rule for the artifact writer; T3 emitter formats specified to the character with golden-test requirement. No TBDs.
- **Type consistency:** `bootstrap_mean_auc_ci` signature consistent T1↔T2; `build_binary_auc_table`/`build_attribution_table` outputs consistent with T3 emitters' inputs; seed-artifact filenames match Phase 3's writer.
- **Known risks, pre-ruled:** (a) 7-seed campaign is long — detached, per-seed artifacts land incrementally, restartable via `missing_seeds`; (b) dominance flip at 10 seeds = STOP-and-report science finding, explicitly not implementer-fixable; (c) pgfplots node-placement convention (mean+std+0.033) is inferred from the paper's manual labels — emitter output is a *candidate* snippet pending author sign-off, stated in MANUSCRIPT_STATS.md.
