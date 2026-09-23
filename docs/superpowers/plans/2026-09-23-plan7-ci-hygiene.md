# CI, Hygiene, Quarantine & Final Docs — Phase 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the roadmap: lint + CI on the fast suite (green on a fresh clone), a Makefile, quarantine of the legacy/leakage directories under `archive/`, a documented immutable dataset, and a final docs sweep — plus the cheap deferred minors accumulated across Phases 3–6.

**Architecture:** Three bounded tasks. No science, no campaigns, no new methodology. The only engineering subtlety is CI portability: a fresh clone has no `data/UAVIDS-2025.csv` (gitignored) and no `outputs/` artifacts, so every fast test that touches local-only files gains an explicit skip guard; the committed oracles (`results/tables/probe_distances*.csv` etc.) ARE in git, so the golden-master suite stays exercised in CI.

**Tech Stack:** Python 3.9, pytest, ruff (dev-extra addition — permitted new DEV dependency, not a runtime one), GitHub Actions, make.

**Spec:** roadmap Phase 7; accumulated deferred-minor ledger entries (Phases 3–6 PR descriptions and plan docs).

## Global Constraints

- Locked surfaces unchanged: `pipeline.py`, `tools/`, published tables, `paper/*.md` result docs, all `uav_tda` computational semantics (hygiene fixes must not change any computed value — the fast suite incl. golden-master tests is the gate).
- Quarantine = `git mv`, never delete: `poster_eda/` (contains a known data-leakage backup — the archive README must warn), `scripts_archive/`, root `poster.jsx` → `archive/`. History preserved.
- `data/UAVIDS-2025.csv` immutable; its sha256 goes in `data/README.md` (compute the full hash at task time; prefix measured 2026-09-23: `d50d339f68be7b23f0bf089dd438b20a…`).
- Skip guards use `pytest.mark.skipif(not <path>.exists(), reason=...)` (or runtime `pytest.skip`) — never delete or weaken a test; on THIS machine (all artifacts present) the fast suite must still run everything: **103 passed, 0 new skips locally**.
- New dev tools go in `[project.optional-dependencies] dev` only; runtime deps unchanged.
- Python 3.9; fast suite < ~2 min; commit per task.

## File Structure

```
pyproject.toml            # MODIFY: [tool.ruff] config; ruff in dev extra
uav_tda/manuscript.py     # MODIFY: thread rebuild_dir through run_missing_seeds (dual-TABLES_DIR minor)
uav_tda/windowed.py       # MODIFY: build_frontier_table accepts precomputed detection frame (M7); module docstring updated (L8); _render_windowed_report unused ws param (L9)
tests/*                   # MODIFY: skip guards; remove dead imports flagged by ruff
.github/workflows/ci.yml  # NEW
Makefile                  # NEW: test / test-all / lint / repro-fast targets
archive/                  # NEW (git mv targets) + archive/README.md
data/README.md            # NEW
README.md                 # MODIFY: final structure
CLAUDE/CLAUDE.md          # MODIFY: file-discipline for archive/; accuracy sweep
CLAUDE/PROJECT_BRIEF.md   # MODIFY: final decision-log entry (Task 3 only)
```

---

### Task 1: Lint + cheap deferred-minor sweep

**Files:** Modify `pyproject.toml`, `uav_tda/manuscript.py`, `uav_tda/windowed.py`, tests as flagged.

- [ ] **Step 1:** Add `ruff>=0.4` to the dev extra and a `[tool.ruff]` block: `line-length = 100`, `target-version = "py39"`, `lint.select = ["E9", "F63", "F7", "F82", "F401", "F811"]` (syntax/undefined-name/unused-import safety tier only — no style churn). Run `ruff check .` and fix every finding (known offenders: unused `pandas` import in tests/test_probe.py; audit the rest). Exclude `archive/`-to-be dirs is unnecessary yet (Task 3 moves them; add `exclude = ["pipeline.py", "tools", "scripts_archive", "poster_eda", "archive"]` so legacy code is out of lint scope from the start).
- [ ] **Step 2:** Deferred minors, cheap tier ONLY:
  - `manuscript.run_missing_seeds(seeds, w2_timeout=30.0, rebuild_dir=None)` — accept the dir, default preserving today's path; `build_manuscript_stats`/CLI pass theirs through (kills the dual-TABLES_DIR trap, Phase-4 final review).
  - `windowed.build_frontier_table(..., detection_df=None)` — accept the precomputed detection frame; `build_windowed_tables` passes its own (kills the double 112-row×B=2000 bootstrap, Phase-5 M7). No recompute of shipped artifacts.
  - `windowed.py` module docstring updated to describe the full module (Phase-5 L8); drop `_render_windowed_report`'s unused `ws` param IF no caller breaks (else leave + note).
  - Do NOT touch: `_NODE_Y_OFFSET`, comparator gates, any shipped CSV/report.
- [ ] **Step 3:** Full fast suite **103 passed** (zero behavior change proof) + `ruff check .` clean. Commit — `"chore: ruff config + deferred-minor sweep (no behavior change)"`.

### Task 2: CI portability + workflow + Makefile

**Files:** Modify tests (guards); Create `.github/workflows/ci.yml`, `Makefile`.

- [ ] **Step 1: Audit** every fast test for local-only dependencies (`data/UAVIDS-2025.csv`, `outputs/**`, absolute anaconda paths). Known: `tests/test_harness.py` (data symlink), `tests/conftest.py` fixtures, `tests/test_cli_phases.py::test_prep_debug_smoke` (data), any spot-check reading `outputs/`. Committed `results/tables/**` oracles are IN git — no guard for those.
- [ ] **Step 2:** Add skip guards with the shared helper pattern (one `requires_local_data = pytest.mark.skipif(...)` per file or in conftest). Locally: **103 passed, 0 skipped** (guard must not fire here — assert by running).
- [ ] **Step 3:** `.github/workflows/ci.yml`: on push + PR to main; ubuntu-latest; python 3.9; `pip install -e ".[dev]"`; `ruff check .`; `python -m pytest -m "not slow" -q`. Note in a comment: gudhi wheels exist for linux/py3.9 (verify at implement time via pip index or a constraint; if gudhi install is flaky on the runner, the workflow may pip-install with a pinned known-good version — document whatever is chosen).
- [ ] **Step 4:** `Makefile`: `test` (fast suite), `test-all` (incl. slow), `lint` (ruff), `repro-fast` (`uav-tda probe --seed 42`), `help` default. `.PHONY` correct.
- [ ] **Step 5:** Fast suite still 103/0-skips locally; commit — `"feat: CI workflow, portable fast suite (skip guards), Makefile"`. (CI's first real run happens on push after merge — state this in the report; do NOT push from this task.)

### Task 3: Quarantine + docs + program close

**Files:** `git mv` targets; Create `archive/README.md`, `data/README.md`; Modify `README.md`, `CLAUDE/CLAUDE.md`, `CLAUDE/PROJECT_BRIEF.md`.

- [ ] **Step 1:** `git mv poster_eda archive/poster_eda && git mv scripts_archive archive/scripts_archive && git mv poster.jsx archive/poster.jsx`. `archive/README.md`: DEPRECATED banner; per-dir one-liner; **explicit warning** that `archive/poster_eda/` contains pre-fix backups with a known data-leakage flaw and must never be used for results; pointer to `uav_tda`/`pipeline.py` as the real implementations. Verify nothing imports from the moved paths (grep uav_tda/ tests/); update `.gitignore` paths referencing poster_eda; ruff excludes already cover archive/ (Task 1).
- [ ] **Step 2:** `data/README.md`: UAVIDS-2025 provenance (Zeng et al., IEEE CNS 2025), 122,171 rows × 23 cols, IMMUTABLE, full sha256, gitignored-by-design note + how a fresh clone obtains it, and the row-order caveat (no timestamps; FlowID order is the temporal proxy — cite the Phase-5 spec).
- [ ] **Step 3:** Top-level `README.md` final structure: what this repo is (paper + package), install, one-command repro paths (fast suite / probe / campaigns), the five results docs in `paper/`, `pipeline.py` = frozen oracle note, disk note (`outputs/` ≈32G, regenerable via `uav-tda prep/tda`, safe to delete locally IF regeneration time is acceptable — user decision), archive/ pointer. Keep it tight (< ~120 lines).
- [ ] **Step 4:** `CLAUDE/CLAUDE.md` accuracy sweep: file-discipline gains `archive/` (read-only legacy, leakage warning); verify every CLI subcommand listed matches `cli.py`; memory-pressure + long-run lessons one-liner (chunked workers, n_jobs≤3, commit-before-launch). `CLAUDE/PROJECT_BRIEF.md`: final dated decision-log entry — 7-phase program complete, pointers to the five results docs, pending-sign-off status of manuscript numbers.
- [ ] **Step 5:** Fast suite green (103, 0 skips); `ruff check .` clean; commit — `"chore: quarantine legacy dirs, dataset README, final docs sweep"`.

---

## Self-Review

- **Spec coverage:** roadmap P7 items all mapped (ruff→T1; CI+Makefile→T2; quarantine→T3; data README→T3; top-level README→T3; CLAUDE.md sweep→T3). mypy deliberately dropped (py3.9 legacy-heavy codebase; YAGNI — noted as an explicit descope for the reviewer).
- **Placeholder scan:** rule lists, filenames, mv commands, README content requirements all concrete. The one implement-time verification (gudhi wheel availability on ubuntu/py3.9) is named with a fallback policy.
- **Type consistency:** `run_missing_seeds(..., rebuild_dir=None)` / `build_frontier_table(..., detection_df=None)` additive-default signatures — no caller breaks; skip-guard helper named consistently.
- **Pre-ruled risks:** (a) CI can't be fully validated pre-merge — first run on push, stated; (b) quarantine is `git mv` only — reversible; (c) zero-behavior-change enforced by the untouched 103-test suite + golden masters.
