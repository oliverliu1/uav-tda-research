# Manuscript Artifacts — Phase 8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every item in `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md` (2026-10-18) that does not need author sign-off: the like-for-like compute frontier (audit I-1), provenance gaps, unique-window counts (I-3), LaTeX snippets for the definitive full-test tables, the five missing manuscript figures, quarantine of the April 2026 PNGs, and one machine-generated report tying it together.

**Architecture:** No new science, no campaigns, no re-timing. Every new artifact is derived from CSVs already committed under `results/tables/rebuild/`. One new module `uav_tda/artifacts.py` (tables, sidecars, snippets, report) and one `uav_tda/figures.py` (matplotlib, Agg backend, vector PDF). One new CLI subcommand `uav-tda artifacts` that builds all of it and writes `paper/MANUSCRIPT_ARTIFACTS.md`. Existing committed artifacts and the five Phase 3–6 results docs are **not modified**; corrections are additive and the new report states what each supersedes.

**Tech Stack:** Python 3.9, pandas, numpy, scikit-learn (`roc_curve`, `roc_auc_score`), matplotlib 3.4 (already a runtime dep), pytest.

**Spec:** `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md` §2 items 2 and 5, §4 (I-1, I-3, missing sidecars, numbers with no CSV), §5, §8 steps 2–3.

## Global Constraints

- Locked surfaces unchanged: `pipeline.py`, `tools/`, `data/`, `outputs/`, `archive/`, top-level `results/tables/*.csv`, every already-committed file under `results/tables/rebuild/` (no overwrite, no re-render), the eight existing `paper/*.md` docs, `paper/historical/`, and golden-master `metrics.py` semantics.
- New tables go under `results/tables/rebuild/artifacts/`; new snippets under `results/tables/rebuild/paper_snippets/` with new filenames only; new figures under `results/figures/manuscript/`. Every new artifact (CSV, JSON, TEX, PDF) gets a `.provenance.json` sidecar via `uav_tda.provenance.write_provenance`.
- Backfilled sidecars for pre-existing files are written next to them with `params` containing `"backfilled": true` and `"introduced_in": <sha of the commit that added the file>` (find with `git log --diff-filter=A --format=%H -- <path>`). The data file itself is not touched.
- **Frontier ruling (Task 1):** the per-flow AUC stays the Phase-4 10-seed probe value (0.8930), because the latency harness timed that same probe configuration (top-K=50, δ=0.2, sparse ε=0.5). Cost for every row comes from the latency harness's full-path `total_s` (single instrument, n=30, same machine and session). The campaign's own windowed marginal is kept as a secondary column; the old W2-only per-flow 0.0602 s is kept as a labelled column, never as the cost. Error bars are bootstrap CIs, not repeat std. The full-test exact AUC (0.9082) is drawn as a horizontal reference, not a frontier point.
- **Figure style:** Agg backend; single-column 3.5 in wide (double-column 7.16 in only where noted); base font 8 pt; vector PDF; colour-blind-safe palette (Okabe–Ito); no titles inside the PDF (captions live in LaTeX). Ordered vs shuffled and per-manifold colours fixed across all figures.
- No test reads `outputs/`; every new test uses committed CSVs or synthetic frames, so CI stays portable. Fast suite must stay < ~2.5 min and go from 103 to 103+N passed, 0 skipped locally.
- Python 3.9 syntax (no `match`, no `X | Y` at runtime). Commit per task.

## File Structure

```
uav_tda/artifacts.py            # NEW: frontier, sidecar backfill, eps-max persist, unique-window counts, FPR/TPR table, exact snippets, report
uav_tda/figures.py              # NEW: five manuscript figures
uav_tda/cli.py                  # MODIFY: `artifacts` subcommand
tests/test_artifacts.py         # NEW
tests/test_figures.py           # NEW
results/tables/rebuild/artifacts/*        # NEW (generated)
results/tables/rebuild/paper_snippets/*   # NEW files only (generated)
results/figures/manuscript/*.pdf          # NEW (generated)
results/figures/legacy_2026-04/           # NEW: git mv of the 15 April PNGs + README.md
paper/MANUSCRIPT_ARTIFACTS.md             # NEW (generated)
CLAUDE.md, README.md, CLAUDE/STATUS.md, CLAUDE/PROJECT_BRIEF.md  # MODIFY (Task 4 only)
```

---

### Task 1: Like-for-like compute frontier + provenance hygiene + tables

**Files:** Create `uav_tda/artifacts.py`, `tests/test_artifacts.py`.

- [ ] **Step 1 (TDD) `build_full_path_frontier(rebuild_dir)`** → DataFrame, one row per `per_flow`, W=25/50/100/200. Inputs: `windowed/compute_frontier.csv` (AUC mean/std/ci, campaign `marginal_s`, `baseline_s`) and `latency_<host>.csv` (summary; pick the single `latency_*.csv` that is not a per-mode detail file — the one with a `mode,w,stage,...` header). Columns: `row, w, auc_mean, auc_ci_lo, auc_ci_hi, cost_s` (= harness `total_s` mean for that mode/W), `cost_p95_s`, `campaign_marginal_s` (windowed rows; NaN for per-flow), `w2_only_per_flow_s` (per-flow row only: the old 0.0602), `baseline_setup_s` (harness `baseline_setup_s`; NaN per-flow), `flows_per_decision` (W, or 1), `cost_per_flow_s` (= cost_s / flows_per_decision), `speedup_vs_per_flow` (= per-flow cost_s / cost_s). Tests: synthetic inputs → exact column values; on the committed CSVs, per-flow `cost_s` == 3.5762238905333343 and W=25 `speedup_vs_per_flow` ≈ 477.7 (assert within 0.5).
- [ ] **Step 2 (TDD) `emit_full_path_frontier_pgfplots(df)`**: same shape as `windowed.emit_frontier_pgfplots` but x = `cost_s`, asymmetric CI error bars `+- (0,hi) -= (0,lo)` via pgfplots `explicit` syntax (`(x, y) += (0, ci_hi-y) -= (0, y-ci_lo)`), plus a commented header line naming the source CSV and stating "x = full-path seconds per decision (latency harness, n=30)". Golden-string test on a 2-row synthetic frame.
- [ ] **Step 3 (TDD) `unique_window_counts(rebuild_dir)`**: from `windowed/run_w{W}_ordered0.csv` (C2/Physical are bit-identical across repeats, so repeat 0 is the unique population) compute per W: total windows, pure-Normal windows (attack_frac == 0), Blackhole-majority windows, and the pooled ×10 figures the report quoted. Read the run CSV's actual column names at implementation time (`head -1`); do not guess. Test: W=200 → 19 Blackhole-majority of 91 windows (audit I-3).
- [ ] **Step 4 (TDD) `operating_point_table(rebuild_dir)`**: from `exact/shard_val_*.csv` (val-Normal) and `exact/shard_test_*.csv` compute, for each of the 7 subsets × {raw, znorm}, τ = Q95 of val-Normal subset score, then test FPR (Normal), overall TPR, and per-attack TPR. Z-norm uses val-Normal mean/std per manifold (same as `metrics.znorm_stats_from_val`/`apply_znorm` — call them, don't reimplement). Reuse `metrics` subset definitions. Test: all-three znorm FPR ≈ 0.056 and TPR ≈ 0.570 (±0.002), Sybil TPR ≈ 0.91.
- [ ] **Step 5 `persist_eps_max(outputs_dir, rebuild_dir)`**: copy `outputs/max_edge_lengths.json` → `rebuild/artifacts/max_edge_lengths.json` + sidecar (skip with a logged warning if the source is absent, so CI never fails). Test with a tmp source.
- [ ] **Step 6 `backfill_sidecars(rebuild_dir)`**: for `znorm_summary.csv`, `znorm_per_attack.csv`, `paper_snippets/{attribution_table_rows,binary_auc_pgfplots,windowed_frontier_pgfplots}.tex`, `exact/manifest.json`, write a sidecar if absent (never overwrite an existing one). Test on tmp files.
- [ ] **Step 7:** Exact-campaign snippets: `emit_exact_attribution_rows(df)` (rows for Table tab:per_attack_auc from `exact/exact_per_attack_auc.csv`: `auc` to 2 dp with `[lo, hi]` CI, dominant bolded, same row order as `manuscript.emit_attribution_rows`) and `emit_exact_binary_pgfplots(df, scoring="znorm")` (7 subsets, asymmetric CI bars, same x coords as the existing figure). Golden-string tests.
- [ ] **Step 8:** Fast suite + ruff clean. Commit — `"feat: like-for-like compute frontier, operating points, exact snippets, provenance backfill"`.

### Task 2: Manuscript figures

**Files:** Create `uav_tda/figures.py`, `tests/test_figures.py`.

Each function takes the relevant DataFrame(s) (not paths) plus an output path, returns the path, writes a PDF + sidecar. A `build_all_figures(rebuild_dir, out_dir)` loads CSVs and calls them.

- [ ] **Step 1 `fig_frontier`** (single column): log-x full-path cost per decision vs znorm all-three AUC, CI bars, per-flow point distinct marker, W labels, dashed horizontal line at the exact full-test AUC labelled "per-flow, full test (δ≤0.01)". Uses Task 1's frontier frame.
- [ ] **Step 2 `fig_detection_vs_w`** (single column): x = W (log2 ticks 25/50/100/200), y = znorm all-three AUC with CI band, ordered vs shuffled lines; per-flow 10-seed probe and exact full-test as horizontal references. Source `windowed/windowed_detection.csv`.
- [ ] **Step 3 `fig_contamination`** (double column, 4 panels one per W): ordered arm, `majority_class == "all"`, x = attack-fraction bin midpoint, y = detection rate; annotate unique n per bin from Task 1 Step 3 (pooled n / 10). Source `windowed/contamination_curve.csv`.
- [ ] **Step 4 `fig_latency`** (single column): horizontal stacked bars of mean seconds per stage (scaler / Rips / slice / W2) per manifold for the per-flow arm, plus a second panel of windowed `total_s` per W on log-x with the per-flow total as a reference line. Source latency summary CSV.
- [ ] **Step 5 `fig_attribution`** (double column): heatmap grid, rows = 4 attack classes, columns = per-flow full-test C2/Network/Physical then W=25…200 × 3 manifolds; cell = AUC (diverging map centred at 0.5, so anti-correlated cells read as such), dominant manifold per block outlined. Sources `exact/exact_per_attack_auc.csv`, `windowed/windowed_attribution.csv`.
- [ ] **Step 6 `fig_roc`** (single column): full-test ROC curves from the exact shards for znorm all-three, N+P, and each single manifold; mark the val-Normal Q95 operating point on all-three (from Task 1 Step 4). AUC in legend.
- [ ] **Step 7 tests:** each figure function on small synthetic frames writes a non-empty PDF and a sidecar; `fig_roc` legend AUC on the committed shards equals `exact_binary_auc.csv` to 4 dp; no test reads `outputs/`.
- [ ] **Step 8:** Fast suite + ruff clean. Commit — `"feat: manuscript figure generators (frontier, detection-vs-W, contamination, latency, attribution, ROC)"`.

### Task 3: CLI, report, generate, quarantine

**Files:** Modify `uav_tda/cli.py`; extend `uav_tda/artifacts.py`; `git mv` legacy PNGs.

- [ ] **Step 1:** `uav-tda artifacts [--root PATH]` runs Task 1 builders (writes `rebuild/artifacts/{full_path_frontier,unique_window_counts,operating_points}.csv`, eps-max copy, backfilled sidecars, new snippets), `figures.build_all_figures`, and `write_artifacts_report` → `paper/MANUSCRIPT_ARTIFACTS.md`. Idempotent: re-running produces byte-identical CSVs/TEX (PDFs may differ in metadata; set `metadata={"CreationDate": None}` to keep them stable).
- [ ] **Step 2:** `write_artifacts_report` (machine-generated, every number formatted from the frames): §1 what this supersedes (frontier 6.6× → full-path ratio; pooled → unique window counts; FPR claim now measured); §2 frontier table; §3 operating points (all-three and N+P rows, both scorings); §4 unique window counts vs pooled; §5 artifact inventory (every new file + its source CSV); §6 "pending author sign-off" footer.
- [ ] **Step 3:** Run `uav-tda artifacts` for real. Check outputs exist, sidecars present, PDFs open (non-zero size, `%PDF` header). Eyeball each PDF by rendering page 1 to PNG in the session scratchpad (`pdftoppm` if available, else matplotlib re-read is not possible — use `sips -s format png` on macOS) and inspecting it.
- [ ] **Step 4:** Quarantine: `git mv` the 15 April 2026 PNGs (all of `results/figures/*.png` except `supervised_comparison.png` and `feature_importance_curated.png`) into `results/figures/legacy_2026-04/` with a README (provenance: pre-monolith scripts, leakage-era lineage, too wide for print per `paper/historical/2026-05-drafting/FIGURE_AUDIT.md`; never use). First `grep -rn` the repo (excluding `archive/`, `pipeline.py`) for each filename; if a test or live module reads one, leave that file in place and note it.
- [ ] **Step 5:** Fast suite + ruff clean. Commit — `"feat: uav-tda artifacts subcommand, MANUSCRIPT_ARTIFACTS report, generated figures; quarantine April PNGs"`.

### Task 4: Docs + decision log

- [ ] `CLAUDE.md` commands block: add `uav-tda artifacts`. README "Results docs": add `paper/MANUSCRIPT_ARTIFACTS.md` and `results/figures/manuscript/`. `CLAUDE/STATUS.md`: tick the closed items, keep the open author decisions. `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md` is a dated snapshot — do not edit; STATUS points to the new report instead. Append one dated decision-log entry to `CLAUDE/PROJECT_BRIEF.md` recording the frontier ruling (Global Constraints) and what was closed. Commit — `"docs: Phase-8 artifacts in CLAUDE.md, README, STATUS, decision log"`.

## Out of scope (needs author sign-off or a decision)

Editing `main.tex`; choosing the headline estimand; re-timing anything; companion-board run; untracking the dataset or `node_modules`; dropping the stash; deleting `outputs/`; replacing `requirements.txt`.
