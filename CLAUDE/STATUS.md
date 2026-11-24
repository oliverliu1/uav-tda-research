# Project Status — 2026-11-23

One page on where everything stands. Framing and decisions: `CLAUDE/PROJECT_BRIEF.md`.
Manuscript audit (dated 2026-10-18 snapshot, not updated): `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md`. Agent rules: root `CLAUDE.md`.
Human entry point: `README.md`.

## Where we are

- **Extended abstract:** submitted to AIAA SciTech 2027 (Intelligent Systems). Source
  `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`, untouched since submission. Full
  manuscript due **December 2026**.
- **Infrastructure:** complete. Seven-phase program (2026-09-19 → 09-23, PRs #11–#17) delivered the
  `uav_tda` package, equivalence tests against the frozen `pipeline.py`, z-normalized scoring, a
  10-seed bootstrap campaign, the time-windowed variant, the full-test high-precision Wasserstein
  campaign, a portable latency harness, CI, lint, and legacy quarantine. No infrastructure work
  remains.
- **Phase 8 (manuscript artifacts):** merged to `main` (PR #19, after the audit/cleanup PR #18;
  2026-11-10) (`uav-tda artifacts`): full-path frontier, unique window counts, operating points,
  six figures, three new snippets, backfilled sidecars. Fast suite 192 passed / 0 skipped; ruff
  clean; CI green (runner pinned to `ubuntu-24.04`). Report:
  `paper/MANUSCRIPT_ARTIFACTS.md`; figures: `results/figures/manuscript/`.

## Results package (all pending author sign-off)

| Doc | What it holds | Headline |
|---|---|---|
| `paper/EXACT_RESULTS.md` | full test split, hera δ≤0.01, no sampling | znorm all-three AUC **0.9082 [0.9033, 0.9131]**; attribution replicates |
| `paper/MANUSCRIPT_STATS.md` | 10 seeds, within-seed bootstrap CIs, probe config, fixed reference set | znorm all-three 0.8930 ± 0.0078 |
| `paper/WINDOWED_RESULTS.md` | W ∈ {25, 50, 100, 200}, 80 runs, shuffle control | W=100 0.9376 ± 0.0005; shuffle Δ +0.32–0.37; Sybil flips to C2 at window level |
| `paper/LATENCY_RESULTS.md` | full inference path, per-flow + windowed | per-flow 3.576 s mean / 3.859 s p95 (Rosetta-emulated M1 Pro); W=25 0.0075 s/decision |
| `paper/ZNORM_RESULTS.md` | 3-seed raw vs znorm; seed-123 reanalysis | 0 timeouts at 30 s |
| `paper/MANUSCRIPT_ARTIFACTS.md` | full-path frontier, unique window counts, FPR/TPR, caveats | W=25 is **478x** cheaper per decision than per-flow; all-three znorm FPR 0.0555 / TPR 0.5703 |

Every number was recomputed from its CSVs during the audit; zero mismatches. As-published numbers
(3-seed raw probe) reproduce bit-exactly from the immutable `results/tables/probe_distances*.csv`.

## What must change in the manuscript (short list; details in the audit)

1. "Exact Wasserstein" → "high-precision (δ≤0.01)"; δ=0 is intractable.
2. "1–3 s per flow" → 3.58 s mean, on an emulated workstation, not companion hardware.
3. "C2 adds noise; N+P is best" → all-three is best under z-normalized scoring.
4. W2 ground metric is L2 not L∞; seed does not control k-medoids or Wasserstein; feature count is
   20 raw / 22 encoded; the val-Normal threshold was never computed in the abstract's lineage
   (now measured: FPR 0.0555, TPR 0.5703; closed).
5. Frontier: fixed in the artifacts (full-path latency cost; **478x** at W=25, old 6.6x
   superseded). The manuscript text still needs the 478x figure and the new frontier plot.
6. Window-level attribution flips (Sybil → C2) must be reconciled with the per-flow Sybil story.
7. Cross-benchmark evaluation was promised and not done.

## Figures (Phase 8, closed)

Six vector PDFs in `results/figures/manuscript/` (frontier, detection-vs-W, contamination, latency,
attribution, ROC); snippets `full_path_frontier_pgfplots.tex`, `exact_attribution_rows.tex`,
`exact_binary_auc_pgfplots.tex` in `results/tables/rebuild/paper_snippets/`. April PNGs moved to
`results/figures/legacy_2026-04/`. Also closed: unique-vs-pooled window counts, missing sidecars
(backfilled), eps-max values persisted. Caption caveats:

- Nine of 16 windowed attribution cells have overlapping CIs (conservative, not a paired test; scan
  in the report): Flooding and Wormhole at every W, Blackhole at W=200. Blackhole's dominant
  (Physical) CI also reaches AUC 0.5 at every W. Do not read these dominances as resolved.
- Windowed CIs were corrected 2026-11-23 (shared-window bootstrap; the old ordered-arm CIs were about
  3x too narrow). W=100 is 0.9376 [0.900, 0.969], which overlaps the per-flow full-test CI, so
  "windowing beats per-flow" is not established. Adjacent windows are correlated (lag-1 0.44 to
  0.70), so these CIs are still optimistic; block-10 bootstrap at W=100 is [0.874, 0.985].
  The per-flow full-test CI has not been checked for the same dependence.
- Also in the report: windowed cost is Rips + W2 (slicing inside W2) with no scaler stage
  (pre-scaled inputs; per-flow scaler ~2.51 ms) and no baseline setup, so the W=25 speedup is
  upper-end; per-flow vs windowed AUC units differ (M2); contamination uses an empirical Q95
  over attack_frac <= 0.05 windows (43/16/9/5 unique at W=25/50/100/200), not val-Normal Q95.
- Per-flow attribution block is a single full-test estimate; windowed blocks are 10-run means.
- Thinnest contamination bin is n=1 (W=100, bin 0); low power.

## Open author decisions

- Per-flow headline estimand: decided 2026-11-11, full-test δ≤0.01 AUC 0.9082; the 10-seed probe
  0.8930 is used only where paired with the latency harness cost (frontier).
- Companion-board latency run: Raspberry Pi 5 (4 GB, active cooler, 27 W supply, USB 3 SSD) is set up
  and passed a 3-sample smoke test 2026-11-23 (not for citation). Next: `uav-tda hang-check` and
  `uav-tda latency --fresh-baselines --exact-arm` (n=30) on the Pi. The report text still calls every
  host "workstation-class", so copy back only the CSVs/sidecars; `artifacts` then needs `--latency-host`.
- Cross-benchmark evaluation: decided, future work (reword the §V promise).

Decided 2026-11-11: April stash dropped; `archive/poster_eda/node_modules` untracked
(still in history); `outputs/` kept until submission; `requirements.txt` replaced with exact
env pins; dataset stays in-repo under its ODbL license with attribution (`data/README.md`).
Stale worktrees and merged branches were pruned 2026-11-10; only `main` remains.

- Sign off (or amend) the six results docs (incl. `MANUSCRIPT_ARTIFACTS.md`).
- `main.tex` rewrite (deferred until after sign-off).

## Regenerating anything

`make test` for the fast suite. `uav-tda <x>-report` rebuilds a results doc from committed CSVs
with no recompute (`znorm-report`, `manuscript-report`, `windowed-report`, `exact-report`, `artifacts`);
`uav-tda latency` re-measures. `outputs/` is complete, so `uav-tda tda` never needs re-running.
Install only with `pip install -e ".[dev]"`.
