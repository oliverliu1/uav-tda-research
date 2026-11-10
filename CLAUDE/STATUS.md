# Project Status — 2026-10-18

One page on where everything stands. Framing and decisions: `CLAUDE/PROJECT_BRIEF.md`.
Manuscript audit: `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md`. Agent rules: root `CLAUDE.md`.
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
- **Repo:** `main` == `origin/main` @ 4390e64. Fast suite 103 passed / 0 skipped; ruff clean;
  CI green. Working branch for today's doc changes: `chore/claude-docs-status-audit` (uncommitted
  at the time of writing).

## Results package (all pending author sign-off)

| Doc | What it holds | Headline |
|---|---|---|
| `paper/EXACT_RESULTS.md` | full test split, hera δ≤0.01, no sampling | znorm all-three AUC **0.9082 [0.9033, 0.9131]**; attribution replicates |
| `paper/MANUSCRIPT_STATS.md` | 10 seeds, cluster-bootstrap CIs, probe config | znorm all-three 0.8930 ± 0.0078 |
| `paper/WINDOWED_RESULTS.md` | W ∈ {25, 50, 100, 200}, 80 runs, shuffle control | W=100 0.9376 ± 0.0005; shuffle Δ +0.32–0.37; Sybil flips to C2 at window level |
| `paper/LATENCY_RESULTS.md` | full inference path, per-flow + windowed | per-flow 3.576 s mean / 3.859 s p95 (Rosetta-emulated M1 Pro); W=25 0.0075 s/decision |
| `paper/ZNORM_RESULTS.md` | 3-seed raw vs znorm; seed-123 reanalysis | 0 timeouts at 30 s |

Every number was recomputed from its CSVs during the audit; zero mismatches. As-published numbers
(3-seed raw probe) reproduce bit-exactly from the immutable `results/tables/probe_distances*.csv`.

## What must change in the manuscript (short list; details in the audit)

1. "Exact Wasserstein" → "high-precision (δ≤0.01)"; δ=0 is intractable.
2. "1–3 s per flow" → 3.58 s mean, on an emulated workstation, not companion hardware.
3. "C2 adds noise; N+P is best" → all-three is best under z-normalized scoring.
4. W2 ground metric is L2 not L∞; seed does not control k-medoids or Wasserstein; feature count is
   20 raw / 22 encoded; the val-Normal threshold was never computed in the abstract's lineage
   (now measured: FPR ≈ 0.056, TPR 0.57).
5. The windowed compute-frontier's per-flow cost (0.0602 s) excludes Rips; like-for-like the W=25
   advantage is ~400×, not 6.6×. Recompute before using the frontier figure.
6. Window-level attribution flips (Sybil → C2) must be reconciled with the per-flow Sybil story.
7. Cross-benchmark evaluation was promised and not done.

## Figures

No current-code generators for: detection-vs-W, contamination curve, latency bars, attribution
heatmaps, full-test ROC. Three LaTeX snippets exist (`results/tables/rebuild/paper_snippets/`).
Fifteen of eighteen PNGs in `results/figures/` are April 2026 legacy output; do not use.

## Open author decisions

- Sign off (or amend) the five results docs; choose the per-flow headline estimand.
- Keep redistributing the third-party dataset in-repo, or untrack it.
- Untrack `archive/poster_eda/node_modules` (3,981 files, 84% of tracked files).
- Delete local `outputs/` (32 GB; ~10 GB disk free) or keep it for re-runs.
- Drop the April 2026 stash (pins ~1.3 GB of leakage-era blobs) and prune five stale worktrees.
- Replace or delete `requirements.txt` (its pins would break the pinned environment).
- Run a companion-board latency measurement before December, or reword the promise.

## What changed in `CLAUDE/` today

- `CLAUDE/CLAUDE.md` → root `CLAUDE.md`, rewritten lean (a subdirectory copy was not auto-loaded).
- `CLAUDE/PROJECT_BRIEF.md` body regenerated; decision log kept verbatim plus a 2026-10-18 entry.
- `CLAUDE/RESEARCH_CONTEXT.md` deleted (May 2026 handoff for a superseded methodology); its
  dataset quirks moved to `data/README.md`, its open ideas and cautions into the brief.
- `CLAUDE/STATUS.md` (this file) and `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md` added.
- Four never-committed May 2026 drafting docs recovered from a stale worktree into
  `paper/historical/2026-05-drafting/`.

## Regenerating anything

`make test` for the fast suite. `uav-tda <x>-report` rebuilds a results doc from committed CSVs
with no recompute (`znorm-report`, `manuscript-report`, `windowed-report`, `exact-report`);
`uav-tda latency` re-measures. `outputs/` is complete, so `uav-tda tda` never needs re-running.
Install only with `pip install -e ".[dev]"`.
