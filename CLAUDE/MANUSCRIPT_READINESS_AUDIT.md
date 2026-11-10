# Manuscript Readiness Audit — 2026-10-18

**Scope.** Can the full AIAA SciTech 2027 manuscript (due December 2026) be written from the
repository as it stands? Audited: the submitted extended abstract
(`CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`, unchanged since submission), the five
machine-generated results docs in `paper/`, the three historical docs, every CSV under
`results/tables/`, the figure inventory, the bibliography and LaTeX build, the full git history
(98 commits, 17 PRs), and the engineering state (tests, CI, environment, disk).

**Method.** Five parallel read-only audits (git history; context docs; results-package integrity;
paper claim-by-claim reconciliation; repo/build readiness), followed by direct spot-checks of
every contradiction that is asserted below as fact (hera ground metric, seed controls, feature
count, zero-distance artifact, frontier timing path). Nothing scientific was re-run.

---

## 1. Verdict

| Area | Readiness | One-line reason |
|---|---|---|
| Infrastructure / reproducibility | **Ready** | PR #17 merged; `main` == `origin/main`; fast suite 103 passed / 0 skipped; ruff clean; CI green; `outputs/` complete so no multi-hour `tda` rerun is needed |
| Results numbers | **Ready, pending sign-off** | Zero numeric mismatches between any `paper/*.md` and its CSVs (every aggregate recomputed from per-seed / per-run / per-shard data); all 9 provenance SHAs exist |
| Results narrative | **One substantive fix needed** | The windowed compute-frontier's "6.6× cheaper" compares a W2-only per-flow cost against a Rips-inclusive windowed cost (see §5, I-1) |
| Abstract text as a base for the manuscript | **Needs significant correction** | 3 superseded headline numbers, 6 contradicted methodological statements, the central "C2 adds noise / N+P is best" interpretation is wrong under every current evaluation |
| §V promises | **4 delivered, 2 partial, 1 not** | 10 seeds + CIs, znorm, seed-123 reanalysis, per-flow vs windowed: done. "Exact" W2 and onboard ARM latency: partial. Cross-benchmark: not done |
| Figures | **Not ready** | 15 of 18 PNGs are April 2026 legacy output; no generator exists for the five figures the manuscript needs |
| LaTeX / bibliography | **Minor** | Builds clean (10 pages) once pgfplots compat matches the local TeX Live; 19/19 cite keys resolve; 6 bib entries render badly |
| Context docs for assistants | **Fixed today** | `CLAUDE/` regenerated; see `CLAUDE/STATUS.md` |

**Bottom line:** the science is finished and traceable. What stands between the repo and a
manuscript is (a) author sign-off on the five results docs and a headline-number decision,
(b) one frontier-cost recomputation, (c) five figure generators, and (d) a rewrite of the
abstract's Results/Limitations text, not an extension of it.

---

## 2. Blocking items, in order

1. **Author sign-off** on `paper/{EXACT_RESULTS,MANUSCRIPT_STATS,WINDOWED_RESULTS,LATENCY_RESULTS,ZNORM_RESULTS}.md`. Decide the per-flow headline: full-test δ≤0.01 (0.9082 [0.9033, 0.9131]) vs 10-seed probe (0.8930 ± 0.0078). They are different estimands; the exact value lies outside the probe CI in 11 of 14 cells, so the probe CI must never be presented as a prediction interval for the exact number.
2. **Recompute the compute frontier on a like-for-like basis** (`uav_tda/windowed.py:782`, `_time_per_flow_probe` times `probe.run_probe`, which loads precomputed diagrams and never builds a Rips complex). Either time the per-flow arm on the full standardize→Rips→W2 path (the latency harness already does: 3.5762 s) or strip Rips from the windowed marginal. As written, W=25 is ~400× cheaper per decision, not 6.6×. Update `compute_frontier.csv`, the frontier pgfplots snippet, and `WINDOWED_RESULTS.md` §3.
3. **Rewrite the C2 story.** Drop "C2 contributes noise more than signal; N+P is best" (§IV.C, Fig. 2 caption, §V.A). All-three beats N+P under z-normalized scoring at 3, 10 seeds and full test (0.908 vs 0.874), and its 10-seed std (0.008) is smaller than N+P's (0.012). The pre-submission Diagnostic B had already called the C2 effect an artifact.
4. **Correct the four methodological misstatements** (§3 below, M5/M8/M13/D4): W2 ground metric is L2 not L∞; the validation-Normal threshold was never computed in the abstract's lineage; the seed does not control k-medoids or Wasserstein tiebreaking; feature count is 20 raw / 22 encoded, not 21.
5. **Write the five figure generators** (detection-vs-W with shuffle, contamination curve, latency bars, attribution heatmap exact + windowed-by-W, full-test ROC) and quarantine the April PNGs.
6. **Reword three §V promises**: "exact" → "high-precision (hera δ≤0.01)" with the intractability finding; "onboard ARM" → Apple-silicon workstation under x86_64 emulation plus a portable harness (or run a Jetson/RPi before December); "cross-benchmark" → future work or run one.
7. **Reconcile window-level attribution flips** (Sybil→C2 at every W, 0.92–0.93; Flooding→C2 at W≤100; Wormhole→Network at W≤50). This reframes the abstract's Sybil-divergence hypothesis: Zeng's addressing prediction is recovered under temporal aggregation, so the per-flow divergence is a resolution effect. Discuss, do not bury.

---

## 3. Claim-by-claim status of `main.tex`

Status key: VERIFIED (true and traceable) · SUPERSEDED (newer definitive number) · CONTRADICTED (repo shows it wrong) · UNCHECKABLE.

### 3.1 Dataset and preprocessing

| # | Claim | Status | Evidence |
|---|---|---|---|
| D1 | 122,171 UDP flows; class counts 26,172 / 26,110 / 26,086 / 24,077 / 19,726 | VERIFIED | `data/UAVIDS-2025.csv` |
| D2 | 70/15/15 → 85,519 / 18,326 / 18,326 | VERIFIED | `outputs/labels_*.csv` |
| D3 | "23 attributes" / "the dataset's 23 features" | CONTRADICTED (off by one) | 23 columns include `label`; 22 attributes |
| D4 | "21 retained flow features", "raw 21-feature RF", figure node "21 features" | CONTRADICTED (off by one) | 22 − FlowID − Protocol = 20 raw; one-hot ports → 22 encoded (7+10+5); RF input has 22 columns |
| D5 | Protocol constant; only ports 9 and 654; C2 7 / Network 10 / Physical 5 lists; train-only scaling | VERIFIED | `uav_tda/config.py`, `uav_tda/data.py` |

### 3.2 Method and hyperparameters

| # | Claim | Status | Evidence |
|---|---|---|---|
| M1 | 500-point k-medoids reference cloud "for each manifold" | VERIFIED, undisclosed detail | k-medoids is run in **C2 space only**; the row positions are reused for Network and Physical (`uav_tda/tda.py:39-75`). Say so |
| M2 | ε_max = Q25; 0.494 / 0.145 / 0.333 | VERIFIED | `outputs/max_edge_lengths.json` 0.4938 / 0.1445 / 0.3332 (file is gitignored; commit the values) |
| M3 | Sparse ε=0.5, k_max=3 (C2, Network); exact k_max=2 (Physical) | VERIFIED | `config.SPARSE_RIPS_EPSILON`, `MAX_HOM_DIM` |
| M4 | Score = W2 summed over dims, summed over manifolds (not max-pool) | VERIFIED | `probe.py`, `metrics._subset_score` |
| M5 | Matching cost = squared **L∞** distance | **CONTRADICTED** | every hera call uses `internal_p=2.0` → L2 ground metric (`probe.py`, `exact.py`, `latency.py`) |
| M6 | Hera via GUDHI | VERIFIED | gudhi 3.11.0 |
| M7 | Raw C2 distances "roughly five times" Network/Physical | VERIFIED (probe) / SUPERSEDED | Diagnostic B 5.4× / 5.2×; full-test val-Normal means 0.079 vs 0.009 / 0.019 → 8.4× / 4.2× |
| M8 | τ = 95th percentile of validation-Normal summed score; "FPR near 0.05" | **CONTRADICTED for the abstract's lineage** → now VERIFIED | `tools/quick_unsup_probe.py` scored test only and never computed a threshold; `pipeline.py` uses per-manifold thresholds + OR. Full-test FPR with τ from 3,926 val-Normal: all-three znorm 0.056, N+P znorm 0.056; TPR 0.57 (Sybil 0.91, Flooding 0.82, Blackhole 0.34, Wormhole 0.30) |
| M9 | No attack labels at any calibration stage | VERIFIED | — |
| M10 | 200/class, K=50, δ=0.2 | VERIFIED | provenance JSONs |
| M11 | "Doubling resolution from K=20, δ=0.5 confirms results are not approximation artifacts" | Overstated | Rankings stable, but `PROBE_RESULTS.md` Diagnostic A verdict was "unstable": C2-only binary AUC moved 0.46 → 0.76; K 20→50 is 2.5× |
| M12 | ~60 s per seed, 2021-era M1 Pro | UNCHECKABLE (plausible) | — |
| M13 | Seed controls sampling, **k-medoids init, and Wasserstein tiebreaking** | **CONTRADICTED (2 of 3)** | Seed → per-class sample only. `reference_indices.npy` is a single file from seed 42; hera has no RNG. The real extra variance is GUDHI sparse-Rips per-call nondeterminism in the recomputed C2/Network baseline |

### 3.3 Results as published (3-seed raw-sum probe)

Every number in Table 1, Fig. 2 and the §IV text **reproduces bit-exactly** from the three
committed probe CSVs through `uav_tda.metrics.aggregate_over_seeds` (e.g. N+P 0.8594 ± 0.0043,
all-three 0.8577 ± 0.0276, Sybil/Network 0.8698 ± 0.0021, max dominant-manifold std 0.0162).

| # | Claim | Status | Current value |
|---|---|---|---|
| R1 | N+P 0.86 ± 0.00 | SUPERSEDED | full test 0.8711 raw / 0.8737 [0.8673, 0.8799] znorm; 10-seed 0.8692 ± 0.0122 znorm. The published ± 0.0043 was a low draw (clean 3-seed raw std 0.0152) |
| R2 | All-three "statistically equivalent (0.86 ± 0.03) with substantially larger variance; C2 contributes noise" | **CONTRADICTED** | full test all-three 0.9037 raw / **0.9082 [0.9033, 0.9131]** znorm; 10-seed znorm 0.8930 ± 0.0078 |
| R3 | "Best-performing subset is N+P" | **CONTRADICTED** | all-three is best under every current evaluation |
| R4 | Sybil/Flooding → Network, Blackhole/Wormhole → Physical | VERIFIED per-flow | full test 0.888 / 0.787 / 0.792 / 0.728; replicates at 3 and 10 seeds. **Window level differs** (Sybil → C2 at all W) |
| R5 | Attribution range 0.74–0.87 | SUPERSEDED | full test 0.728–0.888 |
| R6 | RF 96.3%; TDA features do not help LR/RF/SVM | VERIFIED | `supervised_summary.csv` (curated RF 0.9627; plain RF 0.9618). Legacy `comparison_improvements.csv` / `baseline_metrics.csv` are pre-leakage-fix and must not be cited |
| R7 | 4.6% of C2 calls on seed 123 exceeded the 5 s cap | VERIFIED | 139/3000 per-dimension calls; 7 flows fully zeroed. It is per-call, not per-flow |
| R8 | C2-containing variance "inherits spread from this artifact" | PARTLY CONTRADICTED | at 30 s with 0 timeouts the C2-only std is larger (0.030 / 0.040); sparse-Rips noise dominates |
| R9 | Physical-only 0.61; Sybil/Physical 0.20 | VERIFIED + new diagnostic | **34% of probe flows have W2_physical exactly 0** (Sybil 72%, Normal 43%, Flooding 40%, Wormhole 12%, Blackhole 4%); 0% in the untruncated full-test campaign → top-K=50 truncation artifact on the exact-Rips manifold. Disclose; it explains the weak Physical-only numbers |
| R10 | Cells near 0.5 "indicate no separation" | Wording | Sybil/Physical 0.20 and Wormhole/Network 0.28 are anti-correlated, i.e. informative, not uninformative |

### 3.4 §V promises

| # | Promise | Status | Where |
|---|---|---|---|
| P1 | Exact Wasserstein on the complete test set | **PARTIAL** | delivered as hera δ≤0.01 over 18,326 test + 3,926 val-Normal, 0 exceptions, δ-insensitivity ≤ 0.0002; literal δ=0 intractable (13/13 sampled flows > 120 s, ~30-day projection); POT/LAP backend unavailable. `EXACT_RESULTS.md` |
| P2 | Ten seeds + bootstrap CIs | DELIVERED | `MANUSCRIPT_STATS.md`; snippets `attribution_table_rows.tex`, `binary_auc_pgfplots.tex` |
| P3 | Val-Normal z-normalization replacing the unweighted sum | DELIVERED | `ZNORM_RESULTS.md`, `MANUSCRIPT_STATS.md`, `EXACT_RESULTS.md` |
| P4 | Seed-123 timeout reanalysis | DELIVERED (clean lineage) | 0 timeouts at 30 s; original diagrams no longer exist, so this is a reanalysis on regenerated diagrams |
| P5 | Per-flow vs time-windowed at matched compute | DELIVERED with disclosures | `WINDOWED_RESULTS.md`; W=100 0.9376 ± 0.0005 [0.926, 0.948]; shuffle Δ +0.32–0.37. Disclose: no timestamps (FlowID order), window-majority labels are a different population than flow labels, attribution flips, thin pure-Normal bins. Frontier cost needs item 2 of §2 |
| P6 | Onboard latency on ARM companion hardware | **PARTIAL** | `LATENCY_RESULTS.md`: M1 Pro workstation, Python under Rosetta 2; per-flow 3.576 s mean / 3.859 s p95 (C2 Rips 3.06 s); windowed W=25 0.0075 s/decision + 78 s setup; harness portable, no board run |
| P7 | Cross-benchmark evaluation | **NOT DELIVERED** | only UAVIDS-2025 in the repo |
| P8 | "1–3 s per flow" | **CONTRADICTED** | 3.58 s mean / 3.86 s p95 |

### 3.5 Background claims (not checkable against the repo)

HX-2 "nearly four thousand ... in 2024" and "immune to electronic warfare" (marketing wording;
verify attribution); the India–Pakistan citation (`Sareen` bib entry is suspicious, the unused
`Kronstadt` CRS report is sturdier); "Department of War" (add a parenthetical on first use);
Wang 30.45%, Kacem & Benjamin 62 studies; Zeng RF > 96% (consistent with the repo's 96.2–96.3%).

---

## 4. Results-package integrity

**Verified.** All headline numbers in `EXACT_RESULTS`, `MANUSCRIPT_STATS`, `ZNORM_RESULTS`,
`WINDOWED_RESULTS`, `LATENCY_RESULTS`, and the three historical docs were recomputed from the
underlying per-seed / per-run / per-shard CSVs and agree to 4 dp or better. All three LaTeX
snippets match their source tables. All 9 git SHAs referenced by the 262 provenance sidecars exist.

**Numbers with no CSV behind them** (prose or config comments only; persist as small CSVs if cited):
the δ=0 intractability benchmark (361 / 243 / 211 s per flow); "0 shards in 47 min"; windowed
run-length medians; exact-Rips gate medians and the 35.1 s first trial; the windowed pure-block
gate, Blackhole-majority counts, attack-frac oracle, and determinism check (all report-time, all
reproduced exactly on recompute); the per-flow 0.0602 s one-off timing; the 139 seed-123 timeouts;
the prior K=20 probe pass; ε_max values (`outputs/max_edge_lengths.json` is gitignored).

**Missing provenance sidecars:** `rebuild/znorm_summary.csv`, `rebuild/znorm_per_attack.csv`,
all three `paper_snippets/*.tex`, `exact/manifest.json`.

**Cross-document inconsistencies:**

- **I-1 (substantive).** Per-flow decision cost: `WINDOWED_RESULTS` §3 uses 0.0602 s (W2-only,
  precomputed diagrams) while `LATENCY_RESULTS` measures 3.5762 s on the full path. The "6.6×
  cheaper at W=25" claim is not defensible as written; like-for-like it is ~400×. Also
  `LATENCY` W=25 0.0075 s vs frontier 0.009064 s for the same quantity (17% apart).
- **I-2 (wording).** Exact full-test AUC lies outside the 10-seed probe bootstrap CI in 11/14
  cells (headline 0.9082 vs [0.8849, 0.8999]). Different estimands; `EXACT_RESULTS` §4's "within
  ±0.05" is true but the CI must not be framed as predicting the exact value.
- **I-3 (statistics).** Windowed window counts are pooled ×10 over bit-identical C2/Physical
  repeats (W=50 bin-0 "n=60" is 6 unique windows; W=200 "190/910" is 19/91). The ± 0.0005
  detection std and the frontier's ± 0.000 error bars reflect only Network sparse noise; the
  bootstrap CI is the real uncertainty. Report unique n and use CIs for error bars.
- **I-4 (finding).** Window-level attribution flips vs per-flow (see §2 item 7).
- **I-5.** `main.tex` still carries the 3-seed raw numbers, "1–3 s", and "exact Wasserstein".
- **I-6.** `LATENCY_RESULTS` §6 notes `outputs/scaler_{m}.pkl` are stale foreign-schema files; the
  harness uses `scalers.pkl`. Trap for anyone re-running by hand.

---

## 5. Figures and LaTeX snippets

| Manuscript figure | Data | Generator | Status |
|---|---|---|---|
| Pipeline diagram | — | `pipeline_figure.tex` (TikZ) | exists; node "21 features" → 20 raw / 22 encoded; caption "summed" → "z-scored and summed" if znorm adopted |
| Binary-AUC bars | `rebuild/binary_auc.csv`, `exact/exact_binary_auc.csv` | inline pgfplots + `binary_auc_pgfplots.tex` | 10-seed znorm snippet exists; **no exact-campaign version** |
| Per-attack attribution table | `rebuild/manifold_attribution.csv` | `attribution_table_rows.tex` | 10-seed raw; **no exact or znorm version** |
| Compute frontier (lead figure) | `windowed/compute_frontier.csv` | `windowed_frontier_pgfplots.tex` | exists but blocked on I-1 and I-3 |
| Detection vs W, ordered vs shuffled | `windowed/windowed_detection.csv` | none | **GAP** |
| Contamination curve | `windowed/contamination_curve.csv` | none | **GAP** |
| Latency bars / per-stage breakdown | `rebuild/latency_*.csv` | none | **GAP** |
| Attribution heatmap (exact, and windowed by W) | `exact/exact_per_attack_auc.csv`, `windowed/windowed_attribution.csv` | none | **GAP** (April heatmap is legacy) |
| Full-test ROC curves | `exact/shard_test_*.csv` | none (`plot_unsupervised_roc` reads a different schema) | **GAP** |
| Example barcodes / diagrams | `outputs/persistence_diagrams/*.pkl` | `evaluate.plot_barcode_examples` (never run under current code) | stale PNG only |

`results/figures/`: 15 of 18 PNGs are April 2026 output of the pre-monolith scripts (including
every `wasserstein_*` plot) and must not be used; only `supervised_comparison.png` and
`feature_importance_curated.png` come from current code. The May 2026 figure audit
(`paper/historical/2026-05-drafting/FIGURE_AUDIT.md`) also found every PNG too wide for
single-column print. No module in `uav_tda/{windowed,exact,latency,manuscript,znorm_report}.py`
calls `savefig`; all current figure output is LaTeX snippets. Snippet coverage gaps: exact tables,
exact-vs-probe, δ-insensitivity, detection-vs-W, shuffle, attribution survival, contamination,
latency, znorm delta.

---

## 6. LaTeX build and bibliography

- Toolchain: TeX Live 2021 at `/Library/TeX/texbin` (pdflatex, latexmk, bibtex). As committed the
  build fails on `\pgfplotsset{compat=1.18}` (local pgfplots 1.17); with 1.17 it builds clean:
  10 pages, zero undefined references or citations, zero bad boxes, one `[h]`→`[ht]` warning.
  Keep 1.18 for the publisher; note the local limit. `\lettrine` is provided by `new-aiaa.cls`.
- `pipeline.pdf` in the paper directory is unreferenced; remove or regenerate.
- Bibliography: all 19 cite keys resolve; 6 entries unused (Attali, Fabo, Kirichenko, Kronstadt,
  Molloy, Verkerken). Problems: `Davies` (`@unpublished`, prints its whole abstract in the
  reference list); `Helsing`, `Harper`, `Bendett` (`@misc` with ignored `journal` → ", , 2024."
  with no venue; Helsing has no URL); `Sareen` (truncated Economic Times URL unrelated to the
  stated journal; verify or replace with `Kronstadt`); `Cavanna` (empty note); `Eroglu`, `Kaufman`
  (`@inbook` with both author and editor → use `@incollection`); no access dates on any web source;
  `Davies` cited as both the original argument and a "subsequent review".

---

## 7. Engineering state (for anyone regenerating numbers)

- `main` == `origin/main` @ 4390e64 (PR #17 merged 2026-09-23). All 17 PRs merged; none open.
  CI: 2 runs, both green. One unmerged legacy branch (`claude/labelencoder-audit`, harmless).
- `make test`: 103 passed, 0 failed, 0 skipped, 11 deselected (slow), 1m47s. `make lint`: clean.
  Zero TODO/FIXME in `uav_tda/` and `tests/`.
- Environment: anaconda Python 3.9.7, Mach-O x86_64 under Rosetta 2 on an M1 Pro, 16 GB; numpy
  1.20.3, gudhi 3.11.0, scikit-learn 1.3.2, persim 0.3.8. **`requirements.txt` is hazardous**
  (pins numpy ≥ 1.24, pandas ≥ 2.0); install via `pip install -e ".[dev]"` only.
- Disk: ~10 GiB free. `outputs/` is 32 GB and complete (prep CSVs, reference indices, max-edge
  lengths, scalers, current per-flow diagrams for all manifolds × splits, plus ~5.6 GB of unused
  April-generation diagram files). A from-scratch `tda` rebuild is impossible without first
  deleting `outputs/`. Memory is chronically pressured (swap in use); keep `n_jobs ≤ 3`.
- Dataset sha256 matches `data/README.md`. 3,981 of 4,718 tracked files (84%) are
  `archive/poster_eda/node_modules/`; pack is 107 MiB. An April 2026 stash is the only ref keeping
  ~1.3 GB of leakage-era `.npy` blobs alive in the pack.
- Five stale worktrees under `.claude/worktrees/`. One of them held the only copies of four May
  2026 drafting docs, now rescued to `paper/historical/2026-05-drafting/`.

---

## 8. Suggested manuscript work plan

1. Sign-off meeting: approve or amend the five results docs; pick the per-flow headline (recommend
   full-test δ≤0.01 with the 10-seed probe as the variance study); approve the C2-story reversal.
2. Fix I-1 (frontier per-flow cost), add missing sidecars, persist the report-time diagnostics as
   CSVs, commit the ε_max values. Regenerate `WINDOWED_RESULTS.md` and the frontier snippet.
3. Add snippet emitters for the exact tables and a figure module (`uav_tda/figures.py`) producing
   the five gap figures from committed CSVs, golden-tested like the existing emitters; quarantine
   `results/figures/*.png` from April.
4. Rewrite `main.tex` §III.E–F, §IV, §V against §3 of this audit; add the znorm, windowed,
   latency, and reproducibility sections; fix the bibliography; keep every number traceable to a
   snippet or CSV.
5. Optional before December: a companion-board latency run (harness is one command), a
   threshold / precision–recall sweep, per-class H0/H1 bar-count analysis for the
   topology-simplification hypothesis.

Full agent reports from this audit (git history, results integrity, paper claims, repo readiness)
were produced in the session scratchpad and are summarized above; the decision log entry for this
audit is at the bottom of `CLAUDE/PROJECT_BRIEF.md`.
