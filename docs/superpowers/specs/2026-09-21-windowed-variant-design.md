# Time-Windowed Multi-Manifold Persistence — Phase 5 Design Spec

**Date:** 2026-09-21 · **Status:** approved design, pending implementation plan
**Authors:** Oliver Liu (decisions) with Claude (drafting), per the Phase-5 brainstorming session
**Context:** paper §V names the per-flow vs time-windowed comparison "the lead methodological contribution of the manuscript." This spec defines the windowed variant and its evaluation. Prior phases: `uav_tda` package (P2), val-Normal z-normalized scoring + w2-timeout (P3), 10-seed bootstrap-CI statistics + paper emitters (P4).

## 1. Data facts that constrain the design (measured, 2026-09-21)

- **UAVIDS-2025 has no timestamp column.** Only `FlowID`, a sequential row index. "Temporal" windows are therefore **FlowID-order windows** — row order as a time proxy. This is a manuscript disclosure item.
- **Label run-length structure is extreme** (8,561 contiguous-label runs in 122,171 rows): median run Wormhole 13,043 · Flooding 2,928 · Sybil 2,101 · Normal 2 · **Blackhole 1**. Flooding/Sybil/Wormhole form huge pure blocks; **Blackhole is interleaved with Normal at flow granularity**, so every Blackhole-era window is a mixed window. Consequence: windowing is expected to cost detection precisely on the attack class without temporal clustering — a measurable instance of the §V "attribution resolution" tradeoff, reported via the contamination curve (§4.2).

## 2. Method definition

### 2.1 Windowing
- A window is **W consecutive flows** of the test split in FlowID order; non-overlapping; the trailing partial window is dropped (documented).
- **Shuffle control:** the identical pipeline run after permuting the test-split row order with `np.random.default_rng(order_seed)`. Interpretation: AUC preserved under shuffle ⇒ ordering carries no signal (windowing measures set-composition only); AUC drops ⇒ temporal locality is load-bearing. Either outcome is reported, converting the no-timestamp caveat into a measured robustness check.
- Window sizes: **W ∈ {25, 50, 100, 200}**.

### 2.2 Per-window topology
- Per manifold (c2 / network / physical, the frozen Phase-2 partition), the W standardized feature vectors **are** the point cloud; one Vietoris–Rips complex per (window, manifold). This is the Bruillard-et-al.-2016 construction (the paper's cited closest prior work) and the source of the compute claim: N/W complexes on W-point clouds replace N complexes on 501-point clouds.
- Filtration cap: the existing per-manifold `max_edge_lengths` (consistency with the per-flow method; standardized units).
- Homology dims: the existing `MAX_HOM_DIM` per manifold.
- **Exact-Rips preference:** at W ≤ 200, exact Rips (sparse=None) is expected tractable for all manifolds. The implementation plan includes a benchmark gate: use exact wherever a single (window, manifold) call stays within budget (~2 s at W=200); otherwise sparse ε=0.5 for that manifold only, disclosed. If exact holds everywhere, the windowed arm is **fully deterministic** — eliminating the sparse-Rips process-noise class documented in Phases 2–3 and making repeat runs bit-reproducible (itself a reportable property).

### 2.3 Baseline and scoring
- Validation-split **Normal** flows, in their own row order, are chunked into windows of W; each yields a diagram per manifold.
- Baseline barcode per (manifold, W) = the **W2-medoid** of the val-normal window diagrams (the diagram minimizing summed W2 to the rest; pairwise matrix over the ~⌊3,926/W⌋ windows).
- Test-window score per manifold = W2(window diagram, baseline), summed over homology dims (hera, order=2, internal_p=2).
- **Z-normalization (canonical scoring, per the Phase-3 ruling):** per-manifold z-scores on the val-normal-window score distribution, then summed for subsets. Raw sums also reported. **Same-run coupling:** baseline, z-stats, and test scores come from one invocation (one process), mirroring `run_probe_with_znorm`.

## 3. Evaluation design

### 3.1 Detection
- Window ground truth: **majority label** (>50 % attack flows ⇒ positive). Window metadata retains `attack_frac` and per-class composition.
- Window-level binary AUC for all 7 manifold subsets × {raw, znorm} × W × {ordered, shuffled} — table shape mirrors the per-flow tables for direct comparison.

### 3.2 Contamination curve
- AUC (and score distributions) binned by window `attack_frac` (bins incl. 0, (0,.25], (.25,.5], (.5,.75], (.75,1)); per attack class where sample sizes permit. This is the quantitative home of the Blackhole mixed-window finding.

### 3.3 Attribution survival
- Per-attack one-vs-rest AUC per manifold at window level (windows labeled by majority class): does Sybil→Network / Flooding→Network / Blackhole→Physical / Wormhole→Physical survive aggregation, and at which W does it degrade?

### 3.4 Matched-compute frontier (the manuscript's lead figure)
- Per run: wall-time telemetry (total, per-window, per-Rips-call breakdown). Frontier: detection AUC vs per-decision cost with five points — W ∈ {25, 50, 100, 200} plus the per-flow arm.
- Per-flow arm: AUC from **Phase-4's existing 10-seed tables** (no recompute); per-decision cost from a small timed per-flow sample (same machine, recorded).

### 3.5 Repetitions and uncertainty
- Per W: **10 FlowID-order repeats + 10 seeded shuffle controls** (symmetry with Phase 4's 10 seeds, user decision). If exact Rips holds, ordered repeats should be bit-identical — reported as a determinism result rather than variance; shuffle seeds still provide a distribution.
- CIs: cluster bootstrap reusing `manuscript.bootstrap_mean_auc_ci` (stratified within run, across-run mean, B=2000).

## 4. Implementation architecture

- **New module `uav_tda/windowed.py`:** `make_windows(n_rows, W, order=None)`; `window_diagram(points, max_edge, max_hom_dim, sparse)`; `baseline_medoid_diagram(diagrams)`; `run_windowed(ws, W, split="test", order_seed=None, ...) -> (window_df, stats, timing)` (coupled single-invocation pass; window_df carries scores + `start_idx, majority_label, attack_frac`); `build_windowed_tables(...)`; report generator for `paper/WINDOWED_RESULTS.md`.
- **Reuse, not reimplementation:** `metrics.MANIFOLD_SUBSETS` (subset sums), z-norm helpers, `manuscript.bootstrap_mean_auc_ci`, `provenance.write_provenance`, Phase-4 emitter pattern for the frontier pgfplots snippet.
- **CLI:** `uav-tda windowed --w W [--order-seed N] [--repeat K]`; `uav-tda windowed-report` (ensures the campaign, builds tables + report + snippet).
- **Artifacts:** `results/tables/rebuild/windowed/` — per-run CSVs + `windowed_detection.csv`, `windowed_attribution.csv`, `contamination_curve.csv`, `compute_frontier.csv`, provenance sidecars throughout; `paper_snippets/windowed_frontier_pgfplots.tex`; `paper/WINDOWED_RESULTS.md` (machine-generated, hedged pending author sign-off).
- **Locked surfaces unchanged:** golden-master metrics functions, probe default path, `pipeline.py`, `tools/`, published top-level tables, pre-existing `paper/*.md`.

## 5. Testing

- Unit (fast): windowing arithmetic (counts, trailing-drop, permutation correctness); medoid on hand-built diagrams; majority/attack_frac labeling; **exact-Rips determinism** (same window ⇒ identical diagram — a stronger guarantee than the per-flow arm permits); z-stat coupling; emitter golden strings.
- Benchmark gate (T1 of the plan): per-manifold exact-Rips cost at W=200; decides exact vs sparse per manifold; numbers recorded in the report.
- Integration (slow): W=100 end-to-end on real data with sanity gates — pure-block attacks (Flooding/Sybil/Wormhole) window-AUC high (≥0.85 expected given pure windows); contamination curve broadly monotone; timing captured; gates report-not-loosen on miss.

## 6. Campaign

- 80 runs (4 W × 20). Estimated minutes per run ⇒ ~2–6 h total, detached, incremental, restartable (missing-run guard in the report builder, same pattern as Phase 4).

## 7. Out of scope (deferred)

Overlapping/sliding windows; per-window flow-level localization; online/streaming operation; ARM-hardware timing (Phase 6 — the timing harness built here is what Phase 6 reruns on ARM); train-split windowing; alternative baselines (barycenter, k-medoid sets).

## 8. Decisions log (from the brainstorming session)

1. Window basis: FlowID order **+ shuffle control** (user-selected, recommended).
2. Window TDA: **window-only Rips** (user-selected, recommended; Bruillard-aligned; delivers the compute claim).
3. Window labels: **majority + contamination curve** (user-selected, recommended).
4. W grid: **{25, 50, 100, 200}** (user-selected, recommended).
5. Repetitions: **10 + 10 per W** (user-selected over the recommended 3+3; full symmetry with Phase 4).
