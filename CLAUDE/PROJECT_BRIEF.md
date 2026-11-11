# Project Brief — Multi-Manifold Topological Analysis of UAV Network Attacks

**Last updated:** 2026-10-18 (body regenerated; decision log at the bottom is append-only and verbatim since 2026-05-23)
**Authors:** Oliver Liu (Baylor CS alum; Capital One), advised by Dr. Liang Sun and Dr. Natnael Negash (Baylor ME / AVIA Lab)
**Target venue:** AIAA SciTech 2027, Intelligent Systems track. Extended abstract **submitted** (source: `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`, unchanged since submission). Full manuscript due **December 2026**.
**Status:** Infrastructure program complete (Phases 1–7, PRs #11–#17, 2026-09-19 → 09-23). Results package complete and machine-generated, **all numbers pending author sign-off**. Remaining work is manuscript-side. See `CLAUDE/STATUS.md` for the current readiness audit and open items.

---

## 0. How to use this document

Sections 1–4 are the framing: read them before touching any claim. Section 5 is the methodology
snapshot. Sections 6–8 record what is settled, what was abandoned, and what is genuinely open.
Section 9 is writing guidance. The decision log at the end is the authoritative record of every
framing, numbers, or discipline change; **append, never edit**.

If you are an LLM picking this up: do not relitigate decisions recorded here. Raise disagreements
explicitly and log them. File discipline (frozen paths, write zones, ops rules) lives in the root
`CLAUDE.md`.

---

## 1. The story

### 1.1 As submitted (extended abstract, 3-seed raw-sum probe)

We applied multi-manifold persistent homology to UAV intrusion detection on UAVIDS-2025. The
original framing — "TDA improves classification accuracy" — is dead: raw Random Forest already
hits 96.3% and TDA features do not improve on it. Three surviving findings: (1) a label-free
anomaly detector built on per-flow Vietoris–Rips persistence diagrams compared via Wasserstein-2
distance to a reference-cloud baseline barcode achieves binary AUC 0.86 ± 0.00 on the
Network+Physical subset across three seeds, with no attack labels at calibration; (2) per-attack
manifold attribution reproduces across seeds: Sybil and Flooding → Network (0.87, 0.79), Blackhole
and Wormhole → Physical (0.79, 0.74); (3) three of four attributions agree with Zeng et al. (2025);
Sybil is the single divergence (Zeng predicted C2/source-addressing; topology finds Network).

### 1.2 Since submission (all pending sign-off; every number traces to `results/tables/rebuild/`)

| Result | Value | Source doc | Status |
|---|---|---|---|
| Per-flow, val-Normal z-normalized SUM, 10 seeds, probe (K=50, δ=0.2, 200/class) | all-three AUC **0.8930 ± 0.0078** [0.885, 0.900]; N+P 0.8692 ± 0.0122 | `paper/MANUSCRIPT_STATS.md` | pending sign-off |
| Per-flow, znorm, **full test split** (18,326 test + 3,926 val-Normal), hera δ≤0.01, no sampling/truncation | all-three AUC **0.9082** [0.9033, 0.9131]; raw 0.9037 | `paper/EXACT_RESULTS.md` | pending sign-off; this is the definitive per-flow number |
| Per-attack attribution at full-test scale | Sybil→Network 0.888, Flooding→Network 0.787, Blackhole→Physical 0.792, Wormhole→Physical 0.728 | `paper/EXACT_RESULTS.md` | replicates 3-seed and 10-seed exactly |
| Time-windowed variant (FlowID-order windows, W∈{25,50,100,200}, 80 runs) | best **W=100: 0.9376 ± 0.0005** [0.926, 0.948]; ordered ≥0.869 at every W; shuffle control 0.57 (Δ +0.32–0.37) | `paper/WINDOWED_RESULTS.md` | pending sign-off; lead manuscript contribution |
| Window-level attribution | Blackhole→Physical survives all W; **Sybil flips to C2 at all W**, Flooding→C2 at W≤100, Wormhole→Network at W≤50 then Physical at W≥100 | `paper/WINDOWED_RESULTS.md` | finding, not defect; must be reconciled with per-flow story |
| Per-flow latency (full standardize→Rips→W2 path, n=30) | **3.576 s mean / 3.859 s p95**; C2 Rips ≈97% of cost | `paper/LATENCY_RESULTS.md` | contradicts abstract's "1–3 s"; Rosetta-emulated x86_64 on M1 Pro, workstation-class |
| Windowed marginal cost | W=25: 0.0075 s/decision (+78 s one-time baseline) | `paper/LATENCY_RESULTS.md` | one decision covers W flows |
| Seed-123 "C2 timeout artifact" (4.6% in abstract) | 0 timeouts at 30 s cap (vs 139 at 5 s) | `paper/ZNORM_RESULTS.md` | resolved: compute-budget artifact |
| Literal exact Wasserstein (hera δ=0) | intractable: 13/13 sampled flows timed out at 120 s, ~30-day projection | `paper/EXACT_RESULTS.md` §1 | §V promise must be reworded |

**Supervised context (unchanged):** raw 22 features + RF curated 96.3% accuracy (0.997 weighted
AUC); combined raw+TDA 95.8%; TDA-only 81.0% (`results/tables/supervised_summary.csv`).

---

## 2. Research questions and contributions

**Primary question.** What do topological invariants of multi-manifold network-flow
representations reveal about cyber attacks on UAV swarms, and which functional subsystem
manifests each attack class most strongly?

- **RQ1.** Do distinct attack classes produce topologically distinguishable per-manifold
  signatures, and which manifold is dominant for each?
- **RQ2.** Can label-free detection calibrated on Normal traffic alone achieve usable binary AUC?
- **RQ3.** Do the empirical attributions align with Zeng et al.'s feature-level predictions, and
  what does the divergence indicate?
- **Manuscript addition (Phase 5).** How do per-flow and time-windowed variants trade detection
  AUC, attribution resolution, and per-decision compute?

**Not a research question:** whether TDA improves supervised accuracy. It does not. Settled.

**Contributions (exactly three in the abstract; the manuscript adds the windowed comparison as
the lead methodological contribution):** the three-manifold decomposition aligned with Zeng
Table III (C2 7 / Network 10 / Physical 5, pairwise disjoint); the label-free multi-manifold
Wasserstein detector; the per-attack manifold attribution and its agreement/divergence with Zeng.
The "C2 manifold paradox" (C2 hurts the combined detector) was retired 2026-05-24 as a probe
approximation and scale-mismatch artifact; under z-normalized scoring the all-three detector
beats N+P (0.893 vs 0.869 at 10 seeds; 0.908 vs 0.874 at full test).

---

## 3. Claims and non-claims

**Claims (manuscript-ready, pending sign-off):**
- Per-attack manifold attribution with dominant-manifold AUCs 0.73–0.89, replicating across 3
  seeds, 10 seeds, and the full test split.
- Label-free detection AUC 0.908 (all three manifolds, z-normalized, full test) without attack
  labels at any calibration stage; 0.874 when C2 is unavailable.
- Time-windowed variant reaches 0.938 at W=100 and is orders of magnitude cheaper per decision;
  FlowID-order temporal locality is load-bearing (shuffle control).
- 3-of-4 agreement with Zeng et al.; Sybil is the single per-flow divergence. At window level the
  Sybil signature moves to C2 (i.e. toward Zeng's prediction) — to be discussed, not hidden.

**Explicit non-claims:**
- Not higher accuracy than supervised classifiers (RF 96.3% beats every TDA configuration).
- Not demonstrated real-time or onboard deployment. Per-flow cost is 3.6 s on an emulated
  workstation; companion-class (Jetson/RPi) numbers do not exist yet.
- Not that TDA features complement raw features for supervised classification.
- No generalization beyond UAVIDS-2025 (single simulated benchmark, no timestamps).
- Never cite the poster's "57.1% baseline" (provenance unknown); never claim zero-day detection.

---

## 4. Methodology snapshot (verified against `uav_tda/config.py` and the production run)

- **Dataset:** UAVIDS-2025, 122,171 UDP flows, 23 columns, 5 classes. Quirks in `data/README.md`.
- **Split:** stratified 70/15/15 = 85,519 / 18,326 / 18,326, seed 42.
- **Manifolds:** C2 = last-octet SrcAddr/DstAddr + one-hot ports {9, 654} + FlowDuration (7 cols);
  Network = packet/byte counts and rates + MeanPacketSize (10); Physical = MeanDelay, MeanJitter,
  Throughput, PacketDropRate, AverageHopCount (5). StandardScaler fitted on train only.
- **Reference cloud:** 500 k-medoids Normal-Traffic training flows; medoids selected on C2, same
  row positions reused across manifolds (disclosure item, see `paper/historical/…/METHODS_DISCLOSURES.md`).
- **Filtration:** per-flow Vietoris–Rips on {query} ∪ reference; max edge = 25th percentile of
  reference pairwise distances (C2 0.494, Network 0.145, Physical 0.333; the spec's p95 blew up
  the simplex tree). Sparse Rips ε=0.5 for C2/Network (H0–H2), exact Rips for Physical (H0–H1).
- **Score:** Wasserstein-2 (hera) from the flow's diagram to the manifold's reference-only
  baseline barcode, summed over homology dimensions; **canonical combination = val-Normal
  z-normalized SUM across manifolds**. The abstract reported the raw unweighted sum; the
  monolith's max-pool survives only as the frozen port in `uav_tda/unsupervised.py`.
- **Threshold:** 95th percentile of validation-Normal scores. Label-free = no attack labels.
- **Probe (abstract and 10-seed campaign):** 200 flows/class, top-K=50 bars, hera δ=0.2,
  `w2_timeout` 30 s. **Full-test campaign:** all flows, full diagrams, hera δ≤0.01, no timeout.
- **Windowed variant:** non-overlapping FlowID-order windows; one Rips complex per
  (window, manifold) on window points only; W2-medoid val-Normal baseline; exact Rips for C2 and
  Physical, sparse ε=0.5 for Network; 10 ordered repeats + 10 shuffle controls per W.
- **Seeds:** `PROBE_SEEDS=(42,7,123)`; `MANUSCRIPT_SEEDS=(42,7,123,0,1,2,3,4,5,6)`;
  `SUPERVISED_SEEDS=(42,7,2024)`; `PRIMARY_SEED=42`. CIs are cluster bootstraps, B=2000.
- **Determinism:** GUDHI 3.11 sparse Rips is nondeterministic per call; sparse surfaces are
  statistically reproducible only. Part of the published "seed variance" is process noise.

---

## 5. Settled decisions (do not relitigate)

1. Capability framing, not performance framing; TDA-vs-RF accuracy is not the claim.
2. Exactly three contributions in the abstract; windowed comparison is the manuscript's lead addition.
3. Disjoint 7/10/5 partition per Zeng Table III; "manifold" used loosely, one footnote.
4. Label-free = no attack labels; Normal labels seed the reference cloud and threshold; deployment
   substitute is a trusted-initialization window.
5. Zeng mismatch is a finding, not a flaw; Sybil is the single per-flow divergence.
6. C2 paradox retired; z-normalized SUM is the canonical score.
7. `pipeline.py` is a byte-frozen oracle; `tools/`, top-level `results/tables/*.csv`, and the
   three historical `paper/*.md` are immutable provenance. New numbers only in `results/tables/rebuild/`.
8. "Exact" Wasserstein = hera δ≤0.01; δ=0 and the POT/LAP backend are abandoned.
9. Sparse-Rips comparisons use calibrated tolerances, never bit equality.
10. Environment pins stay (py 3.9.7 / numpy 1.20.3 / gudhi 3.11.0); no new runtime dependencies.
11. Every results doc is a candidate until the authors sign off; no number enters `main.tex` before then.

---

## 6. Tried and abandoned (so it does not get re-proposed)

- "TDA improves supervised accuracy" — dead (96.3% raw vs 95.8% combined vs 81.0% TDA-only).
- "TDA features complement raw features" — dead.
- "Persistence images add discriminative value" — no, curated subset ≈ summary-only.
- "GPS-denied ⇒ drop Physical" — superseded by "no C2 visibility ⇒ drop C2" (N+P subset).
- "C2 hurts the combined detector" (C2 paradox) — probe/scale artifact; retired 2026-05-24.
- Full-test exact W2 "infeasible (~74 h), deferred to journal" — **overtaken**: Phase 6 ran all
  22,252 flows at δ≤0.01 in ~1.0–1.2 s/flow. Only literal δ=0 remains infeasible.
- "Only one seed was run" — overtaken by 3 then 10 seeds.
- Fork-based W2 timeouts inside loky workers; closure-captured diagram dicts in joblib tasks —
  both crashed the Phase-6 campaign; replaced by in-process hera + chunked worker-side loading.
- Pre-campaign hypothesis "Blackhole is never a window majority" — disproved (19–21% of windows).
- "Contamination detection rate is monotone at every W" — pooling artifact; corrected to
  per-arm computation (non-monotone dip at 50–75% contamination for W≤100).

---

## 7. Genuinely open items

**Manuscript corrections (must land in the full manuscript):**
1. §V "exact Wasserstein on the full test set" → "high-precision (δ≤0.01)" plus the intractability finding.
2. §V "1–3 s per flow" → measured 3.58 s mean / 3.86 s p95 (C2 Rips dominates).
3. Latency numbers are Rosetta-emulated workstation figures, not companion-class hardware.
4. Windowed compute frontier: **resolved in `paper/MANUSCRIPT_ARTIFACTS.md`** (full-path latency cost;
   478x at W=25; old 0.0602 s / 6.6x superseded). The manuscript text still needs updating.
5. Reconcile window-level attribution flips (Sybil→C2 etc.) with the per-flow attribution story.

**Author decisions pending:** sign-off on the five results docs; keep vs untrack the third-party
dataset; untrack `archive/poster_eda/node_modules` (3,981 files, 84% of tracked files); delete
local `outputs/` (32 GB) vs keep for re-runs; drop the April 2026 stash that pins 1.3 GB of
leakage-era blobs in the pack; prune the five stale `.claude/worktrees/`.

**Experiments not run (optional):** companion-board (Jetson/RPi) or native-arm64 latency run
(harness is portable, `paper/LATENCY_RESULTS.md` §6); threshold / precision–recall sweep;
per-class H0/H1 bar-count analysis to test the "topology simplification" hypothesis for sub-0.5
attack×manifold AUCs (Blackhole on Network 0.33, Sybil on Physical 0.20); cross-benchmark evaluation
(promised in §V as deferred; no second dataset identified).

**Figures:** no current-code generator exists for the windowed detection-vs-W curve, contamination
curve, latency bars, attribution heatmaps, or full-test ROC curves. All `results/figures/*.png`
dated April 2026 are legacy monolith output and must not be used.

---

## 8. Writing reminders

- Capability framing. Resist any sentence comparing TDA AUC favorably to supervised AUC.
- State once in Methodology what "label-free" means and what substitutes for Normal labels in deployment.
- Disclose: p25 edge cap (why), sparse ε=0.5 and its nondeterminism, C2-only medoid selection,
  essential-H0 truncation, SVM 5k subsample, z-normalized SUM replacing the raw sum, FlowID order as
  temporal proxy, δ≤0.01 rather than exact, Rosetta-emulated latency.
- Zeng agreement is Blackhole, Wormhole, Flooding; Sybil is the divergence. At window level Sybil
  moves to C2 — discuss, do not bury.
- Every number traces to `results/tables/rebuild/**` via one of the five results docs, or to the
  immutable published CSVs for the as-submitted numbers. Do not round beyond two decimals in prose;
  tables may carry four. Use the golden-tested LaTeX snippets in `results/tables/rebuild/paper_snippets/`.
- Limitations go in §V, not the abstract.

---

## 9. Decision authority

Framing decisions (three contributions, manifold partition, label-free framing, Zeng mismatch as
finding) were made jointly by Oliver, Dr. Sun, and the advisory LLM on 2026-05-23/24. Every
infrastructure-era ruling (canonical score, δ≤0.01 redefinition, frozen surfaces) is recorded below
with its date. If new data contradicts anything here, raise it explicitly and append a log entry.

---

## Decision log

- **2026-05-23 (initial pivot):** Abandoned "TDA improves accuracy" framing
  after probe data showed raw RF baseline outperforming all TDA configurations.
  Adopted manifold-attribution + label-free + C2-paradox framing.
- **2026-05-23 (this brief):** Locked three contributions. Identified two
  blocking diagnostics (A and B) before any results finalization. Set venue
  to SciTech 2027 IS, length 5-page extended abstract.
- **2026-05-24 (post-diagnostics revision):** Completed Diagnostics A, B, and C per Section 10. Three substantive revisions to the brief: (1) Retired the C2-manifold-paradox framing entirely — both A and B indicate the original finding was a probe artifact. (2) Updated all empirical numbers to the three-seed means from Diagnostic C, with Network+Physical at 0.86 ± 0.00 (binary AUC) as the headline detection number. (3) Corrected the Zeng-mismatch reading: re-reading Zeng et al. Section II.D.2 shows that Zeng's predicted signature features for Blackhole and Wormhole (PacketDropRate, Throughput, AverageHopCount, MeanDelay) are in our Physical manifold, not Network. Our empirical attributions therefore agree with Zeng on Blackhole, Wormhole, and Flooding. Sybil is the single divergence — Zeng predicted source-addressing (C2) dominance; we observe Network dominance. Section 1, Section 3, Section 4 claims, and Section 8 abstract guidance updated accordingly. The Methodology Note from MULTI_SEED_VARIANCE.md (seed-123 C2 timeouts) is logged as a manuscript-version remediation item, not a current-paper blocker.
- **2026-09-21 (Phase-2 strangler complete):** The `uav_tda` package is now the maintained implementation of the pipeline, ported phase-by-phase from `pipeline.py` and equivalence-tested against it in debug mode using hermetic workspaces, with sparse-Rips nondeterminism handled via documented statistical comparators. `pipeline.py` remains frozen as the equivalence oracle and historical record of the paper's production run; it is not modified going forward. Section 9 file discipline is extended accordingly: `uav_tda/`, `tests/`, and `docs/` (plans and design notes, e.g. `docs/superpowers/plans/`) are now write zones alongside the existing ones.
- **2026-09-21 (Phase-3 znorm scoring + clean-lineage reanalysis):** Implemented §III.E Z-normalized subset scoring on the validation Normal-Traffic population (`metrics.znorm_stats_from_val`/`apply_znorm`/`binary_auc_by_subset_znorm`, `probe.run_probe_with_znorm`), replacing Diagnostic B's test-Normal-substitute lineage (`paper/C2_NORMALIZATION_DIAGNOSTIC.md`) with a same-run-coupled val+test probe pass. Wired `--w2-timeout` through `uav-tda probe --znorm` and re-ran seed 123 at 30s (versus the historical 5s/139-C2-timeouts noted in `paper/MULTI_SEED_VARIANCE.md`). Canonical forward-looking anomaly score is the z-normalized SUM across manifolds (probe/metrics path); `uav_tda/unsupervised.py` remains the frozen monolith-equivalent max-pool port and is not reconciled to match (docstring updated to state the ruling explicitly; no logic change). Refreshed 3-seed raw-vs-znorm numbers, per-attack attribution under both scorings, and the seed-123 timeout reanalysis are written to `paper/ZNORM_RESULTS.md` (generated by `uav-tda znorm-report` from `results/tables/rebuild/*.csv`) as candidates for the revisable abstract, pending author sign-off — the historical `paper/C2_NORMALIZATION_DIAGNOSTIC.md` and `paper/MULTI_SEED_VARIANCE.md` are preserved unmodified and only quoted as literals for comparison.
- **2026-09-21 (Phase-4 ten-seed bootstrap-CI campaign complete):** Ran the manuscript's ten-seed evaluation (seeds 42, 7, 123, 0, 1, 2, 3, 4, 5, 6; `config.MANUSCRIPT_SEEDS`) through the Phase-3 coupled val+test znorm probe at `w2_timeout=30s` (zero W_2 timeouts on every seed and manifold), and computed cluster-bootstrap 95% CIs (B=2000, stratified per-seed resampling) on every binary-detection and per-attack-attribution AUC (`uav_tda/manuscript.py`, `uav-tda manuscript-report`). Dominant-manifold attribution REPLICATES at ten seeds: Sybil→Network, Flooding→Network, Blackhole→Physical, Wormhole→Physical, identical to the published 3-seed finding. Headline 10-seed znorm binary AUC: all_three 0.8930 ± 0.0078 [0.8849, 0.8999], network_physical 0.8692 ± 0.0122 [0.8602, 0.8778]. Results tables are `results/tables/rebuild/binary_auc.csv` and `manifold_attribution.csv`; the full report with 10-seed vs recomputed-3-seed vs published-3-seed (quoted literals from `paper/MULTI_SEED_VARIANCE.md`) comparison is `paper/MANUSCRIPT_STATS.md` — **pending author sign-off** before any number enters the manuscript. LaTeX snippets for Table tab:per_attack_auc and Fig. fig:binary_auc are generated from these tables at report-build time (golden-string-tested emitters, `results/tables/rebuild/paper_snippets/*.tex`) so the paper and the code that produced its numbers cannot drift; they are candidates only, likewise pending author sign-off. Historical paper/*.md diagnostics remain unmodified.
- **2026-09-22 (Phase-5 windowed-variant campaign complete):** Implemented and ran the time-windowed multi-manifold persistence variant per `docs/superpowers/specs/2026-09-21-windowed-variant-design.md` (`uav_tda/windowed.py`, `uav-tda windowed`/`windowed-report`): non-overlapping FlowID-order windows of W ∈ {25, 50, 100, 200} flows, one Rips complex per (window, manifold), W2-medoid val-Normal baseline, same-run-coupled z-norm scoring, 10 FlowID-order repeats + 10 seeded shuffle controls per W (80 runs total), cluster-bootstrap 95% CIs (B=2000). **Headline frontier:** best windowed znorm all_three AUC is **0.9376 ± 0.0005 [0.9262, 0.9480] at W=100**, beating the per-flow arm's 10-seed 0.8930 ± 0.0078; AUC ordered-arm is ≥0.85 at every W (0.869/0.901/0.938/0.910 at W=25/50/100/200). Per-decision marginal cost rises monotonically with W (0.0091s at W=25 to 0.679s at W=200); the cheapest windowed point (W=25) is ~6.6× cheaper per decision than the per-flow arm's marginal 0.0602s (two-point-slope estimate) — frontier table, CI, and pgfplots snippet (`results/tables/rebuild/paper_snippets/windowed_frontier_pgfplots.tex`) all machine-generated from `compute_frontier.csv`. **Shuffle control:** ordered beats shuffled by +0.32 to +0.37 AUC at every W (e.g. W=100: 0.938 ordered vs 0.565 shuffled) — FlowID-order temporal locality is strongly load-bearing, not merely set composition (spec §2.1). **Determinism:** c2 and physical (exact Rips, `WINDOWED_SPARSE=None`) are bit-identical across all 10 ordered repeats at every W, confirmed as a measured result, not an assumption; network (sparse ε=0.5) differs run-to-run as expected (memory: sparse-rips-nondeterminism). **Attribution survival — the headline finding:** Blackhole attribution SURVIVES intact at every W (physical dominant, no flip, no NaN; contrary to the pre-campaign hypothesis that Blackhole's median flow-run-length of 1 would make it rarely/never a window majority — it is in fact a window majority in ~19–21% of ordered-arm windows). Sybil and Flooding instead show a striking, largely W-independent FLIP from the per-flow dominant manifold (network) to c2 at window level (Sybil: c2 dominant at all 4 W, AUC 0.92–0.93 vs network's 0.53–0.61; Flooding: c2 dominant at W=25/50/100, network recovers only at W=200); Wormhole flips to network at W=25/50 and recovers to physical (its per-flow dominant manifold) at W=100/200. None of these flips look like a code defect (systematic, directionally consistent, well-bounded AUCs/CIs) — they are reported as findings for author review, not corrected or hidden. Contamination detection_rate (ordered arm, computed per-arm after the H1 pooling fix) is monotone in `attack_frac` only at W=200; at W=25/50/100 it dips at (0.50,0.75] in lockstep with the mean-score column (mean_znorm_all_three), which is non-monotone at W=25/50/100 with the same dip — the earlier "monotone at every W" claim was an artifact of pooling the ordered and shuffled arms into one bin population, which inflated the (0.50,0.75] bin's high-score/high-detection population with shuffled-only windows; bin-0 (pure-normal) windows are extremely thin-to-empty at larger W (n=0 at W=200 across all 20 runs pooled), consistent with Normal's median flow-run-length of 2 — the empirical near-normal-reference substitute threshold (documented deviation, not the canonical val threshold) is used for `contamination_curve.csv`'s detection-rate column. Full report (config, all four tables, deviations, determinism note): `paper/WINDOWED_RESULTS.md`, generated by `write_windowed_report`, machine-generated from `results/tables/rebuild/windowed/{windowed_detection,windowed_attribution,contamination_curve,compute_frontier}.csv` and the 80 per-run CSVs — **pending author sign-off**, no number here yet approved for the manuscript. Three documented implementation deviations (contamination threshold substitute, attribution NaN-guard, per-flow two-point-slope timing) are listed in the report's §1 and in `windowed.py`'s `WINDOWED_DEVIATIONS`. Operational note: the 80-run campaign was launched detached before this session's report-generation wiring (`write_windowed_report`) landed in `cli.py`; the already-running process therefore wrote the 4 table CSVs but exited without the report/snippet (stale in-process module state, not a code bug in the now-committed `cli.py`) — the report and snippet were generated as a zero-recompute follow-up directly from the already-written table CSVs.
- **2026-09-22 (Phase-6 high-precision Wasserstein-2 campaign complete):** Ran the definitive full-test-split Wasserstein-2 campaign (`uav_tda/exact.py`, `uav-tda exact`/`exact-report`) replacing the probe approximation for the paper's §V "exact full-test" promise, over all 22,252 flows (test 18,326 + val-Normal 3,926, 45 sharded/resumable shards). **Not literally exact**: a pre-launch re-benchmark found hera's true exact mode (`delta=0.0`, auction-LP) timed out on the 120s fork-timeout for ≥1 homology dim of EVERY one of 13 sampled real flows across all 3 manifolds (13/13; not just pathological cases), projecting ~30 days at `--n-jobs 7` (~220× the plan's 3.3s/flow-based estimate) — itself a §V-relevant intractability finding. The alternative exact backend (`gudhi.wasserstein`, POT-backed LAP) was found unusable without a new, disallowed dependency (`import ot` unconditional, POT not installed). Controller ruling redefined the campaign as **high-precision** hera `delta<=0.01` (hera's own default tolerance, confirmed fast) run DIRECTLY IN-PROCESS on the campaign's hot path (`direct_w2_flow`) with **NO wall-clock timeout** — a per-dim call is wrapped only in a Python try/except (a caught exception zeroes that dimension and flags the flow, with no retry attempt); the separate `delta=0.05` timeout-retry machinery (`exact_w2_flow`) was used only for the pre-launch delta=0 intractability benchmark and is never called by the campaign itself. 0 exceptions were caught and 0 flows were flagged across all 22,252 campaign flows. Two further launch-engineering fixes were needed before a shard would complete: (1) `probe._w2_with_timeout`'s per-call fork nested inside a joblib/loky worker is unstable on macOS (first launch: 0 shards in 47min, workers dying) — fixed by calling hera directly in-process (`direct_w2_flow`, mirroring `pipeline.py`'s production `_wasserstein_for_flow`); (2) the parent process holding the full split's diagrams (~250-350MB/manifold) in a closure re-pickled per one of ~500 per-flow tasks exhausted this workstation's memory (second launch: 0 shards in 16min despite fix (1); confirmed via an n_jobs=1 isolation-probe smoke shard that completed cleanly with zero loky processes) — fixed by chunking each shard into ~100-flow, per-manifold worker-side-loading tasks (`_process_chunk_for_manifold`) at `--n-jobs 3`; the campaign then completed all 45 shards with zero worker churn, ~1.0-1.2 s/flow cadence. **Headline**: exact znorm all_three binary AUC = **0.9082 [0.9033, 0.9131]** (raw 0.9037 [0.8982, 0.9093]), vs the probe 10-seed mean **0.8930 ± 0.0078** — delta **+0.0153**, within the ±0.05 science gate (probe approximation measured, not assumed, to be conservative-but-close on this headline number; every subset showed the exact AUC ≥ or ≈ the probe mean, exact-minus-probe deltas ranging +0.0044 to +0.0524 across all 14 subset×scoring cells). Per-attack dominant-manifold attribution REPLICATES exactly at full-test scale: Sybil→network (0.8878), Flooding→network (0.7869), Blackhole→physical (0.7916), Wormhole→physical (0.7281) — identical to the 3-seed and 10-seed findings. A δ=0.01-vs-δ=0.05 insensitivity check on a 500-flow subsample found max |ΔAUC| = 0.0002 across all 7 subsets — the campaign's residual delta-tolerance approximation is negligible for AUC ranking, quantified rather than assumed. All four T3 sanity gates passed (znorm all_three within band; per-attack dominance matches; row counts exact 18,326+3,926; timeout/approx counts 0/0). Full report (config header with the intractability/POT/launch-engineering story, definitive binary+per-attack tables, exact-vs-probe-vs-published comparison, insensitivity check): `paper/EXACT_RESULTS.md`, generated by `write_exact_report`, machine-generated from `results/tables/rebuild/exact/*.csv` (assembled from the 45 shard CSVs + persisted baselines) — **pending author sign-off**, no number here yet approved for the manuscript. Full investigation timeline (three controller interventions, benchmark numbers, kill/fix/relaunch sequence): `.superpowers/sdd/2026-09-22-plan6-exact-w2-latency/task-3-report.md`.
- **2026-09-22 (Phase-6 Track B portable latency harness complete):** Implemented and ran the §V onboard-latency characterization (`uav_tda/latency.py`, `uav-tda latency`): a portable, one-command harness measuring the FULL production inference path (standardize → Rips → slice/truncate → Wasserstein-2) per sampled decision, for the per-flow arm (paper's deployed probe-style config: `SPARSE_RIPS_EPSILON`, `PROBE_TOP_K=50`, `PROBE_DELTA=0.2` — deliberately NOT the Track-A exact-campaign config) and the windowed arm (`WINDOWED_SPARSE` at each W ∈ {25, 50, 100, 200}), n=30 sampled decisions per mode. **Headline**: measured per-flow decision cost (all 3 manifolds summed) mean **3.5762s**, median 3.5855s, p95 **3.8591s** — ABOVE the paper's published "1–3 seconds" per-flow claim (`CLAUDE/PROJECT_BRIEF.md` §4, quoted as a 2020-era-laptop estimate); read per-manifold instead, c2 alone (3.0753s mean) sits almost exactly at the claim's upper bound while network (0.0813s) and physical (0.4196s) are well below it — c2's Rips construction (~3.06s, ~97% of its total) dominates the per-flow cost, consistent with C2's higher-dimensional homology (H2) and sparse-Rips epsilon=0.5. **Fastest available decision mode**: windowed W=25's marginal per-window cost, **0.0075s/decision** (net of its one-time 78.09s val-baseline setup, reported as a separate fixed cost per the matched-compute-frontier convention already used in `paper/WINDOWED_RESULTS.md`) — ~477× cheaper per decision than the per-flow arm, though a windowed decision covers W=25 flows at once, not a single flow, so this is not a like-for-like per-flow comparison (stated explicitly in the report). **Two disclosures**: (1) ARM-ISA-but-workstation-class — this run used the anaconda Python 3.9.7 build's own `platform.machine()`, which reported `x86_64`, NOT arm64; independent verification (`uname -m`, `sysctl -n machdep.cpu.brand_string`) confirmed the underlying silicon is a genuine Apple M1 Pro (arm64) and this Python binary is running under Apple's Rosetta 2 x86_64 translation layer — an emulation detail beyond the plan's original "workstation-class ARM-ISA" framing, now disclosed explicitly in the report (a native-arm64 Python build could measure differently). Regardless of process architecture, this remains a workstation (unified memory, no thermal/power constraints) and not a companion-class board, so all numbers are a lower bound on real onboard latency. (2) A real correctness finding: the per-manifold `outputs/scaler_{m}.pkl` files on this repo are STALE, fitted on an older/different feature schema (e.g. c2's file expects 5 columns named `SrcAddr_octet`/`SrcPort_binary`/... vs the current `config.C2_FEATURES`'s 7 columns) and written by neither `pipeline.py` nor any `uav_tda` module — the harness uses the combined `outputs/scalers.pkl` (written by `data.run_prep`, verified feature-count-correct) exclusively and asserts on any future mismatch. The harness is designed to run unmodified on a companion board (Jetson/RPi): `paper/LATENCY_RESULTS.md` §6 lists the exact small artifacts needed (`outputs/{m}_{train,val,test}.csv`, `reference_indices.npy`, `max_edge_lengths.json`, `labels_{test,val}.csv`, `scalers.pkl`, optionally the persisted exact-campaign baselines) and explicitly excludes the multi-hundred-MB persistence-diagram pickles — every diagram the harness times is computed fresh, on the fly, exactly as a deployed detector would. Full report (machine header, per-stage mean/median/p95 breakdown per mode, paper-claim check, matched-compute fastest-mode comparison, both disclosures, companion-hardware instructions): `paper/LATENCY_RESULTS.md`, generated by `uav-tda latency` from `results/tables/rebuild/latency_Olivers-MacBook-Pro.local.csv` and its five per-mode detail CSVs — **pending author sign-off**, no number here yet approved for the manuscript.
- **2026-09-23 (Phase-7 infrastructure program complete — final entry):** Closed the 7-phase infrastructure roadmap (Phases 1–7, PRs #11–#16 plus this task): the `uav_tda` package strangler-ported from `pipeline.py` with hermetic equivalence tests (Phase 2); z-normalized clean-lineage scoring (Phase 3); the ten-seed bootstrap-CI manuscript campaign (Phase 4); the time-windowed multi-manifold variant (Phase 5); the full-test-split high-precision Wasserstein-2 campaign and the portable onboard-latency harness (Phase 6); and, in Phase 7, ruff lint + a portable CI workflow + Makefile (Tasks 1–2), followed by this task (Task 3): quarantining `poster_eda/`, `scripts_archive/`, and `poster.jsx` under `archive/` (git-mv, history preserved; `archive/README.md` carries the data-leakage warning on `poster_eda/`), a `data/README.md` documenting the immutable UAVIDS-2025 CSV (sha256, no-timestamps/FlowID-order caveat), a rewritten top-level `README.md`, and an accuracy sweep of `CLAUDE/CLAUDE.md` (file discipline, full CLI subcommand list, ops lessons). The five results docs in `paper/` — `EXACT_RESULTS.md`, `LATENCY_RESULTS.md`, `WINDOWED_RESULTS.md`, `MANUSCRIPT_STATS.md`, `ZNORM_RESULTS.md` — constitute the manuscript-ready results package; **all five remain pending author sign-off**, no number in any of them is yet approved for the manuscript. Three outstanding manuscript corrections carried forward from Phase 6, still unresolved as of this entry: (1) the paper's "exact Wasserstein-2" claim must be corrected to high-precision (δ≤0.01, hera's own default tolerance) — literal δ=0 exact mode was measured intractable (13/13 sampled flows timed out at the 120s fork bound across all 3 manifolds, ~30-day campaign projection); (2) the published "1–3 seconds" per-flow latency estimate must be corrected against the measured per-flow decision cost, mean **3.5762s** (median 3.5855s, p95 3.8591s), c2's Rips construction being ~97% of that cost; (3) all latency figures are Rosetta-emulated-x86_64-on-Apple-M1-Pro workstation numbers (unified memory, no thermal/power constraints), not companion-class (Jetson/RPi) hardware — a native-arm64 or companion-board run remains a future-work item, not yet executed. Infrastructure work on this repository is done; remaining work is exclusively manuscript-side (author review of the five pending reports, resolving the three corrections above, and writing).
- **2026-10-18 (manuscript-readiness audit; CLAUDE/ cleanup):** Regenerated this brief's body (sections 0–9) to the post-Phase-7 state; the decision log above is carried verbatim. Moved the agent-instructions file from `CLAUDE/CLAUDE.md` to the repo root `CLAUDE.md` (a subdirectory `CLAUDE.md` is not auto-loaded by Claude Code; verified this session). Deleted `CLAUDE/RESEARCH_CONTEXT.md` (May-2026 handoff for the numbered-scripts era: 5D/15D/2D overlapping manifolds, k-NN TDA, 1,000-flow baseline, 3σ threshold — every methodology, result, and repo-structure statement in it was superseded); its only unique content (dataset quirks → `data/README.md`; the topology-simplification hypothesis, the un-run threshold-sweep and H0/H1-count ideas, and the "57.1%"/"zero-day" cautions → §3 and §7 above) was migrated first. Recovered four never-committed May-2026 drafting docs (`HANDOFF_NOTES.md`, `OUTLINE.md`, `METHODS_DISCLOSURES.md`, `FIGURE_AUDIT.md`) from the stale worktree `.claude/worktrees/affectionate-dewdney-ef7609/paper/` into `paper/historical/2026-05-drafting/` (read-only; same data-loss path that lost `tools/` once). Added `CLAUDE/STATUS.md` (current status of repo, results package, manuscript, open decisions) and `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md` (claim-by-claim audit of `main.tex` against the results package, figure/snippet gap list, prioritized manuscript edits). New finding logged as open correction #4 in §7: the windowed compute-frontier's per-flow reference cost (0.0602 s, "~6.6× cheaper at W=25", `paper/WINDOWED_RESULTS.md` §3) is a W2-only marginal — `windowed._time_per_flow_probe` times `probe.run_probe`, which loads precomputed diagrams and never builds a Rips complex — whereas the windowed marginal includes Rips construction; against the latency harness's full-path per-flow 3.5762 s the ratio is roughly 400×, so the frontier comparison must be recomputed on a like-for-like basis or reworded before use. Note for readers of the 2026-09-22 Phase-6 entry: the cited `.superpowers/sdd/2026-09-22-plan6-exact-w2-latency/task-3-report.md` is gitignored and no longer present on disk; the same investigation summary is in `paper/EXACT_RESULTS.md` §1. Repo state at this entry: PR #17 merged (2026-09-23), `main` == `origin/main` @ 4390e64, fast suite 103 passed / 0 skipped, ruff clean, CI green; `requirements.txt` identified as hazardous for the pinned env (pins numpy≥1.24 / pandas≥2.0) — root `CLAUDE.md` now says to install via `pip install -e ".[dev]"` only. Nothing in this entry changes any scientific number or framing; all five results docs remain pending author sign-off.
- **2026-11-09 (Phase-8 manuscript artifacts):** Added `uav-tda artifacts` (`uav_tda/artifacts.py`, `uav_tda/figures.py`; 16 subcommands now), building tables in `results/tables/rebuild/artifacts/`, three new snippets, six vector figures in `results/figures/manuscript/`, and `paper/MANUSCRIPT_ARTIFACTS.md`, purely from committed CSVs. **Frontier ruling:** per-flow AUC stays the 10-seed probe value 0.8930 because the latency harness timed that same probe configuration (top-K=50, delta=0.2, sparse eps=0.5); cost for every row is the latency harness full-path `total_s` (single instrument, n=30); the campaign marginal and the W2-only 0.0602 s are kept as labelled columns, never as the cost; error bars are bootstrap CIs; the exact full-test AUC (0.9082) is a horizontal reference line, not a frontier point. Result: W=25 is 478x cheaper per decision than per-flow (supersedes 6.6x). **Provenance fixes:** `uav_tda/provenance.py` gained `run_git` (Rosetta-safe; plain `git` failed so sidecars recorded "unknown") and an opt-in `skip_if_unchanged` content-hash mode; missing sidecars backfilled with `backfilled: true` and `introduced_in`. **Closed:** frontier (brief section 7 item 4, audit I-1), unique-vs-pooled window counts (I-3), FPR claim measured (FPR 0.0555 / TPR 0.5703, all-three znorm), eps-max values persisted, exact-campaign snippets, the five gap figures plus frontier, April PNGs quarantined to `results/figures/legacy_2026-04/`. **Still open (author):** sign-off of all six results docs, headline estimand, `main.tex` rewrite (incl. the 478x text), companion-board latency, cross-benchmark, dataset redistribution, `node_modules`, stash, `outputs/` deletion, `requirements.txt`. Caption caveats: Wormhole W=50 effective tie; per-flow attribution is a single full-test estimate vs 10-run windowed means; thinnest contamination bin n=1. Fast suite 145 passed / 0 skipped. All numbers pending author sign-off.
- **2026-11-11 (author decisions):** Per-flow headline estimand is the full-test hera δ≤0.01 z-norm all-three AUC **0.9082** [0.9033, 0.9131] (`paper/EXACT_RESULTS.md`); the 10-seed probe 0.8930 ± 0.0078 is used only where it is paired with the latency harness's cost (the frontier), and the two are always labelled as different estimands. Cross-benchmark evaluation (§V promise P7) becomes future work. A companion-board latency run will be done before December (hardware being chosen). Housekeeping: April 2026 stash dropped; `archive/poster_eda/node_modules` untracked (remains in history); `outputs/` kept until submission; `requirements.txt` replaced with exact pins of the working env (gudhi 3.11.0 has no py3.9 linux/aarch64 PyPI wheel, so ARM boards take gudhi from conda-forge); UAVIDS-2025 stays in-repo under its published license (database ODbL, contents © original authors) with attribution in `data/README.md`. No scientific number changes; all results docs remain pending author sign-off.
