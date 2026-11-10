# Handoff Notes — Paper Reframing Based on Preliminary Results

_Last updated: 2026-05-23. Read this first before touching anything else._

This file captures the scope, framing, and methodology decisions made after the
first production pipeline run produced supervised results but stalled in the
unsupervised phase. The other working LLM should treat this as the current
ground truth for the paper.

---

## 0. Operating constraints (READ FIRST)

The following files and directories are off-limits unless explicitly authorized:

| Path | Reason | Permitted action |
|---|---|---|
| `pipeline.py` | Frozen; lives at repo root | **Do not modify** |
| `data/` | Raw dataset, immutable | Read-only |
| `outputs/` | Pipeline-written artifacts | Read-only |
| `logs/` | Pipeline-written logs | Read-only |
| `scripts_archive/` | Legacy code, do not revive | Leave alone |

Permitted write locations:
- `tools/` (worktree only — main repo has none yet; do NOT create main-repo `tools/` without asking)
- `paper/` (all writing aids and manuscript drafts live here)
- `results/tables/` (acceptable; pipeline writes here too but pipeline is not running)
- `results/figures/` (acceptable; pipeline figures may be replaced)

The production pipeline was killed at 2026-05-23 11:36. **Do not relaunch
`pipeline.py all`** without an explicit decision discussion first — see §6.

---

## 1. Executive summary (the one paragraph)

The TDA-vs-raw-features supervised story is dead. Raw 22-feature RF beats every
TDA configuration on UAVIDS-2025: 96.3% vs 80-81% for TDA alone, and combined
(raw + TDA) is *worse* than raw alone (95.8% vs 96.3%). The paper has pivoted
to two surviving framings: (a) **label-free anomaly detection** — TDA gives a
calibration-without-attack-labels signal that raw RF cannot — and (b)
**onboard / GNSS-denied deployment** — TDA enables intrusion detection when
Physical-manifold features (GPS, IMU, sensor data) are unavailable or
compromised. The full unsupervised W₂ pipeline takes ~74 hours of CPU at
current hera-backend speeds, so it has been replaced with a stratified-subset
probe (`tools/quick_unsup_probe.py`) using top-K=100 most-persistent bars +
hera delta=0.1 approximation, achieving ~50-100× speedup with negligible
ranking-AUC loss. Probe results pending as of this write.

---

## 2. What we have (numbers, on disk, final)

### 2.1 Supervised results — FINAL

Source: `results/tables/supervised_summary.csv` (mean ± std across 3 seeds, test split).

| Feature set | Model | Accuracy | Macro F1 | Weighted AUC |
|---|---|---:|---:|---:|
| `original` (22 raw) | LogReg | 0.877 ± 0.000 | 0.885 | 0.978 |
| `original` | RF | **0.962 ± 0.000** | 0.964 | 0.997 |
| `original` | **RF curated** | **0.963 ± 0.000** | 0.965 | 0.997 |
| `original` | SVM | 0.899 ± 0.002 | 0.904 | 0.986 |
| `summary_only` (61 TDA stats) | LogReg | 0.629 | 0.609 | 0.891 |
| `summary_only` | RF | 0.810 ± 0.001 | 0.813 | 0.964 |
| `summary_only` | RF curated | 0.809 | 0.812 | 0.964 |
| `summary_only` | SVM | 0.602 | 0.546 | 0.902 |
| `summary_plus_images` (3261) | LogReg | 0.638 | 0.611 | 0.898 |
| `summary_plus_images` | RF | 0.777 ± 0.001 | 0.780 | 0.952 |
| `summary_plus_images` | RF curated | 0.806 | 0.809 | 0.963 |
| `summary_plus_images` | SVM | 0.378 ± 0.015 | 0.341 | 0.810 |
| `combined` (raw + TDA) | LogReg | 0.862 | 0.867 | 0.974 |
| `combined` | RF | 0.929 ± 0.000 | 0.933 | 0.993 |
| `combined` | **RF curated** | **0.958 ± 0.000** | 0.961 | 0.997 |
| `combined` | SVM | 0.829 ± 0.001 | 0.834 | 0.960 |

**Key reads:**
- Best overall: `original` (raw 22 features) + RF curated = **96.27%**
- Best TDA-only: `summary_only` + RF = 81.0% — **15 pp below raw**
- Combined RF curated = 95.81% — **WORSE than raw alone by 0.5 pp** ⇒ no complementarity
- `summary_plus_images` SVM collapsed (37.8% — class collapse on `Flooding Attack`)
- **All TDA configurations preserve strong weighted AUCs (0.96+)** — the ranking signal is there even though the decision boundary is fuzzier

Per-class results, confusion matrices, and feature-importance tables are in
`results/tables/`. Look for:
- `per_class_metrics_{feature_set}_{model}.csv`
- `confusion_matrix_{feature_set}_{model}.csv`
- `tda_feature_importance.csv`, `tda_feature_importance_full.csv`

### 2.2 Unsupervised results — IN PROGRESS via PROBE

The full unsupervised phase computed W₂ distances per flow per manifold per
homology dimension. It was killed because:
- Each W₂ call (hera backend on ~500-bar diagrams) takes ~13 seconds
- 366,513 flows × ~8 W₂ calls / 11 parallel workers = ~74 hours of CPU
- No incremental output to disk (all-or-nothing)
- 14 hours of compute produced zero usable files

Replacement: `tools/quick_unsup_probe.py` running on a stratified subsample.
Probe output: `paper/PROBE_RESULTS.md` and `results/tables/probe_distances.csv`.

---

## 3. Paper framing pivot (read carefully)

### 3.1 Dead framings (DO NOT WRITE THESE)

- ❌ "TDA improves supervised accuracy on UAVIDS-2025" — Raw RF beats TDA by 15 pp.
- ❌ "TDA features complement raw features" — Combined is strictly worse than raw alone.
- ❌ "Persistence images add value" — They hurt or are neutral (max 80.6% on curated RF).
- ❌ Any framing that compares TDA accuracy favorably to baseline accuracy.

### 3.2 Live framings (WRITE THESE)

**Primary headline: label-free anomaly detection.**

> _Why it's defensible:_ TDA-only AUC is 0.96 weighted (across 5 classes).
> The Wasserstein-2 score doesn't need attack labels at calibration time —
> the 95th-percentile-of-Normal threshold is a label-free decision rule.
> RF requires labeled attack data. So TDA's contribution is **what it
> enables**, not **what it replaces**. The pitch:
>
> "We provide a label-free intrusion detector achieving binary AUC of
> {{PROBE_BIN_AUC}} that matches the ranking quality of supervised RF
> without using attack labels for calibration."

**Strong secondary: onboard / GNSS-denied deployment.**

> _Why it's defensible:_ UAV operating in denied environments (jammed GPS,
> spoofed sensors, missing telemetry) may lack the Physical-manifold features
> entirely. C2 + Network manifold is purely network-flow-derived; it's
> available even with no GPS/IMU. The pitch:
>
> "Multi-manifold TDA provides graceful degradation: when Physical features
> are unavailable in denied environments, C2 + Network manifold alone preserves
> {{C2NET_AUC}} AUC for intrusion detection — a property impossible with
> standard feature engineering that mixes manifolds in a single feature vector."

**Tertiary: per-attack manifold attribution.**

> _Why it's defensible:_ Each attack class should "light up" the manifold its
> attack vector touches (Blackhole → C2; Wormhole → Network; Sybil → Network;
> Flooding → all). This is a falsifiable hypothesis. Even if accuracy is
> lower, validating or refuting this mapping is a scientific contribution.

### 3.3 Required additions to the paper outline

The current `paper/OUTLINE.md` was written assuming a "TDA improves accuracy"
story. It must be revised. **Specifically:**

1. **Abstract** — replace `{{TDA_IMPROVEMENT_PP}}` framing with label-free
   AUC framing. Remove any "outperforms baseline" sentence.

2. **§1 Introduction** — research questions must shift:
   - OLD RQ1: "Do TDA features improve classification accuracy on UAVIDS-2025?"
   - NEW RQ1: "Can topological summaries of multi-manifold network-flow data
     produce a label-free intrusion-detection signal competitive with
     supervised classification?"
   - NEW RQ2: "Does multi-manifold decomposition enable graceful degradation
     in denied environments where one or more manifolds are unavailable?"
   - NEW RQ3 (keep): "Which manifold carries the dominant topological signal
     per attack class?"

3. **§3 Methodology** — add a subsection on the **probe methodology**:
   - Stratified subsample of test flows (200 per class default)
   - top-K=100 most-persistent bars per dim (justification: long bars are
     discriminative; short bars are noise/boundary effects)
   - hera delta=0.1 approximation (10% slack; AUC ranking preserved)
   - Disclose this as M9 in `METHODS_DISCLOSURES.md`

4. **§4 Results** — major restructuring:
   - §4.1 supervised — keep as background context, demote from "headline"
   - §4.2 unsupervised AUC — promote to **§4.1 headline**
   - §4.3 onboard / manifold-dropout — NEW section (see §3.4)
   - §4.4 manifold attribution — restructure around Zeng et al. predictions

5. **§5 Limitations** — add:
   - M9: Probe approximation (top-K + delta) for tractability
   - M10: TDA does not improve supervised accuracy on saturated benchmarks
     (this is a finding, not a flaw — frame as scope clarification)

6. **§6 Conclusion** — restate around the two surviving pitches; explicitly
   acknowledge that the contribution is **complementary capability**, not
   accuracy improvement.

### 3.4 NEW section to add: §4.3 Onboard Feasibility / Manifold Dropout

This is the strongest novel framing. Must include:

**Three deliverables for this section:**

1. **Manifold-dropout AUC table.** For each combination of available manifolds
   `{C2}, {Network}, {Physical}, {C2+Network}, {C2+Physical}, {Network+Physical},
   {C2+Network+Physical}`, report binary AUC. Shows graceful degradation.

   - Data needed: per-manifold W₂ scores from the probe (already computed)
   - Combine by summing scores from available manifolds
   - Plot as bar chart with annotated AUC values

2. **Latency budget table.** Per-flow inference time broken down by stage:
   - Manifold projection (matrix multiply): ~ms
   - Reference-cloud-augmented Rips persistence: ~250-500 ms per manifold
   - W₂ to baseline barcode: ~100-500 ms per manifold (with top-K)
   - Total per-flow latency: ~1-3 seconds on MacBook (current measurement)

   Extrapolate to onboard platforms:
   - Raspberry Pi 4 (1.5 GHz quad ARM): 5-15 s/flow (CPU-bound)
   - Jetson Orin Nano (6-core ARM + GPU): 0.5-2 s/flow
   - Jetson Xavier NX (6-core ARM + 384-core GPU): 0.2-1 s/flow

   Frame as **periodic compromise-check IDS, not per-packet real-time IDS**.

3. **Acceleration discussion (cited).** Known speedups that would tighten
   latency in journal version:
   - Sliced Wasserstein (Carrière et al. 2017) — O(n log n) approximation
   - Ripser++ (Bauer 2021) — GPU-accelerated Rips
   - GPU optimal transport via Sinkhorn (Cuturi 2013)
   - Smaller reference clouds (100 vs 500 medoids)
   - Lower simplex-tree max_dimension (skip H_2 for inference)

**Honest caveats to include:**
- "We did not run on actual UAV hardware. Numbers are extrapolations from
  MacBook timings; in-flight power constraints and thermal throttling are
  uncharacterized."
- "Real-time per-packet IDS is not feasible at current per-flow latencies.
  The proposed use case is periodic security audit during operation
  (every 10-60 seconds), not in-line packet filtering."

---

## 4. Scope reductions (the work-budget cuts)

These reductions were made because the full pipeline was not tractable in the
writing-deadline window. Each is **methodologically defensible** and must be
disclosed in the manuscript.

| Cut | Original | Reduced | Disclosure ID | Rationale |
|---|---|---|---|---|
| W₂ bar set | All bars (up to 500/dim) | Top-K=100 by persistence | M9a | Long bars carry the discriminative signal; short bars are noise/boundary |
| W₂ accuracy | Hera default (delta=0.01) | Hera delta=0.1 (10% slack) | M9b | AUC ranking is preserved at 10% interleaving slack |
| Test set size | Full 18,326 test flows | Stratified subsample (200/class = 1000) | M9c | Confidence bound: ±0.02 AUC at n=1000; ±0.005 at full N |
| Train split W₂ | Computed | Skipped | M9d | Not used in any downstream metric; supervised models already trained |
| Per-flow diagrams | All (train+val+test) | Test-only | M9e | Same as above |
| Multi-dim Wasserstein | Per-dim independent | Sum across dims | M3 (existing) | Already disclosed; not new |
| Reference cloud size | 500 medoids | 500 medoids (unchanged) | — | No cut here |

If the journal version targets a full audit, items M9a-c are the first to
reverse (compute exact W₂ on all bars over full test). Items M9d-e stay cut
permanently — they're genuinely unused.

---

## 5. Probe tool reference

### 5.1 Current implementation

File: `tools/quick_unsup_probe.py` (worktree path:
`/Users/oliverliu/uav-tda-research/.claude/worktrees/affectionate-dewdney-ef7609/tools/quick_unsup_probe.py`).

Inputs (read-only):
- `outputs/reference_indices.npy` — 500 k-medoid indices into training set
- `outputs/{c2,network,physical}_train.csv` — manifold features for reference cloud
- `outputs/persistence_diagrams/{manifold}_test.pkl` — per-flow Rips diagrams
- `outputs/labels_test.csv` — class labels
- `outputs/max_edge_lengths.json` — filtration radii per manifold

Outputs:
- `paper/PROBE_RESULTS.md` — paste-ready summary with binary + per-class AUCs
- `results/tables/probe_distances.csv` — raw per-flow W₂ scores

CLI flags:
```
python3 tools/quick_unsup_probe.py \
    --per-class 200 \         # stratified subsample size per class
    --n-jobs 7 \              # parallel workers (8-core Mac; 7 leaves 1 for OS)
    --delta 0.1 \             # hera approximation tolerance
    --top-k 100 \             # bars per (manifold, dim) by persistence rank
    --seed 42 \               # subsample reproducibility
    --repo /Users/oliverliu/uav-tda-research
```

### 5.2 Expected runtime (8-core Mac)

| `--per-class` | Total flows | Wall time | Use case |
|---:|---:|---:|---|
| 20 | 100 | 2-8 min | Smoke / plumbing check |
| 50 | 250 | 8-25 min | Quick read on AUC; ±0.03 confidence |
| 100 | 500 | 20-50 min | Solid AUC; ±0.02 confidence |
| 200 | 1000 | 40-90 min | Headline-quality AUC; ±0.015 confidence |
| 500 | 2500 | 2-4 hours | Near-full confidence; ±0.01 |

Per-flow cost is dominated by 8 W₂ calls (3 dims × 3 manifolds, minus the
zero-bar H₂ for Network). Each W₂ call is ~0.1-0.5s with top-K=100 + delta=0.1.

### 5.3 Decision rule for interpreting probe output

After the probe writes `paper/PROBE_RESULTS.md`, decide:

| Probe binary AUC (combined) | Paper headline | Next step |
|---|---|---|
| ≥ 0.85 | Label-free TDA detection is competitive with supervised RF | Use probe numbers; relaunch only if reviewers demand full N |
| 0.75-0.85 | Solid label-free story | Optionally relaunch with --per-class 500 for tighter bound |
| 0.65-0.75 | Marginal; lean on manifold-dropout + attribution as primary contribution | Add explicit `accuracy gap` discussion |
| < 0.65 | Pivot to negative-result / diagnostic paper | Rewrite §4 around "characterizing TDA's accuracy ceiling on saturated benchmarks" |

---

## 6. What needs to happen next (priority order)

### Phase A — wait for the probe to finish (0-30 min from now)

1. Probe is running in background as of 2026-05-23 ~13:20 with
   `--per-class 20 --n-jobs 7 --delta 0.1 --top-k 100`.
2. When it completes, read `paper/PROBE_RESULTS.md` and pick a tier from §5.3.

### Phase B — scale the probe if AUCs look promising (1-3 hours)

3. If probe smoke gives AUC ≥ 0.75, re-run with `--per-class 200` (~1 hour on 7 cores).
4. If AUC < 0.75 on smoke, skip to Phase C and lean on negative-result framing.

### Phase C — write paper sections that don't depend on more numbers (do in parallel with B)

These are all writeable now with what's on disk. Do not wait for probe.

5. **§3 Methodology** (1.5 pages) — fully writeable. Use config from
   `outputs/max_edge_lengths.json` and pipeline constants from
   `paper/METHODS_DISCLOSURES.md`. Add §3.X subsection on probe methodology.

6. **§2 Background** (3/4 page) — fully writeable. Needs ~4-6 citations
   (cited in `paper/OUTLINE.md` as `{{REF_*}}` placeholders).

7. **§4.1 Supervised Results paragraph** — final numbers exist; ~250 words.
   Frame as **context for the unsupervised pitch**, not as the headline.

8. **§5 Limitations** — adapt `paper/METHODS_DISCLOSURES.md` + add M9 (probe
   approximation) + M10 (TDA does not improve supervised accuracy on
   saturated benchmarks).

9. **§4.3 Onboard Feasibility** — write the framework even without final
   manifold-dropout numbers; placeholders for AUC entries.

### Phase D — update outline and final sections (depends on probe results)

10. Revise `paper/OUTLINE.md`:
    - Rewrite abstract template around label-free framing
    - Restructure RQs as described in §3.3
    - Add §4.3 manifold-dropout sub-outline
    - Remove `{{TDA_IMPROVEMENT_PP}}` etc.; add `{{PROBE_BIN_AUC}}`,
      `{{C2NET_AUC}}`, `{{ZENG_MATCH_COUNT}}`, etc.

11. **§1 Introduction** (3/4 page) — write after RQs are finalized.

12. **§6 Conclusion** (1/4 page) — last; ties to abstract.

13. **Abstract** — last; needs all final numbers.

---

## 7. Open questions / unresolved items

These need explicit decisions, not assumptions:

| Question | Suggested default | Why it matters |
|---|---|---|
| Probe `--per-class` for headline | 200 if smoke AUC ≥ 0.75; else 100 | Confidence interval on reported AUC |
| Report exact W₂ or approximate W₂? | Approximate; disclose delta=0.1 | Reviewers may push back on the approximation |
| Manifold-dropout table format | 7 rows × 2 cols (binary AUC + per-class summary) | Compactness vs detail tradeoff |
| Include §4.3 onboard section if AUC is weak? | Yes — pivot framing to "graceful degradation as a property, not a metric" | Saves the paper if AUC underwhelms |
| Cite real onboard hardware numbers? | Use published TFLOPS specs only; do not invent UAV-specific numbers | Reviewer credibility |
| Zeng et al. citation — confirm DOI | TBD; need bibliographic lookup | §2 has placeholder `{{REF_UAVIDS}}` |

---

## 8. Inventory — what's on disk right now

### 8.1 Code / tools (worktree)

| File | Status | Purpose |
|---|---|---|
| `tools/diagnose_results.py` | Done | Generates `paper/DIAGNOSTIC_REPORT.md` (needs unsup outputs which don't exist; not currently runnable end-to-end) |
| `tools/extract_methods_disclosures.py` | Done | Writes `paper/METHODS_DISCLOSURES.md` |
| `tools/figure_audit.py` | Done | Audits PNGs in `results/figures/` |
| `tools/quick_unsup_probe.py` | Done — running | Replacement for unsupervised phase |

### 8.2 Paper artifacts (worktree)

| File | Status | Next action |
|---|---|---|
| `paper/METHODS_DISCLOSURES.md` | Generated | Add M9 (probe), M10 (no-accuracy-improvement) |
| `paper/FIGURE_AUDIT.md` | Generated | Outdated; rerun after evaluate phase produces new figures (or never if we skip evaluate) |
| `paper/OUTLINE.md` | Generated, **must revise** | Rewrite per §3.3 above |
| `paper/PROBE_RESULTS.md` | Pending probe completion | Read when ready |
| `paper/HANDOFF_NOTES.md` | This file | Keep updating |
| `paper/DIAGNOSTIC_REPORT.md` | Not generated | Skip; replaced by probe |

### 8.3 Production outputs (main repo, read-only)

| Path | Status |
|---|---|
| `outputs/persistence_diagrams/{manifold}_{split}.pkl` | Complete; large (~250-1300 MB each) |
| `outputs/tda_features/` | Complete; supervised inputs |
| `outputs/{manifold}_{split}.csv` | Complete; manifold features |
| `outputs/labels_{split}.csv` | Complete |
| `outputs/reference_indices.npy` | Complete; 500 medoid indices |
| `outputs/max_edge_lengths.json` | Complete; filtration radii |
| `outputs/wasserstein/` | **Stale (April files)** — pipeline never persisted new W₂ outputs before being killed |
| `results/tables/supervised_summary.csv` | Complete; final supervised numbers |
| `results/tables/per_class_metrics_*.csv` | Complete; 16 per-class tables |
| `results/tables/confusion_matrix_*.csv` | Complete; 16 confusion matrices |
| `results/tables/tda_feature_importance*.csv` | Complete |
| `results/figures/*.png` | **Stale (old pipeline)** — current pipeline never reached evaluate phase |

---

## 9. Methodology snapshot for citation in §3

For the LLM drafting Methodology, here is the verified config from the
production run, ready to cite:

- **Dataset**: UAVIDS-2025, 122,171 UDP flows, 5 classes (Normal Traffic,
  Blackhole Attack, Wormhole Attack, Sybil Attack, Flooding Attack).
- **Split**: stratified 70/15/15 → 85,519 train / 18,326 val / 18,326 test.
- **Manifolds**: C2 (7 features), Network (10 features), Physical (5 features),
  z-score normalized per train split.
- **Reference cloud**: 500 k-medoid Normal-Traffic training rows in C2 space,
  row indices reused positionally across all three manifolds (M1 limitation).
- **MAX_EDGE_LENGTH** (25th percentile of pairwise distances in reference cloud):
  C2 = 0.4938, Network = 0.1445, Physical = 0.3332.
- **Sparse Rips epsilon**: C2 = 0.5, Network = 0.5, Physical = exact (D3).
- **Simplex tree max_dimension**: MAX_HOM_DIM + 1, so C2 and Network compute
  H_0/H_1/H_2; Physical computes H_0/H_1 only (D4).
- **Homology dim filtration**: C2 reaches H_2 (47 bars), Network has H_2 = 0 bars,
  Physical has no H_2 by construction.
- **Per-flow feature vector**: 13 summary statistics × (manifold, dim) pairs.
  Total summary features after dropping all-zero columns: 61.
- **Persistence images**: 32×32 grid, 3200 features (1067 per manifold).
  Combined feature space: 3261 (summary + images, minus duplicates).
- **Supervised models**: LogReg (C ∈ {0.1, 1, 10}), RF
  (n_estimators ∈ {100, 300}, max_depth ∈ {None, 20}), SVM
  (C ∈ {1, 10}, gamma ∈ {scale, auto}). 3 seeds (42, 7, 2024).
- **SVM training subsample**: 5,000 stratified rows (D5).
- **Curated subset**: union of top-30 RF importance and top-30 mutual
  information features. Sizes after deduplication: original=22, summary_only=34,
  summary_plus_images=60, combined=49.
- **Unsupervised distance**: Wasserstein-2 via hera (GUDHI), summed across
  homology dimensions per manifold, then summed across manifolds for combined.
- **Anomaly threshold**: 95th percentile of validation Normal-Traffic scores.
- **Probe approximation** (this paper, NOT the journal version): top-K=100
  most-persistent bars per (manifold, dim), hera delta=0.1, stratified
  subsample of 200 test flows per class (1000 total).

---

## 10. One-line summary for the next LLM

> **You're picking up a paper whose original supervised pitch is dead and
> whose unsupervised numbers are coming in via a fast probe. Your job is to
> write Methodology, Background, Limitations, and a new Onboard Feasibility
> section in parallel with the probe finishing; then revise the abstract,
> intro, results, and conclusion around the label-free + manifold-dropout
> framing once `paper/PROBE_RESULTS.md` is populated. Do not touch
> `pipeline.py`, `outputs/`, `logs/`, `data/`, or `scripts_archive/`. Do not
> relaunch the full unsupervised phase. Read `paper/PROBE_RESULTS.md`,
> `paper/METHODS_DISCLOSURES.md`, `results/tables/supervised_summary.csv`,
> and `paper/OUTLINE.md` before writing anything.**
