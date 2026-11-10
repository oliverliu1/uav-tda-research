# Paper Outline — Extended Abstract
<!-- AIAA SciTech 2027 Intelligent Systems TC — 5-page extended abstract -->
<!-- Fill every {{...}} token from paper/DIAGNOSTIC_REPORT.md once the production run completes. -->
<!-- This file is a scaffold only. Do not paste section bullets verbatim into the manuscript. -->

---

## Title (placeholder)

**{{PAPER_TITLE}}**
_Working title: "Multi-Manifold Persistent Homology for UAV Network Intrusion Detection"_

---

## Abstract (~200 words)

> Replace `{{...}}` tokens with values from `paper/DIAGNOSTIC_REPORT.md`.

We investigate whether persistent homology features extracted from three disjoint
network-flow manifolds — C2 (command-and-control), Network, and Physical —
improve detection of UAV intrusion attacks in the UAVIDS-2025 dataset of
{{N_SAMPLES}} labeled UDP flows across {{N_CLASSES}} traffic classes. We define
per-flow point clouds by augmenting a shared reference scaffold of 500 k-medoid
Normal-Traffic flows with each query flow, compute Vietoris-Rips persistence
diagrams through homology dimension H_{{MAX_HOM_DIM}}, and derive {{N_TDA_FEATURES}}
summary statistics per flow. Supervised classifiers trained on TDA features alone
achieve {{HEADLINE_ACC}}% accuracy (best model: {{BEST_MODEL}}) versus
{{BASELINE_ACC}}% for the raw-feature baseline, a gain of {{TDA_IMPROVEMENT_PP}}
percentage points. The Random Forest on the curated TDA subset reaches
{{CURATED_RF_ACC}}% accuracy and {{CURATED_RF_F1}} macro-F1. For unsupervised
anomaly detection, Wasserstein-2 distances to a Normal-Traffic reference barcode
yield a binary AUC of {{UNSUP_BIN_AUC}} (threshold derived from the 95th percentile
of validation Normal-Traffic distances). Manifold-ablation results show
{{DOMINANT_MANIFOLD}} features are most discriminative, consistent with the topology
of UAV network anomalies. These results demonstrate that topological data analysis
provides a viable, model-agnostic signal for airborne network intrusion detection.

_Target: 200 words. Actual at fill-in time: {{ABSTRACT_WORD_COUNT}}._

---

## Section 1 — Introduction

**Target: 3/4 page (~375 words prose)**
**AIAA SciTech 2027 coverage: (a) research questions, (b) technical challenges, (d) specific contributions**

### Bullets

1. **Opening hook / problem statement** — UAV fleet security: control-link hijacking and
   intra-swarm protocol attacks (Blackhole, Wormhole, Sybil, Flooding) threaten civilian
   and defense UAV operations; cite recent incidents and standards gap; motivate the need
   for topology-aware detection that does not depend on attack signatures.

2. **Research questions (RQ)**
   - RQ1: Do persistent homology features extracted from disjoint network-flow manifolds
     separate attack classes from Normal Traffic in the UAVIDS-2025 dataset?
   - RQ2: Which manifold (C2, Network, Physical) carries the most discriminative
     topological signal for each attack type?
   - RQ3: Do Wasserstein-2 distances to a reference barcode provide a viable unsupervised
     anomaly score without labeled attack data at threshold-calibration time?

3. **Technical challenges** — high-dimensional flow data with multi-modal class structure;
   memory and compute cost of exact Vietoris-Rips at scale (19M+ simplices at 95th-percentile
   radius); manifold non-independence; sparse and noisy higher-order homology (H_2);
   aggregating multi-dimensional Wasserstein into a single anomaly score.

4. **Specific contributions**
   - First application of multi-manifold Rips persistence to UAVIDS-2025 with explicit
     C2/Network/Physical decomposition.
   - Curated TDA feature subset of {{N_CURATED_FEATURES}} features achieving
     {{CURATED_RF_ACC}}% accuracy — competitive with raw-feature baselines.
   - Unsupervised detection AUC of {{UNSUP_BIN_AUC}} requiring no labeled attacks at
     threshold-calibration time.
   - Reproducible open-source pipeline; all results regenerable from a single command.

5. **Paper roadmap** — one sentence each pointing to §2 (background and related work),
   §3 (three-manifold TDA methodology), §4 (supervised and unsupervised results),
   §5 (limitations and full-paper plan), §6 (conclusion).

### Placeholders needed
`{{N_CURATED_FEATURES}}`, `{{CURATED_RF_ACC}}`, `{{UNSUP_BIN_AUC}}`

---

## Section 2 — Background and Related Work

**Target: 3/4 page (~375 words prose)**
**AIAA SciTech 2027 coverage: (b) technical challenges, (c) up-to-date references**

### Bullets

1. **Persistent homology primer** — Vietoris-Rips filtration over a finite metric space;
   persistence diagrams encoding birth/death of connected components (H_0), loops (H_1),
   and voids (H_2); Wasserstein and bottleneck stability; summary statistics as finite
   feature vectors. Key citations: Edelsbrunner and Harer (2010), Carlsson (2009),
   GUDHI library (Maria et al.); note sparse Rips approximation (Gudhi implementation,
   Boissonnat and Pritam 2020).

2. **TDA in network intrusion detection** — prior applications of persistence to network
   flow graphs and packet-timing distributions (cite {{REF_TDA_IDS_1}}, {{REF_TDA_IDS_2}});
   contrast with our per-flow point-cloud formulation and disjoint manifold decomposition
   which avoids graph-construction overhead.

3. **UAV intrusion detection landscape** — UAVIDS-2025 dataset and benchmark (Zeng et al.
   {{REF_UAVIDS}}); RF/SVM baselines on raw features; known class-imbalance challenges
   with Sybil and Flooding; prior topological or geometric intrusion studies on UAV
   networks (cite {{REF_UAV_SECURITY_SURVEY}}).

4. **Manifold decomposition motivation** — C2, Network, and Physical features capture
   qualitatively different geometric structure (control-link timing vs. routing topology
   vs. sensor-layer anomalies); treating them as separate Rips spaces avoids feature-scale
   interference; cite manifold hypothesis (Fefferman et al. 2016) and per-manifold
   TDA precedents in sensor fusion.

5. **Gap statement** — no prior work applies multi-manifold Rips persistence to UAV flow
   data; cross-manifold Wasserstein scoring for unsupervised detection without labeled
   attacks is unexplored; this paper closes that gap on the UAVIDS-2025 benchmark.

### Placeholders needed
`{{REF_TDA_IDS_1}}`, `{{REF_TDA_IDS_2}}`, `{{REF_UAVIDS}}`, `{{REF_UAV_SECURITY_SURVEY}}`

_Note: fill references from literature search before submission; use AIAA numbered citation style._

---

## Section 3 — Methodology

**Target: 1.5 pages (~600 words prose + figures/tables)**
**AIAA SciTech 2027 coverage: (b) technical challenges, (d) specific contributions**

### Sub-outline: figures and tables

| # | Item | Caption stub | Placement |
|---|------|-------------|-----------|
| Fig. 1 | `methodology_flowchart.png` | End-to-end pipeline: raw flows → manifold split → Rips persistence → supervised and unsupervised scoring | §3 opening, full column width |
| Fig. 2 | `persistence_barcode_examples.png` | Representative barcodes for Normal vs. Blackhole on each manifold (H_0, H_1, H_2 side-by-side) | §3.3, after Rips description |
| Table 1 | Feature-set composition | {{N_C2_FEATURES}} C2 / {{N_NETWORK_FEATURES}} Network / {{N_PHYSICAL_FEATURES}} Physical / {{N_COMBINED_FEATURES}} Combined feature counts and names | §3.2, inline |

### Bullets

1. **Dataset** — UAVIDS-2025: {{N_SAMPLES}} UDP flows, 5 classes (Normal, Blackhole,
   Wormhole, Sybil, Flooding); {{N_TRAIN}} / {{N_VAL}} / {{N_TEST}} stratified 70/15/15
   split; {{N_RAW_FEATURES}} raw columns → 22 numeric features after dropping non-numeric;
   class distribution: {{CLASS_DIST_SUMMARY}}.

2. **Manifold construction** — C2 ({{N_C2_FEATURES}} features), Network
   ({{N_NETWORK_FEATURES}}), Physical ({{N_PHYSICAL_FEATURES}}); z-score normalisation
   fit on training split; reference cloud = 500 k-medoid Normal-Traffic flows selected
   in C2 space, row indices reused positionally across all three manifolds (see §5 for
   the per-manifold diversity limitation).

3. **Persistent homology extraction** — per-flow point cloud: reference cloud (500 pts)
   augmented by one query flow; Vietoris-Rips up to H_{{MAX_HOM_DIM}}; MAX_EDGE_LENGTH =
   25th-percentile pairwise distance on reference cloud (D2: 95th percentile exhausted
   memory at >19M simplices); sparse Rips epsilon = 0.5 for C2 and Network, exact for
   Physical (D3); simplex-tree max_dimension = MAX_HOM_DIM + 1 (D4); 13 summary
   statistics per (manifold, dimension) pair → {{N_TDA_FEATURES}} total TDA features.
   Feature names: mean/std/max persistence, number of bars, total persistence, birth/death
   mean and std, Betti number, persistence entropy, normalized lifetime.

4. **Supervised classification** — 4 feature sets (C2, Network, Physical, Combined) x
   3 models (Logistic Regression, Random Forest, SVM) x 3 seeds; hyperparameter grid
   search by single validation-accuracy fit per combination (D6); SVM trained on
   stratified 5,000-row subsample for tractability (D5; LR and RF use full {{N_TRAIN}}
   rows); curated RF: top-{{N_CURATED_FEATURES}} features by RF importance, fixed
   (n_estimators=300, max_depth=None) (D8).

5. **Unsupervised anomaly detection** — baseline barcode: persistence diagram of the
   reference cloud alone; per-flow Wasserstein-2 score = sum_k W_2(D_k(flow),
   D_k(baseline)); anomaly threshold = 95th percentile of validation Normal-Traffic
   scores; pattern-based multi-class rule: for each manifold, flag flow as anomalous if
   score exceeds threshold; modal flag pattern per test flow mapped to attack class
   (tie-break alphabetical, M4 limitation).

### Placeholders needed
`{{N_SAMPLES}}`, `{{N_TRAIN}}`, `{{N_VAL}}`, `{{N_TEST}}`, `{{N_RAW_FEATURES}}`,
`{{N_C2_FEATURES}}`, `{{N_NETWORK_FEATURES}}`, `{{N_PHYSICAL_FEATURES}}`,
`{{N_COMBINED_FEATURES}}`, `{{MAX_HOM_DIM}}`, `{{N_TDA_FEATURES}}`,
`{{N_CURATED_FEATURES}}`, `{{CLASS_DIST_SUMMARY}}`

---

## Section 4 — Results

**Target: 1.25 pages (~500 words prose + tables/figures)**
**AIAA SciTech 2027 coverage: (e) initial evidence**

### Sub-outline: figures and tables

| # | Item | Caption stub | Placement |
|---|------|-------------|-----------|
| Table 2 | `headline_results.csv` → inline table | Best accuracy, macro-F1, and binary AUC per feature-set x model combination (bold best row; dagger SVM subsample note) | §4.1 opening |
| Fig. 3 | `headline_results.png` | Accuracy by feature set and model; dashed raw-feature baseline; error bars = 1 SD across seeds | §4.1 |
| Fig. 4 | `supervised_comparison.png` | Multi-metric comparison (accuracy, macro-F1, per-class F1) across all models and feature sets | §4.1 |
| Fig. 5 | `feature_importance_curated.png` | Top-{{N_CURATED_FEATURES}} feature importances colour-coded by manifold (C2 / Network / Physical) | §4.2 |
| Table 3 | Manifold and group ablation | Accuracy delta vs. no_ablation baseline when each manifold or feature group removed | §4.2 |
| Fig. 6 | `unsupervised_distance_distributions.png` | Wasserstein-2 score distributions for Normal vs. each attack class; threshold line | §4.3 |
| Fig. 7 | `unsupervised_roc_curves.png` | ROC curves (binary Normal vs. Attack) for C2, Network, Physical, and Combined | §4.3 |
| Fig. 8 | `unsupervised_pattern_heatmap.png` | Per-class modal-flag heatmap; inference-rule pattern matrix with unmapped fraction | §4.3 |

### Bullets

1. **Supervised headline** — best result: {{BEST_FEATURE_SET}} features + {{BEST_MODEL}}
   achieves {{HEADLINE_ACC}}% accuracy and {{HEADLINE_F1}} macro-F1 (seed {{BEST_SEED}});
   mean ± SD across 3 seeds: {{HEADLINE_ACC_MEAN}} ± {{HEADLINE_ACC_STD}}%; comparison
   to raw-feature baseline ({{BASELINE_ACC}}%): TDA gain = {{TDA_IMPROVEMENT_PP}} pp
   (Q2 verdict: {{Q2_VERDICT}}); curated {{N_CURATED_FEATURES}}-feature RF achieves
   {{CURATED_RF_ACC}}%.

2. **Manifold contribution and feature importance** — top-{{N_CURATED_FEATURES}} features:
   {{TOP_MANIFOLD_FRACTION}}% from {{TOP_MANIFOLD}} manifold; manifold-ablation worst drop:
   removing {{ABLATION_WORST_GROUP}} reduces accuracy by {{ABLATION_WORST_DROP}} pp
   (Q3 verdict: {{Q3_VERDICT}}); each manifold contributes independently (removing any
   one degrades performance).

3. **Per-class supervised analysis** — hardest class: {{HARDEST_CLASS}} (F1 =
   {{HARDEST_CLASS_F1}}); dominant confusion pair: {{CONF_PAIR}}; Sybil/Flooding
   separability: {{SYBIL_FLOOD_COMMENT}}; note SVM comparison is approximate due to
   5,000-row training cap (M6 limitation).

4. **Unsupervised detection** — binary AUC = {{UNSUP_BIN_AUC}} (Q1 verdict:
   {{Q1_VERDICT}}); best single-manifold AUC: {{UNSUP_BEST_MANIFOLD}} at
   {{UNSUP_BEST_MANIFOLD_AUC}}; 95th-percentile threshold yields precision
   {{UNSUP_PREC}} / recall {{UNSUP_RECALL}} on the test set.

5. **Unsupervised pattern rule** — pattern-rule multi-class accuracy = {{PATTERN_ACC}}%;
   unmapped fraction = {{UNMAPPED_FRAC}} (Q4 verdict: {{Q4_VERDICT}}); dominant
   manifold per attack class matches Zeng et al. predictions for
   {{ZENG_MATCH_COUNT}}/{{N_ATTACK_CLASSES}} attack types (Q5 verdict: {{Q5_VERDICT}}).

### Placeholders needed
`{{BEST_FEATURE_SET}}`, `{{BEST_MODEL}}`, `{{HEADLINE_ACC}}`, `{{HEADLINE_F1}}`,
`{{BEST_SEED}}`, `{{HEADLINE_ACC_MEAN}}`, `{{HEADLINE_ACC_STD}}`, `{{BASELINE_ACC}}`,
`{{TDA_IMPROVEMENT_PP}}`, `{{Q2_VERDICT}}`, `{{N_CURATED_FEATURES}}`,
`{{CURATED_RF_ACC}}`, `{{TOP_MANIFOLD}}`, `{{TOP_MANIFOLD_FRACTION}}`,
`{{ABLATION_WORST_GROUP}}`, `{{ABLATION_WORST_DROP}}`, `{{Q3_VERDICT}}`,
`{{HARDEST_CLASS}}`, `{{HARDEST_CLASS_F1}}`, `{{CONF_PAIR}}`,
`{{SYBIL_FLOOD_COMMENT}}`, `{{UNSUP_BIN_AUC}}`, `{{UNSUP_BEST_MANIFOLD}}`,
`{{UNSUP_BEST_MANIFOLD_AUC}}`, `{{UNSUP_PREC}}`, `{{UNSUP_RECALL}}`,
`{{Q1_VERDICT}}`, `{{PATTERN_ACC}}`, `{{UNMAPPED_FRAC}}`, `{{Q4_VERDICT}}`,
`{{ZENG_MATCH_COUNT}}`, `{{N_ATTACK_CLASSES}}`, `{{Q5_VERDICT}}`

---

## Section 5 — Limitations and Final Manuscript Plan

**Target: 1/2 page (~250 words prose)**
**AIAA SciTech 2027 coverage: (b) technical challenges, (f) deliverables sketch**

### Bullets

1. **Known limitations to disclose** (full text in `paper/METHODS_DISCLOSURES.md`):
   - M1: Reference cloud sampled by k-medoids in C2 space only; Network/Physical
     reference diversity not guaranteed.
   - M2: Essential H_0 bar truncated to MAX_EDGE_LENGTH; mean and max persistence
     biased upward by a manifold-specific constant.
   - M3: Wasserstein aggregation by sum across homology dimensions; max-pooling and
     multi-dimensional Wasserstein alternatives unexplored.
   - M6: SVM trained on 5,000-row subsample vs. full training set for LR/RF; cross-model
     accuracy comparisons are approximate.
   - M8: Variance estimated over 3 seeds; reported standard deviations carry wide
     uncertainty.

2. **Filtration scope limitation** — MAX_EDGE_LENGTH capped at 25th-percentile pairwise
   distance (not 95th) to stay within memory; sparse Rips epsilon = 0.5 introduces an
   interleaving-distance approximation error bounded by epsilon per dimension; full-radius
   exact Rips deferred to journal version with GPU acceleration.

3. **Final manuscript extensions (full journal version)**:
   - Per-manifold k-medoid reference-cloud sampling (independent diversity in each space).
   - Larger filtration radius with Ripser++ (GPU-accelerated Rips) or CUDA Gudhi.
   - Multi-dimensional and max-pooling Wasserstein aggregation comparison.
   - 5-fold inner cross-validation inside the hyperparameter grid search.
   - Live-stream inference latency profiling for real-time onboard deployment.

4. **Dataset scope** — UAVIDS-2025 is a simulation dataset; real-world distributional
   shift is uncharacterized; single swarm topology limits generalization claims.

### Placeholders needed
_None — this section draws entirely from the hardcoded limitation catalog in
`paper/METHODS_DISCLOSURES.md`._

---

## Section 6 — Conclusion

**Target: 1/4 page (~125 words prose)**
**AIAA SciTech 2027 coverage: (d) specific contributions, (e) initial evidence**

### Bullets

1. **Restate core finding** — multi-manifold persistent homology features are discriminative
   for UAV intrusion detection; three-manifold Rips pipeline achieves {{HEADLINE_ACC}}%
   supervised accuracy and {{UNSUP_BIN_AUC}} unsupervised binary AUC on UAVIDS-2025.

2. **Contribution summary** — three-manifold C2/Network/Physical Rips pipeline;
   curated {{N_CURATED_FEATURES}}-feature RF subset; unsupervised Wasserstein scoring
   without labeled attacks at calibration time; reproducible open-source implementation.

3. **Dominant manifold finding** — {{DOMINANT_MANIFOLD}} carries the strongest topological
   signal; consistent with Zeng et al. predictions for {{ZENG_MATCH_COUNT}}/{{N_ATTACK_CLASSES}}
   attack types; discrepancies suggest attack-specific manifold sensitivity worth
   further investigation.

4. **Forward look** — per-manifold reference sampling and full-scale Rips on GPU hardware
   are the two highest-leverage paths to tighter bounds and a full-length journal submission.

### Placeholders needed
`{{HEADLINE_ACC}}`, `{{UNSUP_BIN_AUC}}`, `{{N_CURATED_FEATURES}}`,
`{{DOMINANT_MANIFOLD}}`, `{{ZENG_MATCH_COUNT}}`, `{{N_ATTACK_CLASSES}}`

---

## Placeholder master index

| Token | Source | Section(s) used |
|-------|--------|----------------|
| `{{PAPER_TITLE}}` | Author decision | Title |
| `{{N_SAMPLES}}` | Dataset / run summary | Abstract, §3 |
| `{{N_CLASSES}}` | Dataset | Abstract |
| `{{N_TRAIN}}` / `{{N_VAL}}` / `{{N_TEST}}` | Run summary | §3 |
| `{{N_RAW_FEATURES}}` | Dataset | §3 |
| `{{N_C2_FEATURES}}` | Manifold definition | §3, Table 1 |
| `{{N_NETWORK_FEATURES}}` | Manifold definition | §3, Table 1 |
| `{{N_PHYSICAL_FEATURES}}` | Manifold definition | §3, Table 1 |
| `{{N_COMBINED_FEATURES}}` | Sum of above | §3, Table 1 |
| `{{MAX_HOM_DIM}}` | Pipeline config | Abstract, §3 |
| `{{N_TDA_FEATURES}}` | Feature extraction | Abstract, §3 |
| `{{N_CURATED_FEATURES}}` | Feature importance | §1, §3, §4, §6 |
| `{{CLASS_DIST_SUMMARY}}` | Dataset stats | §3 |
| `{{BEST_FEATURE_SET}}` | DIAGNOSTIC_REPORT §2 | §4, Table 2 |
| `{{BEST_MODEL}}` | DIAGNOSTIC_REPORT §2 | Abstract, §4 |
| `{{HEADLINE_ACC}}` | DIAGNOSTIC_REPORT §2 | Abstract, §4, §6 |
| `{{HEADLINE_F1}}` | DIAGNOSTIC_REPORT §2 | Abstract, §4 |
| `{{BEST_SEED}}` | DIAGNOSTIC_REPORT §2 | §4 |
| `{{HEADLINE_ACC_MEAN}}` / `{{HEADLINE_ACC_STD}}` | DIAGNOSTIC_REPORT §2 | §4 |
| `{{BASELINE_ACC}}` | DIAGNOSTIC_REPORT §2 | Abstract, §4 |
| `{{TDA_IMPROVEMENT_PP}}` | DIAGNOSTIC_REPORT Q2 | Abstract, §4 |
| `{{CURATED_RF_ACC}}` | DIAGNOSTIC_REPORT §2 | §1, Abstract, §4 |
| `{{CURATED_RF_F1}}` | DIAGNOSTIC_REPORT §2 | Abstract |
| `{{Q1_VERDICT}}` | DIAGNOSTIC_REPORT Q1 | §4 |
| `{{Q2_VERDICT}}` | DIAGNOSTIC_REPORT Q2 | §4 |
| `{{Q3_VERDICT}}` | DIAGNOSTIC_REPORT Q3 | §4 |
| `{{Q4_VERDICT}}` | DIAGNOSTIC_REPORT Q4 | §4 |
| `{{Q5_VERDICT}}` | DIAGNOSTIC_REPORT Q5 | §4 |
| `{{UNSUP_BIN_AUC}}` | DIAGNOSTIC_REPORT Q1 | Abstract, §1, §4, §6 |
| `{{UNSUP_BEST_MANIFOLD}}` / `{{UNSUP_BEST_MANIFOLD_AUC}}` | DIAGNOSTIC_REPORT §4 | §4 |
| `{{UNSUP_PREC}}` / `{{UNSUP_RECALL}}` | DIAGNOSTIC_REPORT §4 | §4 |
| `{{PATTERN_ACC}}` / `{{UNMAPPED_FRAC}}` | DIAGNOSTIC_REPORT §5 | §4 |
| `{{TOP_MANIFOLD}}` / `{{TOP_MANIFOLD_FRACTION}}` | Feature importance table | §4, Fig. 5 |
| `{{ABLATION_WORST_GROUP}}` / `{{ABLATION_WORST_DROP}}` | DIAGNOSTIC_REPORT Q3 | §4, Table 3 |
| `{{HARDEST_CLASS}}` / `{{HARDEST_CLASS_F1}}` | Per-class metrics | §4 |
| `{{CONF_PAIR}}` | Confusion matrix | §4 |
| `{{SYBIL_FLOOD_COMMENT}}` | Per-class analysis | §4 |
| `{{DOMINANT_MANIFOLD}}` | Ablation / Q3 | §4, §6 |
| `{{ZENG_MATCH_COUNT}}` / `{{N_ATTACK_CLASSES}}` | DIAGNOSTIC_REPORT Q5 | §4, §6 |
| `{{ABSTRACT_WORD_COUNT}}` | Author count at fill-in | Abstract note |
| `{{REF_TDA_IDS_1}}` / `{{REF_TDA_IDS_2}}` | Literature search | §2 |
| `{{REF_UAVIDS}}` / `{{REF_UAV_SECURITY_SURVEY}}` | Literature search | §2 |

---

## Page budget summary

| Section | Target pages | Target words (prose) | Figures / tables |
|---------|-------------|---------------------|-----------------|
| Abstract | ~0.10 | ~200 | — |
| §1 Introduction | 0.75 | ~375 | — |
| §2 Background | 0.75 | ~375 | — |
| §3 Methodology | 1.50 | ~600 | Fig. 1, Fig. 2, Table 1 |
| §4 Results | 1.25 | ~500 | Fig. 3–8, Table 2, Table 3 |
| §5 Limitations | 0.50 | ~250 | — |
| §6 Conclusion | 0.25 | ~125 | — |
| **Total** | **5.10** | **~2425** | **6 figures, 3 tables** |

_Note: 0.10 page over target. If layout is tight, trim §2 to ~325 words or merge
the shortest §6 bullets into a single paragraph._

---

_Last updated: {{OUTLINE_DATE}}.
Fill `{{...}}` tokens from `paper/DIAGNOSTIC_REPORT.md` after the production run completes._
