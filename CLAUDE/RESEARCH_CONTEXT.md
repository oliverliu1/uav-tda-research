# Project Context: Multi-Manifold Persistent Homology for UAV Swarm Intrusion Detection
## Handoff Document for Claude Code — May 2026

---

## 1. PROJECT OVERVIEW

**Author:** Oliver Liu (Dept. of Computer Science, Baylor University)  
**Advisor:** Dr. Liang Sun (Dept. of Mechanical Engineering)  
**Goal:** Publish at AIAA SciTech 2027 (extended abstract due ~mid-2026, full paper due late 2026)  
**Current status:** Experiments partially complete, poster presented at ECS Summit 2026, results have known issues that must be resolved before paper writing begins.

**One-sentence summary of the research:**  
Apply multi-manifold persistent homology (Topological Data Analysis) to detect adversarial attacks in UAV drone swarm networks, using the UAVIDS-2025 benchmark dataset, and compare against classical ML baselines.

---

## 2. DATASET

**UAVIDS-2025** — 122,171 network flow records, 23 columns, no missing values.  
Source: Zeng et al., IEEE CNS 2025. Publicly available.

**Label distribution:**
- Normal Traffic: 26,172 (21.4%)
- Blackhole Attack: 26,110 (21.4%)
- Wormhole Attack: 26,086 (21.4%)
- Sybil Attack: 24,077 (19.7%)
- Flooding Attack: 19,726 (16.1%)

**Key dataset quirks:**
- Protocol column is entirely UDP (zero variance — dropped)
- FlowID is a sequential row index with no semantic meaning (dropped)
- Normal Traffic records are interleaved with attack records throughout — NOT contiguous. Baseline construction must filter by label, not row position.
- IP addresses follow 192.168.0.X pattern — only last octet varies (range 1–255)
- Only two port values exist: 9 (Discard) and 654 (AODV routing protocol)

---

## 3. METHODOLOGY (DESIGNED AND PARTIALLY IMPLEMENTED)

### Feature Partitioning into Three Manifolds

After dropping FlowID and Protocol, the remaining 21 features are partitioned:

| Manifold | Dimensions | Features |
|---|---|---|
| C2 (Command & Control) | 5D | SrcAddr_octet, SrcPort_binary, DstAddr_octet, DstPort_binary, FlowDuration/s |
| Network | 15D | TxPackets, RxPackets, LostPackets, TxBytes, RxBytes, TxPacketRate/s, RxPacketRate/s, TxByteRate/s, RxByteRate/s, MeanPacketSize, MeanDelay/s, MeanJitter/s, Throughput/Kbps, PacketDropRate, AverageHopCount |
| Physical (proxy) | 2D | MeanDelay/s, AverageHopCount (proxies for spatial position — actual kinematics not in dataset) |

**Note:** MeanDelay/s and AverageHopCount appear in BOTH the Network and Physical manifolds. This creates duplicate columns in the combined feature matrix (appearing as `MeanDelay/s` and `MeanDelay/s.1`, `AverageHopCount` and `AverageHopCount.1`). This is a known issue to fix.

### Preprocessing
- C2: IP addresses → last octet integer extraction. Ports → binary flag (9→0, 654→1). Then StandardScaler.
- Network: StandardScaler only.
- Physical: StandardScaler only.

### TDA Pipeline (Two Parallel Approaches)

**Approach A — Supervised (Scripts 01–08):**  
Per-sample local topology using k-nearest neighbors (k=100). For each sample, build a Vietoris-Rips complex from its 100 nearest neighbors (within same train/test split). Compute persistence diagrams (H0, H1, H2). Extract statistical features from diagrams (count, mean/std persistence, entropy, total persistence, mean birth/death per dimension). Concatenate TDA features with original features. Train classifiers (LR, RF, SVM) on combined features. Compare against baselines trained on original features only.

**Approach B — Unsupervised Wasserstein (Scripts 2W–6W):**  
Build a "healthy baseline" barcode from 1,000 sampled Normal Traffic flows (one Rips complex from all 1,000 points). For each flow, build a Rips complex from a 50-point sequential neighborhood (±25 rows). Compute Wasserstein distance between each flow's barcode and the healthy baseline. Z-score normalize distances using Normal Traffic statistics. Flag as anomalous if Z-score > 3σ. Evaluate with AUC-ROC.

**TDA Library:** GUDHI  
**Wasserstein distance:** GUDHI's built-in `wasserstein_distance` (NOT Hera — the methodology doc mentions Hera but it was not used in implementation)  
**Parameters:** max_edge_length=5.0, max_dimension=2 (H0, H1, H2), neighborhood k=100 (supervised), neighborhood=50 sequential rows (unsupervised)

---

## 4. CURRENT RESULTS (ALL REAL, FROM ACTUAL RUNS)

### Supervised Baseline (No TDA, Script 02):
| Model | CV Mean | Test Accuracy | Test F1 | Test AUC |
|---|---|---|---|---|
| Logistic Regression | 0.766 | 75.8% | 0.760 | 0.945 |
| Random Forest | 0.961 | 96.2% | 0.962 | 0.997 |
| SVM | 0.616 | 62.1% | 0.610 | 0.897 |

### TDA-Enhanced Models (Script 07):
| Model | Test Accuracy | Test F1 | Test AUC |
|---|---|---|---|
| Logistic Regression (TDA) | 82.5% | 0.818 | 0.972 |
| Random Forest (TDA) | 91.1% | 0.910 | 0.989 |
| SVM (TDA) | 77.3% | 0.779 | 0.954 |

### Per-Class TDA RF Classification Report:
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Blackhole Attack | 0.755 | 0.965 | 0.847 |
| Flooding Attack | 0.980 | 0.990 | 0.985 |
| Normal Traffic | 0.995 | 0.968 | 0.982 |
| Sybil Attack | 1.000 | 0.973 | 0.986 |
| Wormhole Attack | 0.891 | 0.683 | 0.773 |

### Unsupervised Wasserstein (Scripts 2W–6W):
| Metric | Value |
|---|---|
| Overall AUC | 0.844 |
| Accuracy (at 3σ threshold) | 36.6% |
| Precision | 0.975 |
| Recall | **0.198** |
| F1 | 0.329 |
| C2 manifold AUC | 0.811 |
| Network manifold AUC | 0.618 |
| Physical manifold AUC | 0.639 |

### Per-Attack Wasserstein AUC:
| Attack | Overall | C2 AUC | Network AUC | Physical AUC |
|---|---|---|---|---|
| Sybil | 0.712 | **0.921** | 0.843 | 0.655 |
| Flooding | 0.659 | 0.730 | 0.439 | 0.385 |
| Wormhole | 0.634 | 0.497 | 0.447 | 0.596 |
| Blackhole | **0.384** | 0.232 | 0.398 | 0.490 |

---

## 5. KNOWN ISSUES — MUST FIX BEFORE PAPER

### Issue 1 (CRITICAL): Random Forest TDA Regression
The TDA-enhanced RF (91.1%) is **worse** than baseline RF (96.2%). This is the most important problem to diagnose. Random Forests are generally robust to irrelevant features, so a 5-point accuracy drop when adding features is anomalous and likely indicates a bug rather than a true finding.

**Suspected causes (investigate in this order):**
1. **Train/test split inconsistency.** Script 02 (`baseline_models.py`) calls `train_test_split` directly on `original_features.csv` with `random_state=42`. Script 07 (`tda_enhanced_models.py`) loads pre-split files from Script 06. Script 06 reconstructs splits from `train_indices.npy` / `test_indices.npy` saved by Script 01. If the data passed into `train_test_split` in Script 02 has different column ordering or row count than the data used to generate the indices in Script 01, the test sets differ and the comparison is invalid.
   - **Diagnostic:** Print `y_test.value_counts()` from both Script 02 and Script 07 and confirm they are identical. Also confirm `len(X_test)` matches.
2. **Duplicate columns in combined feature matrix.** `MeanDelay/s` and `AverageHopCount` exist in both Network and Physical manifolds, creating `MeanDelay/s.1` and `AverageHopCount.1` duplicates in `combined_features.csv`. This introduces collinearity that may degrade RF performance.
3. **Scaling applied twice or inconsistently.** Script 01 scales each manifold separately and saves scaled CSVs. Script 07 then applies `StandardScaler` again to the combined features. The original features in `original_features.csv` are saved *before* scaling (in Script 01). Confirm that the combined features file uses the pre-scale originals concatenated with TDA features, not double-scaled data.
4. **Label encoding inconsistency.** Script 07 uses `LabelEncoder` to encode `y_train` and `y_test`. Script 02 uses string labels directly. Confirm the class ordering is consistent.

### Issue 2 (IMPORTANT): per_class_performance.csv is all zeros
The file `results/tables/per_class_performance.csv` contains all zeros for Precision, Recall, F1, and Support. This means `create_detailed_confusion_matrix.py` failed silently (likely because it recreates the train/test split independently rather than loading saved indices, causing a mismatch with the saved RF model). This script needs to be rewritten to use the canonical split indices.

### Issue 3 (MODERATE): Wasserstein recall is 0.198
The 3σ threshold is far too conservative. The detector has near-perfect precision (0.975) but catches only 19.8% of attacks. This is a threshold calibration issue, not a fundamental failure of the approach. The AUC of 0.844 shows the Wasserstein distances are discriminative — the threshold just needs to be optimized. A threshold sweep with precision-recall curve analysis is needed.

### Issue 4 (MODERATE): Blackhole detection is below random (AUC 0.384)
The Blackhole attack has AUC below 0.5 on all manifolds, meaning the Wasserstein distance is *inversely* correlated with Blackhole detection. Blackhole attacks manifest primarily through PacketDropRate and reduced Throughput — behaviors that may actually *simplify* the topology (fewer connected components, simpler structure) rather than making it more complex, thus producing lower Wasserstein distances than normal traffic. This is a genuine scientific finding worth analyzing and reporting. It does NOT need to be fixed — it needs to be understood and explained.

### Issue 5 (MINOR): Sequential neighborhood in Wasserstein pipeline is semantically wrong
Script 3W builds per-flow neighborhoods using `all_points[idx-25:idx+25]` — sequential row indices. This mixes attack and normal records arbitrarily based on dataset ordering rather than temporal or spatial proximity. This undermines the "healthy baseline comparison" framing.

---

## 6. NEXT EXPERIMENTS (PRIORITY ORDER)

### Priority 1: Audit and Fix the RF Regression (before any new experiments)
Diagnose Issue 1 above. This is a half-day of work. The outcome determines whether the supervised story is "TDA hurts RF" (a real finding requiring explanation) or "there was a bug" (fix it and rerun). Do not proceed to new experiments until this is resolved.

**Specific tasks:**
- Add a canonical `load_split()` utility that ALL scripts use to load `train_indices.npy` and `test_indices.npy` from Script 01. Eliminate all independent `train_test_split` calls in Scripts 02 and beyond.
- Remove duplicate columns from combined feature matrix (drop Physical manifold raw features before combining, since they already exist in Network).
- Audit the scaling pipeline: confirm original features enter the combined matrix unscaled, and only one StandardScaler is applied in Script 07.
- Rerun Scripts 02, 07, 08 and compare.

### Priority 2: Temporal Windowing for Wasserstein Pipeline (new experiment)
Replace the per-flow sequential-neighborhood approach with semantically meaningful time windows.

**Design:**
- Sort flows by FlowID (already sequential). Group into non-overlapping windows of size k.
- Test k ∈ {50, 100, 200} flows per window.
- For each window, build one Rips complex from all k points in the window.
- Compute Wasserstein distance between that window's barcode and the healthy baseline barcode.
- Z-score normalize. Apply threshold sweep (not just 3σ) and report precision-recall curve.
- Evaluate: does windowing improve Blackhole and Wormhole detection? Does it improve recall?

**Why this matters for the paper:** The current per-flow approach treats each network flow as isolated. Real attacks unfold over time. Windowing is operationally realistic (you'd monitor the swarm in time windows, not per-packet) and is what the methodology document describes. It also directly addresses the low recall problem.

**Computational note:** 122,171 flows ÷ 100 = ~1,222 windows. Building 1,222 Rips complexes on 100-point clouds is dramatically faster than 122,171 complexes on 50-point clouds. This experiment should run in minutes, not hours.

### Priority 3: Threshold Optimization for Wasserstein (quick analysis)
Run a precision-recall curve sweep across Z-score thresholds [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] for the per-attack Wasserstein distances. Find the threshold that maximizes F1 per attack type. Report optimal thresholds and the resulting precision/recall. This costs nothing computationally — all distances are already computed.

### Priority 4 (Optional, if time allows): Investigate Blackhole Topology
Analyze whether Blackhole attack topology is genuinely simpler than normal (lower Wasserstein distance from baseline). Compute mean H0 count and H1 count per attack class from the persistence diagrams. If Blackhole shows fewer connected components and fewer loops than Normal Traffic, this confirms the "topology simplification" hypothesis and provides a principled explanation for the below-random AUC.

---

## 7. REPO STRUCTURE (current state)

```
project/
├── data/
│   └── UAVIDS-2025.csv
├── scripts/
│   ├── 01_data_scripts.py       # Data prep, manifold partitioning, train/test split
│   ├── 02_baseline_models.py    # LR, RF, SVM on original features (BUG: own split)
│   ├── 03_tda_c2_manifold.py    # GUDHI persistence for C2 manifold
│   ├── 04_tda_network_manifold.py
│   ├── 05_tda_physical_manifold.py
│   ├── 06_tda_features_extraction.py  # Statistical features from persistence diagrams
│   ├── 07_tda_enhanced_models.py      # LR, RF, SVM on original + TDA features
│   ├── 08_comparative_analysis.py     # Comparison plots and tables
│   ├── 2W_wasserstein_baseline.py     # Healthy baseline barcode from Normal Traffic
│   ├── 3W_wasserstein_per_flow.py     # Per-flow persistence (BUG: sequential neighborhood)
│   ├── 4W_wasserstein_distances.py    # Wasserstein distance computation
│   ├── 5W_wasserstein_detection.py    # Z-score normalization, 3σ detection
│   ├── 6W_wasserstein_evaluation.py   # AUC-ROC, visualizations
│   ├── create_detailed_confusion_matrix.py  # BUG: all-zero output
│   ├── create_feature_importance_viz.py
│   ├── create_methodology_flowchart.py
│   ├── create_persistence_diagram_examples.py
│   └── run_wasserstein_pipeline.sh
├── outputs/
│   ├── c2_manifold_scaled.csv
│   ├── network_manifold_scaled.csv
│   ├── physical_manifold_scaled.csv
│   ├── labels.csv
│   ├── original_features.csv
│   ├── train_indices.npy          # CANONICAL split — all scripts must use this
│   ├── test_indices.npy           # CANONICAL split — all scripts must use this
│   ├── persistence_diagrams/      # .npy and .pkl files for all three manifolds
│   ├── tda_features/              # Extracted statistical features + combined
│   └── wasserstein/               # Baseline barcodes, distances, detection results
└── results/
    ├── figures/
    ├── models/
    └── tables/
```

---

## 8. REFACTORING RECOMMENDATIONS FOR CLAUDE CODE

The codebase works but has structural issues that will cause problems as experiments expand. Recommended refactoring before new experiments:

1. **Create `utils/split.py`** with a single `load_canonical_split(output_dir)` function that loads `train_indices.npy` and `test_indices.npy`. Every script that needs train/test split calls this. Remove all independent `train_test_split` calls from Scripts 02, 06, 07, and the visualization scripts.

2. **Create `utils/config.py`** with all shared constants: `DATA_DIR`, `OUTPUT_DIR`, `RESULTS_DIR`, `RANDOM_STATE=42`, `N_NEIGHBORS=100`, `MAX_EDGE_LENGTH=5.0`, `MAX_DIMENSION=2`, manifold feature lists. Currently these are defined redundantly in every script.

3. **Fix the combined feature matrix.** In Script 06 (`tda_features_extraction.py`), the `original_features.csv` loaded from Script 01 contains raw (unscaled) features including both MeanDelay/s and AverageHopCount from the Network manifold. When Physical manifold is separately defined with the same two features, they get concatenated twice. Fix: either (a) do not include a separate Physical manifold in the supervised combined features, or (b) deduplicate columns after concat.

4. **Rewrite `create_detailed_confusion_matrix.py`** to load the canonical test split and the saved RF model, rather than re-splitting. This will fix the all-zero output.

5. **Add a `scripts/run_supervised_pipeline.sh`** similar to the existing Wasserstein shell script, so the full supervised pipeline can be re-run end-to-end after refactoring.

6. **For new Wasserstein windowing experiment:** create `3W_wasserstein_windowed.py` as a new script rather than modifying the existing per-flow script. Keep the original for comparison.

---

## 9. SCIENTIFIC FRAMING FOR THE PAPER (advisory context)

The research advisor has recommended the following honest framing for AIAA SciTech 2027:

**Primary contribution:** A multi-manifold persistent homology framework for UAV swarm anomaly detection that produces attack-type-specific topological signatures. This is a methodological contribution paper, not a "we beat all baselines" paper.

**Honest findings to report:**
- C2 manifold Wasserstein distance is strongly discriminative for Sybil attacks (AUC 0.921), validating the hypothesis that identity-based attacks disrupt communication topology geometry.
- TDA feature augmentation most benefits weaker baselines (LR: +8.8%, SVM: +24.5% accuracy), suggesting topological features add information not captured by simpler learners.
- TDA augmentation hurts the strongest baseline (RF: −5.2%) — either a bug (must diagnose) or a finding about feature redundancy with ensemble methods.
- The 3σ threshold is too conservative (recall 0.198) — threshold calibration is needed and should be reported as a finding.
- Blackhole attacks are geometrically underdetected (AUC 0.384) — hypothesized explanation is topology simplification (fewer connected components during packet drops), not a failure of the framework.

**Claims to avoid:**
- "Zero-day detection" — not validated (requires out-of-distribution attack evaluation).
- Uniform improvement across all models — RF regression is real and must be acknowledged.
- The 57.1% "baseline" cited on the poster — this number's provenance is unclear and should not appear in the paper without precise definition.

**Target venue:** AIAA SciTech 2027. The forum values novel methods applied to aerospace/defense problems. The multi-manifold architecture applied to UAV swarms is novel. The honest, nuanced treatment of where TDA helps and where it doesn't is scientifically credible and appropriate for this venue.

---

## 10. KEY REFERENCES IN PROJECT

1. Zeng et al. (2025). UAVIDS-2025: A Benchmark Dataset for Intrusion Detection in UAV Networks. IEEE CNS 2025, pp. 97–105. — *The dataset paper. Their best ML result is XGBoost at 96% accuracy, AUC 0.99 on raw features.*
2. Bruillard, Nowak, Purvine (2016). Anomaly Detection Using Persistent Homology. IEEE Cybersecurity Symposium. — *Direct precedent for TDA-based network anomaly detection. Single manifold, simpler network.*
3. The GUDHI Project (2020). GUDHI User and Reference Manual 3.1.0. — *Library used for all Rips complex and persistence computations.*
4. Attali, Lieutier, Salinas (2011). Vietoris-Rips complexes also provide topologically correct reconstructions of sampled shapes. SoCG '11. — *Theoretical justification for Rips complex use.*
5. Boissonnat & Maria (2014). The Simplex Tree: An Efficient Data Structure for General Simplicial Complexes. Algorithmica. — *Data structure underlying GUDHI implementation.*
6. Kerber, Morozov, Nigmetjanov (2017). Geometry Helps to Compare Persistence Diagrams. ACM JEA. — *Geometric Wasserstein distance algorithm (Hera library) — cited in methodology but not yet used in implementation.*

---

*Document prepared May 2026. All results are from actual pipeline runs on UAVIDS-2025. No results are placeholder or projected.*
