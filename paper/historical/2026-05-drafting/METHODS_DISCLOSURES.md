# Methodology Disclosures

_Three paragraph blocks suitable for direct paste into the paper's Methods, Limitations, and Future Work sections. The source-index table below the paragraphs is for traceability only and should not appear in the manuscript._

## Methods deviations

Several methodology choices warrant explicit disclosure. We set MAX_EDGE_LENGTH to the 25th percentile of pairwise distances within each reference cloud rather than the 95th percentile of our initial design, because the 95th percentile produced simplex trees exceeding 19 million simplices per query and exhausted memory at simplex-dimension three. We applied a sparse Rips approximation (epsilon = 0.5) for C2 and Network while Physical remained exact, and set the simplex-tree max_dimension to MAX_HOM_DIM + 1 so the highest-order homology features had finite death times. The Support Vector Machine was trained on a stratified 5,000-row subsample for tractability; Logistic Regression and Random Forest used all 85,519 training rows. We searched the hyperparameter grid by single fit per combination, selecting by validation accuracy; the curated Random Forest variant used fixed (n_estimators = 300, max_depth = None) rather than re-tuning. Ablation deltas were reported against a no_ablation row trained with the same fixed RF, so deltas reflect feature contribution rather than hyperparameter mismatch. Debug mode used a per-class stratified head sample because UAVIDS-2025 is row-sorted by class.

## Limitations

The reference cloud was sampled by k-medoids on the C2 manifold only; the same row positions were reused for Network and Physical, guaranteeing a shared scaffold but not diversity in the latter two feature spaces. The single essential H_0 feature per Rips diagram was truncated to MAX_EDGE_LENGTH before summary statistics, biasing mean and max persistence upward by a manifold-specific constant. Per-flow Wasserstein distances were aggregated by summing across homology dimensions; max-pooling or multi-dimensional Wasserstein could change relative rankings. The pattern-based inference rule resolves ties by class-name order, potentially disadvantaging rarer attack types. The Support Vector Machine was trained on a 5,000-row stratified subsample because the RBF kernel does not scale to 85,519 rows, making its comparison with the linear and random-forest models approximate. Variance was estimated over three seeds, so reported standard deviations carry wide uncertainty.

## Future work

Three immediate extensions follow from these limitations. First, a larger filtration radius combined with a denser sparse Rips approximation would capture topological features at coarser scales than the present 25th-percentile cutoff permits. Second, per-manifold reference-cloud sampling, rather than reusing C2-derived medoids across all three manifolds, would ensure that the reference scaffold is locally representative of each manifold's geometry. Third, alternative Wasserstein aggregations across homology dimensions, including max-pooling for sensitivity to single-dimension anomalies and concatenated multi-dimensional Wasserstein for joint structure, merit empirical comparison against the current sum-aggregation.

---

## Source index (not for paper text)

Below maps every catalogued deviation and limitation to its coverage in the paragraphs above. Two limitations (M5, M7) were resolved during the post-refactor audit and are not paper limitations.

### Deviations (D1-D8) — all covered in Methods deviations

| ID | Title |
| :--- | :--- |
| D1 | Debug-mode stratified head sample |
| D2 | MAX_EDGE_PERCENTILE lowered from 95 to 25 |
| D3 | Sparse Rips approximation (epsilon = 0.5) for C2 and Network |
| D4 | Simplex-tree max_dimension = MAX_HOM_DIM + 1 |
| D5 | SVM trained on a stratified 5,000-row subsample |
| D6 | Grid search picks best by val accuracy, no inner CV |
| D7 | Ablations use fixed RF; no_ablation row included |
| D8 | Curated RF subset uses fixed hyperparameters |

### Limitations (M1-M8)

| ID | Title | Coverage |
| :--- | :--- | :--- |
| M1 | Reference cloud sampled by KMedoids on C2 only | Limitations |
| M2 | Essential H_0 truncated to MAX_EDGE_LENGTH | Limitations |
| M3 | Wasserstein aggregation by sum across homology dimensions | Limitations |
| M4 | Inference-rule tie-break by class-name order | Limitations |
| M5 | Inference-rule unmapped-pattern fraction now reported | Audit fix M5 (not a paper limitation) |
| M6 | SVM training subsample (5,000 rows) vs LR/RF full 85k | Limitations |
| M7 | Ablations now compared against no_ablation baseline row | Audit fix C3+C4+M7 (not a paper limitation) |
| M8 | Variance estimated over only three seeds | Limitations |

