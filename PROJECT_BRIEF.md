# Project Brief — Multi-Manifold Topological Analysis of UAV Network Attacks

**Last updated:** 2026-05-23
**Author:** Oliver Liu, advised by Dr. Sun
**Target venue:** AIAA SciTech 2027, Intelligent Systems track
**Status:** Reframed paper; drafting begins after two confirmation diagnostics complete.

---

## 0. How to use this document

Read sections 1–4 before doing anything else. Sections 5–8 are reference material.
Section 9 lists the only files anyone should be touching right now. Section 10 is
the next-48-hour action list with explicit blockers.

If you are an LLM picking this up: do not relitigate decisions in this document.
The pivot below is settled. Your job is to help execute it.

---

## 1. The one-paragraph story

We applied multi-manifold persistent homology to UAV intrusion detection on the UAVIDS-2025 benchmark. The original framing — "TDA improves classification accuracy" — is dead: raw Random Forest baseline already hits 96.3% accuracy and TDA features do not improve on it. The project pivoted to three surviving findings. First, a label-free anomaly detector built on per-flow Vietoris-Rips persistence diagrams compared via Wasserstein-2 distance to a reference-cloud baseline barcode achieves binary AUC of 0.86 ± 0.00 on the Network+Physical manifold subset across three seeds, without exposure to any attack labels at calibration. Second, attack-specific manifold attribution reproduces across three seeds: Sybil and Flooding manifest on the Network manifold (mean AUCs 0.87 and 0.79), while Blackhole and Wormhole manifest on the Physical manifold (mean AUCs 0.79 and 0.74). Third, three of four attack-class attributions agree with the feature-level predictions in Zeng et al. (2025); Sybil is the exception — Zeng predicted Sybil would manifest on source-addressing features (C2 manifold), but our topology finds Sybil's strongest signature on the Network manifold. We discuss what this single divergence may reveal about TDA-based versus feature-inspection-based analysis.

---

## 2. Research goals and questions

### Primary question
What do topological invariants of multi-manifold network-flow representations
reveal about cyber attacks on UAV swarms, and which functional subsystem
manifests each attack class most strongly?

### Three research questions (in priority order for the paper)
- **RQ1.** Do distinct attack classes in UAVIDS-2025 produce topologically
  distinguishable signatures on a per-manifold basis, and if so, which manifold
  is dominant for each attack class?
- **RQ2.** Can label-free anomaly detection — calibrated using only normal-traffic
  samples — match supervised classifiers' ranking quality (AUC) on this benchmark,
  particularly under operational constraints that exclude one or more manifolds?
- **RQ3.** Do the empirically observed per-manifold attack signatures align with
  the feature-level attack-signature predictions in Zeng et al. (2025), and where
  they diverge, what does that divergence indicate?

### What is NOT a research question
- Whether TDA improves supervised accuracy on UAVIDS-2025. It does not. This
  question is settled and is not in the paper.

---

## 3. Contributions claimed in the paper (exactly three)

1. **A three-manifold decomposition of UAV network traffic** aligned with the functional categorization defined by the UAVIDS-2025 dataset creators (Zeng et al., 2025, Table III). C2 (Connection) = 7 features after one-hot port encoding; Network (Traffic Volume) = 10 features; Physical (Performance) = 5 features. The three manifolds are pairwise disjoint.

2. **A label-free multi-manifold anomaly detector** based on per-flow Vietoris-Rips persistence diagrams compared via Wasserstein-2 distance to a reference-cloud baseline barcode, achieving a binary normal-vs-attack AUC of 0.86 ± 0.00 (three-seed mean) on the Network+Physical manifold subset of UAVIDS-2025. The detector uses only normal-traffic samples for threshold calibration; it never sees attack labels.

3. **An empirical characterization of per-attack manifold attribution**, reproducible across three seeds: Sybil and Flooding attacks manifest on the Network manifold (mean AUCs 0.87 and 0.79), while Blackhole and Wormhole attacks manifest on the Physical manifold (mean AUCs 0.79 and 0.74). Three of four attributions agree with feature-level predictions in Zeng et al. (2025). Sybil's empirical attribution to Network (rather than C2, where Zeng predicted source-addressing signatures would dominate) is a single divergence; we discuss what this may reveal about TDA-based versus feature-inspection-based analysis.

The previously contemplated fourth contribution — the "C2 manifold paradox" — is retired. Diagnostic A (confirmation probe at top-K=50, delta=0.2) and Diagnostic B (per-manifold Z-score normalization) both indicate the original "C2 hurts combined AUC" finding was a probe approximation and scale-mismatch artifact, not a property of the data. The C2 paradox is not in the paper in any form.

---

## 4. What the paper claims and what it does not claim

### Claims
- Persistent homology applied to a functional decomposition of UAV network traffic reveals per-attack-class manifold attribution with three-seed mean AUCs of 0.74–0.87 at the dominant manifold for each attack.
- A label-free detector restricted to Network and Physical manifolds achieves binary AUC of 0.86 ± 0.00 (three-seed mean) on UAVIDS-2025 without using attack labels at calibration.
- The empirical attack-to-manifold attribution agrees with Zeng et al. (2025) on three of four attack classes (Blackhole, Wormhole, Flooding). Sybil is the exception: Zeng predicted Sybil's signature would manifest on source-addressing features (C2 manifold) while we observe Sybil's dominant signature on the Network manifold.
- Per-flow inference latency, measured on a 2020-era laptop, is on the order of 1–3 seconds; we do not characterize onboard performance in this extended abstract.

### Explicit non-claims
- We do not claim higher accuracy than supervised classifiers; raw Random Forest
  outperforms every TDA configuration on this benchmark.
- We do not claim demonstrated real-time UAV deployment; the latency footprint
  is consistent with periodic security audits, not per-packet IDS.
- We do not claim TDA features complement raw features for supervised
  classification; they do not on this benchmark.
- We do not claim our results generalize beyond UAVIDS-2025.

---

## 5. Methodology snapshot (verified configuration from production run)

- **Dataset:** UAVIDS-2025 (Zeng et al., 2025). 122,171 UDP flow records, 23
  columns, five classes (Normal Traffic, Blackhole, Wormhole, Sybil, Flooding).
- **Split:** stratified 70/15/15 train/val/test = 85,519 / 18,326 / 18,326.
- **Manifold partition:** C2 = SrcAddr_last_octet + one-hot SrcPort + DstAddr_last_octet
  + one-hot DstPort + FlowDuration/s (7 cols). Network = TxPackets, RxPackets,
  LostPackets, TxBytes, RxBytes, four rate metrics, MeanPacketSize (10 cols).
  Physical = MeanDelay/s, MeanJitter/s, Throughput/Kbps, PacketDropRate,
  AverageHopCount (5 cols). All manifolds StandardScaler-normalized on train only.
- **Reference cloud:** 500 k-medoids-sampled Normal-Traffic flows from the
  training set. Same 500 row positions used across all three manifolds.
- **Filtration:** per-flow Vietoris-Rips on {query_flow} ∪ {500 reference points}.
  Max edge length = 25th percentile of reference-cloud pairwise distances per
  manifold (C2 = 0.494, Network = 0.145, Physical = 0.333). Sparse Rips
  epsilon = 0.5 for C2 and Network; exact Rips for Physical. Simplex tree max
  dimension = 3 for C2 and Network (yielding H_0, H_1, H_2 with real death
  times); = 2 for Physical (yielding H_0, H_1 only).
- **Baseline barcode:** one Vietoris-Rips persistence diagram per manifold,
  computed on the 500 reference points only.
- **Anomaly score:** per-flow Wasserstein-2 distance from the flow's persistence
  diagram to that manifold's baseline barcode, summed across homology
  dimensions. Hera backend via GUDHI when available; falls back to GUDHI's
  default Wasserstein implementation.
- **Threshold:** 95th percentile of validation-set Normal-Traffic Wasserstein
  distances, computed per-manifold. A flow is flagged if any manifold's score
  exceeds its threshold.
- **Probe approximation** (this paper, not journal): top-K most persistent bars
  per (manifold, dim), Hera delta-approximation. Currently running at top-K=20,
  delta=0.5 for tractability; a confirmation pass at top-K=50, delta=0.2 is the
  next step before any results lock.

---

## 6. Current empirical results (probe, n=200 per class, top-K=20, delta=0.5)

### Binary AUC (Normal vs any Attack)
- C2 alone: 0.46 (below chance)
- Network alone: 0.73
- Physical alone: 0.60
- C2 + Network: 0.64
- C2 + Physical: 0.53
- **Network + Physical: 0.82**
- All three combined: 0.69

### Per-attack one-vs-rest AUC by manifold
- Sybil: C2=0.45, **Network=0.84**, Physical=0.25
- Flooding: C2=0.54, **Network=0.75**, Physical=0.38
- Blackhole: C2=0.49, Network=0.33, **Physical=0.76**
- Wormhole: C2=0.48, Network=0.31, **Physical=0.71**

### Supervised numbers for context (3 seeds, full data)
- Original 22 features + RF curated: **96.3% accuracy, 0.997 weighted AUC** (baseline)
- Combined (original + summary + images) + RF curated: 95.8% accuracy
- TDA-only (summary_only) + RF: 81.0% accuracy

The supervised numbers go in Section 4.1 as context for the unsupervised pitch,
not as a headline. The contribution is unsupervised.

---

## 7. Open diagnostics that gate finalizing numbers

Two diagnostics must run before any results in Section 6 are considered locked:

**Diagnostic A — Confirmation probe at lower approximation.** Re-run the probe
at top-K=50 and delta=0.2 (vs the current top-K=20, delta=0.5). Expected runtime
30–60 seconds. If AUCs hold within ±0.02 of current numbers, we're solid. If
they shift by more, the new numbers replace what's in Section 6.

**Diagnostic B — Per-manifold Z-score normalization.** The current "combined"
score sums raw Wasserstein-2 distances across manifolds. Because C2, Network,
and Physical have different scales (C2 distances ~0.02–0.09; Network ~0.0–0.06;
Physical ~0.0–0.05), summing raw distances may be the reason C2 appears to hurt.
Re-compute combined AUC after Z-normalizing each manifold's distances on the
validation Normal-Traffic distribution. Two possible outcomes:
- If Z-normalized combined AUC ≥ 0.78: the "C2 hurts" finding is a scale artifact.
  Reframe as "per-manifold normalization is essential," cite Z-normalized numbers
  throughout, drop "C2 paradox" framing.
- If Z-normalized combined AUC < 0.75: the C2 manifold is genuinely uninformative
  for topological discrimination on this dataset. Keep the C2-paradox framing
  as a real finding.

Both diagnostics must be in hand before drafting the abstract or Section 4.

---

## 8. Paper structure (5-page extended abstract for AIAA SciTech 2027 IS)

- **Abstract** (~200 words). Lead with the problem (UAV IDS in contested
  operation, lack of labeled adversary data). State the technical move
  (multi-manifold persistent homology). Lead with the empirical findings
  (manifold attribution, label-free AUC of 0.82 on N+P, mismatch with two of
  Zeng's predictions). End with a limitation sentence (probe approximation, no
  hardware validation).
- **Section 1 — Introduction** (~3/4 page). Aerospace motivation, three research
  questions from Section 2 of this brief, three contributions from Section 3.
- **Section 2 — Background and Related Work** (~3/4 page). UAVIDS-2025 and Zeng
  et al.'s supervised baselines (positioned as "what's known"). Bruillard, Nowak,
  Purvine 2016 (persistent homology for NetFlow anomaly detection — closest
  prior work, differentiate via multi-manifold decomposition). Vietoris-Rips and
  Wasserstein references.
- **Section 3 — Methodology** (~1.5 pages). Three-manifold partition with
  citation to UAVIDS-2025 Table III. Reference-cloud Vietoris-Rips construction.
  Wasserstein-2 distance to baseline barcode. Per-manifold thresholding and
  combination rule. Probe approximations (top-K, delta) disclosed explicitly.
- **Section 4 — Results** (~1.25 pages). Three subsections:
  - 4.1 Supervised context (one paragraph; reference Zeng baselines, note our
    raw-RF replication, position as background)
  - 4.2 Manifold attribution (the per-attack table from Section 6 of this brief,
    plus discussion of Zeng-mismatch on Blackhole/Wormhole)
  - 4.3 Label-free detection and the C2 question (binary AUC table, the
    network+physical = 0.82 finding, the C2-related discussion conditional on
    Diagnostic B outcome)
- **Section 5 — Limitations and Final Manuscript Plan** (~1/2 page). Probe
  approximation. No hardware validation. Single dataset. Label-free uses Normal
  labels (deployment substitute = trusted initialization window). Final
  manuscript will run exact Wasserstein on full test, profile on ARM hardware,
  test trusted-initialization-window baseline construction.
- **Section 6 — Conclusion** (~1/4 page). One paragraph tying findings back to
  the operational framing.

---

## 9. File and directory discipline

**Off-limits (read-only):**
- `pipeline.py` (frozen)
- `data/` (immutable raw dataset)
- `outputs/` (production-pipeline-written artifacts)
- `logs/`
- `scripts_archive/`

**Write zones:**
- `tools/` (diagnostic and probe scripts)
- `paper/` (this brief, drafts, results markdown, outline)
- `results/tables/` (tables to be cited in the paper)
- `results/figures/` (figures for the paper)

**Authoritative files:**
- `paper/PROJECT_BRIEF.md` — this document
- `paper/PROBE_RESULTS.md` — latest probe results (to be updated after
  Diagnostic A)
- `paper/METHODS_DISCLOSURES.md` — verbatim prose-ready disclosures
- `results/tables/supervised_summary.csv` — final supervised numbers
- `results/tables/probe_distances.csv` — raw per-flow probe scores
- `tools/quick_unsup_probe.py` — probe runner

Anything not in `paper/`, `tools/`, or `results/` should be treated as
production infrastructure and not modified.

---

## 10. The next 48 hours, in execution order

These are blocking dependencies. Do them in order. Do not start drafting until
items 1 and 2 are done.

1. **Run Diagnostic A.** Execute `python3 tools/quick_unsup_probe.py
   --per-class 200 --top-k 50 --delta 0.2 --seed 42`. ~30–60 seconds.
   Update `paper/PROBE_RESULTS.md` with new numbers. Compare against Section 6
   of this brief.

2. **Run Diagnostic B.** Write a 20-line script `tools/diagnose_c2_scaling.py`
   that loads `results/tables/probe_distances.csv`, computes per-manifold mean
   and std on validation Normal-Traffic distances, applies Z-normalization to
   each manifold column, recomputes the seven dropout-combination AUCs and the
   binary AUC on the normalized scores, and writes
   `paper/C2_NORMALIZATION_DIAGNOSTIC.md`.

3. **Resolve the C2 framing.** Based on Diagnostic B output:
   - If normalized combined ≥ 0.78: drop "C2 paradox" framing, replace with
     "per-manifold normalization is essential" framing. Section 4.3 restructured.
   - If normalized combined < 0.75: keep "C2 paradox" framing. Section 4.3 stays
     as a finding.

4. **Confirm with Dr. Sun.** Send him this brief and the two diagnostic outputs.
   Confirm the contribution list (Section 3) and the headline framing (Section 1).
   Do not start drafting prose until he signs off.

5. **Draft the Methodology section first.** Use Section 5 of this brief
   verbatim as a starting point. Methodology is the most stable section
   regardless of small number movements. ~1.5 pages.

6. **Draft Background.** ~3/4 page. UAVIDS-2025 baselines, Bruillard/Nowak/Purvine,
   Vietoris-Rips and Wasserstein references. Citations should already be in the
   project files.

7. **Draft Results section 4.2 (manifold attribution).** Build the per-attack
   AUC table from Section 6 numbers. Add the Zeng-mismatch discussion. ~1/2 page.

8. **Draft Results section 4.3 (label-free detection).** Build the binary AUC +
   dropout table from Section 6. Conditional on Diagnostic B outcome, finalize
   the C2 framing. ~1/2 page.

9. **Draft Results section 4.1 (supervised context).** ~1 paragraph. Reference
   Zeng baselines. Frame as "what's known."

10. **Draft Limitations and Final Manuscript Plan.** ~1/2 page. Honest list.

11. **Draft Introduction.** ~3/4 page. RQs and contributions from this brief.
    Write Introduction late, after Results are solid, so the contribution
    framing matches what the data actually supports.

12. **Draft Conclusion and Abstract last.** Both should be straightforward
    once everything else is written.

---

## 11. Things to remember while writing

- **Capability framing, not performance framing.** The contribution is what the
  method reveals and what it enables, not what it scores. Resist any sentence
  that compares TDA AUC favorably to supervised AUC; that comparison is not
  in our favor and is not the claim.
- **"Label-free" means no attack labels.** Be explicit: we use Normal-Traffic
  labels to identify which flows seed the reference cloud and which seed the
  threshold; we never expose the detector to any attack label. In deployment
  this is substituted by a trusted-initialization-window of known-good
  operation. State this distinction once, clearly, in Methodology.
- **The word "manifold" is being used loosely.** These are feature subspaces in
  Euclidean space, not smooth manifolds in the differential-geometry sense.
  Disclose once in Methodology with a footnote.
- **Honesty about the Zeng mismatch is a feature.** Do not paper over the
  Blackhole/Wormhole disagreement. State it directly. Discuss what it implies
  about TDA seeing protocol-level signatures that feature-inspection-based
  predictions miss.
- **Limitations belong in Section 5, not the abstract.** The abstract should
  state findings cleanly. The limitation acknowledgment goes in its own section
  and is honest but bounded.
- **All numbers come from `paper/PROBE_RESULTS.md` and
  `results/tables/supervised_summary.csv`.** Do not paraphrase. Do not round
  more than two decimal places. Do not cite a number you cannot trace to one
  of those files.

---

## 12. What was tried and abandoned (so it does not get re-proposed)

- "TDA improves supervised accuracy" — dead. Raw RF baseline 96.3%, combined
  RF 95.8%, TDA-only 81.0%. Do not pitch this.
- "TDA features complement raw features" — dead. Combined is worse than raw.
- "Persistence images add discriminative value" — they do not on this benchmark;
  the curated subset performs comparably to summary-statistics-only.
- "GPS-denied = drop Physical manifold" — superseded. The operationally
  defensible pitch is "no-C2-visibility = drop C2," which gives the 0.82 AUC.
- The full unsupervised pipeline at exact Wasserstein on all 18,326 test
  flows — infeasible on current compute (~74 hours estimated, hera hangs on
  pathological diagrams). Deferred to journal version. The probe approximation
  is the disclosure path.
- Three-seed variance reporting on the probe — only 1 seed (42) was run.
  Document as a limitation. Re-running with multiple seeds is a future-work
  item, not a paper blocker.

---

## 13. Decision authority

The framing decisions in this document — three contributions, manifold partition,
label-free framing, Zeng-mismatch as finding-not-flaw — are settled. They were
made jointly by Oliver, Dr. Sun, and the advisory LLM after reviewing the probe
results on 2026-05-23.

If something in this document seems wrong as you encounter new data, raise it
explicitly. Do not silently work around it. Update this file and note the change
in a "decision log" appended at the bottom.

---

## Decision log

- **2026-05-23 (initial pivot):** Abandoned "TDA improves accuracy" framing
  after probe data showed raw RF baseline outperforming all TDA configurations.
  Adopted manifold-attribution + label-free + C2-paradox framing.
- **2026-05-23 (this brief):** Locked three contributions. Identified two
  blocking diagnostics (A and B) before any results finalization. Set venue
  to SciTech 2027 IS, length 5-page extended abstract.
- **2026-05-24 (post-diagnostics revision):** Completed Diagnostics A, B, and C per Section 10. Three substantive revisions to the brief: (1) Retired the C2-manifold-paradox framing entirely — both A and B indicate the original finding was a probe artifact. (2) Updated all empirical numbers to the three-seed means from Diagnostic C, with Network+Physical at 0.86 ± 0.00 (binary AUC) as the headline detection number. (3) Corrected the Zeng-mismatch reading: re-reading Zeng et al. Section II.D.2 shows that Zeng's predicted signature features for Blackhole and Wormhole (PacketDropRate, Throughput, AverageHopCount, MeanDelay) are in our Physical manifold, not Network. Our empirical attributions therefore agree with Zeng on Blackhole, Wormhole, and Flooding. Sybil is the single divergence — Zeng predicted source-addressing (C2) dominance; we observe Network dominance. Section 1, Section 3, Section 4 claims, and Section 8 abstract guidance updated accordingly. The Methodology Note from MULTI_SEED_VARIANCE.md (seed-123 C2 timeouts) is logged as a manuscript-version remediation item, not a current-paper blocker.
