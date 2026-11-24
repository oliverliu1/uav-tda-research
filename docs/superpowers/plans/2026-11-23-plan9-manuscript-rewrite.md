# Plan 9: full-manuscript rewrite of `main.tex`

**Status:** approved by the author 2026-11-23. Phase 0 done (working copy, numbers file, build).

**Goal.** Turn the submitted extended abstract
(`CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`, 319 lines, about 10 pages) into the full
SciTech 2027 manuscript, using only numbers from the six results docs signed off 2026-11-23 (as
committed at `68dae1b`).

**Inputs.** `CLAUDE/MANUSCRIPT_READINESS_AUDIT.md` §3 (claim-by-claim status of the abstract),
`CLAUDE/PROJECT_BRIEF.md` §1–§8 (framing, settled decisions, writing reminders), the six
`paper/*.md` results docs, six figures in `results/figures/manuscript/`, and the snippets in
`results/tables/rebuild/paper_snippets/`.

**Rules that bind every section** (from the brief; not reopened here):

- Capability framing. No sentence compares TDA AUC favourably with supervised accuracy.
- Canonical score is the val-Normal z-normalized sum over manifolds. All three manifolds is the
  best subset; the "C2 adds noise" story is retired.
- "Exact" becomes "high-precision (hera δ≤0.01)".
- 0.9082 is quoted with the 250-flow block interval [0.885, 0.930]. Windowed AUCs show the
  block-10 interval beside the independent-window one.
- 0.9082 pairs with the full-diagram latency arm; 0.8930 pairs with the probe arm.
- Windowed is never claimed to beat per-flow on accuracy (intervals overlap).
- Prose carries at most two decimals for AUCs; tables may carry four.

---

## 1. Section-by-section plan

Legend: **Keep** (light edits), **Fix** (correct a wrong statement), **Rewrite**, **New**.

### Front matter

| Part | Action | Change |
|---|---|---|
| Title | Keep | "Label-Free UAV Intrusion Detection via Multi-Manifold Persistent Homology". Optional subtitle on the per-flow / windowed trade-off; author choice. |
| Authors, footnotes | Keep | Confirm affiliations are still current. |
| Abstract | Rewrite | Replace N+P 0.86 ± 0.00 with all-three 0.91 on the full test split; attribution range 0.73–0.89; add the windowed variant (0.87–0.94 across W, order-dependent, per-decision cost two to three orders of magnitude lower at W=25); add measured latency on a Raspberry Pi 5; drop "the full manuscript will…". Limitations stay out of the abstract. |

### I. Introduction

| Part | Action | Change |
|---|---|---|
| Motivation (¶1–3) | Keep | Check three background claims flagged by the audit: HX-2 wording ("immune to electronic warfare" is vendor language), the `Sareen` citation, "Department of War" (parenthetical on first use). |
| Approach (¶4) | Fix | Add that scores are z-normalized on validation Normal traffic and that a time-windowed variant is also evaluated. |
| Findings (¶5) | Rewrite | New headline numbers; all-three is best; windowed result; Sybil moves to C2 at window level. |
| Research questions | Rewrite | Keep RQ1–RQ3; add RQ4: how do per-flow and time-windowed variants trade detection AUC, attribution resolution and per-decision compute? "Extended abstract" becomes "paper". |
| Roadmap | Fix | New section list. |

### II. Background and Related Work

| Part | Action | Change |
|---|---|---|
| A. UAV intrusion detection | Keep | No number changes. |
| B. Persistent homology for cybersecurity | Fix | `Davies` is cited as both the original argument and a "subsequent review"; fix. Note that Bruillard et al. used time windows, which positions our windowed variant. |
| C. Multi-manifold decomposition | Fix | "23 features" becomes 22 attributes plus the label. |

### III. Methodology

| Part | Action | Change |
|---|---|---|
| Pipeline figure | Fix | Node "21 features" becomes 20 raw / 22 encoded; caption "summed" becomes "z-scored and summed". |
| A. Dataset | Fix | 22 attributes plus label. Add: no timestamps, so FlowID order is the temporal proxy; Normal and Blackhole rows interleave while the other attacks form long contiguous blocks. |
| B. Three-manifold decomposition | Fix | "21 retained features" becomes 20 raw, 22 after encoding (7 + 10 + 5). One footnote on the loose use of "manifold". |
| C. Reference cloud | Fix | Disclose that k-medoids runs in C2 space and the same 500 rows are reused for Network and Physical; one reference set (seed 42) is used throughout. State once what "label-free" means and the trusted-initialization substitute in deployment. |
| D. Per-flow Vietoris–Rips | Keep + add | Add: why the 25th-percentile cap (the larger cap was intractable); GUDHI sparse Rips is nondeterministic per call, so C2 and Network diagrams are statistically, not bit, reproducible. |
| E. Anomaly scoring | Fix + rewrite | Ground metric is L2, not L∞. New equation for the z-normalized sum, replacing the raw sum as the reported score. Threshold at the 95th percentile of validation-Normal scores (now actually measured). |
| F. Time-windowed variant | **New** | Non-overlapping FlowID-order windows, W ∈ {25, 50, 100, 200}; one Rips complex per window and manifold on the window's own points; W2-medoid validation-Normal baseline; exact Rips for C2 and Physical, sparse for Network; window label = majority class; shuffle control. |
| G. Evaluation protocol | Rewrite (replaces "Computational Approximation") | Three tiers: as-submitted 3-seed probe; 10-seed probe (200 flows per class, top-K 50, δ=0.2); full test split at δ≤0.01 with no sampling or truncation. δ=0 intractability (13 of 13 sampled flows timed out). Seeds control flow sampling only (correcting the abstract's k-medoids and tiebreaking claim). Bootstrap intervals and the block bootstrap for correlated flows and windows. |
| H. Latency measurement | **New** | Full path per decision (standardize → Rips → slice → W2), n=30, two per-flow configurations, two hosts; baselines built on the timed machine. |

### IV. Results

| Part | Action | Change | Figure / table |
|---|---|---|---|
| A. Supervised context | Fix | 22-feature input; otherwise unchanged (RF 96.3%; topological features do not help). | none |
| B. Label-free detection, per-flow | Rewrite | All-three 0.91 [0.885, 0.930] on 18,326 test flows; N+P 0.87 when C2 is unavailable; z-norm vs raw (0.9082 vs 0.9037); 10-seed probe 0.89 ± 0.01 as the sampling-stability study, labelled a different estimand. Operating point: FPR 0.056, TPR 0.57 overall, 0.91 Sybil, 0.82 Flooding, 0.34 Blackhole, 0.30 Wormhole. Short paragraph on what changed from the abstract and why. | Subset bars from `exact_binary_auc_pgfplots.tex` (replaces Fig. 2); `fig_roc.pdf` |
| C. Per-attack attribution, per-flow | Rewrite | Full-test table (dominant cells 0.89 / 0.79 / 0.79 / 0.73); replicates at 3 and 10 seeds; cells well below 0.5 are anti-correlated, not uninformative; Zeng agreement on three of four. | Table from `exact_attribution_rows.tex` (replaces Table 1) |
| D. Time-windowed variant | **New** | Detection by W (0.87 / 0.90 / 0.94 / 0.91) with both intervals; shuffle control drops AUC to 0.53–0.59; window AUC uses majority labels, so it is not one-to-one with flow AUC; contamination curve; window-level attribution: Sybil → C2 at every W (0.92–0.93), nine of sixteen cells unresolved, Blackhole near chance. | `fig_detection_vs_w.pdf`, `fig_contamination.pdf`, `fig_attribution.pdf` |
| E. Compute cost and latency | **New** | Two-host table: per-flow probe 3.58 s (Mac) / 8.36 s (Pi 5); 0.9082 configuration 5.79 s / 10.55 s; windowed 7 ms to 1 s per decision. Speedups reported per host (W=25: 478× Mac, 1,148× Pi; W=100: 31×, 51×) with their scope caveats. C2 Rips is about 85% of per-flow cost. Corrects the abstract's "1–3 s". | `fig_frontier.pdf` (lead figure), `fig_latency.pdf`, one new table |

### V. Discussion — **New**, short

- The Sybil divergence: per-flow topology finds Sybil on Network, while windows find it on C2,
  which is where Zeng et al. predicted. Offered as a hypothesis that addressing structure only
  becomes visible across many flows. Wording needs co-author agreement.
- Which variant suits which use: per-flow gives flow-level attribution at seconds per decision;
  windows give milliseconds per decision and coarser attribution.
- What the Pi 5 numbers do and do not show (one board, mains power, Python reference code).

### VI. Limitations and Future Work — Rewrite

Remove the four abstract-era limitations that are now resolved (probe approximation, scale
heterogeneity, seed-123 timeout, pending extensions). Replace with:

1. Single simulated benchmark; cross-benchmark evaluation is future work (rewording the promise).
2. No timestamps; FlowID order stands in for time, and the windowed result depends on it.
3. Correlated flows and windows: intervals are block-bootstrap and indicative (few contiguous runs).
4. High-precision, not exact, Wasserstein.
5. One fixed reference set; sparse-Rips nondeterminism.
6. Weak detection of Blackhole and Wormhole at the 5% false-positive operating point.
7. Latency scope; hera can fail to terminate on diagrams that differ only by rounding, so
   baselines must be built on the machine that runs the detector.

### VII. Conclusion — Rewrite

Three contributions plus the windowed comparison, with the new numbers. No forward references
to a later manuscript.

### Back matter

| Part | Action | Change |
|---|---|---|
| Acknowledgments | Fix | "The present extended abstract" becomes "this paper". AI-use statement: confirm the wording still describes how the work was done. |
| Bibliography | Fix | `Davies` (prints its abstract), `Helsing` / `Harper` / `Bendett` (no venue), `Sareen` (verify or replace with `Kronstadt`), `Cavanna` (empty note), `Eroglu` / `Kaufman` (entry type), access dates on web sources; drop or use the six uncited entries. |
| Appendix (optional) | New | δ-insensitivity table, full seven-subset tables, reproducibility statement (commit, environment, commands). |

---

## 2. Cross-cutting work

1. **Number provenance.** Done for prose: `uav-tda manuscript-numbers` writes
   `paper_snippets/manuscript_numbers.tex` (one `\newcommand` per cited number, with its CSV cell),
   and a test fails if the committed file drifts from the CSVs. It is a separate command, not part
   of `uav-tda artifacts`, so the signed-off artifacts report is not re-rendered. Row snippets for
   the new tables are added in Phase 2 with the tables.
   Known wiring issue for Phase 2: `exact_binary_auc_pgfplots.tex` starts with `coordinates {`
   and ends with `\node` lines, so it cannot simply be `\input` after `\addplot`; either split it
   or draw the subset bars as a generated PDF.
2. **Build.** Done: `make manuscript`. `compat` lowered to 1.17, which also builds on newer TeX
   Live. The copy builds to 10 pages with zero undefined references. A self-contained upload
   bundle (flattened inputs) is a Phase 4 item.
3. **Brief housekeeping.** `PROJECT_BRIEF.md` §3 still says companion-class numbers do not exist
   and cites 3.6 s only; update the body when the latency section is written.

## 3. Order of work and checkpoints

| Phase | Content | Author checkpoint |
|---|---|---|
| 0 | Manuscript location, numbers file, build check | Approve the mechanism |
| 1 | §III Methodology and pipeline figure | Review the methods text |
| 2 | §IV Results with figures and tables | Review results and the Sybil wording |
| 3 | §V–§VII, abstract, introduction | Review framing with co-authors |
| 4 | Bibliography, appendix, final number audit against the CSVs | Final read |

Each phase is one branch and pull request.

## 4. Decisions (made 2026-11-23)

1. **Where the manuscript lives:** `CLAUDE/SciTech2027_Manuscript/`; the submitted abstract
   directory stays untouched.
2. **How numbers enter the text:** generated macros and snippets.
3. **Venue constraints:** AIAA format, the template already in use, no page limit. The exact
   December deadline is still not recorded.
4. **Discussion** is its own short section.
