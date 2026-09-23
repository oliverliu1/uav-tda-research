# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for a paper on **multi-manifold persistent homology (TDA) for UAV intrusion detection** on the UAVIDS-2025 benchmark, targeting AIAA SciTech 2027. The code is experiment infrastructure, not a product. Understanding the *research framing* matters as much as the code — read `PROJECT_BRIEF.md` before making claims or changing analysis. It records what the paper does and does not claim, decisions that are settled (do not relitigate them), and approaches already tried and abandoned (§12) so they don't get re-proposed.

The headline framing (as of the 2026-05-24 decision log): the "TDA improves supervised accuracy" story is **dead** (raw Random Forest already hits 96.3%). The three surviving contributions are a **label-free Wasserstein anomaly detector**, **per-attack manifold attribution**, and **agreement/divergence with Zeng et al. (2025)** feature-level predictions.

## Running the pipeline

The `uav_tda` package is the maintained implementation. `pipeline.py` is retained verbatim as the
frozen equivalence oracle — every `uav_tda` phase is tested against it (debug mode, hermetic
workspaces) — and as the historical record of the paper's production run. Do not modify `pipeline.py`;
see "File discipline". Phases are `uav-tda` argparse subcommands, run in order:

```bash
uav-tda prep           # Phase 2: load CSV, stratified 70/15/15 split, per-manifold StandardScaler, write outputs/*.csv
uav-tda tda             # Phase 3: per-flow Vietoris-Rips persistence diagrams vs reference cloud
uav-tda features        # Phase 4: summary stats (8 per manifold/dim) + persistence images
uav-tda supervised      # Phase 5: grid-searched LogReg/RF/SVM over 4 feature sets, curated RF
uav-tda unsupervised    # Phase 6: Wasserstein-2 distances, thresholding, per-attack AUC
uav-tda evaluate        # Phase 7: ablations, final tables, paper figures
uav-tda all             # every phase in order
```

Additional analysis commands (not part of the ordered `prep`..`evaluate` pipeline above):

```bash
uav-tda probe [--seed N] [--per-class N] [--top-k N] [--delta F] [--w2-timeout F] [--out PATH] [--znorm]
                                                         # Run the unsupervised Wasserstein-2 probe (the tractable approximation to `unsupervised`); --znorm runs the coupled test+val Z-normalized scoring (paper Sec III.E) instead of the raw sum-scored probe
uav-tda znorm-report                                    # Build the 3-seed znorm-vs-raw reanalysis report, write paper/ZNORM_RESULTS.md
uav-tda manuscript-report [--seeds S] [--bootstrap B] [--w2-timeout F]
                                                         # Phase 4: ensure the ten-seed campaign artifacts and build bootstrap-CI stats tables, write paper/MANUSCRIPT_STATS.md
uav-tda windowed --w W [--order-seed N] [--repeat K]  # Phase 5: run one windowed-variant campaign-grid entry (time-windowed multi-manifold persistence; see docs/superpowers/specs/2026-09-21-windowed-variant-design.md)
uav-tda windowed-report [--bootstrap B]                 # Phase 5: ensure the 80-run windowed campaign, build detection/attribution/contamination/frontier tables, write paper/WINDOWED_RESULTS.md
uav-tda exact [--n-jobs N] [--shard-size N]             # Phase 6 Track A: sharded resumable high-precision (delta<=0.01) Wasserstein-2 campaign over the full test split (see uav_tda/exact.py)
uav-tda exact-report [--bootstrap B]                    # Phase 6 Track A: assemble the exact-W2 campaign shards into definitive AUC tables, write paper/EXACT_RESULTS.md
uav-tda latency [--n N]                                 # Phase 6 Track B: portable onboard-latency harness (full standardize->Rips->slice->W2 path, per-flow + windowed arms), write paper/LATENCY_RESULTS.md
```

Shared flags on every subcommand: `--debug` (fast stratified-sample smoke test — use this for any quick
iteration), `--root PATH` (workspace root, default: repo root via `Workspace.default()`). `tda`, `unsupervised`, and `all` also take `--n-jobs`; `tda` and `all` also take `--manifold {c2,network,physical,all}`, `--split {train,val,test,all}`, and `--seed`.

**Legacy, equivalent:** the original monolith subcommands (`python pipeline.py prep`, `tda`, `features`,
`supervised`, `unsupervised`, `evaluate`, `all`, with `--debug`/`-v/--verbose` and the same `tda` flags)
still work identically against `pipeline.py` and remain the oracle these are checked against.

There is no dedicated pipeline.py-level test suite, but `uav_tda` phases each carry equivalence tests
against `pipeline.py`'s debug-mode output. Validation is via `validate_*` / `assert_*` functions
embedded in each phase that raise on invariant violations (disjoint manifolds, class balance, no NaNs,
expected feature counts). Full runs are slow — the unsupervised phase at exact Wasserstein on all test
flows is infeasible (~74h); a probe approximation is used instead.

Dependencies: `pip install -r requirements.txt`. Note `gudhi` (Rips + Wasserstein, uses the Hera backend when available) and `persim` (PersistenceImager) are the load-bearing TDA libraries.

## Architecture

**Three disjoint feature manifolds** partition the 22 encoded columns (defined in `pipeline.py` §3, `MANIFOLDS`):
- **C2** (7 cols): source/dest last-octet + one-hot ports {9, 654} + flow duration — the "connection" manifold.
- **Network** (10 cols): packet/byte counts and rates, mean packet size — "traffic volume."
- **Physical** (5 cols): delay, jitter, throughput, drop rate, hop count — "performance/physical proxy."

Each is scaled independently on train only. The partition mirrors Zeng et al. (2025) Table III; "manifold" is used loosely (feature subspaces, not smooth manifolds).

**Data flow** is file-based between phases, all under `outputs/` (gitignored, reproducible from seed 42 + the CSV): `prep` writes `{c2,network,physical,original_features}_{train,val,test}.csv` + labels + scalers → `tda` writes persistence diagrams (`outputs/persistence_diagrams/`) → `features` writes `outputs/tda_features/` → `supervised`/`unsupervised` read those and write `results/tables/` + `results/figures/`.

**TDA method:** a 500-point k-medoids reference cloud of Normal-Traffic training flows (same row positions across manifolds). Per query flow, Vietoris-Rips filtration on `{flow} ∪ {500 reference points}`; `max_edge = 25th percentile` of reference pairwise distances (config note in §4 explains why p25 not the spec's p95 — tractability). Sparse Rips ε=0.5 for C2/Network, exact for Physical. Anomaly score = Wasserstein-2 distance from the flow's diagram to the manifold's baseline barcode; threshold = 95th percentile of validation Normal distances; flagged if any manifold exceeds its threshold.

**Config lives at the top of `pipeline.py` (§1–4)** — paths, dataset schema, manifold definitions, all hyperparameters, `SEEDS = (42, 7, 2024)`, `PRIMARY_SEED = 42`. Change experiment parameters there, not scattered in function bodies.

**Dataset quirks** (see `RESEARCH_CONTEXT.md` §2): Protocol is all-UDP (dropped), FlowID is a meaningless row index (dropped), Normal records are interleaved not contiguous (baselines must filter by label, never row position), only two ports exist.

## File discipline (from PROJECT_BRIEF §9)

**Read-only / frozen — do not modify:** `pipeline.py` (frozen equivalence oracle and historical record — do not modify), `data/` (immutable raw dataset), `outputs/` (pipeline-written), `logs/`, `archive/` (quarantined legacy code — `poster_eda/`, `scripts_archive/`, `poster.jsx`; read-only history, not maintained; `archive/poster_eda/` in particular contains pre-fix backups with a **known data-leakage flaw** — never use it for results, see `archive/README.md`).

**Write zones:** `uav_tda/` (the maintained package), `tests/`, `docs/` (plans and design notes, e.g. `docs/superpowers/plans/`), `paper/` (brief, drafts, diagnostics markdown), `results/tables/`, `results/figures/`, and `tools/` (diagnostic/probe scripts — note this dir may not exist yet and is created as needed).

**Number provenance:** every number cited in the paper must trace to `paper/PROBE_RESULTS.md` or a `results/tables/*.csv`. Do not paraphrase or invent results.

**Ops lessons (memory-pressured box):** chunk worker payloads instead of closing over large arrays/diagrams in the parent process (a re-pickled multi-hundred-MB closure per task exhausts memory before it exhausts CPU); keep `n_jobs<=3` for heavy campaigns (Rips/W2 at scale) even though more cores are visible; never fork inside a `loky` worker (nested `subprocess`/fork-based timeouts inside a joblib/loky worker are unstable on macOS — call the underlying library directly in-process instead); commit before launching any long detached run so the launched process's code state is reconstructable if the session ends first.
