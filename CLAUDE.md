# CLAUDE.md

Agent instructions for this repository. Humans start at `README.md`; the research framing,
settled decisions, and dated decision log are in `CLAUDE/PROJECT_BRIEF.md`; the current
manuscript-readiness status is `CLAUDE/STATUS.md`. Read the brief before making or changing
any scientific claim. Do not relitigate decisions recorded there.

## What this is

Research code for **multi-manifold persistent homology (TDA) for UAV intrusion detection** on
UAVIDS-2025, targeting AIAA SciTech 2027 (extended abstract submitted; full manuscript due
December 2026; LaTeX source in `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex`). Experiment
infrastructure, not a product. The "TDA improves supervised accuracy" framing is **dead** (raw RF
hits 96.3%). The paper claims label-free Wasserstein-2 detection, per-attack manifold attribution,
and 3-of-4 agreement with Zeng et al. (2025). Infrastructure work is complete (Phases 1–7, PRs
#11–#17); remaining work is manuscript-side.

## Environment (pinned; do not upgrade)

anaconda Python 3.9.7 running under **Rosetta 2** on an Apple M1 Pro, 16 GB RAM, chronically
memory-pressured, ~10 GB free disk. numpy 1.20.3, gudhi 3.11.0 (conda-only; PyPI resolves 3.10.1),
scikit-learn 1.3.2. Install with `pip install -e ".[dev]"` (pyproject bounds are correct).
**Do not use `requirements.txt`** — its pins would upgrade numpy/pandas past the working env.
No new runtime dependencies (e.g. POT) without an explicit decision.

## Commands

```bash
make test        # fast suite (pytest -m "not slow"), ~2 min, 162 tests — run before claiming anything works
make test-all    # includes slow monolith-equivalence tests (needs .cache/monolith-debug/)
make lint        # ruff, safety tier
uav-tda --help   # 16 subcommands; README "Quickstart" + "Results docs" list which command regenerates which paper/*.md
uav-tda artifacts [--root PATH]   # Phase 8: figures/tables/snippets + paper/MANUSCRIPT_ARTIFACTS.md from committed CSVs, no recompute
```

Every subcommand supports `--debug` (fast stratified smoke) except `probe`, `znorm-report`,
`manuscript-report`, `artifacts`. `outputs/` (32 GB, gitignored) already holds the prep CSVs and per-flow
persistence diagrams; `uav-tda tda` never needs re-running for probe/exact/windowed/latency.

## File discipline

**Frozen — never modify:** `pipeline.py` (byte-frozen equivalence oracle + historical record of
the paper's production run), `tools/` (recovered originals, verbatim), `data/`, `outputs/`,
`logs/`, `archive/` (quarantined legacy; `archive/poster_eda/` has a **known data-leakage flaw**,
see `archive/README.md`), the published `results/tables/*.csv` at top level (golden-master
oracles), and the historical `paper/{PROBE_RESULTS,MULTI_SEED_VARIANCE,C2_NORMALIZATION_DIAGNOSTIC}.md`
and `paper/historical/`.

**Write zones:** `uav_tda/`, `tests/`, `docs/`, `paper/` (new docs only), `CLAUDE/`,
`results/tables/rebuild/` (all new tables; every artifact gets a `.provenance.json` sidecar),
`results/figures/manuscript/` (generated manuscript figures, each with a sidecar).
Golden-master `metrics.py` functions and the probe default path must not change behavior.

## Scientific rules

- **Number provenance:** every number cited anywhere must trace to a committed CSV or
  machine-generated doc. Current candidates: `paper/{EXACT,WINDOWED,LATENCY}_RESULTS.md`,
  `paper/MANUSCRIPT_STATS.md`, `paper/ZNORM_RESULTS.md`, `paper/MANUSCRIPT_ARTIFACTS.md` —
  **all pending author sign-off**.
- **Canonical score** = val-Normal z-normalized **SUM** across manifolds (`metrics.apply_znorm`,
  `probe.run_probe_with_znorm`). `uav_tda/unsupervised.py` keeps the monolith's max-pool as a
  frozen port only. Val-stats and test scores must come from the same baseline realization.
- **Seeds:** `PROBE_SEEDS=(42,7,123)` (paper §IV), `SUPERVISED_SEEDS=(42,7,2024)` (monolith
  supervised tables), `MANUSCRIPT_SEEDS` = ten seeds, `PRIMARY_SEED=42`. Live config is
  `uav_tda/config.py`, not `pipeline.py`.
- **Sparse Rips (GUDHI 3.11, ε=0.5) is nondeterministic per call.** Per-flow C2/Network and
  windowed Network diagrams are statistically, not bit, reproducible; exact-Rips surfaces
  (per-flow Physical; windowed C2/Physical) are bit-deterministic. Compare with calibrated
  tolerances, never equality. Manuscript disclosure item.
- "Exact" Wasserstein means hera **δ≤0.01**; literal δ=0 is intractable (13/13 sampled flows
  timed out, ~30-day projection). The dataset has **no timestamps**; FlowID order is the
  temporal proxy (disclosed).

## Ops lessons (memory-pressured box)

Chunk worker payloads; never closure-capture large arrays/diagram dicts into joblib tasks.
`n_jobs<=3` for heavy campaigns. Never fork inside a loky worker (call hera in-process).
Commit all code before launching any detached run; detach anything >10 min; shard + manifest
for resumability. Check free disk before any `tda`/`features` rebuild. Git from this Python must go through
`uav_tda.provenance.run_git` (plain `git` fails under Rosetta; sidecars used to record "unknown").

## Process

Superpowers workflow: brainstorm → spec → TDD plan → subagent-driven implementation with
reviewer, final whole-branch review. Plans/specs live in `docs/superpowers/`. Append dated
entries to the decision log at the bottom of `CLAUDE/PROJECT_BRIEF.md` for any decision that
changes framing, numbers, or file discipline.
