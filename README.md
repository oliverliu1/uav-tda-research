# Multi-Manifold Persistent Homology for UAV Intrusion Detection

Research codebase for a paper (target: AIAA SciTech 2027) applying persistent
homology to three independent feature manifolds — C2, Network, Physical —
extracted from UAV network traffic (UAVIDS-2025). Three surviving
contributions: a label-free Wasserstein anomaly detector, per-attack manifold
attribution, and agreement/divergence with Zeng et al. (2025)'s feature-level
predictions. See `CLAUDE/PROJECT_BRIEF.md` for the full research framing,
decisions, and abandoned approaches.

The `uav_tda/` package is the maintained implementation. `pipeline.py` is
retained, unmodified, as the frozen equivalence oracle every `uav_tda` phase
is tested against (and as the historical record of the paper's production
run) — see `CLAUDE/CLAUDE.md` for details. Do not modify either.

## Install

```bash
pip install -e ".[dev]"     # python >=3.9 (developed against anaconda 3.9.7)
```

## Quickstart

```bash
make test              # fast suite: python -m pytest -m "not slow" -q (~103 tests, ~1 min)
make lint               # ruff check .
uav-tda --help           # full subcommand list
uav-tda probe --seed 42  # one-command repro: writes results/tables/rebuild/probe_distances_seed42.csv
```

Each `probe` run writes a `.provenance.json` sidecar (seed, git SHA, library
versions). The published `results/tables/probe_distances*.csv` are the
immutable as-submitted record and are never overwritten by reruns.

Full pipeline phases (`prep`, `tda`, `features`, `supervised`, `unsupervised`,
`evaluate`, or `all` in order) and analysis commands (`windowed`,
`windowed-report`, `exact`, `exact-report`, `latency`, `znorm-report`,
`manuscript-report`) are documented in `CLAUDE/CLAUDE.md`. Run
`uav-tda <command> --help` for any subcommand's flags.

## Dataset

`data/UAVIDS-2025.csv` (122,171 rows × 23 cols, Zeng et al., IEEE CNS 2025) is
immutable and tracked in git (~19MB) — it ships with every clone. See
`data/README.md` for provenance and the sha256 to verify any copy against.
Prerequisites for most commands: `data/UAVIDS-2025.csv` present, then
`uav-tda prep && uav-tda tda`.

Note: this is a third-party benchmark dataset; whether to continue
redistributing it in this repository is an author decision.

## Results docs

`paper/` also holds three historical provenance docs, kept as the
as-published record rather than regenerated: `PROBE_RESULTS.md` (the
published probe record `CLAUDE/CLAUDE.md`'s number-provenance rule points
at), `MULTI_SEED_VARIANCE.md`, and `C2_NORMALIZATION_DIAGNOSTIC.md`.

Alongside those, five machine-generated reports under `paper/`, each built
from committed `results/tables/rebuild/*.csv`. **All pending author sign-off**
— no number in any of them is yet approved for the manuscript:

- `paper/EXACT_RESULTS.md` — full-test-split, high-precision (δ≤0.01)
  Wasserstein-2 campaign (Phase 6 Track A).
- `paper/LATENCY_RESULTS.md` — portable onboard-latency harness, per-flow and
  windowed arms (Phase 6 Track B).
- `paper/WINDOWED_RESULTS.md` — time-windowed multi-manifold persistence
  variant, detection/attribution/contamination/frontier tables (Phase 5).
- `paper/MANUSCRIPT_STATS.md` — ten-seed bootstrap-CI evaluation of the probe
  approximation (Phase 4).
- `paper/ZNORM_RESULTS.md` — clean-lineage Z-normalized-scoring reanalysis
  vs. raw scoring, 3 seeds (Phase 3).

## Disk

`outputs/` is pipeline-written and gitignored, currently ~32G of regenerable
artifacts (persistence diagrams, features, scalers). `uav-tda prep && uav-tda
tda` (plus downstream phases as needed) rebuilds it from `data/` + seed 42.
Deleting it locally to reclaim space is a local decision — nothing in `results/`
or `paper/` depends on it surviving on disk.

## Archive

`archive/` holds quarantined legacy code (`poster_eda/`, `scripts_archive/`,
`poster.jsx`) — read-only history, not maintained, not safe to run. See
`archive/README.md`, in particular the data-leakage warning on
`archive/poster_eda/`.

## CI

`.github/workflows/ci.yml` runs `ruff check .` and the fast test suite
(`pytest -m "not slow"`) on push/PR to `main`, python 3.9, ubuntu-latest.
