# Multi-Manifold Persistent Homology for UAV Intrusion Detection

**Anomaly Detection in Contested ISR Military Drone Swarms**

This repository contains the implementation of a novel intrusion detection system for UAV networks using multi-manifold topological data analysis (TDA) with persistent homology.

## Overview

We apply persistent homology to three independent feature manifolds (C2, Network, Physical) extracted from UAV network traffic data. Topological features are computed via Vietoris-Rips filtration and combined with traditional ML classifiers to detect intrusion attacks.

**Dataset**: UAVIDS-2025 benchmark (122,171 network flow records, 5 attack types)

**Key Innovation**: Multi-manifold TDA pipeline that captures topological structure across command-control, network traffic, and physical proxy spaces.

## Reproducing the paper's results

The `uav-tda` CLI (installed via `pip install -e ".[dev]"`) is the maintained way to run the pipeline.
`pipeline.py` is retained, unmodified, as the frozen equivalence oracle each `uav_tda` phase is tested
against, and as the historical record of the paper's production run — it is not the recommended entry
point going forward.

Prerequisites: `data/UAVIDS-2025.csv` present, and Phase-2/3 artifacts built
(`uav-tda prep && uav-tda tda`).

```bash
pip install -e ".[dev]"
python -m pytest -m "not slow"          # fast: config + metrics locked to published oracle CSVs
uav-tda probe --seed 42                  # writes results/tables/rebuild/probe_distances_seed42.csv
uav-tda probe --seed 7
uav-tda probe --seed 123
python -m pytest -m slow                 # verifies rebuilt AUCs match the paper within seed variance (±0.05)
```

Each run writes a `.provenance.json` sidecar recording seed, git SHA, and library versions. The published
`results/tables/probe_distances*.csv` are the immutable as-submitted record and are never overwritten.

### Pipeline phase commands

`uav-tda` exposes each pipeline phase as a subcommand, run in order:

```bash
uav-tda prep           # Phase 2: load CSV, stratified 70/15/15 split, per-manifold StandardScaler, write outputs/*.csv
uav-tda tda             # Phase 3: per-flow Vietoris-Rips persistence diagrams vs reference cloud
uav-tda features        # Phase 4: summary stats (8 per manifold/dim) + persistence images
uav-tda supervised      # Phase 5: grid-searched LogReg/RF/SVM over 4 feature sets, curated RF
uav-tda unsupervised    # Phase 6: Wasserstein-2 distances, thresholding, per-attack AUC
uav-tda evaluate        # Phase 7: ablations, final tables, paper figures
uav-tda all             # every phase in order
```

Shared flags: `--debug` (fast smoke-test sample), `--root PATH` (workspace root, default: repo root).
`tda` (and `all`) also take `--manifold {c2,network,physical,all}`, `--split {train,val,test,all}`,
`--seed`; `unsupervised` (and `all`) also take `--n-jobs`.

**Legacy, equivalent:** the original monolith commands still work identically against `pipeline.py`
and are kept for reference / oracle reproduction:

```bash
python pipeline.py prep
python pipeline.py tda
python pipeline.py features
python pipeline.py supervised
python pipeline.py unsupervised
python pipeline.py evaluate
python pipeline.py all
```
