# Multi-Manifold Persistent Homology for UAV Intrusion Detection

**Anomaly Detection in Contested ISR Military Drone Swarms**

This repository contains the implementation of a novel intrusion detection system for UAV networks using multi-manifold topological data analysis (TDA) with persistent homology.

## Overview

We apply persistent homology to three independent feature manifolds (C2, Network, Physical) extracted from UAV network traffic data. Topological features are computed via Vietoris-Rips filtration and combined with traditional ML classifiers to detect intrusion attacks.

**Dataset**: UAVIDS-2025 benchmark (122,171 network flow records, 5 attack types)

**Key Innovation**: Multi-manifold TDA pipeline that captures topological structure across command-control, network traffic, and physical proxy spaces.

## Reproducing the paper's results

Prerequisites: `data/UAVIDS-2025.csv` present, and Phase-2/3 artifacts built
(`python pipeline.py prep && python pipeline.py tda`).

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
