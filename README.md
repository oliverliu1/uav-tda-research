# Multi-Manifold Persistent Homology for UAV Intrusion Detection

**Anomaly Detection in Contested ISR Military Drone Swarms**

This repository contains the implementation of a novel intrusion detection system for UAV networks using multi-manifold topological data analysis (TDA) with persistent homology.

## Overview

We apply persistent homology to three independent feature manifolds (C2, Network, Physical) extracted from UAV network traffic data. Topological features are computed via Vietoris-Rips filtration and combined with traditional ML classifiers to detect intrusion attacks.

**Dataset**: UAVIDS-2025 benchmark (122,171 network flow records, 5 attack types)

**Key Innovation**: Multi-manifold TDA pipeline that captures topological structure across command-control, network traffic, and physical proxy spaces.
