# Quick Unsupervised W_2 Probe

_Stratified subset of 1000 test flows (200 per class, seed=42). Baselines recomputed from `reference_indices.npy`. Two probe runs preserved here: a confirmation pass at less-aggressive approximation (top-K=50, delta=0.2) and the prior pass that informed the initial PROJECT_BRIEF.md draft (top-K=20, delta=0.5). The confirmation pass values supersede the prior values for all results-locking purposes (see Diagnostic A summary at bottom)._

---

## Confirmation probe (top-K=50, delta=0.2, seed=42)

### Binary AUC (Normal Traffic vs any Attack)

#### Per manifold and combined (all three)

| Score column | AUC |
| :--- | ---: |
| c2 | 0.7599 |
| network | 0.7481 |
| physical | 0.6192 |
| combined | 0.8712 |

#### Manifold-dropout combinations (onboard / denied-environment)

| Available manifolds | Scenario | Binary AUC |
| :--- | :--- | ---: |
| c2_only | C2 link only (no telemetry, no network captures) | 0.7599 |
| network_only | Network captures only (no C2 visibility, no sensors) | 0.7481 |
| physical_only | Sensor / telemetry only (no traffic capture) | 0.6192 |
| c2_network | **GPS / sensor denied (no Physical)** | 0.8438 |
| c2_physical | Mid-network compromised (no Network) | 0.7680 |
| network_physical | C2 unobservable (e.g., encrypted control) | 0.8550 |
| all_three | Full instrumentation (baseline) | 0.8712 |

### Per-class AUC (one-vs-rest)

#### Per manifold and combined

| Class | C2 | Network | Physical | Combined |
| :--- | ---: | ---: | ---: | ---: |
| Blackhole Attack | 0.4473 | 0.3106 | 0.8032 | 0.4857 |
| Flooding Attack | 0.6333 | 0.7919 | 0.3784 | 0.6999 |
| Normal Traffic | 0.2401 | 0.2519 | 0.3808 | 0.1287 |
| Sybil Attack | 0.6443 | 0.8717 | 0.2119 | 0.6910 |
| Wormhole Attack | 0.5350 | 0.2739 | 0.7256 | 0.4946 |

#### Per-class AUC under dropout (Zeng-attribution check)

| Class | c2_only | network_only | physical_only | c2_network | c2_physical | network_physical | all_three |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Blackhole Attack | 0.4473 | 0.3106 | 0.8032 | 0.3393 | 0.6340 | 0.5321 | 0.4857 |
| Flooding Attack | 0.6333 | 0.7919 | 0.3784 | 0.7704 | 0.5343 | 0.7001 | 0.6999 |
| Normal Traffic | 0.2401 | 0.2519 | 0.3808 | 0.1562 | 0.2320 | 0.1450 | 0.1287 |
| Sybil Attack | 0.6443 | 0.8717 | 0.2119 | 0.8517 | 0.4279 | 0.6982 | 0.6910 |
| Wormhole Attack | 0.5350 | 0.2739 | 0.7256 | 0.3824 | 0.6718 | 0.4246 | 0.4946 |

---

## Prior probe (top-K=20, delta=0.5)

### Binary AUC (Normal Traffic vs any Attack)

#### Per manifold and combined (all three)

| Score column | AUC |
| :--- | ---: |
| c2 | 0.4609 |
| network | 0.7268 |
| physical | 0.6001 |
| combined | 0.6940 |

#### Manifold-dropout combinations (onboard / denied-environment)

| Available manifolds | Scenario | Binary AUC |
| :--- | :--- | ---: |
| c2_only | C2 link only (no telemetry, no network captures) | 0.4609 |
| network_only | Network captures only (no C2 visibility, no sensors) | 0.7268 |
| physical_only | Sensor / telemetry only (no traffic capture) | 0.6001 |
| c2_network | **GPS / sensor denied (no Physical)** | 0.6386 |
| c2_physical | Mid-network compromised (no Network) | 0.5349 |
| network_physical | C2 unobservable (e.g., encrypted control) | 0.8199 |
| all_three | Full instrumentation (baseline) | 0.6940 |

### Per-class AUC (one-vs-rest)

#### Per manifold and combined

| Class | C2 | Network | Physical | Combined |
| :--- | ---: | ---: | ---: | ---: |
| Blackhole Attack | 0.4906 | 0.3305 | 0.7605 | 0.4949 |
| Flooding Attack | 0.5404 | 0.7454 | 0.3772 | 0.6501 |
| Normal Traffic | 0.5391 | 0.2732 | 0.3999 | 0.3060 |
| Sybil Attack | 0.4511 | 0.8418 | 0.2484 | 0.5994 |
| Wormhole Attack | 0.4788 | 0.3092 | 0.7140 | 0.4496 |

#### Per-class AUC under dropout (Zeng-attribution check)

| Class | c2_only | network_only | physical_only | c2_network | c2_physical | network_physical | all_three |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Blackhole Attack | 0.4906 | 0.3305 | 0.7605 | 0.3609 | 0.6487 | 0.4848 | 0.4949 |
| Flooding Attack | 0.5404 | 0.7454 | 0.3772 | 0.7090 | 0.4786 | 0.6711 | 0.6501 |
| Normal Traffic | 0.5391 | 0.2732 | 0.3999 | 0.3614 | 0.4651 | 0.1801 | 0.3060 |
| Sybil Attack | 0.4511 | 0.8418 | 0.2484 | 0.7289 | 0.2968 | 0.7288 | 0.5994 |
| Wormhole Attack | 0.4788 | 0.3092 | 0.7140 | 0.3398 | 0.6108 | 0.4351 | 0.4496 |

---

## Sample composition (both runs)

| Class | Count |
| :--- | ---: |
| Blackhole Attack | 200 |
| Flooding Attack | 200 |
| Normal Traffic | 200 |
| Sybil Attack | 200 |
| Wormhole Attack | 200 |

## Verdict heuristic (confirmation probe)

- **Binary AUC (all_three) = 0.8712**
- **Binary AUC (network_physical) = 0.8550**
- **Binary AUC (c2_network — GPS-denied) = 0.8438**

Thresholds (probe-only; full-test results may differ by +-0.02-0.05):

- **>= 0.85** -> label-free anomaly detection is a viable paper headline.
- **0.75-0.85** -> story is solid; relaunch full unsupervised on val+test only.
- **0.65-0.75** -> marginal; consider negative-result / diagnostic framing.
- **< 0.65** -> pivot to negative-result paper.

---

### Diagnostic A summary

- Binary AUC (Network+Physical): prior 0.82 → new 0.86 (delta: +0.04)
- Sybil Network-AUC: prior 0.84 → new 0.87 (delta: +0.03)
- Flooding Network-AUC: prior 0.75 → new 0.79 (delta: +0.05)
- Blackhole Physical-AUC: prior 0.76 → new 0.80 (delta: +0.04)
- Wormhole Physical-AUC: prior 0.71 → new 0.73 (delta: +0.01)
- C2-alone binary AUC: prior 0.46 → new 0.76 (delta: +0.30)
- Verdict: **unstable (one or more deltas exceed ±0.02)**. Five of six tracked metrics shifted by more than ±0.02 at the less-aggressive approximation, and C2-alone in particular swung by +0.30 — moving from below-chance to genuinely informative. The prior `top-K=20, delta=0.5` numbers cannot be relied on for results-locking; the confirmation-pass numbers (`top-K=50, delta=0.2`) supersede them throughout. Notably, the prior "C2-paradox" framing (where dropping C2 raised combined AUC from 0.69 to 0.82) does not survive at this less-aggressive approximation: confirmation-pass combined-AUC (0.87) now exceeds network+physical (0.86), so C2 is contributing useful signal rather than degrading it. Diagnostic B (Z-normalization check) becomes correspondingly less critical for distinguishing artifact-from-finding, but should still be run to confirm the C2 contribution holds under per-manifold standardization.
