# LATENCY_RESULTS: Portable Onboard-Latency Harness (Phase 6, Track B)

_Rendered by `uav-tda latency-report` (`uav_tda/latency.py`) from the per-host CSVs and sidecars in `results/tables/rebuild/`; each host's numbers are machine-measured by `uav-tda latency` on that host (full inference path: standardize -> Rips -> slice/truncate -> Wasserstein-2). Nothing here is re-measured or re-typed._

## 1. Per-decision cost by host (mean total_s / p95, seconds)

Ratios are each host's mean divided by the first host's. They are not pure hardware comparisons: software stacks and baseline sources differ (see each host's section). Speedups inside a host compare one window with one flow.

| Decision | Apple M1 Pro laptop | Raspberry Pi 5 (4 GB, active cooler, 27 W supply) | Raspberry Pi 5 (4 GB, active cooler, 27 W supply) / Apple M1 Pro laptop |
| :--- | ---: | ---: | ---: |
| Per-flow, probe (0.8930) | 3.5762 / 3.8591 | 8.3569 / 8.7932 | 2.34 |
| Per-flow, full diagrams (0.9082) | 5.7888 / 7.1369 | 10.5456 / 11.5621 | 1.82 |
| Windowed W=25 | 0.0075 / 0.0096 | 0.0073 / 0.0093 | 0.97 |
| Windowed W=50 | 0.0266 / 0.0387 | 0.0284 / 0.0473 | 1.07 |
| Windowed W=100 | 0.1157 / 0.2053 | 0.1629 / 0.4014 | 1.41 |
| Windowed W=200 | 0.6036 / 1.3033 | 1.0144 / 2.8012 | 1.68 |

## 2. Host: Apple M1 Pro laptop

_Sidecar note: final-review F1/F2: corrected hardware_arch detection (sysctl hw.optional.arm64 instead of a Rosetta-translated `uname -m`) and parameterized machine narrative; measurement NOT re-run, original measurement timestamp_utc=2026-09-23T03:01:55.725644+00:00_

| Field | Value |
| :--- | :--- |
| platform | Darwin |
| machine | x86_64 |
| hardware_arch | arm64 |
| rosetta_translated | True |
| cpu_brand | Apple M1 Pro |
| physical_cores | 8 |
| logical_cores | 8 |
| python_version | 3.9.7 |
| gudhi_version | 3.11.0 |
| hostname | Olivers-MacBook-Pro.local |
| timestamp_utc | 2026-09-23T03:01:55.725644+00:00 |

This Python process reports `x86_64`, not ARM64: the Python 3.9.7 build is a x86_64 binary running under Rosetta 2 on ARM64 hardware (`hardware_arch` = `arm64` from a kernel-level sysctl; `rosetta_translated` = True). The measurements ran under x86_64-on-ARM64 emulation; a native build on the same hardware could measure differently, and this report does not assume faster or slower. This is a workstation-class machine: its power budget, thermal envelope, core count and clock behaviour exceed a companion board's, so these numbers are a LOWER BOUND on onboard latency, not a companion-hardware measurement.

### 2.1 Per-flow arm, probe configuration

n=30 sampled test flows, `SPARSE_RIPS_EPSILON={'c2': 0.5, 'network': 0.5, 'physical': None}`, `PROBE_TOP_K=50`, `PROBE_DELTA=0.2` (pairs with the 0.8930 ten-seed probe AUC, not with 0.9082). The scaler stage is timed on the genuine pre-scaler features, recovered by the scaler's exact inverse.

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_scaler_s | 0.0008 | 0.0008 | 0.0010 | 30 |
| c2_rips_s | 3.0556 | 3.0626 | 3.3204 | 30 |
| c2_slice_s | 0.0002 | 0.0002 | 0.0002 | 30 |
| c2_w2_s | 0.0187 | 0.0199 | 0.0217 | 30 |
| c2_total_s | 3.0753 | 3.0839 | 3.3353 | 30 |
| network_scaler_s | 0.0009 | 0.0008 | 0.0009 | 30 |
| network_rips_s | 0.0544 | 0.0538 | 0.0585 | 30 |
| network_slice_s | 0.0001 | 0.0001 | 0.0002 | 30 |
| network_w2_s | 0.0259 | 0.0248 | 0.0358 | 30 |
| network_total_s | 0.0813 | 0.0802 | 0.0928 | 30 |
| physical_scaler_s | 0.0008 | 0.0007 | 0.0009 | 30 |
| physical_rips_s | 0.4133 | 0.4048 | 0.4815 | 30 |
| physical_slice_s | 0.0002 | 0.0002 | 0.0002 | 30 |
| physical_w2_s | 0.0054 | 0.0067 | 0.0118 | 30 |
| physical_total_s | 0.4196 | 0.4102 | 0.4876 | 30 |
| total_s | 3.5762 | 3.5855 | 3.8591 | 30 |

Baseline source: **committed** (committed = the exact campaign's persisted barcodes; fresh = built on this machine before timing).

### 2.2 Per-flow arm, 0.9082 configuration

Same sampled flows and Rips construction, scored like the full-test campaign behind the 0.9082 headline AUC (`paper/EXACT_RESULTS.md`): full diagrams (no top-K) and hera delta<=0.01. Rips is re-run, not reused. Baseline source: **committed**.

_Timed in a separate session from the probe arm (2026-11-24 vs 2026-09-23, UTC); compare the two arms with that in mind._

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_scaler_s | 0.0011 | 0.0009 | 0.0016 | 30 |
| c2_rips_s | 2.8625 | 2.8708 | 3.0859 | 30 |
| c2_slice_s | 0.0001 | 0.0001 | 0.0002 | 30 |
| c2_w2_s | 0.7566 | 0.7031 | 1.1474 | 30 |
| c2_total_s | 3.6204 | 3.5854 | 3.9980 | 30 |
| network_scaler_s | 0.0012 | 0.0011 | 0.0017 | 30 |
| network_rips_s | 0.0536 | 0.0536 | 0.0573 | 30 |
| network_slice_s | 0.0001 | 0.0001 | 0.0001 | 30 |
| network_w2_s | 1.0484 | 0.7797 | 2.2843 | 30 |
| network_total_s | 1.1032 | 0.8357 | 2.3383 | 30 |
| physical_scaler_s | 0.0012 | 0.0011 | 0.0017 | 30 |
| physical_rips_s | 0.3781 | 0.3765 | 0.3984 | 30 |
| physical_slice_s | 0.0002 | 0.0001 | 0.0004 | 30 |
| physical_w2_s | 0.6857 | 0.7182 | 0.8019 | 30 |
| physical_total_s | 1.0651 | 1.1034 | 1.1814 | 30 |
| total_s | 5.7888 | 5.6030 | 7.1369 | 30 |

### 2.3 Windowed arm

n=30 sampled test-split windows per W (FlowID order). The val-Normal medoid baseline is built once per W outside the timed loop and reported separately as `baseline_setup_s`.

#### W=25 (baseline_setup_s = 78.0921s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.0003 | 0.0001 | 0.0008 | 30 |
| c2_w2_s | 0.0022 | 0.0020 | 0.0029 | 30 |
| c2_total_s | 0.0024 | 0.0023 | 0.0030 | 30 |
| network_rips_s | 0.0005 | 0.0003 | 0.0013 | 30 |
| network_w2_s | 0.0022 | 0.0022 | 0.0032 | 30 |
| network_total_s | 0.0027 | 0.0025 | 0.0045 | 30 |
| physical_rips_s | 0.0003 | 0.0001 | 0.0007 | 30 |
| physical_w2_s | 0.0021 | 0.0020 | 0.0028 | 30 |
| physical_total_s | 0.0023 | 0.0023 | 0.0030 | 30 |
| total_s | 0.0075 | 0.0076 | 0.0096 | 30 |

#### W=50 (baseline_setup_s = 64.6144s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.0018 | 0.0005 | 0.0070 | 30 |
| c2_w2_s | 0.0069 | 0.0069 | 0.0080 | 30 |
| c2_total_s | 0.0087 | 0.0082 | 0.0131 | 30 |
| network_rips_s | 0.0030 | 0.0015 | 0.0109 | 30 |
| network_w2_s | 0.0073 | 0.0072 | 0.0098 | 30 |
| network_total_s | 0.0103 | 0.0093 | 0.0188 | 30 |
| physical_rips_s | 0.0011 | 0.0004 | 0.0039 | 30 |
| physical_w2_s | 0.0064 | 0.0062 | 0.0085 | 30 |
| physical_total_s | 0.0076 | 0.0077 | 0.0104 | 30 |
| total_s | 0.0266 | 0.0246 | 0.0387 | 30 |

#### W=100 (baseline_setup_s = 57.3783s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.0275 | 0.0037 | 0.1199 | 30 |
| c2_w2_s | 0.0248 | 0.0245 | 0.0349 | 30 |
| c2_total_s | 0.0522 | 0.0303 | 0.1418 | 30 |
| network_rips_s | 0.0108 | 0.0044 | 0.0400 | 30 |
| network_w2_s | 0.0240 | 0.0238 | 0.0311 | 30 |
| network_total_s | 0.0348 | 0.0300 | 0.0645 | 30 |
| physical_rips_s | 0.0071 | 0.0028 | 0.0271 | 30 |
| physical_w2_s | 0.0217 | 0.0209 | 0.0335 | 30 |
| physical_total_s | 0.0287 | 0.0255 | 0.0502 | 30 |
| total_s | 0.1157 | 0.0937 | 0.2053 | 30 |

#### W=200 (baseline_setup_s = 49.4467s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.2071 | 0.0365 | 0.8428 | 30 |
| c2_w2_s | 0.1092 | 0.1036 | 0.1620 | 30 |
| c2_total_s | 0.3162 | 0.1551 | 0.9597 | 30 |
| network_rips_s | 0.0529 | 0.0273 | 0.2408 | 30 |
| network_w2_s | 0.0981 | 0.0840 | 0.1669 | 30 |
| network_total_s | 0.1510 | 0.1308 | 0.3295 | 30 |
| physical_rips_s | 0.0611 | 0.0182 | 0.2333 | 30 |
| physical_w2_s | 0.0752 | 0.0770 | 0.1011 | 30 |
| physical_total_s | 0.1363 | 0.1034 | 0.3249 | 30 |
| total_s | 0.6036 | 0.4568 | 1.3033 | 30 |

### 2.4 The paper's "1-3 s per flow" claim vs measured

Published claim (`CLAUDE/PROJECT_BRIEF.md`): "Per-flow inference latency, measured on a 2020-era laptop, is on the order of 1-3 seconds; we do not characterize onboard performance in this extended abstract."

Per-flow decision (probe configuration, all three manifolds): mean **3.5762s**, median 3.5855s, p95 3.8591s -- ABOVE the claimed 1-3s band (3.5762s > 3.0s).
 0.9082 configuration: mean **5.7888s**, p95 7.1369s -- ABOVE the claimed 1-3s band (5.7888s > 3.0s).

| Manifold | mean total_s | median | p95 | vs 1-3s band |
| :--- | ---: | ---: | ---: | :--- |
| c2 | 3.0753 | 3.0839 | 3.3353 | ABOVE the claimed 1-3s band (3.0753s > 3.0s) |
| network | 0.0813 | 0.0802 | 0.0928 | BELOW the claimed 1-3s band (0.0813s < 1.0s) |
| physical | 0.4196 | 0.4102 | 0.4876 | BELOW the claimed 1-3s band (0.4196s < 1.0s) |

## 3. Host: Raspberry Pi 5 (4 GB, active cooler, 27 W supply)

| Field | Value |
| :--- | :--- |
| platform | Linux |
| machine | aarch64 |
| hardware_arch | arm64 |
| rosetta_translated | False |
| cpu_brand | unknown |
| physical_cores | 4 |
| logical_cores | 4 |
| python_version | 3.9.23 |
| gudhi_version | 3.11.0 |
| hostname | uav-pi5 |
| timestamp_utc | 2026-11-24T00:24:26.082550+00:00 |

`platform.machine()` reports `aarch64` (ARM64) directly and `rosetta_translated` = False: the process ran natively. This is a companion-class board, so these are measured companion-hardware latencies, not a lower bound inferred from another machine. They are for this board and software stack only, on mains power with its own cooler; they say nothing about a battery-powered airframe or other boards.

### 3.1 Per-flow arm, probe configuration

n=30 sampled test flows, `SPARSE_RIPS_EPSILON={'c2': 0.5, 'network': 0.5, 'physical': None}`, `PROBE_TOP_K=50`, `PROBE_DELTA=0.2` (pairs with the 0.8930 ten-seed probe AUC, not with 0.9082). The scaler stage is timed on the genuine pre-scaler features, recovered by the scaler's exact inverse.

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_scaler_s | 0.0016 | 0.0016 | 0.0017 | 30 |
| c2_rips_s | 7.0308 | 6.9971 | 7.4645 | 30 |
| c2_slice_s | 0.0004 | 0.0004 | 0.0004 | 30 |
| c2_w2_s | 0.0181 | 0.0190 | 0.0216 | 30 |
| c2_total_s | 7.0509 | 7.0143 | 7.4863 | 30 |
| network_scaler_s | 0.0016 | 0.0016 | 0.0017 | 30 |
| network_rips_s | 0.0926 | 0.0928 | 0.0953 | 30 |
| network_slice_s | 0.0003 | 0.0003 | 0.0003 | 30 |
| network_w2_s | 0.0210 | 0.0207 | 0.0253 | 30 |
| network_total_s | 0.1155 | 0.1159 | 0.1218 | 30 |
| physical_scaler_s | 0.0016 | 0.0016 | 0.0016 | 30 |
| physical_rips_s | 1.1833 | 1.1842 | 1.2029 | 30 |
| physical_slice_s | 0.0004 | 0.0004 | 0.0004 | 30 |
| physical_w2_s | 0.0052 | 0.0068 | 0.0119 | 30 |
| physical_total_s | 1.1905 | 1.1906 | 1.2068 | 30 |
| total_s | 8.3569 | 8.3166 | 8.7932 | 30 |

Baseline source: **fresh** (committed = the exact campaign's persisted barcodes; fresh = built on this machine before timing).

### 3.2 Per-flow arm, 0.9082 configuration

Same sampled flows and Rips construction, scored like the full-test campaign behind the 0.9082 headline AUC (`paper/EXACT_RESULTS.md`): full diagrams (no top-K) and hera delta<=0.01. Rips is re-run, not reused. Baseline source: **fresh**.

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_scaler_s | 0.0016 | 0.0016 | 0.0017 | 30 |
| c2_rips_s | 6.9423 | 6.9672 | 7.2779 | 30 |
| c2_slice_s | 0.0003 | 0.0003 | 0.0003 | 30 |
| c2_w2_s | 0.5955 | 0.5570 | 0.8233 | 30 |
| c2_total_s | 7.5397 | 7.5538 | 8.0244 | 30 |
| network_scaler_s | 0.0016 | 0.0016 | 0.0017 | 30 |
| network_rips_s | 0.0922 | 0.0919 | 0.0947 | 30 |
| network_slice_s | 0.0002 | 0.0002 | 0.0002 | 30 |
| network_w2_s | 1.1579 | 1.1587 | 2.1586 | 30 |
| network_total_s | 1.2520 | 1.2527 | 2.2552 | 30 |
| physical_scaler_s | 0.0016 | 0.0016 | 0.0016 | 30 |
| physical_rips_s | 1.1850 | 1.1860 | 1.1993 | 30 |
| physical_slice_s | 0.0003 | 0.0003 | 0.0003 | 30 |
| physical_w2_s | 0.5671 | 0.5028 | 0.9156 | 30 |
| physical_total_s | 1.7540 | 1.6944 | 2.0987 | 30 |
| total_s | 10.5456 | 10.5196 | 11.5621 | 30 |

### 3.3 Windowed arm

n=30 sampled test-split windows per W (FlowID order). The val-Normal medoid baseline is built once per W outside the timed loop and reported separately as `baseline_setup_s`.

#### W=25 (baseline_setup_s = 70.6505s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.0004 | 0.0002 | 0.0010 | 30 |
| c2_w2_s | 0.0020 | 0.0019 | 0.0028 | 30 |
| c2_total_s | 0.0024 | 0.0023 | 0.0031 | 30 |
| network_rips_s | 0.0007 | 0.0003 | 0.0018 | 30 |
| network_w2_s | 0.0020 | 0.0019 | 0.0029 | 30 |
| network_total_s | 0.0027 | 0.0023 | 0.0042 | 30 |
| physical_rips_s | 0.0003 | 0.0002 | 0.0009 | 30 |
| physical_w2_s | 0.0019 | 0.0018 | 0.0024 | 30 |
| physical_total_s | 0.0022 | 0.0021 | 0.0030 | 30 |
| total_s | 0.0073 | 0.0069 | 0.0093 | 30 |

#### W=50 (baseline_setup_s = 59.4362s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.0027 | 0.0008 | 0.0095 | 30 |
| c2_w2_s | 0.0067 | 0.0064 | 0.0089 | 30 |
| c2_total_s | 0.0094 | 0.0077 | 0.0154 | 30 |
| network_rips_s | 0.0045 | 0.0022 | 0.0161 | 30 |
| network_w2_s | 0.0068 | 0.0067 | 0.0093 | 30 |
| network_total_s | 0.0113 | 0.0097 | 0.0253 | 30 |
| physical_rips_s | 0.0017 | 0.0006 | 0.0059 | 30 |
| physical_w2_s | 0.0061 | 0.0060 | 0.0077 | 30 |
| physical_total_s | 0.0078 | 0.0073 | 0.0114 | 30 |
| total_s | 0.0284 | 0.0253 | 0.0473 | 30 |

#### W=100 (baseline_setup_s = 55.0266s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.0653 | 0.0055 | 0.3051 | 30 |
| c2_w2_s | 0.0241 | 0.0234 | 0.0334 | 30 |
| c2_total_s | 0.0894 | 0.0316 | 0.3255 | 30 |
| network_rips_s | 0.0171 | 0.0061 | 0.0735 | 30 |
| network_w2_s | 0.0233 | 0.0233 | 0.0314 | 30 |
| network_total_s | 0.0404 | 0.0318 | 0.0953 | 30 |
| physical_rips_s | 0.0125 | 0.0044 | 0.0488 | 30 |
| physical_w2_s | 0.0205 | 0.0195 | 0.0293 | 30 |
| physical_total_s | 0.0330 | 0.0253 | 0.0721 | 30 |
| total_s | 0.1629 | 0.1042 | 0.4014 | 30 |

#### W=200 (baseline_setup_s = 49.0355s, fixed one-time cost)

| Stage | mean (s) | median (s) | p95 (s) | n |
| :--- | ---: | ---: | ---: | ---: |
| c2_rips_s | 0.5076 | 0.0612 | 2.1638 | 30 |
| c2_w2_s | 0.0926 | 0.0875 | 0.1267 | 30 |
| c2_total_s | 0.6003 | 0.1541 | 2.2811 | 30 |
| network_rips_s | 0.0991 | 0.0422 | 0.5032 | 30 |
| network_w2_s | 0.0919 | 0.0846 | 0.1568 | 30 |
| network_total_s | 0.1910 | 0.1310 | 0.6250 | 30 |
| physical_rips_s | 0.1478 | 0.0298 | 0.6122 | 30 |
| physical_w2_s | 0.0754 | 0.0726 | 0.1007 | 30 |
| physical_total_s | 0.2231 | 0.1143 | 0.6998 | 30 |
| total_s | 1.0144 | 0.5318 | 2.8012 | 30 |

### 3.4 The paper's "1-3 s per flow" claim vs measured

Published claim (`CLAUDE/PROJECT_BRIEF.md`): "Per-flow inference latency, measured on a 2020-era laptop, is on the order of 1-3 seconds; we do not characterize onboard performance in this extended abstract."

Per-flow decision (probe configuration, all three manifolds): mean **8.3569s**, median 8.3166s, p95 8.7932s -- ABOVE the claimed 1-3s band (8.3569s > 3.0s).
 0.9082 configuration: mean **10.5456s**, p95 11.5621s -- ABOVE the claimed 1-3s band (10.5456s > 3.0s).

| Manifold | mean total_s | median | p95 | vs 1-3s band |
| :--- | ---: | ---: | ---: | :--- |
| c2 | 7.0509 | 7.0143 | 7.4863 | ABOVE the claimed 1-3s band (7.0509s > 3.0s) |
| network | 0.1155 | 0.1159 | 0.1218 | BELOW the claimed 1-3s band (0.1155s < 1.0s) |
| physical | 1.1905 | 1.1906 | 1.2068 | within the claimed 1-3s band |

## Companion-hardware one-command instructions

The harness runs unmodified on a companion-class board given these artifacts copied from this workspace (small: no persistence-diagram pickles):

- `outputs/{m}_train.csv, outputs/{m}_test.csv for m in {c2, network, physical} (the 500-point reference cloud is drawn from *_train.csv; the sampled flows are drawn from *_test.csv)`
- `outputs/{m}_val.csv for m in {c2, network, physical} (windowed arm only -- val-Normal windows build the per-W medoid baseline)`
- `outputs/reference_indices.npy`
- `outputs/max_edge_lengths.json`
- `outputs/labels_test.csv, outputs/labels_val.csv`
- `outputs/scalers.pkl (the combined per-manifold StandardScaler dict written by data.run_prep; the per-manifold outputs/scaler_{m}.pkl files on this repo are STALE -- fitted on a different, older feature schema -- and are NOT used)`
- `results/tables/rebuild/exact/baselines_*.npy + baselines_manifest.json (OPTIONAL, and NOT to be used on a different machine: pass --fresh-baselines there, see uav_tda/hangcheck.py)`

The harness needs NONE of outputs/persistence_diagrams/*.pkl (hundreds of MB to 1.5GB each) -- every diagram it times is computed FRESH, on the fly, from the tiny CSV/npy/json artifacts above, exactly as a deployed detector would.

```bash
pip install -e .
uav-tda latency --fresh-baselines --exact-arm
```

Output files are keyed to the host name, so hosts never overwrite each other; `uav-tda latency-report` then renders every host's section here.

## Pending sign-off

These latencies are candidates for the manuscript's §V characterization and are pending author sign-off. Every number traces to `results/tables/rebuild/latency_<host>.csv`, its per-mode detail CSVs and sidecars.
