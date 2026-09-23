# data/

## `UAVIDS-2025.csv`

**Source:** UAVIDS-2025 benchmark (Zeng et al., IEEE CNS 2025) — UAV network
traffic dataset with 5 attack types (Blackhole, Flooding, Sybil, Wormhole,
Normal-Traffic).

**Shape:** 122,171 rows × 23 columns.

**Status: IMMUTABLE.** This file is never modified in place by any code in
this repository — `pipeline.py` and every `uav_tda` phase treat it as
read-only input. Do not edit, re-sort, re-encode, or overwrite it.

**sha256:**
```
d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0
```
Verify with `shasum -a 256 data/UAVIDS-2025.csv` after obtaining the file.

**Gitignored by design.** The CSV (~20MB) is excluded from version control
(`data/*.csv` in `.gitignore`) to keep the repo lean. A fresh clone does
**not** include it — obtain it from the original UAVIDS-2025 source/authors
and verify the sha256 above before running anything (`uav-tda prep` and every
downstream phase depend on it).

**No-timestamps caveat.** UAVIDS-2025 has no timestamp column — only
`FlowID`, a sequential row index (dropped as a feature, since it carries no
signal by value). Row order is the only temporal proxy available in the
dataset. Any "time-windowed" analysis (see `uav_tda/windowed.py`,
`docs/superpowers/specs/2026-09-21-windowed-variant-design.md` §1) therefore
uses **FlowID order** as a stand-in for real time; this is a disclosed
manuscript caveat, not an assumption hidden in the code.
