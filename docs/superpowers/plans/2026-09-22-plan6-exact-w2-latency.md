# Exact Wasserstein Campaign + Latency Harness — Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the probe approximation with definitive exact-Wasserstein AUCs on the full test split (checkpointed ~3h campaign; the old ~74h estimate is obsolete — measured 2026-09-22: exact W2 ≈ 3.3 s/flow across manifolds on the clean diagrams), and deliver the §V latency characterization via a portable one-command benchmark harness run on this arm64 machine (disclosed as ARM-ISA/workstation-class) and ready for companion-class hardware.

**Architecture:** Track A (exact W2): new `uav_tda/exact.py` — a checkpointed, resumable, sharded runner over the stored clean diagrams with persisted baselines (so resume never re-rolls the sparse-Rips baseline realization), per-call fork-timeout reusing `probe._w2_with_timeout`, and a retry-once δ=0.01 fallback recorded per flow; then definitive tables + `paper/EXACT_RESULTS.md`. Track B (latency): new `uav_tda/latency.py` — measures the FULL inference path (standardize → Rips → slice → W2), per-flow and per-window, machine-info header, `paper/LATENCY_RESULTS.md`. Reuse throughout: `probe` internals, `metrics`, `windowed`, `manuscript.bootstrap`, `provenance`.

**Tech Stack:** Python 3.9 (`from __future__ import annotations`), numpy, pandas, gudhi/hera, pytest. No new dependencies.

**Spec:** roadmap Phase 6; paper §V ("Exact Wasserstein computation on the full test set will replace the probe approximation"; "onboard latency profiling on ARM-class hardware"); user decision 2026-09-22: M1-only + portable harness (companion-class run deferred to the user, one-command ready).

## Global Constraints

- **Same-run baseline coupling, resume-safe:** a campaign computes per-manifold baselines ONCE at start and **persists them** (`results/tables/rebuild/exact/baselines_{m}_dim{k}.npy` + manifest + provenance); every shard and every resume reloads the persisted baselines — never recompute (sparse-Rips noise must not differ between val and test or across resumes). Z-norm stats come from the full val-Normal distances of the SAME campaign.
- **Exact means exact:** hera `order=2, internal_p=2.0`, NO delta, NO top-K, on `probe._slice_dim`-clamped full diagrams. Per-call fork-timeout 120 s (reuse `probe._w2_with_timeout` machinery); on timeout, retry ONCE with `delta=0.01`; record per (flow, manifold): `n_timeouts`, `approx_flag` (True iff any dim fell back). Definitive AUCs are computed over all flows; the report states the approx-flag count and, if > 0, also reports AUCs excluding flagged flows.
- **Sharding/resume:** shards of 500 flows per (split, manifold-set) written as `exact/shard_{split}_{start:05d}.csv`; a `manifest.json` tracks completed shards; the runner skips completed shards on re-invocation. Campaign scope: `test` (all 18,326) + `val` Normal-only (3,926).
- **Latency harness measures the full path:** per-flow = StandardScaler transform (from stored scalers) + Rips({q} ∪ 500 ref) + slice + exact-config W2 per manifold (probe production config: sparse per `config.SPARSE_RIPS_EPSILON`, δ=0.2, top-K=50 — i.e., the PAPER's deployed configuration, not the exact campaign's); per-window = window Rips + W2 vs medoid per `config.WINDOWED_SPARSE` at each W. N=30 sampled decisions per mode; report mean/median/p95 + machine header (platform, machine, CPU brand, core count, Python/gudhi versions). Output: `results/tables/rebuild/latency_{hostname}.csv` + `paper/LATENCY_RESULTS.md` with the ARM-ISA-but-workstation-class disclosure and one-command companion-hardware instructions.
- Locked surfaces unchanged (golden-master metrics fns, probe default path, pipeline.py, tools/, published top-level tables, pre-existing paper/*.md). New artifacts only under `results/tables/rebuild/exact/`, `results/tables/rebuild/latency_*.csv`, `paper/{EXACT_RESULTS,LATENCY_RESULTS}.md`.
- Slow tests `@pytest.mark.slow`; fast suite < ~1 min; runs > 10 min detached AND all campaign code committed BEFORE launch (Phase-5 process-staleness lesson); report-not-loosen; provenance sidecars everywhere; Python 3.9.

## File Structure

```
uav_tda/
  exact.py         # NEW — Track A: sharded runner, assembly, definitive tables, report
  latency.py       # NEW — Track B: full-path benchmark + report
  cli.py           # MODIFY: `exact`, `exact-report`, `latency` subcommands
tests/
  test_exact.py    # NEW
  test_latency.py  # NEW
results/tables/rebuild/exact/     # OUTPUT (shards, baselines, manifest, distances, tables)
paper/EXACT_RESULTS.md            # OUTPUT
paper/LATENCY_RESULTS.md          # OUTPUT
CLAUDE/PROJECT_BRIEF.md           # MODIFY in T3/T4 only: decision-log entries
CLAUDE/CLAUDE.md                  # MODIFY in T4 only: CLI list
```

---

### Task 1: Sharded exact-W2 runner with persisted baselines

**Files:** Create `uav_tda/exact.py`; Test `tests/test_exact.py`.

**Interfaces (produced):**
- `persist_baselines(ws, exact_dir) -> dict` — computes per-manifold per-dim baselines (via `probe._load_baseline_barcodes` semantics) ONCE, saves each dim slice as `.npy` + a `baselines_manifest.json` (shapes, date, provenance); returns the loaded dict. `load_baselines(exact_dir) -> dict` reloads; `ensure_baselines(ws, exact_dir)` = load-if-present-else-persist.
- `exact_w2_flow(diagram, baselines_m, max_edge, max_hom_dim, timeout_s=120.0) -> tuple[float, int, bool]` — (summed distance, n_timeouts, approx_flag): exact per dim via `probe._w2_with_timeout`; NaN → retry once `delta=0.01`; still NaN → contribute 0.0, count.
- `run_shard(ws, exact_dir, split, start, size, baselines, class_filter=None, n_jobs=-1) -> Path` — computes rows `[row_idx, label, W2_<m>×3, n_timeouts, approx_flag]` for flows [start, start+size) of the (optionally class-filtered) split, joblib-parallel; writes `shard_{split}_{start:05d}.csv` + updates `manifest.json` atomically (write tmp, rename).
- `run_exact_campaign(ws, n_jobs=-1, shard_size=500) -> None` — ensure_baselines → all val-Normal shards → all test shards, skipping manifest-completed ones. Resumable by construction.

- [ ] **Step 1: Failing unit tests** — `exact_w2_flow` on tiny hand-built diagrams (finite result, no timeout, approx_flag False); timeout path: monkeypatch `probe._w2_subprocess_target` to hang → with timeout_s=0.3 and a monkeypatched retry (also hanging) → returns (0.0-contribution, counts, approx_flag True); manifest resume logic with fake shard files (completed shards skipped); baseline persist/load roundtrip on synthetic diagrams (npy equality).
- [ ] **Step 2: fail → implement → pass; fast suite green; commit** — `"feat: sharded resumable exact-W2 runner with persisted baselines"`

### Task 2: Assembly, definitive tables, CLI

**Files:** Modify `uav_tda/exact.py`, `uav_tda/cli.py`; Test `tests/test_exact.py` (append).

**Interfaces:**
- `assemble_distances(exact_dir, split) -> pd.DataFrame` — concat shards in order, verify row count vs manifest, add the 7 subset columns (`metrics.MANIFOLD_SUBSETS` sums).
- `build_exact_tables(ws, B=2000, bootstrap_seed=0) -> dict` — from assembled val-Normal + test frames: znorm stats (`metrics.znorm_stats_from_val` on the val frame); definitive `exact_binary_auc.csv` (7 subsets × {raw, znorm}: AUC + flow-level stratified bootstrap CI via `manuscript.bootstrap_mean_auc_ci` with a single-"seed" cluster); `exact_per_attack_auc.csv` (one-vs-rest, raw, per manifold + CI); `exact_vs_probe.csv` comparison (exact vs Phase-4 10-seed probe mean±std vs published 3-seed values quoted from `paper/MULTI_SEED_VARIANCE.md` as labeled literals).
- CLI: `uav-tda exact [--n-jobs N] [--shard-size N]` (campaign, resumable) and `uav-tda exact-report [--bootstrap B]` (assemble → tables → CSVs + provenance → `paper/EXACT_RESULTS.md`).
- [ ] **Step 1: Failing tests** — synthetic shards in tmp exact_dir: assembly order/count checks + subset sums; tables: shapes, ci bounds ordered, comparison frame has exact/probe/published columns; CLI registration (3 subcommands, flags). No real computation in tests.
- [ ] **Step 2: fail → implement → pass; fast suite; commit** — `"feat: exact-W2 assembly, definitive tables, CLI"`

### Task 3: The exact campaign + EXACT_RESULTS report

- [ ] **Step 1: COMMIT-BEFORE-LAUNCH check** — all campaign code committed (T1+T2 done); only then launch detached: `nohup python3 -m uav_tda.cli exact --n-jobs 7 > <scratchpad>/exact_campaign.log 2>&1 &`. Expected ≈ 3–4 h (measured 3.3 s/flow ÷ 7 cores × 22,252 flows). Poll shards/manifest; resumable on any interruption.
- [ ] **Step 2: SANITY GATES** (report-not-loosen): timeout/approx counts reported (expected ≈ 0 at 120 s); definitive znorm all_three AUC within ±0.05 of the probe 10-seed mean (0.8930) — outside band = STOP-and-report science finding; per-attack dominant manifolds match the per-flow attribution (Sybil/Flooding→network, Blackhole/Wormhole→physical) — flip = STOP-and-report; row counts exact (18,326 + 3,926).
- [ ] **Step 3: `uav-tda exact-report`** → verify every table value traces; `paper/EXACT_RESULTS.md` sections: config header (exact definition, timeout policy, baseline persistence + which manifolds' baselines carry sparse noise); definitive binary table (raw+znorm+CI); per-attack table; exact-vs-probe-vs-published comparison with discussion (the probe approximation's measured error, §V promise discharged); approx-flag disclosure; pending-sign-off framing.
- [ ] **Step 4: decision-log entry (dated) + fast suite + commit** — `"feat: exact Wasserstein campaign — definitive full-test AUCs (EXACT_RESULTS)"`

### Task 4: Latency harness + LATENCY_RESULTS

**Files:** Create `uav_tda/latency.py`; Modify `uav_tda/cli.py`, `CLAUDE/CLAUDE.md`, `CLAUDE/PROJECT_BRIEF.md`; Test `tests/test_latency.py`.

**Interfaces:** `machine_info() -> dict` (platform/machine/CPU brand via sysctl-or-/proc fallback/cores/python/gudhi versions/hostname); `time_per_flow_decision(ws, n=30, rng_seed=0) -> pd.DataFrame` (per sampled test flow: scaler transform + Rips({q}∪ref, production sparse config) + slice + W2 δ=0.2/top-K=50 per manifold; columns per-stage seconds + total); `time_windowed_decision(ws, w, n=30, rng_seed=0) -> pd.DataFrame` (window Rips per `WINDOWED_SPARSE` + W2 vs a persisted-or-fresh medoid; per-stage + total; baseline setup timed separately, reported as fixed cost); `run_latency(ws) -> Path` (all modes: per-flow + W∈WINDOW_SIZES; writes `results/tables/rebuild/latency_{hostname}.csv` + provenance + `paper/LATENCY_RESULTS.md`: machine header, mean/median/p95 table per mode with per-stage breakdown, the paper's "1–3 s per flow" claim checked against measured, ARM-ISA-but-workstation-class disclosure, one-command instructions for Jetson/RPi (`pip install -e . && uav-tda latency` after prep+tda artifacts or with the shipped reference data — state exactly what artifacts a companion box needs: outputs/{m}_{train,test}.csv, reference_indices.npy, max_edge_lengths.json, labels — NOT the diagram pkls)); CLI `uav-tda latency [--n N]`.
- [ ] **Step 1: Failing tests** — machine_info keys; harness on synthetic tiny data via monkeypatched Rips/W2 (frame schema, per-stage columns sum ≈ total); CLI registration. Fast only.
- [ ] **Step 2: fail → implement → pass → RUN `uav-tda latency` on this machine (~10–20 min: 30×(per-flow ~1-3s + windows)) — detach if needed.**
- [ ] **Step 3: verify report renders with real numbers; measured per-flow total vs the paper's 1–3 s claim explicitly stated; decision-log entry; CLAUDE.md CLI list; fast suite; commit** — `"feat: portable latency harness + workstation-ARM measurements (LATENCY_RESULTS)"`

---

## Self-Review

- **Spec coverage:** §V exact-full-test → T1–T3 (definitive AUCs + comparison) ✓; §V latency → T4 (full-path measurement + companion-ready harness, per the user's M1-only decision) ✓; roadmap batching/caching/timeout-fallback/checkpointing → T1 (shards, persisted baselines, fork-timeout, δ-retry, manifest resume) ✓.
- **Placeholder scan:** interfaces exact; gates numeric; artifact names concrete. Test steps name their fixtures and assertions. No TBDs.
- **Type consistency:** `exact_w2_flow` 3-tuple consistent T1↔T2 assembly columns; `ensure_baselines`/`load_baselines` used by T1 runner and T3 campaign; latency frame schema consistent T4 internal↔report.
- **Pre-ruled risks:** (a) exact campaign slower than measured → shards+resume make interruption cheap; report actual; (b) any timeout/approx flow → disclosed count + AUC-excluding variant; (c) AUC or attribution outside band → STOP-and-report; (d) commit-before-launch enforced (Phase-5 lesson); (e) latency per-flow uses the PAPER's production config (δ=0.2/top-K) not exact config — deliberate, stated in the report.
