# Z-Normalized Scoring + W2 Timeout — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the paper's §V-promised scoring corrections as tested code: per-manifold Z-score normalization computed on held-out **validation** Normal-Traffic W2 distances (replacing the unweighted sum), a working per-call `w2_timeout` (removing the seed-123 C2 artifact), and a regenerated Z-normalization results document on clean lineage — giving refreshable numbers for the revisable abstract.

**Architecture:** All new behavior extends `uav_tda/metrics.py` and `uav_tda/probe.py`; existing golden-master-locked functions are untouched. One probe invocation computes baselines ONCE and reuses them for both the val-Normal distance pass (→ Z-stats) and the test pass (→ scores), so Z-stats and scores share the same sparse-Rips baseline realization (process noise cannot decouple them — see the sparse-Rips memory/plan-2 findings). The timeout mechanism is ported verbatim-in-spirit from the recovered original `tools/quick_unsup_probe.py` (forked subprocess per hera call; timed-out dims contribute 0.0 and are counted).

**Tech Stack:** Python 3.9 (anaconda; `from __future__ import annotations`), numpy, pandas, gudhi/hera, pytest. No new dependencies.

**Spec:** `docs/superpowers/plans/2026-09-19-roadmap.md` (Phase 3); paper `CLAUDE/SciTech2027_IntelligentSystems_Liu/main.tex` §III.E ("per-manifold Z-score normalization computed on held-out validation Normal Traffic") and §V ("reanalyzed under an extended per-call timeout"); recovered original `tools/quick_unsup_probe.py` (timeout mechanism, lines ~190–246); `paper/C2_NORMALIZATION_DIAGNOSTIC.md` (the historical test-Normal-substitute diagnostic this supersedes).

## Global Constraints

- **Do not modify** the golden-master-locked functions in `uav_tda/metrics.py` (`binary_auc_by_subset`, `per_attack_auc`, `aggregate_over_seeds`, `MANIFOLD_SUBSETS`) or their tests; do not modify `uav_tda/{data,tda,features,supervised,unsupervised,evaluate}.py` (equivalence-locked ports), `pipeline.py`, `tools/`, `data/`, published `results/tables/*.csv` (top level), or existing `paper/*.md` files. New results go to `results/tables/rebuild/`; the new results document is a NEW file `paper/ZNORM_RESULTS.md`.
- **Scoring reconciliation ruling (final):** the canonical forward-looking scoring is the probe path — per-manifold W2 summed across manifolds, now optionally Z-normalized on validation Normal. `uav_tda/unsupervised.py`'s max-pool + flag semantics are the frozen-equivalent monolith port and stay as-is; this relationship must be stated in both modules' docstrings (Task 3).
- **Same-run coupling:** Z-stats and the test distances they normalize MUST come from the same probe invocation (same baseline realization). Never normalize one run's distances with another run's stats — enforce by API shape (one function returns both) and say so in docstrings.
- Val-Normal pass uses **all 3,926 validation Normal-Traffic flows** (paper wording: "computed on held-out validation Normal Traffic"), not a sample.
- Degenerate-σ guard: if a manifold's val-Normal W2 std is 0 (or < 1e-12), z-normalization for that manifold divides by 1.0 instead and logs a warning — never NaN/inf.
- Probe seeds for the reanalysis: `(42, 7, 123)` (paper's). `--w2-timeout` default for the reanalysis runs: 30.0 seconds (the "extended" timeout; original artifact used 5s).
- Slow tests marked `@pytest.mark.slow`; fast suite stays under ~1 min. Runs >10 min inside implementer shells must be detached (nohup+log+poll).
- Python 3.9 compatible.

## File Structure

```
uav_tda/
  metrics.py       # MODIFY (append only): znorm_stats_from_val, apply_znorm, binary_auc_by_subset_znorm
  probe.py         # MODIFY: _w2_with_timeout port; distances_for_indices refactor; run_probe_with_znorm; wire w2_timeout in run_probe
  cli.py           # MODIFY: probe subcommand gains --znorm; --w2-timeout becomes functional; new `znorm-report` subcommand
tests/
  test_metrics_znorm.py   # NEW
  test_probe_znorm.py     # NEW (unit + slow integration)
results/tables/rebuild/   # OUTPUT (gitignored? NO — rebuild/ is not ignored; these are the refreshed citable tables)
paper/ZNORM_RESULTS.md    # OUTPUT of the report generator (committed)
CLAUDE/PROJECT_BRIEF.md   # MODIFY in Task 3 only: append one decision-log entry
```

---

### Task 1: Z-normalization math in `metrics.py`

**Files:**
- Modify: `uav_tda/metrics.py` (append-only)
- Test: `tests/test_metrics_znorm.py`

**Interfaces:**
- Consumes: probe-distance dataframes (columns `label`, `W2_c2`, `W2_network`, `W2_physical`).
- Produces (exact signatures, consumed by Tasks 2–3):
  - `znorm_stats_from_val(val_df: pd.DataFrame) -> dict[str, tuple[float, float]]` — `{manifold: (mean, std)}` of `W2_<m>` over rows where `label == "Normal Traffic"`; applies the degenerate-σ guard (std < 1e-12 → std = 1.0, log warning).
  - `apply_znorm(df: pd.DataFrame, stats: dict) -> pd.DataFrame` — returns a COPY with `W2_<m>` columns replaced by `(x - mean)/std` and all 7 `W2_<subset>` columns recomputed as SUMS of the z-scored manifolds.
  - `binary_auc_by_subset_znorm(df, stats) -> dict[str, float]` — convenience: `binary_auc_by_subset(apply_znorm(df, stats))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_znorm.py
import numpy as np
import pandas as pd
import pytest

from uav_tda import metrics


def _toy_df():
    # 4 Normal + 4 attack rows with hand-computable stats.
    return pd.DataFrame({
        "label": ["Normal Traffic"] * 4 + ["Sybil Attack"] * 4,
        "W2_c2":       [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
        "W2_network":  [0.0, 0.0, 0.0, 0.0,  1.0,  1.0,  1.0,  1.0],
        "W2_physical": [5.0, 5.0, 5.0, 5.0,  5.0,  5.0,  5.0,  5.0],
    })


def test_znorm_stats_from_val_normal_only():
    stats = metrics.znorm_stats_from_val(_toy_df())
    mean, std = stats["c2"]
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))  # population std
    # degenerate sigma guard: network/physical Normal values are constant
    assert stats["network"][1] == 1.0
    assert stats["physical"][1] == 1.0


def test_apply_znorm_recomputes_subset_sums():
    df = _toy_df()
    stats = metrics.znorm_stats_from_val(df)
    z = metrics.apply_znorm(df, stats)
    # original untouched (copy semantics)
    assert df["W2_c2"].iloc[0] == 1.0
    # z-scored manifold column
    expected_c2 = (1.0 - 2.5) / np.std([1.0, 2.0, 3.0, 4.0])
    assert z["W2_c2"].iloc[0] == pytest.approx(expected_c2)
    # subset columns are sums of z-scored manifolds
    assert z["W2_all_three"].iloc[0] == pytest.approx(
        z["W2_c2"].iloc[0] + z["W2_network"].iloc[0] + z["W2_physical"].iloc[0])
    assert z["W2_network_physical"].iloc[0] == pytest.approx(
        z["W2_network"].iloc[0] + z["W2_physical"].iloc[0])


def test_znorm_auc_shifts_scale_not_ranking_per_manifold():
    # Z-normalization is monotone per manifold → single-manifold AUCs unchanged.
    df = _toy_df()
    stats = metrics.znorm_stats_from_val(df)
    raw = metrics.binary_auc_by_subset(df)
    z = metrics.binary_auc_by_subset_znorm(df, stats)
    for subset in ("c2_only", "network_only", "physical_only"):
        assert z[subset] == pytest.approx(raw[subset])


def test_existing_golden_master_functions_untouched():
    # The locked API surface must still exist unmodified.
    for name in ("binary_auc_by_subset", "per_attack_auc", "aggregate_over_seeds"):
        assert hasattr(metrics, name)
```

- [ ] **Step 2: Verify fail** — `python3 -m pytest tests/test_metrics_znorm.py -v` → AttributeError on `znorm_stats_from_val`.
- [ ] **Step 3: Implement (append to `uav_tda/metrics.py`)**

```python
# --- Phase 3: Z-normalized scoring (paper §III.E / §V correction) -----------
# Canonical forward-looking scoring: per-manifold W2, Z-normalized on held-out
# VALIDATION Normal-Traffic distances, then summed across manifolds. The raw
# unweighted-sum functions above are frozen (golden-master-locked to the
# published extended-abstract numbers) and must not change.
# NOTE: stats and the distances they normalize must come from the SAME probe
# run (same sparse-Rips baseline realization) — see run_probe_with_znorm.

import logging

_log = logging.getLogger("uav_tda.metrics")


def znorm_stats_from_val(val_df: pd.DataFrame) -> dict:
    """Per-manifold (mean, std) of W2 over validation Normal-Traffic rows."""
    normal = val_df[val_df["label"] == NORMAL]
    stats: dict = {}
    for m in MANIFOLDS:
        x = normal[f"W2_{m}"].to_numpy(dtype=float)
        mean, std = float(np.mean(x)), float(np.std(x))
        if std < 1e-12:
            _log.warning("znorm: manifold %s has degenerate std=%.3g; using 1.0", m, std)
            std = 1.0
        stats[m] = (mean, std)
    return stats


def apply_znorm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Copy of df with manifold W2 columns z-scored and subset sums recomputed."""
    out = df.copy()
    for m in MANIFOLDS:
        mean, std = stats[m]
        out[f"W2_{m}"] = (out[f"W2_{m}"].astype(float) - mean) / std
    for subset, manifolds in MANIFOLD_SUBSETS.items():
        out[f"W2_{subset}"] = sum(out[f"W2_{m}"] for m in manifolds)
    return out


def binary_auc_by_subset_znorm(df: pd.DataFrame, stats: dict) -> dict:
    """Normal-vs-attack AUC per subset on z-normalized scores."""
    return binary_auc_by_subset(apply_znorm(df, stats))
```

- [ ] **Step 4: All tests pass** — the new file AND the untouched `tests/test_metrics.py` (4 golden-master tests) both green.
- [ ] **Step 5: Commit** — `git add uav_tda/metrics.py tests/test_metrics_znorm.py && git commit -m "feat: val-Normal Z-normalized subset scoring (paper §III.E correction)"`

---

### Task 2: Probe extensions — w2_timeout + val-Normal pass + coupled Z-norm run

**Files:**
- Modify: `uav_tda/probe.py`
- Test: `tests/test_probe_znorm.py`

**Interfaces:**
- Consumes: Task 1's `znorm_stats_from_val`/`apply_znorm`; existing probe internals (`_load_baseline_barcodes`, `_slice_dim`, `truncate_top_k`); artifacts `outputs/persistence_diagrams/{m}_{split}.pkl` (test AND val, both present on clean lineage), `outputs/labels_{split}.csv`, `outputs/max_edge_lengths.json`, `outputs/reference_indices.npy`.
- Produces:
  - `_w2_with_timeout(a, b, order, internal_p, delta, timeout_sec) -> float` — hera call in a forked subprocess (`multiprocessing.get_context("fork")`, `Process` + `Queue`, `join(timeout_sec)`, `terminate()` on overrun); returns `float("nan")` on timeout. Port the mechanism from `tools/quick_unsup_probe.py` (read its `_w2_with_timeout` and mirror it; cite the line range in a comment).
  - `run_probe(..., w2_timeout: float | None = None, ...)` — the EXISTING no-op parameter becomes functional: when set, every per-dim hera call goes through `_w2_with_timeout`; NaN results contribute 0.0 to the flow's sum and increment a timeout counter. `run_probe` gains a `split: str = "test"` parameter and an optional `class_filter: str | None = None` + `per_class: int | None` semantics (`per_class=None` = take ALL flows of the filtered class; with `class_filter=None`, `per_class` must be an int, sampling as today). Returns the DataFrame as today; timeout counts exposed via a new attribute-free mechanism: an optional `stats_out: dict | None = None` parameter that, when a dict is passed, is filled with `{"n_timeouts_<manifold>": int}`.
  - `run_probe_with_znorm(seed, per_class=200, top_k=50, delta=0.2, w2_timeout=None, n_jobs=-1) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]` — ONE invocation that: loads baselines ONCE; computes val-split distances for ALL validation Normal flows (`split="val"`, `class_filter="Normal Traffic"`, `per_class=None`); computes the seeded test probe as today; derives `stats = znorm_stats_from_val(val_df)`; returns `(test_df, val_df, stats, timeout_counts)`. Test/val passes MUST share the same in-memory baselines (refactor `run_probe` so baseline loading is separable — e.g. internal `_probe_distances(baselines, max_edge_lengths, split, indices, ...)` used by both).

- [ ] **Step 1: Failing unit tests**

```python
# tests/test_probe_znorm.py
import numpy as np
import pytest

from uav_tda import probe


def test_w2_with_timeout_returns_value_for_fast_call():
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.0, 1.1]])
    w = probe._w2_with_timeout(a, b, order=2.0, internal_p=2.0, delta=0.2, timeout_sec=30.0)
    assert np.isfinite(w) and w >= 0.0


def test_w2_with_timeout_kills_slow_call(monkeypatch):
    # Simulate a hang: patch the subprocess target's worker to sleep past the timeout.
    # Implementation must expose the worker as probe._w2_subprocess_target so this
    # test can wrap it.
    import time

    real_target = probe._w2_subprocess_target

    def slow_target(queue, *args):
        time.sleep(5.0)
        real_target(queue, *args)

    monkeypatch.setattr(probe, "_w2_subprocess_target", slow_target)
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.0, 1.1]])
    w = probe._w2_with_timeout(a, b, order=2.0, internal_p=2.0, delta=0.2, timeout_sec=0.5)
    assert np.isnan(w)
```

(Note for the implementer: for the monkeypatch to take effect the fork must resolve the target through the module attribute at call time — structure `_w2_with_timeout` to call `Process(target=_module_lookup)` accordingly, e.g. `target=globals()["_w2_subprocess_target"]` resolved inside the function, or simply `target=probe._w2_subprocess_target` via module self-import. Verify the test actually exercises the kill path.)

- [ ] **Step 2: Verify fail → implement `_w2_subprocess_target` + `_w2_with_timeout` (mirroring `tools/quick_unsup_probe.py`), wire into the per-dim loop (NaN → +0.0, counter++), refactor baseline loading out of `run_probe`, add `split`/`class_filter`/`per_class=None`/`stats_out`, implement `run_probe_with_znorm`. Unit tests pass.**
- [ ] **Step 3: Slow integration test (append)**

```python
# tests/test_probe_znorm.py (append)
from uav_tda import metrics
from uav_tda.paths import PERSISTENCE_DIR


@pytest.mark.slow
def test_run_probe_with_znorm_end_to_end_seed42():
    """Clean-lineage sanity gates for the §III.E-corrected scoring.

    Gates (not oracle-exact — clean lineage + process noise; see plan):
    - val df: exactly the 3,926 validation Normal flows, no other labels;
    - stats: 3 manifolds, std > 0;
    - znorm all_three AUC within [raw_all_three - 0.02, 1.0] (historical
      Diagnostic B: znorm improved all_three 0.8712 -> 0.8983 on old lineage);
    - N+P raw AUC in [0.80, 0.92] (clean-lineage band around published 0.86);
    - Sybil dominant manifold remains network under BOTH scorings;
    - with w2_timeout=30.0, zero timeouts on network/physical.
    """
    if not (PERSISTENCE_DIR / "c2_val.pkl").exists():
        pytest.skip("val diagrams absent")
    test_df, val_df, stats, tcounts = probe.run_probe_with_znorm(
        seed=42, w2_timeout=30.0)
    assert set(val_df["label"]) == {"Normal Traffic"}
    assert len(val_df) == 3926
    assert set(stats) == {"c2", "network", "physical"}
    assert all(s[1] > 0 for s in stats.values())
    raw = metrics.binary_auc_by_subset(test_df)
    z = metrics.binary_auc_by_subset_znorm(test_df, stats)
    assert z["all_three"] >= raw["all_three"] - 0.02
    assert 0.80 <= raw["network_physical"] <= 0.92
    pa_raw = metrics.per_attack_auc(test_df).set_index(["attack_class", "manifold"])["auc"]
    pa_z = metrics.per_attack_auc(metrics.apply_znorm(test_df, stats)).set_index(
        ["attack_class", "manifold"])["auc"]
    assert pa_raw.xs("Sybil Attack").idxmax() == "network"
    assert pa_z.xs("Sybil Attack").idxmax() == "network"
    assert tcounts.get("n_timeouts_network", 0) == 0
    assert tcounts.get("n_timeouts_physical", 0) == 0
```

- [ ] **Step 4: Run it** — `python3 -m pytest tests/test_probe_znorm.py -m slow -v`, expect ~10–15 min (3,926 val + 1,000 test flows × 3 manifolds, plus fork overhead from the timeout wrapper). Detach if needed. If a gate fails, report the measured values — do not loosen without a controller ruling.
- [ ] **Step 5: Full fast suite; commit** — `"feat: wire w2 timeout and val-Normal coupled znorm probe run"`

---

### Task 3: CLI, three-seed reanalysis artifacts, and docs

**Files:**
- Modify: `uav_tda/cli.py` (probe subcommand: `--znorm` flag; new `znorm-report` subcommand)
- Modify: `CLAUDE/PROJECT_BRIEF.md` (append ONE decision-log entry), `uav_tda/unsupervised.py` docstring + `uav_tda/metrics.py` docstring only if Task 1 didn't already state the reconciliation (no logic changes)
- Test: `tests/test_cli_znorm.py`
- Output artifacts (committed): `results/tables/rebuild/probe_distances_seed{42,7,123}.csv` (+ `_znorm` variants), `results/tables/rebuild/val_normal_distances_seed{N}.csv`, `results/tables/rebuild/znorm_stats_seed{N}.json`, `results/tables/rebuild/znorm_summary.csv`, `paper/ZNORM_RESULTS.md`

**Interfaces:**
- `uav-tda probe --seed N --znorm [--w2-timeout SEC]` → runs `run_probe_with_znorm`, writes to `results/tables/rebuild/`: raw test CSV (schema as today), `probe_distances_seed{N}_znorm.csv` (z-scored columns), `val_normal_distances_seed{N}.csv`, `znorm_stats_seed{N}.json` (stats + timeout counts + provenance sidecars). Without `--znorm`, behavior identical to today.
- `uav-tda znorm-report` → reads the three seeds' rebuild CSVs (running any missing seed with `--w2-timeout 30`), then writes:
  - `results/tables/rebuild/znorm_summary.csv` — per subset × {raw, znorm}: mean, std over seeds (42, 7, 123); plus per-attack dominant-manifold table under both scorings;
  - `paper/ZNORM_RESULTS.md` — a markdown report: (1) header stating clean-lineage + val-Normal-based stats + extended timeout, superseding the test-Normal-substitute `C2_NORMALIZATION_DIAGNOSTIC.md` (link it, do not modify it); (2) raw vs znorm binary-AUC table (3-seed mean±std, all 7 subsets); (3) per-attack attribution table under both scorings; (4) seed-123 timeout reanalysis paragraph: timeout counts per manifold at 30s (expected ≈0 vs the historical 139 at 5s) and the resulting C2 numbers; (5) explicit "numbers available for the revisable abstract" framing with the paper's published values quoted alongside for comparison. Every number in the tables generated from the CSVs — no hand-typed values.

- [ ] **Step 1: Failing CLI test**

```python
# tests/test_cli_znorm.py
from uav_tda import cli


def test_probe_has_znorm_flag_and_znorm_report_registered():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    assert "znorm-report" in sub.choices
    probe_parser = sub.choices["probe"]
    opts = {s for a in probe_parser._actions for s in a.option_strings}
    assert "--znorm" in opts and "--w2-timeout" in opts
```

- [ ] **Step 2: Implement CLI wiring; unit test passes.**
- [ ] **Step 3: Run the reanalysis** (detached): `uav-tda probe --seed 42 --znorm --w2-timeout 30`, then seeds 7 and 123, then `uav-tda znorm-report`. Total ≈ 30–45 min. Verify: `znorm_summary.csv` has 7 subsets × 2 scorings; `ZNORM_RESULTS.md` tables render and every value traces to the CSVs; seed-123 timeout counts reported.
- [ ] **Step 4: Docs.** Append the Phase-3 decision-log entry to `CLAUDE/PROJECT_BRIEF.md` (dated, stating: znorm scoring implemented per §III.E on clean lineage; w2-timeout wired and seed-123 reanalyzed at 30s; canonical scoring = z-normalized sum, `unsupervised.py` remains the frozen monolith-equivalent port; `paper/ZNORM_RESULTS.md` holds the refreshed numbers pending author sign-off for the revisable abstract). Ensure the reconciliation note exists in `unsupervised.py`'s module docstring (add one sentence if Phase-2's docstring doesn't already say it — docstring-only change is permitted here) and in `metrics.py`'s Phase-3 comment block.
- [ ] **Step 5: Full fast suite; commit everything incl. artifacts** — `"feat: znorm CLI + three-seed clean-lineage reanalysis (ZNORM_RESULTS)"`

---

## Self-Review

- **Spec coverage:** §III.E val-Normal Z-normalization → T1 (math) + T2 (val pass, same-run coupling) ✓; §V extended-timeout seed-123 reanalysis → T2 (mechanism) + T3 (30s runs + report) ✓; roadmap "reconcile sum vs max-pool" → Global-Constraints ruling + T3 docstrings (docs-level, equivalence-locked ports untouched) ✓; "regenerate C2_NORMALIZATION_DIAGNOSTIC from committed code" → superseding `paper/ZNORM_RESULTS.md` generated by `uav-tda znorm-report`, historical file preserved ✓; refreshed abstract numbers → `znorm_summary.csv` + report §5 ✓.
- **Placeholder scan:** all test steps carry runnable code; T2's implementation references the recovered original's exact mechanism with a named function to port; T3's report contents enumerated section-by-section. No TBDs.
- **Type consistency:** `znorm_stats_from_val`/`apply_znorm`/`binary_auc_by_subset_znorm` consistent T1→T2→T3; `run_probe_with_znorm` 4-tuple return consistent between T2 Interfaces, T2 slow test, and T3 CLI; `stats_out`/timeout-count keys (`n_timeouts_<manifold>`) consistent T2↔T3 report.
- **Known risks, pre-ruled:** (a) fork-per-hera-call overhead could slow the val pass — acceptable (timeout only used when requested; reanalysis runs are detached); if >2× slowdown at timeout=30 vs no-timeout, implementer reports it, controller decides whether timeout applies only to c2; (b) clean-lineage gates in T2's slow test are bands, not oracle equality — any gate miss is reported with measured values for a controller ruling; (c) sparse-Rips process noise means each seed's stats differ slightly — same-run coupling (enforced by API) is the mitigation.
