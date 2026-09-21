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


from uav_tda import metrics  # noqa: E402
from uav_tda.paths import PERSISTENCE_DIR  # noqa: E402


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
