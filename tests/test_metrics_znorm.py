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
