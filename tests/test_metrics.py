import pandas as pd
import pytest

from uav_tda import metrics
from uav_tda.paths import probe_distances_csv

# --- Oracles: PROBE_RESULTS.md confirmation pass, seed 42 ---
SEED42_BINARY = {
    "c2_only": 0.7599, "network_only": 0.7481, "physical_only": 0.6192,
    "c2_network": 0.8438, "c2_physical": 0.7680,
    "network_physical": 0.8550, "all_three": 0.8712,
}
SEED42_PER_ATTACK = {  # (attack, manifold): auc
    ("Blackhole Attack", "c2"): 0.4473, ("Blackhole Attack", "network"): 0.3106, ("Blackhole Attack", "physical"): 0.8032,
    ("Flooding Attack", "c2"): 0.6333, ("Flooding Attack", "network"): 0.7919, ("Flooding Attack", "physical"): 0.3784,
    ("Sybil Attack", "c2"): 0.6443, ("Sybil Attack", "network"): 0.8717, ("Sybil Attack", "physical"): 0.2119,
    ("Wormhole Attack", "c2"): 0.5350, ("Wormhole Attack", "network"): 0.2739, ("Wormhole Attack", "physical"): 0.7256,
}
# --- Oracle: MULTI_SEED_VARIANCE.md, mean over seeds 42/7/123 ---
MULTISEED_BINARY_MEAN = {
    "c2_only": 0.7488, "network_only": 0.7610, "physical_only": 0.6114,
    "c2_network": 0.8304, "c2_physical": 0.7542,
    "network_physical": 0.8594, "all_three": 0.8577,
}


def test_binary_auc_matches_seed42_oracle():
    df = pd.read_csv(probe_distances_csv(42))
    got = metrics.binary_auc_by_subset(df)
    for subset, expected in SEED42_BINARY.items():
        assert got[subset] == pytest.approx(expected, abs=1e-3), subset


def test_per_attack_auc_matches_seed42_oracle():
    df = pd.read_csv(probe_distances_csv(42))
    tbl = metrics.per_attack_auc(df).set_index(["attack_class", "manifold"])["auc"]
    for (attack, manifold), expected in SEED42_PER_ATTACK.items():
        assert tbl[(attack, manifold)] == pytest.approx(expected, abs=1e-3), (attack, manifold)


def test_subset_columns_equal_sum_of_single_manifolds():
    # Invariant that defines the probe's combined score (SUM, not max-pool).
    df = pd.read_csv(probe_distances_csv(42))
    assert (df["W2_c2_network"] - (df["W2_c2"] + df["W2_network"])).abs().max() < 1e-9
    assert (df["W2_all_three"] - (df["W2_c2"] + df["W2_network"] + df["W2_physical"])).abs().max() < 1e-9


def test_multiseed_binary_mean_matches_oracle():
    dfs = {s: pd.read_csv(probe_distances_csv(s)) for s in (42, 7, 123)}
    agg = metrics.aggregate_over_seeds(dfs)["binary"].set_index("subset")["mean"]
    for subset, expected in MULTISEED_BINARY_MEAN.items():
        assert agg[subset] == pytest.approx(expected, abs=1e-3), subset
