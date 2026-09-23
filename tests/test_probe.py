import numpy as np
import pytest

from uav_tda import probe


def test_truncate_top_k_keeps_longest_bars():
    # persistence = death - birth: bars have lengths 1, 5, 2, 0.5
    diag = np.array([[0.0, 1.0], [0.0, 5.0], [1.0, 3.0], [2.0, 2.5]])
    kept = probe.truncate_top_k(diag, k=2)
    lengths = np.sort(kept[:, 1] - kept[:, 0])
    assert np.allclose(lengths, [2.0, 5.0])


def test_truncate_top_k_handles_fewer_than_k():
    diag = np.array([[0.0, 1.0]])
    assert probe.truncate_top_k(diag, k=50).shape == (1, 2)


def test_truncate_top_k_empty():
    assert probe.truncate_top_k(np.empty((0, 2)), k=50).shape == (0, 2)


from uav_tda import metrics  # noqa: E402
from uav_tda.paths import PERSISTENCE_DIR  # noqa: E402

# Oracle: PROBE_RESULTS.md confirmation pass, seed 42.
ORACLE_BINARY_SEED42 = {
    "network_physical": 0.8550, "all_three": 0.8712, "network_only": 0.7481,
}
ORACLE_SYBIL_NETWORK_SEED42 = 0.8717


@pytest.mark.slow
def test_run_probe_reproduces_seed42_oracle_aucs():
    # Validates the reconstructed runner on the CLEAN rebuilt diagrams. Tolerance
    # is seed-variance scale (0.05), NOT 0.01: the rebuilt diagrams use the correct
    # 15% split, so exact reproduction of the old-split published numbers is not
    # expected (see the plan's Lineage note). We assert the finding is preserved:
    # Network+Physical strong, Sybil dominant on Network.
    if not (PERSISTENCE_DIR / "network_test.pkl").exists():
        pytest.skip("Phase-3 persistence diagrams not present; run `pipeline.py tda --split test` first.")
    df = probe.run_probe(seed=42)
    binary = metrics.binary_auc_by_subset(df)
    for subset, expected in ORACLE_BINARY_SEED42.items():
        assert binary[subset] == pytest.approx(expected, abs=0.05), subset
    pa = metrics.per_attack_auc(df).set_index(["attack_class", "manifold"])["auc"]
    assert pa[("Sybil Attack", "network")] == pytest.approx(ORACLE_SYBIL_NETWORK_SEED42, abs=0.05)
    # Attribution must hold: Sybil's dominant manifold is Network.
    sybil = pa.xs("Sybil Attack", level="attack_class")
    assert sybil.idxmax() == "network"
