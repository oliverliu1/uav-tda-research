from pathlib import Path

import pandas as pd
import pytest

from uav_tda import cli
from uav_tda.paths import PERSISTENCE_DIR


def test_cli_probe_writes_csv_and_provenance(tmp_path: Path):
    if not (PERSISTENCE_DIR / "network_test.pkl").exists():
        pytest.skip("Phase-3 persistence diagrams not present.")
    out = tmp_path / "probe.csv"
    rc = cli.main(["probe", "--seed", "42", "--per-class", "5", "--out", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert {"label", "W2_c2", "W2_network", "W2_physical", "W2_all_three"} <= set(df.columns)
    assert (out.with_name(out.name + ".provenance.json")).exists()
