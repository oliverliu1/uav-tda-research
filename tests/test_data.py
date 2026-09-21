from pathlib import Path

import pytest

from tests import monolith_harness as mh
from uav_tda.workspace import Workspace

PREP_CSVS = [f"{m}_{s}.csv" for m in ("c2", "network", "physical", "original_features")
             for s in ("train", "val", "test")] + [f"labels_{s}.csv" for s in ("train", "val", "test")]


@pytest.mark.slow
def test_prep_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "prep")
    from uav_tda.data import run_prep
    ws = Workspace.at(tmp_path)
    (tmp_path / "data").symlink_to(repo_root / "data")
    ws.ensure()
    run_prep(ws, debug=True)
    for name in PREP_CSVS:
        mh.assert_csvs_equal(oracle / "outputs" / name, ws.outputs_dir / name)
    import numpy as np
    for name in ("train_indices.npy", "val_indices.npy", "test_indices.npy"):
        assert np.array_equal(np.load(oracle / "outputs" / name),
                              np.load(ws.outputs_dir / name))
