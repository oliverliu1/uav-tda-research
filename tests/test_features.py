import pytest

from tests import monolith_harness as mh
from uav_tda.workspace import Workspace


@pytest.mark.slow
def test_features_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "features")
    from uav_tda.features import run_features
    ws = Workspace.at(tmp_path)
    ws.ensure()
    mh.stage_oracle(oracle, ws, include=("outputs",))
    # remove the oracle's own features output so we prove OUR phase rebuilds it
    import shutil
    shutil.rmtree(ws.tda_features_dir, ignore_errors=True)
    (ws.outputs_dir / "persistence_imagers.pkl").unlink()
    run_features(ws, debug=True)
    oracle_csvs = sorted((oracle / "outputs" / "tda_features").glob("*.csv"))
    assert oracle_csvs, "oracle produced no feature CSVs — harness bug"
    for oc in oracle_csvs:
        mh.assert_csvs_equal(oc, ws.tda_features_dir / oc.name)
