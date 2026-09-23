import pandas as pd
import pytest

from tests import monolith_harness as mh
from tests.conftest import requires_data


def test_pipeline_hash_stable(repo_root):
    h1 = mh.pipeline_hash(repo_root)
    assert h1 == mh.pipeline_hash(repo_root)
    assert len(h1) == 12


@requires_data
def test_make_monolith_workspace(tmp_path, repo_root):
    ws = mh.make_monolith_workspace(tmp_path / "ws", repo_root)
    assert (ws / "pipeline.py").is_file()
    assert (ws / "data" / "UAVIDS-2025.csv").exists()  # via symlink


def test_assert_csvs_equal_detects_difference(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"x": [1.0, 2.0]}).to_csv(a, index=False)
    pd.DataFrame({"x": [1.0, 2.0]}).to_csv(b, index=False)
    mh.assert_csvs_equal(a, b)  # equal → no raise
    pd.DataFrame({"x": [1.0, 2.1]}).to_csv(b, index=False)
    with pytest.raises(AssertionError):
        mh.assert_csvs_equal(a, b)


@pytest.mark.slow
def test_ensure_debug_prep_builds_and_caches(monolith_cache, repo_root):
    ws = mh.ensure_debug_phase(monolith_cache, repo_root, "prep")
    assert (ws / ".done-prep").exists()
    assert (ws / "outputs" / "c2_train.csv").is_file()
    labels = pd.read_csv(ws / "outputs" / "labels_train.csv")
    assert len(labels) < 10000  # debug sample, not the full 85k split
    mtime = (ws / "outputs" / "c2_train.csv").stat().st_mtime
    ws2 = mh.ensure_debug_phase(monolith_cache, repo_root, "prep")  # cached: no re-run
    assert ws2 == ws
    assert (ws / "outputs" / "c2_train.csv").stat().st_mtime == mtime
