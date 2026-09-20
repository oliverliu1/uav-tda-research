import importlib.util
from pathlib import Path

import uav_tda


def test_package_imports_and_has_version():
    assert isinstance(uav_tda.__version__, str)
    assert uav_tda.__version__


def _load_pipeline(repo_root: Path):
    """Import the frozen monolith as a module without running its CLI."""
    spec = importlib.util.spec_from_file_location("pipeline_frozen", repo_root / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_matches_frozen_pipeline(repo_root):
    p = _load_pipeline(repo_root)
    assert uav_tda.config.MANIFOLDS == p.MANIFOLDS
    assert uav_tda.config.EXPECTED_CLASSES == p.EXPECTED_CLASSES
    assert uav_tda.config.MAX_HOM_DIM == p.MAX_HOM_DIM
    assert uav_tda.config.SPARSE_RIPS_EPSILON == p.SPARSE_RIPS_EPSILON
    assert uav_tda.config.REFERENCE_CLOUD_SIZE == p.REFERENCE_CLOUD_SIZE
    assert uav_tda.config.MAX_EDGE_PERCENTILE == p.MAX_EDGE_PERCENTILE
    assert uav_tda.config.THRESHOLD_PERCENTILE == p.THRESHOLD_PERCENTILE
    assert uav_tda.config.PRIMARY_SEED == p.PRIMARY_SEED


def test_probe_seeds_are_paper_seeds_not_monolith_seeds(repo_root):
    p = _load_pipeline(repo_root)
    assert uav_tda.config.PROBE_SEEDS == (42, 7, 123)          # paper §IV
    assert uav_tda.config.SUPERVISED_SEEDS == p.SEEDS          # monolith's (42, 7, 2024)
    assert uav_tda.config.PROBE_SEEDS != uav_tda.config.SUPERVISED_SEEDS
