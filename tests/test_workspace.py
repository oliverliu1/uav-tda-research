from pathlib import Path

from uav_tda import paths
from uav_tda.workspace import Workspace


def test_default_matches_paths_constants():
    ws = Workspace.default()
    assert ws.root == paths.REPO_ROOT
    assert ws.data_csv == paths.DATA_PATH
    assert ws.outputs_dir == paths.OUTPUTS_DIR
    assert ws.persistence_dir == paths.PERSISTENCE_DIR
    assert ws.tables_dir == paths.TABLES_DIR
    assert ws.figures_dir == paths.FIGURES_DIR


def test_at_rebases_everything(tmp_path: Path):
    ws = Workspace.at(tmp_path)
    assert ws.root == tmp_path
    assert ws.outputs_dir == tmp_path / "outputs"
    assert ws.persistence_dir == tmp_path / "outputs" / "persistence_diagrams"
    assert ws.tda_features_dir == tmp_path / "outputs" / "tda_features"
    assert ws.tables_dir == tmp_path / "results" / "tables"
    assert ws.models_dir == tmp_path / "results" / "models"
    assert ws.logs_dir == tmp_path / "logs"
    assert ws.data_csv == tmp_path / "data" / "UAVIDS-2025.csv"


def test_ensure_creates_dirs(tmp_path: Path):
    ws = Workspace.at(tmp_path)
    ws.ensure()
    for d in (ws.outputs_dir, ws.persistence_dir, ws.tda_features_dir,
              ws.tables_dir, ws.figures_dir, ws.models_dir, ws.logs_dir):
        assert d.is_dir()
