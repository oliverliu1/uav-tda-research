import json
from pathlib import Path

from uav_tda.provenance import write_provenance


def test_write_provenance_creates_sidecar(tmp_path: Path):
    target = tmp_path / "result.csv"
    target.write_text("x\n1\n")
    side = write_provenance(target, {"seed": 42, "top_k": 50})
    assert side == tmp_path / "result.csv.provenance.json"
    meta = json.loads(side.read_text())
    assert meta["params"]["seed"] == 42
    assert "git_sha" in meta and "created_utc" in meta
    assert "numpy" in meta["library_versions"]
    assert meta["uav_tda_version"]
