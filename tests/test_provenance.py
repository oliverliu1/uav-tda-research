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


def test_git_sha_is_full_hex_when_git_reachable():
    import re

    import pytest

    from uav_tda.provenance import _git_sha, run_git

    try:
        run_git(["rev-parse", "HEAD"])
    except Exception:
        pytest.skip("no git candidate reachable")
    assert re.fullmatch(r"[0-9a-f]{40}", _git_sha())


def test_run_git_raises_when_no_candidate_works(monkeypatch):
    import pytest

    from uav_tda import provenance

    monkeypatch.setattr(provenance, "_GIT_CANDIDATES", ("/nonexistent/git",))
    with pytest.raises(Exception):
        provenance.run_git(["rev-parse", "HEAD"])
    assert provenance._git_sha() == "unknown"


def test_write_provenance_skip_if_unchanged_keeps_sidecar_stable(tmp_path: Path):
    target = tmp_path / "t.csv"
    target.write_text("x\n1\n")
    side = write_provenance(target, {"a": [1, 2]}, skip_if_unchanged=True)
    first = side.read_bytes()
    meta = json.loads(first)
    assert len(meta["artifact_sha256"]) == 64
    write_provenance(target, {"a": [1, 2]}, skip_if_unchanged=True)
    assert side.read_bytes() == first  # created_utc untouched


def test_write_provenance_skip_if_unchanged_rewrites_on_change(tmp_path: Path):
    target = tmp_path / "t.csv"
    target.write_text("x\n1\n")
    side = write_provenance(target, {"a": 1}, skip_if_unchanged=True)
    sha1 = json.loads(side.read_text())["artifact_sha256"]
    target.write_text("x\n2\n")
    write_provenance(target, {"a": 1}, skip_if_unchanged=True)
    assert json.loads(side.read_text())["artifact_sha256"] != sha1
    write_provenance(target, {"a": 2}, skip_if_unchanged=True)
    assert json.loads(side.read_text())["params"] == {"a": 2}
