"""Write a provenance sidecar next to every generated artifact."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict[str, str]:
    out = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "gudhi"):
        try:
            out[pkg.replace("scikit-learn", "sklearn")] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def write_provenance(target: Path, params: dict) -> Path:
    side = target.with_name(target.name + ".provenance.json")
    meta = {
        "artifact": target.name,
        "params": params,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": _lib_versions(),
        "uav_tda_version": __version__,
    }
    side.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return side
