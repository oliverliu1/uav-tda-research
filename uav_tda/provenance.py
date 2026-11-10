"""Write a provenance sidecar next to every generated artifact."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__


# Under Rosetta (x86_64 Python on Apple silicon) the /usr/bin/git xcrun shim
# can fail to load an arm64-only libxcrun; fall back to the Command Line
# Tools git binary directly.
_GIT_CANDIDATES = ("git", "/Library/Developer/CommandLineTools/usr/bin/git")
GIT_TIMEOUT_S = 30


def run_git(args, cwd=None) -> str:
    """Run ``git *args`` via the first working candidate binary; return stripped stdout.

    ``cwd`` defaults to this package's directory. Each call is bounded by
    `GIT_TIMEOUT_S`; a timeout counts as that candidate failing. Raises the
    last error if every candidate fails.
    """
    cwd = Path(__file__).parent if cwd is None else cwd
    last_exc: Exception = RuntimeError("no git candidate")
    for git in _GIT_CANDIDATES:
        try:
            return subprocess.check_output(
                [git, *args], cwd=cwd, stderr=subprocess.DEVNULL, timeout=GIT_TIMEOUT_S,
            ).decode().strip()
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _git_sha() -> str:
    try:
        return run_git(["rev-parse", "HEAD"])
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


def write_provenance(target: Path, params: dict, skip_if_unchanged: bool = False) -> Path:
    """Write ``<target>.provenance.json`` and return its path.

    With ``skip_if_unchanged`` the sidecar also records ``artifact_sha256``
    (the target's content hash), and an existing sidecar whose every field
    except ``created_utc`` already matches (same params, content hash,
    git_sha, library versions) is left untouched, so idempotent reruns are
    byte-stable. Default behaviour (always rewrite, no hash) is unchanged.
    """
    side = target.with_name(target.name + ".provenance.json")
    meta = {
        "artifact": target.name,
        "params": params,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": _lib_versions(),
        "uav_tda_version": __version__,
    }
    if skip_if_unchanged:
        meta["artifact_sha256"] = hashlib.sha256(Path(target).read_bytes()).hexdigest()
        if side.exists():
            try:
                old = json.loads(side.read_text())
            except ValueError:
                old = None
            new = json.loads(json.dumps(meta, sort_keys=True))
            if isinstance(old, dict):
                old.pop("created_utc", None)
                new.pop("created_utc", None)
                if old == new:
                    return side
    side.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return side
