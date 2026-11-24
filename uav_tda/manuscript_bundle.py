"""Self-contained copy of the manuscript for Overleaf (or any upload).

The working copy in `CLAUDE/SciTech2027_Manuscript/` reads the generated numbers, snippets
and figures from elsewhere in the repo, through the two definitions in its `paths.tex`.
`uav-tda manuscript-bundle` copies the sources plus those generated files into one folder
and writes a `paths.tex` that points inside it, so `main.tex` is byte-identical in the repo
and in the bundle and can be copied back and forth.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .paths import REPO_ROOT

MANUSCRIPT_REL = Path("CLAUDE") / "SciTech2027_Manuscript"
SOURCE_SUFFIXES = (".tex", ".cls", ".bst", ".bib")
_MARKER = "% Written by `uav-tda manuscript-bundle`"
BUNDLE_PATHS_TEX = (
    f"{_MARKER}: every path stays inside this folder.\n"
    "\\newcommand{\\rebuild}{.}\n"
    "\\graphicspath{{figures/}}\n"
)


def _uncommented(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _check_references(out: Path) -> None:
    missing = []
    for tex in out.glob("*.tex"):
        body = _uncommented(tex.read_text())
        for rel in re.findall(r"\\input\{\\rebuild/([^}]+)\}", body):
            if not (out / rel).exists():
                missing.append(f"{tex.name}: \\input {rel}")
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
            if not any((out / "figures" / (name + ext)).exists() for ext in ("", ".pdf")):
                missing.append(f"{tex.name}: figure {name}")
    if missing:
        raise FileNotFoundError("manuscript bundle is missing: " + "; ".join(missing))


def build_bundle(root=None, out_dir=None) -> Path:
    """Write the bundle folder and a zip of it; returns the zip path.

    Default folder: ``<manuscript>/build/overleaf`` (untracked). A previous bundle there is
    replaced; any other non-empty folder is refused.
    """
    root = Path(root) if root is not None else REPO_ROOT
    src = root / MANUSCRIPT_REL
    out = Path(out_dir) if out_dir is not None else src / "build" / "overleaf"
    if out.exists() and any(out.iterdir()):
        marker = out / "paths.tex"
        if not (marker.exists() and marker.read_text().startswith(_MARKER)):
            raise FileExistsError(f"{out} is not empty and was not written by this command")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.iterdir()):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            shutil.copyfile(path, out / path.name)
    (out / "paths.tex").write_text(BUNDLE_PATHS_TEX)

    for sub, pattern, source in (
        ("paper_snippets", "*.tex", root / "results" / "tables" / "rebuild" / "paper_snippets"),
        ("figures", "*.pdf", root / "results" / "figures" / "manuscript"),
    ):
        (out / sub).mkdir()
        for path in sorted(source.glob(pattern)):
            shutil.copyfile(path, out / sub / path.name)

    _check_references(out)
    return Path(shutil.make_archive(str(out), "zip", root_dir=out))
