"""Self-contained manuscript bundle (for Overleaf): no path leaves the bundle folder."""
import zipfile

import pytest

from uav_tda import cli
from uav_tda import manuscript_bundle as mb

_MAIN = (
    "\\documentclass{article}\n\\input{paths}\n"
    "\\input{\\rebuild/paper_snippets/a.tex}\n"
    "\\begin{document}\\includegraphics[width=1in]{fig_x}\n"
    "% \\includegraphics{fig_commented_out}\n"
    "\\bibliography{references}\\end{document}\n"
)


def _root(tmp_path):
    src = tmp_path / "CLAUDE" / "SciTech2027_Manuscript"
    (src / "build").mkdir(parents=True)
    (src / "main.tex").write_text(_MAIN)
    (src / "paths.tex").write_text("\\newcommand{\\rebuild}{../../results/tables/rebuild}\n")
    (src / "references.bib").write_text("@misc{a, title={t}}\n")
    (src / "new-aiaa.cls").write_text("% cls\n")
    (src / "README.md").write_text("not part of the bundle\n")
    (src / "build" / "main.aux").write_text("junk\n")
    snippets = tmp_path / "results" / "tables" / "rebuild" / "paper_snippets"
    snippets.mkdir(parents=True)
    (snippets / "a.tex").write_text("1 & 2 \\\\\n")
    (snippets / "a.tex.provenance.json").write_text("{}")
    figures = tmp_path / "results" / "figures" / "manuscript"
    figures.mkdir(parents=True)
    (figures / "fig_x.pdf").write_bytes(b"%PDF-1.4 x")
    (figures / "fig_x.pdf.provenance.json").write_text("{}")
    return tmp_path, src


def test_bundle_is_flat_and_main_tex_is_byte_identical(tmp_path):
    root, src = _root(tmp_path)
    zip_path = mb.build_bundle(root)
    out = src / "build" / "overleaf"
    assert zip_path == src / "build" / "overleaf.zip"
    names = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert names == ["figures/fig_x.pdf", "main.tex", "new-aiaa.cls",
                     "paper_snippets/a.tex", "paths.tex", "references.bib"]
    # main.tex is the same file in the repo and on Overleaf; only paths.tex differs.
    assert (out / "main.tex").read_bytes() == (src / "main.tex").read_bytes()
    paths = (out / "paths.tex").read_text()
    assert "\\newcommand{\\rebuild}{.}" in paths and "\\graphicspath{{figures/}}" in paths
    assert ".." not in paths
    with zipfile.ZipFile(zip_path) as z:
        assert sorted(n for n in z.namelist() if not n.endswith("/")) == names


def test_bundle_rebuild_replaces_the_previous_bundle(tmp_path):
    root, src = _root(tmp_path)
    mb.build_bundle(root)
    stale = src / "build" / "overleaf" / "stale.tex"
    stale.write_text("old\n")
    mb.build_bundle(root)
    assert not stale.exists()


def test_bundle_refuses_to_overwrite_a_folder_it_did_not_write(tmp_path):
    root, _ = _root(tmp_path)
    other = tmp_path / "mine"
    other.mkdir()
    (other / "thesis.tex").write_text("keep me\n")
    with pytest.raises(FileExistsError, match="mine"):
        mb.build_bundle(root, out_dir=other)
    assert (other / "thesis.tex").read_text() == "keep me\n"


@pytest.mark.parametrize("line,missing", [
    ("\\input{\\rebuild/paper_snippets/nope.tex}", "nope.tex"),
    ("\\includegraphics{fig_missing}", "fig_missing"),
])
def test_bundle_fails_on_a_reference_it_cannot_satisfy(tmp_path, line, missing):
    root, src = _root(tmp_path)
    (src / "main.tex").write_text(_MAIN.replace("\\begin{document}", line + "\n\\begin{document}"))
    with pytest.raises(FileNotFoundError, match=missing):
        mb.build_bundle(root)


def test_real_manuscript_bundles(tmp_path):
    """Every snippet and figure the working copy references exists."""
    out = tmp_path / "overleaf"
    mb.build_bundle(out_dir=out)
    assert (out / "main.tex").exists() and (out / "paper_snippets" / "manuscript_numbers.tex").exists()
    assert len(list((out / "figures").glob("*.pdf"))) >= 6


def test_cli_manuscript_bundle(tmp_path, capsys):
    root, src = _root(tmp_path)
    assert cli.main(["manuscript-bundle", "--root", str(root)]) == 0
    assert "overleaf.zip" in capsys.readouterr().out
    assert (src / "build" / "overleaf.zip").exists()
