import pathlib

from uav_tda import cli


def test_all_phase_subcommands_registered():
    parser = cli.build_parser()
    subactions = next(a for a in parser._actions if getattr(a, "choices", None))
    for cmd in ("probe", "prep", "tda", "features", "supervised",
                "unsupervised", "evaluate", "all"):
        assert cmd in subactions.choices, cmd


def test_prep_debug_smoke(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    (tmp_path / "data").symlink_to(repo_root / "data")
    rc = cli.main(["prep", "--debug", "--root", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "outputs" / "labels_train.csv").is_file()
