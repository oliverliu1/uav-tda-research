from uav_tda import cli


def test_probe_has_znorm_flag_and_znorm_report_registered():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    assert "znorm-report" in sub.choices
    probe_parser = sub.choices["probe"]
    opts = {s for a in probe_parser._actions for s in a.option_strings}
    assert "--znorm" in opts and "--w2-timeout" in opts


def test_probe_znorm_flag_defaults_false():
    args = cli.build_parser().parse_args(["probe", "--seed", "42"])
    assert args.znorm is False


def test_probe_znorm_flag_can_be_set():
    args = cli.build_parser().parse_args(["probe", "--seed", "42", "--znorm", "--w2-timeout", "30"])
    assert args.znorm is True
    assert args.w2_timeout == 30.0


def test_znorm_report_subcommand_dispatches_to_func():
    args = cli.build_parser().parse_args(["znorm-report"])
    assert args.func == cli._cmd_znorm_report
