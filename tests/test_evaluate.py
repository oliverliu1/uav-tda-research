import pytest

from tests import monolith_harness as mh
from uav_tda.workspace import Workspace

# cmd_evaluate (pipeline.py 2182-2214) reads:
#   TABLES_DIR/{supervised_summary,supervised_metrics,
#               unsupervised_per_class_auc,unsupervised_overall_metrics}.csv
# and writes:
#   TABLES_DIR/{final_supervised,final_unsupervised,
#               ablation_manifolds,ablation_features}.csv
#   FIGURES_DIR/{methodology_flowchart,persistence_barcode_examples,
#                headline_results}.png
WRITTEN_TABLES = (
    "final_supervised.csv",
    "final_unsupervised.csv",
    "ablation_manifolds.csv",
    "ablation_features.csv",
)
WRITTEN_FIGURES = (
    "methodology_flowchart.png",
    "persistence_barcode_examples.png",
    "headline_results.png",
)


@pytest.mark.slow
def test_evaluate_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "evaluate")
    from uav_tda.evaluate import run_evaluate

    ws = Workspace.at(tmp_path)
    ws.ensure()
    # evaluate reads earlier phases' tables (supervised_summary.csv etc.) AND
    # the outputs/ feature splits used by the ablations + barcode figure, so
    # stage both subtrees from the oracle rather than regenerating upstream.
    mh.stage_oracle(oracle, ws, include=("outputs", "results"))

    # Delete the files evaluate itself writes so the test proves OUR phase
    # regenerates them (rather than just re-reading oracle leftovers).
    for name in WRITTEN_TABLES:
        (ws.tables_dir / name).unlink()
    for name in WRITTEN_FIGURES:
        (ws.figures_dir / name).unlink()

    run_evaluate(ws, debug=True)

    oracle_tables = oracle / "results" / "tables"
    oracle_figures = oracle / "results" / "figures"

    for name in WRITTEN_TABLES:
        # NOTE: RF n_jobs=-1 ablation cells can flake at 1e-6 if the oracle cache is rebuilt (cross-process float nondeterminism; see task-8 forensics in the SDD ledger).
        mh.assert_csvs_equal(oracle_tables / name, ws.tables_dir / name, float_rtol=1e-6)

    for name in WRITTEN_FIGURES:
        fig = ws.figures_dir / name
        assert fig.is_file() and fig.stat().st_size > 0, name
        assert (oracle_figures / name).is_file(), name
