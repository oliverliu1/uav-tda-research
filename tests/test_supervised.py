import pytest

from tests import monolith_harness as mh
from uav_tda.workspace import Workspace

# cmd_supervised (pipeline.py 1323-1358) writes:
#   TABLES_DIR/supervised_metrics.csv, supervised_summary.csv
#   TABLES_DIR/per_class_metrics_{feature_set}_{model}[_curated].csv  (seed=42 only)
#   TABLES_DIR/confusion_matrix_{feature_set}_{model}[_curated].csv  (seed=42 only)
# for feature_set in FEATURE_SETS x model in MODEL_NAMES, plus rf_curated per feature_set.
FEATURE_SETS = ("original", "summary_only", "summary_plus_images", "combined")
MODEL_NAMES = ("logreg", "rf", "svm")

STEMS = [f"{fs}_{m}" for fs in FEATURE_SETS for m in MODEL_NAMES] + [
    f"{fs}_rf_curated" for fs in FEATURE_SETS
]


@pytest.mark.slow
def test_supervised_debug_equivalent_to_monolith(monolith_cache, repo_root, tmp_path):
    oracle = mh.ensure_debug_phase(monolith_cache, repo_root, "supervised")
    from uav_tda.supervised import run_supervised

    ws = Workspace.at(tmp_path)
    ws.ensure()
    mh.stage_oracle(oracle, ws, include=("outputs",))

    run_supervised(ws, debug=True)

    oracle_tables = oracle / "results" / "tables"

    for name in ("supervised_metrics.csv", "supervised_summary.csv"):
        mh.assert_csvs_equal(oracle_tables / name, ws.tables_dir / name, float_rtol=1e-6)

    for stem in STEMS:
        mh.assert_csvs_equal(
            oracle_tables / f"confusion_matrix_{stem}.csv",
            ws.tables_dir / f"confusion_matrix_{stem}.csv",
        )
