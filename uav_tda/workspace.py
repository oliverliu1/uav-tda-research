"""Rebasable filesystem layout mirroring pipeline.py SECTION 1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Workspace:
    root: Path
    data_csv: Path
    outputs_dir: Path
    persistence_dir: Path
    tda_features_dir: Path
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    logs_dir: Path

    @classmethod
    def at(cls, root: Path) -> "Workspace":
        root = Path(root)
        outputs = root / "outputs"
        results = root / "results"
        return cls(
            root=root,
            data_csv=root / "data" / "UAVIDS-2025.csv",
            outputs_dir=outputs,
            persistence_dir=outputs / "persistence_diagrams",
            tda_features_dir=outputs / "tda_features",
            results_dir=results,
            tables_dir=results / "tables",
            figures_dir=results / "figures",
            models_dir=results / "models",
            logs_dir=root / "logs",
        )

    @classmethod
    def default(cls) -> "Workspace":
        return cls.at(paths.REPO_ROOT)

    def ensure(self) -> None:
        for d in (self.outputs_dir, self.persistence_dir, self.tda_features_dir,
                  self.results_dir, self.tables_dir, self.figures_dir,
                  self.models_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
