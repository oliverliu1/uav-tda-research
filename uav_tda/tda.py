"""Phase 3 — TDA persistence, ported verbatim from pipeline.py SECTION 7.

Computes a per-flow Vietoris-Rips persistence diagram against a fixed
reference cloud of REFERENCE_CLOUD_SIZE training-Normal medoids. Each
query flow's point cloud is {query} ∪ reference so the diagram describes
how that flow sits relative to a canonical normal-traffic shape rather
than a same-class neighborhood.

Note on the simplex-tree dimension: the prompt caps homology at H_k for
each manifold (k=2 for C2/Network, k=1 for Physical). Reliably computing
H_k via Rips requires (k+1)-simplices to fill k-cycles, so the simplex
tree's max_dimension is MAX_HOM_DIM[manifold] + 1.
"""

from __future__ import annotations

import logging

from . import config
from .workspace import Workspace

log = logging.getLogger("uav_tda.tda")


def load_split_manifold(ws: Workspace, manifold: str, split: str):
    """Return the (n_flows, n_features) array for one manifold-split CSV."""
    import pandas as pd

    return pd.read_csv(ws.outputs_dir / f"{manifold}_{split}.csv").values


def load_labels_for_split(ws: Workspace, split: str):
    """Return the label Series for one split CSV."""
    import pandas as pd

    return pd.read_csv(ws.outputs_dir / f"labels_{split}.csv")[config.LABEL_COLUMN]


def sample_reference_indices(ws: Workspace, seed: int):
    """Return REFERENCE_CLOUD_SIZE train positions chosen by k-medoids on Normal flows.

    Runs KMedoids on c2_train restricted to Normal-Traffic rows when
    scikit-learn-extra is importable; otherwise falls back to KMeans plus
    nearest-training-point projection. Returns absolute positions into the
    training-split row order (shared across all three manifold CSVs).
    """
    import numpy as np

    labels = load_labels_for_split(ws, "train")
    normal_positions = np.where(labels.values == "Normal Traffic")[0]
    if len(normal_positions) < config.REFERENCE_CLOUD_SIZE:
        raise AssertionError(
            f"only {len(normal_positions)} Normal Traffic rows in train; "
            f"need >= {config.REFERENCE_CLOUD_SIZE}"
        )
    c2_train = load_split_manifold(ws, "c2", "train")
    normal_c2 = c2_train[normal_positions]
    try:
        from sklearn_extra.cluster import KMedoids

        log.info("k-medoids backend: sklearn_extra.KMedoids")
        km = KMedoids(
            n_clusters=config.REFERENCE_CLOUD_SIZE,
            random_state=seed,
            method="alternate",
        )
        km.fit(normal_c2)
        local_idx = km.medoid_indices_
    except ImportError:
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans

        log.info("k-medoids backend: KMeans + nearest-point fallback")
        km = KMeans(n_clusters=config.REFERENCE_CLOUD_SIZE, random_state=seed, n_init=10)
        km.fit(normal_c2)
        local_idx = cdist(km.cluster_centers_, normal_c2).argmin(axis=1)
    abs_idx = np.sort(normal_positions[np.asarray(local_idx)])
    log.info(
        "sampled %d reference indices from %d Normal train rows",
        len(abs_idx), len(normal_positions),
    )
    return abs_idx


def compute_reference_clouds(ws: Workspace, reference_idx) -> dict:
    """Return {manifold: (REFERENCE_CLOUD_SIZE, n_features) array} by indexing train."""
    return {
        manifold: load_split_manifold(ws, manifold, "train")[reference_idx]
        for manifold in config.MANIFOLDS
    }


def compute_max_edge_lengths(reference_clouds: dict) -> dict:
    """Return {manifold: MAX_EDGE_PERCENTILE-th pairwise distance within its reference cloud}."""
    import numpy as np
    from scipy.spatial.distance import pdist

    return {
        manifold: float(np.percentile(pdist(cloud), config.MAX_EDGE_PERCENTILE))
        for manifold, cloud in reference_clouds.items()
    }


def _persistence_for_point(query_point, reference_points, max_edge, max_simplex_dim, sparse):
    """Compute one (sparse) Rips persistence diagram for {reference} ∪ {query}."""
    import gudhi
    import numpy as np

    point_cloud = np.vstack([reference_points, query_point.reshape(1, -1)])
    kwargs: dict = {"points": point_cloud, "max_edge_length": max_edge}
    if sparse is not None:
        kwargs["sparse"] = sparse
    rips = gudhi.RipsComplex(**kwargs)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_simplex_dim)
    raw = simplex_tree.persistence()
    if not raw:
        return np.empty((0, 3), dtype=float)
    return np.array(
        [[float(dim), float(birth), float(death)] for dim, (birth, death) in raw],
        dtype=float,
    )


def compute_diagrams_for_split(
    ws: Workspace,
    manifold: str,
    split: str,
    reference_points,
    max_edge: float,
    max_simplex_dim: int,
    sparse,
    n_jobs: int,
) -> list:
    """Compute per-flow persistence diagrams for one (manifold, split) in parallel."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    points = load_split_manifold(ws, manifold, split)
    log.info(
        "computing: manifold=%s split=%s flows=%d max_edge=%.4f simplex_dim=%d sparse=%s",
        manifold, split, len(points), max_edge, max_simplex_dim, sparse,
    )
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    gen = parallel(
        delayed(_persistence_for_point)(p, reference_points, max_edge, max_simplex_dim, sparse)
        for p in points
    )
    return list(tqdm(gen, total=len(points), desc=f"{manifold}/{split}"))


def validate_diagrams(diagrams, manifold: str, split: str) -> None:
    """Raise unless every feature has birth >= 0, death >= 0, death > birth."""
    for i, diag in enumerate(diagrams):
        if diag.size == 0:
            continue
        births, deaths = diag[:, 1], diag[:, 2]
        if (births < 0).any():
            raise AssertionError(f"{manifold}/{split} flow {i}: negative birth")
        if (deaths < 0).any():
            raise AssertionError(f"{manifold}/{split} flow {i}: negative death")
        if (deaths <= births).any():
            raise AssertionError(f"{manifold}/{split} flow {i}: death <= birth")


def save_diagrams(ws: Workspace, manifold: str, split: str, diagrams: list) -> None:
    """Persist diagrams as both pickle (raw list) and numpy object array."""
    import pickle

    import numpy as np

    with (ws.persistence_dir / f"{manifold}_{split}.pkl").open("wb") as fh:
        pickle.dump(diagrams, fh)
    array = np.empty(len(diagrams), dtype=object)
    for i, diag in enumerate(diagrams):
        array[i] = diag
    np.save(ws.persistence_dir / f"{manifold}_{split}.npy", array, allow_pickle=True)


def summarize_diagram_counts(diagrams: list, max_hom_dim: int) -> dict:
    """Return {f'mean_H{k}': mean per-flow count} for k in [0, max_hom_dim]."""
    import numpy as np

    counts: dict = {k: [] for k in range(max_hom_dim + 1)}
    for diag in diagrams:
        if diag.size == 0:
            for k in counts:
                counts[k].append(0)
            continue
        dims = diag[:, 0].astype(int)
        for k in counts:
            counts[k].append(int((dims == k).sum()))
    return {f"mean_H{k}": float(np.mean(vals)) for k, vals in counts.items()}


def run_tda(
    ws: Workspace,
    manifold: str = "all",
    split: str = "all",
    seed: int = config.PRIMARY_SEED,
    debug: bool = False,
    n_jobs: int = -1,
) -> None:
    """Phase 3: compute per-flow persistence diagrams against the reference cloud."""
    import json

    import numpy as np
    import pandas as pd

    reference_idx = sample_reference_indices(ws, seed)
    np.save(ws.outputs_dir / "reference_indices.npy", reference_idx)
    reference_clouds = compute_reference_clouds(ws, reference_idx)
    max_edge_lengths = compute_max_edge_lengths(reference_clouds)
    with (ws.outputs_dir / "max_edge_lengths.json").open("w") as fh:
        json.dump(max_edge_lengths, fh, indent=2)
    log.info("max edge lengths: %s", max_edge_lengths)

    manifolds = list(config.MANIFOLDS) if manifold == "all" else [manifold]
    splits = ["train", "val", "test"] if split == "all" else [split]

    summary_rows: list = []
    for m in manifolds:
        max_simplex_dim = config.MAX_HOM_DIM[m] + 1
        ref_points = reference_clouds[m]
        max_edge = max_edge_lengths[m]
        sparse = config.SPARSE_RIPS_EPSILON.get(m)
        for s in splits:
            diagrams = compute_diagrams_for_split(
                ws, m, s, ref_points, max_edge, max_simplex_dim, sparse, n_jobs=n_jobs,
            )
            validate_diagrams(diagrams, m, s)
            save_diagrams(ws, m, s, diagrams)
            stats = summarize_diagram_counts(diagrams, config.MAX_HOM_DIM[m])
            summary_rows.append(
                {"manifold": m, "split": s, "n_flows": len(diagrams), **stats}
            )
            log.info("done %s/%s: %s", m, s, stats)

    table = pd.DataFrame(summary_rows)
    log.info("tda summary:\n%s", table.to_string(index=False))
    table.to_csv(ws.outputs_dir / "tda_summary.csv", index=False)
