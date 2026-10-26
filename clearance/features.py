"""Per-point geometric features from the local neighbourhood covariance."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Features:
    linearity: np.ndarray  # (l1 - l2) / l1: ~1 on wires and pylon members
    planarity: np.ndarray  # (l2 - l3) / l1: ~1 on roofs and ground
    scattering: np.ndarray  # l3 / l1: high in tree crowns
    dir_z: np.ndarray  # |z| of the main direction: ~0 on wires, ~1 on poles
    normal_z: np.ndarray  # |z| of the normal: ~1 on flat roofs
    gap_below: np.ndarray  # vertical gap to the next lower point in the same column


def eigen_features(points: np.ndarray, k: int = 16, chunk: int = 400_000) -> tuple:
    n = len(points)
    tree = cKDTree(points)
    lin = np.empty(n, np.float32)
    plan = np.empty(n, np.float32)
    scat = np.empty(n, np.float32)
    dz = np.empty(n, np.float32)
    nz = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, idx = tree.query(points[s:e], k=k, workers=-1)
        nb = points[idx].astype(np.float64)
        nb -= nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / k
        w, v = np.linalg.eigh(cov)  # ascending eigenvalues
        w = np.clip(w, 1e-12, None)
        l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
        lin[s:e] = (l1 - l2) / l1
        plan[s:e] = (l2 - l3) / l1
        scat[s:e] = l3 / l1
        dz[s:e] = np.abs(v[:, 2, 2])
        nz[s:e] = np.abs(v[:, 2, 0])
    return lin, plan, scat, dz, nz


def gap_below(points: np.ndarray, cell: float = 0.5, join: float = 0.5) -> np.ndarray:
    """Empty space below each point's vertical run in its xy column (inf for the lowest run).

    Points in a column closer than `join` apart form one run, so a sloping wire (several
    points stacked in one column) still sees the open air below it rather than itself.
    """
    cx = np.floor(points[:, 0] / cell).astype(np.int64)
    cy = np.floor(points[:, 1] / cell).astype(np.int64)
    col = (cx - cx.min()) * (cy.max() - cy.min() + 1) + (cy - cy.min())
    order = np.lexsort((points[:, 2], col))
    z = points[order, 2]
    c = col[order]
    n = len(z)
    step = np.full(n, np.inf, dtype=np.float32)  # gap to the point just below
    same = c[1:] == c[:-1]
    step[1:] = np.where(same, z[1:] - z[:-1], np.inf)
    starts = step > join  # first point of each run
    run = np.cumsum(starts) - 1
    gap = np.empty(n, dtype=np.float32)
    gap[order] = step[np.nonzero(starts)[0]][run]
    return gap


def compute(points: np.ndarray, k: int = 16) -> Features:
    lin, plan, scat, dz, nz = eigen_features(points, k=k)
    return Features(lin, plan, scat, dz, nz, gap_below(points))
