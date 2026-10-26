"""Ground filtering: progressive morphological filter (Zhang et al. 2003) on a raster.

Produces a terrain model (DTM) and height above ground (HAG) for every point.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Raster:
    z: np.ndarray  # (rows, cols), rows along y
    x0: float
    y0: float
    cell: float

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear sample at xy (n, 2)."""
        col = (xy[:, 0] - self.x0) / self.cell - 0.5
        row = (xy[:, 1] - self.y0) / self.cell - 0.5
        return ndimage.map_coordinates(self.z, [row, col], order=1, mode="nearest")


def _fill_nearest(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def min_z_raster(points: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    x0, y0 = float(points[:, 0].min()), float(points[:, 1].min())
    cols = ((points[:, 0] - x0) / cell).astype(np.int64)
    rows = ((points[:, 1] - y0) / cell).astype(np.int64)
    shape = (rows.max() + 1, cols.max() + 1)
    flat = rows * shape[1] + cols
    grid = np.full(shape[0] * shape[1], np.inf, dtype=np.float32)
    # Sort by z descending so the last write per cell is the minimum.
    order = np.argsort(points[:, 2])[::-1]
    grid[flat[order]] = points[order, 2]
    grid = grid.reshape(shape)
    return grid, np.isfinite(grid), x0, y0


def pmf_dtm(
    points: np.ndarray,
    cell: float = 0.5,
    max_window: float = 24.0,
    slope: float = 0.3,
    dh0: float = 0.3,
    dh_max: float = 2.5,
) -> Raster:
    zmin, has_data, x0, y0 = min_z_raster(points, cell)
    surface = _fill_nearest(zmin, has_data)
    # Remove low outliers (multipath "holes") before opening, or they drag the DTM down.
    med = ndimage.median_filter(surface, size=5)
    low = (med - surface) > 1.0
    surface = np.where(low, med, surface)

    nonground = ~has_data | low
    prev_w = 1
    w = 3
    while w * cell <= max_window:
        opened = ndimage.grey_opening(surface, size=(w, w))
        dh = min(dh0 + slope * (w - prev_w) * cell, dh_max)
        nonground |= (surface - opened) > dh
        surface = opened
        prev_w, w = w, 2 * w - 1

    ground = ~nonground
    dtm = _fill_nearest(zmin, ground)
    dtm = ndimage.uniform_filter(dtm, size=3)
    return Raster(dtm.astype(np.float32), x0, y0, cell)


def height_above_ground(points: np.ndarray, dtm: Raster) -> np.ndarray:
    return (points[:, 2] - dtm.sample(points[:, :2])).astype(np.float32)
