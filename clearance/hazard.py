"""Hazard (fall-in) trees: trees outside the clearance zone that could hit the line falling.

Treetops are local maxima of a canopy height model. Each vegetation point is given to its
nearest treetop (within a reach that grows with the tree's height), which yields per tree:
- the crown radius: spread in plan of its points in the upper half of the tree;
- the base: where its low points stand (stem, lower branches), if there are enough of
  them; otherwise the tree is taken as vertical. Top minus base is the lean.
A tree pivots at its base: the stem (base to top) plus the crown at its top can reach
anything within sqrt(stem^2 + crown_radius^2) of the base, in any direction, including
downhill (the reach is a sphere in 3D). Which way a leaning tree is likely to fall is not
modelled; the sphere covers every direction.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .ground import Raster

STRIKE, FLASHOVER, SAFE = "strike", "within_mvcd", "safe"


def tree_tops(
    veg: np.ndarray,
    hag: np.ndarray,
    cell: float = 0.5,
    min_height: float = 3.0,
    window_m: float = 3.0,
):
    """(xy (n, 2), height (n,)) of treetops from vegetation points and their HAG."""
    if len(veg) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    x0, y0 = veg[:, 0].min(), veg[:, 1].min()
    c = ((veg[:, 0] - x0) / cell).astype(int)
    r = ((veg[:, 1] - y0) / cell).astype(int)
    chm = np.zeros((r.max() + 1, c.max() + 1), dtype=np.float32)
    np.maximum.at(chm, (r, c), hag)
    smooth = ndimage.gaussian_filter(chm, sigma=1.0)
    size = max(3, int(round(window_m / cell)) | 1)
    peaks = (smooth == ndimage.maximum_filter(smooth, size=size)) & (chm >= min_height)
    rr, cc = np.nonzero(peaks)
    # Height from the unsmoothed CHM around the peak.
    h = ndimage.maximum_filter(chm, size=3)[rr, cc]
    xy = np.column_stack([x0 + (cc + 0.5) * cell, y0 + (rr + 0.5) * cell])
    return xy, h


def crowns(
    veg: np.ndarray, hag: np.ndarray, xy: np.ndarray, height: np.ndarray, min_base_points: int = 10
):
    """Per tree: (crown radius (n,), base xy (n, 2)). Points go to the nearest treetop in
    plan, within 0.4 * height + 2 m (a crown is rarely wider than that)."""
    n = len(xy)
    radius = np.zeros(n)
    base = np.array(xy, dtype=np.float64, copy=True)
    if n == 0 or len(veg) == 0:
        return radius, base
    d, owner = cKDTree(xy).query(veg[:, :2], k=1, workers=-1)
    ok = d <= 0.4 * height[owner] + 2.0
    order = np.argsort(owner[ok], kind="stable")
    pts, hh, own = veg[ok][order], hag[ok][order], owner[ok][order]
    ids, starts = np.unique(own, return_index=True)
    for t, sel in zip(ids, np.split(np.arange(len(own)), starts[1:])):
        p, z = pts[sel], hh[sel]
        upper = z >= 0.5 * height[t]
        if upper.sum() >= 5:
            radius[t] = float(np.percentile(np.linalg.norm(p[upper, :2] - xy[t], axis=1), 90))
        low = z <= max(2.0, 0.25 * height[t])
        if low.sum() >= min_base_points:
            b = np.median(p[low, :2], axis=0)
            # A lean beyond half the height is a neighbour's understorey, not this stem.
            if np.linalg.norm(b - xy[t]) <= 0.5 * height[t]:
                base[t] = b
    return radius, base


def fall_in(
    xy: np.ndarray,
    height: np.ndarray,
    dtm: Raster,
    wire_samples: np.ndarray,
    mvcd: float,
    base_xy: np.ndarray | None = None,
    crown_radius: np.ndarray | None = None,
):
    """Per tree: base z, distance from its base to the nearest conductor sample, reach
    margin (distance - reach), status. Without base_xy / crown_radius: a vertical stick."""
    if len(xy) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.array([], dtype=object)
    bxy = xy if base_xy is None else base_xy
    base = np.column_stack([bxy, dtm.sample(bxy)])
    stem = np.hypot(np.linalg.norm(np.asarray(xy) - bxy, axis=1), height)
    reach = stem if crown_radius is None else np.hypot(stem, crown_radius)
    d, _ = cKDTree(wire_samples).query(base, k=1, workers=-1)
    margin = d - reach  # < 0: the tree is long enough to land on the wire
    status = np.full(len(xy), SAFE, dtype=object)
    status[margin < mvcd] = FLASHOVER
    status[margin < 0] = STRIKE
    return base[:, 2], d, margin, status
