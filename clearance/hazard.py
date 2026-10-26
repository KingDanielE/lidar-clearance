"""Hazard (fall-in) trees: trees outside the clearance zone that could hit the line falling.

Treetops are local maxima of a canopy height model. A tree is taken as vertical, pivoting
at its base: it can reach anything within its height of the base. Leaning trees, crown
width and the ground slope (a tree falling downhill reaches further) are not modelled.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .ground import Raster

STRIKE, FLASHOVER, SAFE = "strike", "within_mvcd", "safe"


def tree_tops(veg: np.ndarray, hag: np.ndarray, cell: float = 0.5, min_height: float = 3.0, window_m: float = 3.0):
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


def fall_in(xy: np.ndarray, height: np.ndarray, dtm: Raster, wire_samples: np.ndarray, mvcd: float):
    """Per tree: distance from its base to the nearest conductor sample, reach margin, status."""
    if len(xy) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.array([], dtype=object)
    base = np.column_stack([xy, dtm.sample(xy)])
    d, _ = cKDTree(wire_samples).query(base, k=1, workers=-1)
    margin = d - height  # < 0: the tree is long enough to land on the wire
    status = np.full(len(xy), SAFE, dtype=object)
    status[margin < mvcd] = FLASHOVER
    status[margin < 0] = STRIKE
    return base[:, 2], d, margin, status
