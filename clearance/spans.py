"""Towers, line route and span assignment.

The route is a polyline through the tower centres, extended past the first and last tower
to where the wire points end (tiles cut spans off at their edges). Each wire point belongs
to the route segment nearest to it in plan view; a segment is a span.
"""

from dataclasses import dataclass

import numpy as np

from .classes import PYLON
from .classify import _groups, _xy_axis, components


@dataclass
class Tower:
    id: int
    xy: np.ndarray  # (2,) centre in plan
    top_z: float
    n_points: int


@dataclass
class Span:
    id: int
    a: np.ndarray  # (2,) start in plan
    b: np.ndarray  # (2,) end in plan
    tower_a: int | None  # None where the span runs off the tile
    tower_b: int | None
    idx: np.ndarray  # indices of its wire points (into the wire point array)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def full(self) -> bool:
        return self.tower_a is not None and self.tower_b is not None


def find_towers(points, labels, min_points=200, merge_distance=10.0) -> list[Tower]:
    idx, comp = components(points, labels == PYLON, size=1.0)
    found = []
    for _, pos in _groups(comp):
        sel = idx[pos]
        if len(sel) >= min_points:
            found.append(sel)
    # Pieces of one tower (a gap in the lattice) are merged by centre distance.
    merged: list[np.ndarray] = []
    for sel in sorted(found, key=len, reverse=True):
        c = np.median(points[sel, :2], axis=0)
        for k, m in enumerate(merged):
            if np.linalg.norm(np.median(points[m, :2], axis=0) - c) < merge_distance:
                merged[k] = np.concatenate([m, sel])
                break
        else:
            merged.append(sel)
    return [
        Tower(i, np.median(points[s, :2], axis=0), float(points[s, 2].max()), len(s))
        for i, s in enumerate(merged)
    ]


def _seg_distance(xy, a, b):
    d = b - a
    t = np.clip(((xy - a) @ d) / max(d @ d, 1e-9), 0.0, 1.0)
    return np.linalg.norm(xy - (a + t[:, None] * d), axis=1)


def build_spans(wire_xy: np.ndarray, towers: list[Tower], min_end_points: int = 50) -> list[Span]:
    axis = _xy_axis(wire_xy)[0]
    s = wire_xy @ axis
    order = sorted(towers, key=lambda t: float(t.xy @ axis))
    nodes: list[tuple[np.ndarray, int | None]] = [(t.xy, t.id) for t in order]

    if not nodes:
        lo, hi = np.percentile(s, [0.5, 99.5])
        nodes = [(wire_xy[np.argmin(np.abs(s - lo))], None), (wire_xy[np.argmin(np.abs(s - hi))], None)]
    else:
        # Extend to the wire ends beyond the first and last towers.
        s_first, s_last = float(order[0].xy @ axis), float(order[-1].xy @ axis)
        before = s < s_first - 5
        after = s > s_last + 5
        if before.sum() >= min_end_points:
            far = np.argsort(s[before])[: max(5, before.sum() // 100)]
            nodes.insert(0, (wire_xy[before][far].mean(axis=0), None))
        if after.sum() >= min_end_points:
            far = np.argsort(s[after])[-max(5, after.sum() // 100) :]
            nodes.append((wire_xy[after][far].mean(axis=0), None))

    segs = list(zip(nodes, nodes[1:]))
    dist = np.column_stack([_seg_distance(wire_xy, a, b) for (a, _), (b, _) in segs])
    nearest = np.argmin(dist, axis=1)
    return [
        Span(i, a, b, ta, tb, np.nonzero(nearest == i)[0])
        for i, ((a, ta), (b, tb)) in enumerate(segs)
    ]
