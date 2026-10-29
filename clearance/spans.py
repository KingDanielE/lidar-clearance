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


def _kinks(s: np.ndarray, z: np.ndarray, step: float = 1.0, window: float = 15.0):
    """Bin centres and the slope drop across each (left slope - right slope) of the wires'
    upper envelope. Catenaries curve upwards (negative drop); a tower is a sharp positive one.
    """
    edges = np.arange(s.min(), s.max() + step, step)
    b = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 1)
    centre = edges + step / 2
    env = np.full(len(edges), np.nan)
    for i in np.unique(b):
        env[i] = np.percentile(z[b == i], 90)
    ok = ~np.isnan(env)
    drop = np.full(len(edges), np.nan)
    for i in range(len(edges)):
        left = ok & (centre >= centre[i] - window) & (centre < centre[i])
        right = ok & (centre > centre[i]) & (centre <= centre[i] + window)
        # Both sides nearly full: a kink measured on a few metres at the end of the data
        # (a tile edge) fired on a tree at wire height there.
        if left.sum() >= 0.8 * window / step and right.sum() >= 0.8 * window / step:
            drop[i] = (
                np.polyfit(centre[left], env[left], 1)[0]
                - np.polyfit(centre[right], env[right], 1)[0]
            )
    return centre, drop


def towers_from_wires(
    points,
    labels,
    min_kink: float = 0.03,
    min_steel: int = 300,
    column_r: float = 2.5,
    slide: float = 8.0,
    separation: float = 50.0,
    wire_mask=None,
) -> list[Tower]:
    """Towers the pylon classification missed (in forest a tower is fused with the canopy).

    A tower is where (1) the conductors' upper envelope kinks downwards, and (2) the line's
    centre column, between the lowest conductor and the top wire, holds structure that is
    neither ground nor wire. At mid-span that column is open air: vegetation must stay below
    the conductors. The column is slid along the line to find the tower body exactly.

    `wire_mask` restricts the wires to one direction group (graph.py), so that lines
    running in different directions each get their own axis.
    """
    from scipy.spatial import cKDTree

    from .classes import GROUND, WIRE

    w = points[(labels == WIRE) if wire_mask is None else wire_mask]
    other = points[(labels != WIRE) & (labels != GROUND)]
    if len(w) < 200 or len(other) == 0:
        return []
    axis = _xy_axis(w[:, :2])[0]
    c0 = w[:, :2].mean(axis=0)
    s = (w[:, :2] - c0) @ axis
    centre, drop = _kinks(s, w[:, 2])
    tree = cKDTree(other[:, :2])

    def column(sk):
        m = np.abs(s - sk) < 3.0
        if m.sum() < 10:
            return 0, None
        xy = np.median(w[m, :2], axis=0)
        lo, hi = np.percentile(w[m, 2], [5, 95])
        nb = np.asarray(tree.query_ball_point(xy, column_r), dtype=int)
        zz = other[nb, 2]
        return int(np.sum((zz > lo - 3.0) & (zz < hi - 1.0))), xy

    found = []
    for i in np.argsort(-np.nan_to_num(drop, nan=-1.0)):
        if not drop[i] > min_kink:
            break
        if any(abs(centre[i] - f[0]) < separation for f in found):
            continue
        best = max(
            (column(centre[i] + d) + (centre[i] + d,) for d in np.arange(-slide, slide + 1, 1.0)),
            key=lambda r: r[0],
        )
        n, xy, sk = best
        if n >= min_steel:
            found.append((sk, xy, n))
    out = []
    for k, (_, xy, n) in enumerate(found):
        nb = np.asarray(tree.query_ball_point(xy, 6.0), dtype=int)
        out.append(Tower(k, xy, float(other[nb, 2].max()), n))
    return out


def merge_towers(primary: list[Tower], extra: list[Tower], distance: float = 20.0) -> list[Tower]:
    """Primary towers plus any extra tower not within `distance` of one; ids renumbered."""
    keep = list(primary)
    for t in extra:
        if all(np.linalg.norm(t.xy - p.xy) > distance for p in keep):
            keep.append(t)
    return [Tower(i, t.xy, t.top_z, t.n_points) for i, t in enumerate(keep)]


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
        nodes = [
            (wire_xy[np.argmin(np.abs(s - lo))], None),
            (wire_xy[np.argmin(np.abs(s - hi))], None),
        ]
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
