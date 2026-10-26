"""Rule-based classification: ground, wires, pylons, buildings, vegetation.

Order matters: ground first (from the DTM), then wires (thin, linear, horizontal, with
empty space below), then pylons (tall, regular structures under the wires), then
buildings (planar), and everything else above ground is vegetation.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .classes import BUILDING, GROUND, PYLON, UNCLASSIFIED, VEGETATION, WIRE
from .features import Features


@dataclass
class Params:
    # Points within this of the DTM are ground. Rough terrain and grass sit 0.3-0.5 m above
    # a raster DTM; anything this low can never reach a conductor anyway.
    ground_hag: float = 0.5
    wire_min_hag: float = 3.0
    wire_linearity: float = 0.8
    # |z| of the main direction: spans down a mountainside run at ~30 deg (0.5).
    wire_max_dir_z: float = 0.6
    wire_min_gap: float = 1.0  # empty space below a wire point
    wire_min_length: float = 15.0  # a real span is long
    wire_min_aspect: float = 4.0  # length / width of a wire cluster
    wire_fragment_length: float = 3.0  # short pieces kept if they line up with known wires
    wire_fragment_cos: float = 0.98  # |cos| between fragment and nearby wire direction (~11 deg)
    wire_grow_radius: float = 0.3
    wire_grow_linearity: float = 0.5
    pylon_min_height: float = 10.0
    # Lattice towers scatter less than tree crowns (~0.11-0.12 vs 0.17-0.3 on GridNet-HD).
    pylon_max_scattering: float = 0.14
    pylon_wire_distance: float = 3.0
    pylon_radius: float = 15.0  # cross-arm half-width; farther points are not the tower
    building_planarity: float = 0.4
    building_max_scattering: float = 0.05
    building_min_points: int = 100
    component_voxel: float = 1.0


def components(points: np.ndarray, mask: np.ndarray, size: float):
    """Connected components of masked points on a voxel grid (26-connectivity).

    Returns (indices of masked points, component id per masked point starting at 1).
    """
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return idx, np.zeros(0, dtype=np.int64)
    k = np.floor(points[idx] / size).astype(np.int64)
    k -= k.min(axis=0)
    grid = np.zeros(k.max(axis=0) + 1, dtype=bool)
    grid[k[:, 0], k[:, 1], k[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    return idx, lab[k[:, 0], k[:, 1], k[:, 2]].astype(np.int64)


def _groups(comp: np.ndarray):
    """Yield (component id, positions into comp) for each component."""
    order = np.argsort(comp, kind="stable")
    ids, starts = np.unique(comp[order], return_index=True)
    for cid, chunk in zip(ids, np.split(order, starts[1:])):
        yield cid, chunk


def _xy_axis(xy: np.ndarray) -> np.ndarray:
    c = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 5000)], full_matrices=False)
    return vt


def _xy_shape(xy: np.ndarray) -> tuple[float, float]:
    """Length along the main xy axis and width (4 std) across it."""
    vt = _xy_axis(xy)
    c = xy - xy.mean(axis=0)
    return float(np.ptp(c @ vt[0])), float(4 * (c @ vt[1]).std())


def _add_fragments(points, comp_idx, comp, wire, p: Params):
    """Short candidate pieces (sparse spans) that line up with nearby accepted wires."""
    if not wire.any():
        return
    wire_idx = np.nonzero(wire)[0]
    tree = cKDTree(points[wire_idx, :2])
    for _, pos in _groups(comp):
        sel = comp_idx[pos]
        if wire[sel[0]] or len(sel) < 5:
            continue
        length, width = _xy_shape(points[sel, :2])
        if length < p.wire_fragment_length or length < p.wire_min_aspect * max(width, 0.2):
            continue
        near = tree.query_ball_point(points[sel, :2].mean(axis=0), r=40.0)
        if len(near) < 20:
            continue
        a = _xy_axis(points[sel, :2])[0]
        b = _xy_axis(points[wire_idx[near], :2])[0]
        if abs(a @ b) >= p.wire_fragment_cos:
            wire[sel] = True


def find_wires(points, hag, f: Features, ground, p: Params) -> np.ndarray:
    cand = (
        ~ground
        & (hag > p.wire_min_hag)
        & (f.linearity > p.wire_linearity)
        & (f.dir_z < p.wire_max_dir_z)
        & (f.gap_below > p.wire_min_gap)
    )
    wire = np.zeros(len(points), dtype=bool)
    idx, comp = components(points, cand, p.component_voxel)
    for _, pos in _groups(comp):
        if len(pos) < 20:
            continue
        length, width = _xy_shape(points[idx[pos], :2])
        if length >= p.wire_min_length and length >= p.wire_min_aspect * max(width, 0.5):
            wire[idx[pos]] = True
    strict = wire.copy()
    _add_fragments(points, idx, comp, wire, p)

    # Grow into nearby linear points the strict rules missed (near clamps, low density).
    for _ in range(3):
        if not wire.any():
            break
        tree = cKDTree(points[wire])
        cand = ~wire & ~ground & (hag > p.wire_min_hag) & (f.linearity > p.wire_grow_linearity)
        ci = np.nonzero(cand)[0]
        d, _ = tree.query(points[ci], k=1, distance_upper_bound=p.wire_grow_radius, workers=-1)
        added = ci[np.isfinite(d)]
        if len(added) == 0:
            break
        wire[added] = True
    return wire, strict


def find_pylons(points, hag, f: Features, ground, wire, p: Params):
    """Tall, regular structures whose top reaches the wires.

    `wire` should be the strict long-span wires only: points added by growing are allowed
    into towers, since lattice members are linear too. Returns (pylon mask, extra wire
    mask): sparse wire points that the voxel grid joined to a tower are handed back.
    """
    pylon = np.zeros(len(points), dtype=bool)
    extra_wire = np.zeros(len(points), dtype=bool)
    if not wire.any():
        return pylon, extra_wire
    wire_idx = np.nonzero(wire)[0]
    wire_xy = cKDTree(points[wire_idx, :2])
    cand = ~ground & ~wire & (hag > 1.0)
    idx, comp = components(points, cand, p.component_voxel)
    for _, pos in _groups(comp):
        if len(pos) < 50:
            continue
        sel = idx[pos]
        # The tower is the dense core; sparse wires attached to it stretch far away.
        centre = np.median(points[sel, :2], axis=0)
        far = np.linalg.norm(points[sel, :2] - centre, axis=1) > p.pylon_radius
        core = sel[~far]
        if len(core) < 50 or hag[core].max() < p.pylon_min_height:
            continue
        if np.median(f.scattering[core]) > p.pylon_max_scattering:
            continue
        top = core[hag[core] > hag[core].max() - 3.0]
        d, j = wire_xy.query(points[top, :2], k=1, workers=-1)
        if d.min() > p.pylon_wire_distance:
            continue
        # Wires hang from the tower, so its top is at or above the nearby wires.
        near = wire_xy.query_ball_point(centre, r=p.pylon_radius)
        if near and points[top, 2].max() < points[wire_idx[near], 2].min() - 1.0:
            continue
        pylon[core] = True
        rest = sel[far]
        extra_wire[rest[(f.linearity[rest] > p.wire_grow_linearity) & (f.dir_z[rest] < p.wire_max_dir_z)]] = True
    return pylon, extra_wire


def find_buildings(points, hag, f: Features, taken, p: Params) -> np.ndarray:
    building = np.zeros(len(points), dtype=bool)
    # Seed on smooth surfaces (roofs and walls scatter ~10x less than tree crowns), then
    # require each cluster to be mostly planar.
    cand = ~taken & (hag > 1.0) & (f.scattering < p.building_max_scattering)
    idx, comp = components(points, cand, p.component_voxel)
    for _, pos in _groups(comp):
        sel = idx[pos]
        if len(sel) < p.building_min_points:
            continue
        if np.median(f.planarity[sel]) < p.building_planarity:
            continue
        length, width = _xy_shape(points[sel, :2])
        if width < 2.0:  # walls of single thin things (fences) are not buildings
            continue
        building[sel] = True
    return building


def classify(points: np.ndarray, hag: np.ndarray, f: Features, p: Params | None = None):
    p = p or Params()
    labels = np.full(len(points), VEGETATION, dtype=np.uint8)
    ground = hag < p.ground_hag
    labels[ground] = GROUND
    labels[hag < -1.0] = UNCLASSIFIED  # low noise

    wire, strict = find_wires(points, hag, f, ground, p)
    pylon, extra_wire = find_pylons(points, hag, f, ground, strict, p)
    wire = (wire & ~pylon) | extra_wire
    building = find_buildings(points, hag, f, ground | wire | pylon, p)

    labels[building] = BUILDING
    labels[pylon] = PYLON
    labels[wire] = WIRE
    return labels
