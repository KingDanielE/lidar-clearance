"""Tower and line graph: towers from several kinds of evidence, spans as edges between them.

The first version assumed one line per tile: towers ordered along one axis, spans as the
segments of one route. A junction (t2z5: four towers within 60 m, a second line leaving)
breaks that. Here:

Towers (nodes), merged within `merge_m`, earlier sources winning:
1. pylon clusters from Phase 1 (`spans.find_towers`);
2. per direction group, the kink + centre-column test (`spans.towers_from_wires`): wires
   are grouped by their direction in plan, so lines running different ways each get their
   own axis;
3. direction changes: wires do not bend in mid-air, so where nearby wire points run in
   clearly different plan directions there is an angle or junction tower.

Spans (edges):
1. two towers are joined if wire points running along the line between them cover most of
   it; an edge passing close to a third tower is dropped (it is two spans);
2. wires leaving a tower that no edge explains become partial spans (rays) to where they
   end, usually the tile edge;
3. wires near no tower become tower-less spans.
"""

import math

import numpy as np
from scipy.spatial import cKDTree

from .classes import GROUND, WIRE
from .classify import _groups, _xy_axis, components
from .spans import Span, Tower, find_towers, towers_from_wires


def wire_directions(w: np.ndarray, k: int = 12, chunk: int = 200_000):
    """Unit plan direction of each wire point's local line (sign-free) and a validity mask
    (linear, and not steeper than ~60 degrees: insulator strings and steel are excluded)."""
    tree = cKDTree(w)
    k = min(k, len(w))
    dirs = np.zeros((len(w), 2))
    ok = np.zeros(len(w), dtype=bool)
    for s in range(0, len(w), chunk):
        e = min(s + chunk, len(w))
        _, idx = tree.query(w[s:e], k=k, workers=-1)
        nb = w[idx].astype(np.float64)
        nb -= nb.mean(axis=1, keepdims=True)
        vals, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
        main = vecs[:, :, 2]
        lin = (vals[:, 2] - vals[:, 1]) / np.maximum(vals[:, 2], 1e-12)
        xy = main[:, :2]
        n = np.linalg.norm(xy, axis=1)
        dirs[s:e] = xy / np.maximum(n, 1e-9)[:, None]
        ok[s:e] = (lin > 0.8) & (n > 0.5)
    return dirs, ok


def direction_groups(dirs: np.ndarray, ok: np.ndarray, tol_deg: float = 7.5, min_points: int = 300):
    """Masks of wire points sharing a plan direction (peaks of a circular histogram)."""
    ang = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
    hist, _ = np.histogram(ang[ok], bins=180, range=(0, 180))
    smooth = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(5) / 5, mode="same")[3:-3]
    peaks = []
    for i in np.argsort(-smooth):
        if smooth[i] * 5 < min_points:
            break
        if all(min(abs(i - p), 180 - abs(i - p)) > 2 * tol_deg for p in peaks):
            peaks.append(i)
    out = []
    for p in peaks:
        diff = np.abs(ang - (p + 0.5))
        diff = np.minimum(diff, 180 - diff)
        m = ok & (diff <= tol_deg)
        if m.sum() >= min_points:
            out.append(m)
    return out


def towers_from_angles(
    w: np.ndarray, dirs: np.ndarray, ok: np.ndarray, other: np.ndarray, radius: float = 6.0,
    max_r: float = 0.9, min_neighbours: int = 20, min_cluster: int = 15, step: int = 3,
) -> list[Tower]:
    """Towers where wires change direction. In doubled-angle form a single direction has
    mean resultant length ~1; two directions `d` apart give ~|cos d|. The 3D radius keeps
    lines crossing at different heights apart."""
    if len(w) < 100:
        return []
    d2 = np.column_stack([dirs[:, 0] ** 2 - dirs[:, 1] ** 2, 2 * dirs[:, 0] * dirs[:, 1]])
    tree = cKDTree(w)
    sel = np.nonzero(ok)[0][::step]
    cand = []
    for i, nb in zip(sel, tree.query_ball_point(w[sel], r=radius, workers=-1)):
        nb = np.asarray(nb, dtype=int)
        nb = nb[ok[nb]]
        if len(nb) >= min_neighbours and np.linalg.norm(d2[nb].mean(axis=0)) < max_r:
            cand.append(i)
    if len(cand) < min_cluster:
        return []
    cand = np.asarray(cand)
    idx, comp = components(np.column_stack([w[cand, :2], np.zeros(len(cand))]), np.ones(len(cand), bool), 5.0)
    other_tree = cKDTree(other[:, :2]) if len(other) else None
    out = []
    for _, pos in _groups(comp):
        sel_c = cand[idx[pos]]
        if len(sel_c) < min_cluster:
            continue
        xy = np.median(w[sel_c, :2], axis=0)
        if _is_crossing(w, dirs, ok, xy):
            continue
        out.append(Tower(len(out), xy, float(w[sel_c, 2].max()), len(sel_c)))
    return out


def _is_crossing(w, dirs, ok, xy, reach: float = 30.0, gap: float = 6.0, min_side: int = 30) -> bool:
    """True if every wire direction near `xy` continues on both sides of it: two lines
    crossing (or one passing straight through), not an angle or junction tower, where at
    least one direction stops."""
    v = w[:, :2] - xy
    r = np.linalg.norm(v, axis=1)
    near = ok & (r < reach)
    if near.sum() < 2 * min_side:
        return False
    ang = np.degrees(np.arctan2(dirs[near, 1], dirs[near, 0])) % 180.0
    hist = np.histogram(ang, bins=36, range=(0, 180))[0]
    sides = []
    for b in np.argsort(-hist)[:4]:
        if hist[b] < min_side:
            break
        th = math.radians(b * 5 + 2.5)
        u = np.array([math.cos(th), math.sin(th)])
        diff = np.abs(ang - (b * 5 + 2.5))
        sel = np.minimum(diff, 180 - diff) < 7.5
        s = v[near][sel] @ u
        sides.append((np.sum(s < -gap) >= min_side) and (np.sum(s > gap) >= min_side))
    return len(sides) >= 2 and all(sides)


def _steel_centre(xy, wire_z, other, other_tree, search: float = 8.0, r: float = 2.5):
    """Move a tower position onto the densest column of non-ground, non-wire structure at
    wire height within `search` m (the tower body), if there is one."""
    if other_tree is None:
        return xy
    # Conductor height only: vegetation has to stay below the wires, a tower body does not.
    lo, hi = np.percentile(wire_z, [5, 95])
    best, best_n = xy, 0
    for dx in np.arange(-search, search + 1, 2.0):
        for dy in np.arange(-search, search + 1, 2.0):
            p = xy + np.array([dx, dy])
            nb = np.asarray(other_tree.query_ball_point(p, r), dtype=int)
            n = int(np.sum((other[nb, 2] > lo - 3) & (other[nb, 2] < hi + 2))) if len(nb) else 0
            if n > best_n:
                best, best_n = p, n
    return best if best_n >= 50 else xy


def merge(sources: list[list[Tower]], distance: float = 10.0) -> list[Tower]:
    """Union of tower lists; a tower within `distance` of an earlier one is the same tower."""
    keep: list[Tower] = []
    for towers in sources:
        for t in towers:
            if all(np.linalg.norm(t.xy - k.xy) > distance for k in keep):
                keep.append(t)
    return [Tower(i, t.xy, t.top_z, t.n_points) for i, t in enumerate(keep)]


def find_all_towers(points, labels, dirs, ok, wire_idx, merge_m: float = 8.0) -> list[Tower]:
    w = points[wire_idx]
    other = points[(labels != WIRE) & (labels != GROUND)]
    kinks = []
    for g in direction_groups(dirs, ok):
        mask = np.zeros(len(points), dtype=bool)
        mask[wire_idx[g]] = True
        kinks += towers_from_wires(points, labels, wire_mask=mask)
    # Kink and angle towers land within ~1 m of the true towers on the real tiles, so a
    # plain 8 m merge keeps junction towers (14-18 m apart) separate. Moving them onto the
    # densest "steel" column first was tried and pulled forest towers 8-9 m off, onto canopy.
    return merge([find_towers(points, labels), kinks, towers_from_angles(w, dirs, ok, other)], merge_m)


def terminal_towers(
    points, labels, wire_idx, spans: list[Span], towers: list[Tower] | None = None,
    beyond: float = 15.0, radius: float = 10.0, min_ground: int = 20, min_steel: int = 30,
    min_ray_m: float = 30.0, min_separation: float = 25.0, max_scattering: float = 0.2,
) -> list[Tower]:
    """Towers where a line ends. A ray (partial span) that stops while the survey goes on
    (ground beyond its end) ends on a structure, if there is steel at wire height there.
    At a tile edge there is no ground beyond; at a sparse stretch of wire, no steel."""
    ground = points[labels == GROUND]
    other = points[(labels != WIRE) & (labels != GROUND)]
    if not len(ground) or not len(other):
        return []
    gtree = cKDTree(ground[:, :2])
    otree = cKDTree(other[:, :2])
    w = points[wire_idx]
    out = []
    known = np.array([t.xy for t in towers], dtype=np.float64).reshape(-1, 2) if towers else np.zeros((0, 2))
    for s in spans:
        # Short rays are hardware and jumper pieces just past a tower, not a line ending.
        if s.tower_a is None or s.tower_b is not None or s.length < min_ray_m:
            continue
        if len(known) and np.min(np.linalg.norm(known - s.b, axis=1)) < min_separation:
            continue
        u = (s.b - s.a) / s.length
        if len(gtree.query_ball_point(s.b + u * beyond, radius)) < min_ground:
            continue  # the tile ends there
        end = w[s.idx][np.linalg.norm(w[s.idx, :2] - s.b, axis=1) < 6.0]
        if len(end) < 5:
            continue
        lo, hi = end[:, 2].min(), end[:, 2].max()
        nb = np.asarray(otree.query_ball_point(s.b, 4.0), dtype=int)
        nb = nb[(other[nb, 2] > lo - 3.0) & (other[nb, 2] < hi + 3.0)]
        if len(nb) < min_steel:
            continue
        # Steel, not canopy: in forest, trees reach wire height where a line leaves the
        # survey (t2z5), and Phase 1 labels tower steel vegetation anyway, so labels cannot
        # tell them apart. Shape can: lattice scatters ~0.11, crowns ~0.28 (t1z6a).
        if _median_scattering(other, nb) > max_scattering:
            continue
        xy = _steel_centre(s.b.copy(), end[:, 2], other, otree)
        out.append(Tower(len(out), xy, float(hi), len(end)))
    return out


def _median_scattering(pts: np.ndarray, idx: np.ndarray, k: int = 16) -> float:
    """Median l3/l1 of the neighbourhood covariance over pts[idx] (neighbours from pts)."""
    tree = cKDTree(pts)
    _, nn = tree.query(pts[idx], k=min(k, len(pts)), workers=-1)
    nb = pts[nn].astype(np.float64)
    nb -= nb.mean(axis=1, keepdims=True)
    vals = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", nb, nb))
    return float(np.median(vals[:, 0] / np.maximum(vals[:, 2], 1e-12)))


def build_line_graph(points, labels, wire_idx):
    """Towers and spans for the wire points `wire_idx`: two passes, the second with the
    terminal towers found at the ends of the first pass's rays."""
    dirs, ok = wire_directions(points[wire_idx])
    towers = find_all_towers(points, labels, dirs, ok, wire_idx)
    spans = build_graph(points, wire_idx, towers, dirs, ok)
    extra = terminal_towers(points, labels, wire_idx, spans, towers)
    if extra:
        towers = merge([towers, extra])
        spans = build_graph(points, wire_idx, towers, dirs, ok)
    return towers, spans


def _span_cos(length: float, base_deg: float, arm: float = 20.0) -> float:
    """Cosine of the largest angle a span's wires may make with the line between its tower
    centres. Conductors hang from cross-arm ends up to ~10 m either side of a centre, so on
    a 15 m span they run at ~50 degrees to it (t2z5 junction); on long spans `base_deg`."""
    return math.cos(math.radians(max(base_deg, math.degrees(math.atan(arm / max(length, 1e-9))))))


def _seg(xy, a, b, clamp: bool = True):
    """Distance to segment a-b and position along it (m), per point. With clamp=False the
    position is the unclamped projection (negative before a, > length past b)."""
    d = b - a
    L = float(np.hypot(*d))
    raw = ((xy - a) @ d) / max(L * L, 1e-9)
    t = np.clip(raw, 0.0, 1.0)
    dist = np.linalg.norm(xy - (a + t[:, None] * d), axis=1)
    return dist, (t if clamp else raw) * L


def build_graph(
    points, wire_idx, towers: list[Tower], dirs, ok, lateral: float = 12.0, min_cover: float = 0.6,
    through: float | None = None, max_span: float = 800.0, bin_m: float = 2.0, min_ray_points: int = 50,
) -> list[Span]:
    """Spans as graph edges. Span.idx indexes the wire point array (`wire_idx` order)."""
    w = points[wire_idx]
    wxy = w[:, :2].astype(np.float64)
    T = np.array([t.xy for t in towers], dtype=np.float64).reshape(-1, 2)
    # An edge passing a third tower within the corridor it gathers wires from is a bypass
    # (A-C past B, covered by the wires of A-B and B-C), not a span.
    through = lateral if through is None else through
    cos20, cos30 = math.cos(math.radians(20)), math.cos(math.radians(30))

    # 1. Tower-to-tower edges covered by wire running along them.
    edges = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T[i], T[j]
            L = float(np.hypot(*(b - a)))
            if L < 8.0 or L > max_span:
                continue
            u = (b - a) / L
            dist, s = _seg(wxy, a, b)
            m = (dist <= lateral) & ok & (np.abs(dirs @ u) >= _span_cos(L, 20.0)) & (s > 4.0) & (s < L - 4.0)
            nbins = max(1, int((L - 8.0) // bin_m))
            counts = np.histogram(s[m], bins=nbins, range=(4.0, L - 4.0))[0]
            if np.mean(counts >= 3) >= min_cover:
                edges.append((i, j))
    # Bypasses: A-C is covered by the wires of A-B and B-C when B lies near the line A-C.
    # (1) B within the corridor of A-C; (2) a triangle A-B-C of edges with B close to A-C
    # relative to its length (a corridor distance alone is fragile: 12.07 m vs 12 m).
    edge_set = set(edges)
    kept = []
    for i, j in edges:
        a, b = T[i], T[j]
        L = float(np.hypot(*(b - a)))
        others = np.array([k for k in range(len(T)) if k not in (i, j)], dtype=int)
        if len(others):
            d, s = _seg(T[others], a, b)
            interior = (s > 0.05 * L) & (s < 0.95 * L)
            if np.any(interior & (d < through)):
                continue
            tri = np.array([((min(i, k), max(i, k)) in edge_set) and ((min(j, k), max(j, k)) in edge_set) for k in others])
            if np.any(tri & interior & (d < 0.15 * L)):
                continue
        kept.append((i, j))

    # Assign wire points to the nearest edge (aligned, or close to a tower: hardware).
    n = len(w)
    best = np.full(n, np.inf)
    owner = np.full(n, -1)
    near_tower = np.zeros(n, dtype=bool)
    if len(T):
        near_tower = cKDTree(T).query(wxy)[0] < 8.0
    for e, (i, j) in enumerate(kept):
        a, b = T[i], T[j]
        L = max(float(np.hypot(*(b - a))), 1e-9)
        u = (b - a) / L
        d, s = _seg(wxy, a, b, clamp=False)
        # Only points beside the edge, not past its towers: a span continuing almost
        # straight beyond a tower must not lose its first metres to this one.
        inside = (s >= -3.0) & (s <= L + 3.0)
        okk = inside & (d <= lateral) & ((ok & (np.abs(dirs @ u) >= _span_cos(L, 30.0))) | near_tower) & (d < best)
        best[okk] = d[okk]
        owner[okk] = e
    spans = [Span(e, T[i], T[j], towers[i].id, towers[j].id, np.nonzero(owner == e)[0]) for e, (i, j) in enumerate(kept)]

    # 2. Rays: wire leaving a tower that no edge explains (partial spans to the tile edge).
    left = owner < 0
    for k in range(len(T)):
        tried = np.zeros(n, dtype=bool)  # bearings from this tower with nothing coherent
        while True:
            cand = np.nonzero(left & ok & ~tried)[0]
            if len(cand) < min_ray_points:
                break
            v = wxy[cand] - T[k]
            r = np.linalg.norm(v, axis=1)
            near = (r > 3.0) & (r < 80.0)
            radial = np.abs(np.einsum("ij,ij->i", v / np.maximum(r, 1e-9)[:, None], dirs[cand])) >= cos20
            sel = cand[near & radial]
            if len(sel) < min_ray_points:
                break
            bearing = np.degrees(np.arctan2(*(wxy[sel] - T[k])[:, ::-1].T)) % 360.0
            hist = np.histogram(bearing, bins=72, range=(0, 360))[0]
            peak = int(np.argmax(hist))
            if hist[peak] < min_ray_points:
                break
            th = math.radians(peak * 5 + 2.5)
            u = np.array([math.cos(th), math.sin(th)])
            # Collect the whole ray: aligned leftover points ahead of the tower, contiguous.
            v = wxy - T[k]
            along = v @ u
            lat = np.abs(v @ np.array([-u[1], u[0]]))
            on = left & (along > 0) & (lat <= lateral) & ((ok & (np.abs(dirs @ u) >= cos30)) | near_tower)
            if on.sum() < min_ray_points:
                tried[sel] = True  # nothing coherent along this bearing; do not retry it
                continue
            sorted_s = np.sort(along[on])
            gaps = np.nonzero(np.diff(sorted_s) > 30.0)[0]
            far = sorted_s[gaps[0]] if len(gaps) else sorted_s[-1]
            on &= along <= far + 1.0
            # Refine the direction to the points themselves (the bearing bin is 5 degrees).
            u = _xy_axis(wxy[on])[0]
            if u @ (wxy[on].mean(axis=0) - T[k]) < 0:
                u = -u
            far = float(((wxy[on] - T[k]) @ u).max())
            spans.append(Span(len(spans), T[k].copy(), T[k] + u * far, towers[k].id, None, np.nonzero(on)[0]))
            left &= ~on
            tried |= on

    # 3. Wires near no tower: each connected piece long enough is a tower-less span.
    rest = np.nonzero(left)[0]
    if len(rest):
        idx, comp = components(w[rest], np.ones(len(rest), dtype=bool), 2.0)
        for _, pos in _groups(comp):
            sel = rest[idx[pos]]
            if len(sel) < 200:
                continue
            ax = _xy_axis(wxy[sel])[0]
            c = wxy[sel].mean(axis=0)
            s = (wxy[sel] - c) @ ax
            if np.ptp(s) < 20.0:
                continue
            spans.append(Span(len(spans), c + s.min() * ax, c + s.max() * ax, None, None, sel))
    return spans
