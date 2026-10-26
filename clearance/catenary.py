"""Split a span's wire points into conductors and fit a catenary to each.

In the vertical plane of a conductor, with u the horizontal distance along it:

    z(u) = z0 + c * (cosh((u - u0) / c) - 1)

u0 / z0 is the lowest point (it may lie outside the span on a steep slope) and c = H / w,
the horizontal tension over the weight per metre: a bigger c is a tighter wire.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .classify import _xy_axis
from .spans import Span


@dataclass
class Conductor:
    span_id: int
    id: int  # within the span, ordered by lateral offset then height
    idx: np.ndarray  # wire point indices
    origin: np.ndarray  # (2,) xy where u = 0 (the span start, projected onto this wire)
    axis: np.ndarray  # (2,) unit direction of the wire in plan
    z0: float
    u0: float
    c: float
    u_min: float  # extent the model is valid over
    u_max: float
    rmse_z: float
    rmse_xy: float  # lateral scatter about the straight plan line
    coverage: float  # share of 1 m bins along the extent that have points
    full_span: bool  # data reaches both towers: modelled tower to tower
    at_tower_a: bool = False  # data reaches the span's start tower (u = 0)
    at_tower_b: bool = False  # data reaches the span's end tower
    sag: float | None = None  # mid-span drop below the chord; full spans only
    min_ground_clearance: float | None = None
    outliers: int = 0  # points dropped by the residual trim
    rejected: str | None = None  # why this cluster is not a usable conductor

    def z(self, u):
        return self.z0 + self.c * (np.cosh((np.asarray(u) - self.u0) / self.c) - 1.0)

    def sample(self, step: float = 0.25) -> np.ndarray:
        u = np.arange(self.u_min, self.u_max + step / 2, step)
        xy = self.origin + u[:, None] * self.axis
        return np.column_stack([xy, self.z(u)])


def split_conductors(points: np.ndarray, span: Span, eps: float = 0.5, min_samples: int = 10):
    """Cluster in the span cross-section: lateral offset and height above a common curve.

    Parallel wires keep a constant offset from each other along the span, so each wire
    collapses to a small blob in (offset, residual height) whatever gaps it has along u.
    Bundled sub-conductors (< eps apart) stay together as one conductor.
    """
    p = points[span.idx]
    d = (span.b - span.a) / max(span.length, 1e-9)
    n = np.array([-d[1], d[0]])
    u = (p[:, :2] - span.a) @ d
    v = (p[:, :2] - span.a) @ n
    ref = np.polyval(np.polyfit(u, p[:, 2], 2), u)
    feats = np.column_stack([v, p[:, 2] - ref])
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
    groups = [span.idx[lab == k] for k in np.unique(lab) if k >= 0]
    keys = [(round(float(np.median(feats[lab == k, 0])), 1), float(np.median(feats[lab == k, 1]))) for k in np.unique(lab) if k >= 0]
    return [g for _, g in sorted(zip(keys, groups), key=lambda kg: kg[0])]


def catenary_z(u, z0, u0, c):
    # Junk clusters can push the optimiser to tiny c; inf residuals just make it back off.
    with np.errstate(over="ignore", invalid="ignore"):
        return z0 + c * (np.cosh((u - u0) / c) - 1.0)


def fit(u: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Robust least-squares catenary fit; returns (z0, u0, c)."""
    with warnings.catch_warnings():
        # Short clusters (later rejected) make the starting parabola ill-conditioned.
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        a, b, d = np.polyfit(u, z, 2)
    a = max(a, 1e-5)  # straight or slightly convex data: start from a tight wire
    x0 = [d - b * b / (4 * a), -b / (2 * a), 1 / (2 * a)]
    x0[2] = float(np.clip(x0[2], 50.0, 5e4))

    def resid(x):
        return catenary_z(u, *x) - z

    # Divergence on junk clusters shows up as non-finite parameters, caught by validate().
    with np.errstate(all="ignore"):
        r = least_squares(
            resid,
            x0,
            loss="soft_l1",
            f_scale=0.1,  # residuals beyond ~10 cm count linearly: stray points don't drag it
            bounds=([-np.inf, -np.inf, 20.0], [np.inf, np.inf, 1e5]),
            x_scale=[10.0, 50.0, 500.0],
        )
    return tuple(float(v) for v in r.x)


def _rms(res: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):  # diverged fits; validate() rejects
        return float(np.sqrt(np.mean(res**2)))


def fit_conductor(
    points: np.ndarray, idx: np.ndarray, span: Span, cid: int, ground_tree=None, ground_z=None,
    tower_reach: float = 8.0,
) -> Conductor:
    p = points[idx]
    axis = _xy_axis(p[:, :2])[0]
    d = (span.b - span.a) / max(span.length, 1e-9)
    if axis @ d < 0:
        axis = -axis
    # u = 0 at the span start projected onto this wire's own plan line.
    c_xy = p[:, :2].mean(axis=0)
    origin = c_xy + ((span.a - c_xy) @ axis) * axis
    u = (p[:, :2] - origin) @ axis
    lateral = (p[:, :2] - origin) @ np.array([-axis[1], axis[0]])
    z = p[:, 2].astype(np.float64)
    keep = np.ones(len(u), dtype=bool)
    z0, u0, c = fit(u, z)
    # Trim what the robust loss only damps (points of other objects in the cluster), refit.
    for _ in range(2):
        res = catenary_z(u, z0, u0, c) - z
        if not np.all(np.isfinite(res)):
            break  # diverged; validate() rejects it
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) < max(0.5, 3 * mad)
        if new.sum() < 0.5 * len(u) or np.array_equal(new, keep):
            break
        keep = new
        z0, u0, c = fit(u[keep], z[keep])
    outliers = int((~keep).sum())
    idx, u, lateral, p = idx[keep], u[keep], lateral[keep], p[keep]
    res = catenary_z(u, z0, u0, c) - p[:, 2]

    end = float((span.b - origin) @ axis)
    # Model a wire only where its data supports it: it reaches a tower only if its points
    # come within `tower_reach` of it (past the tower margin left out of the fit). A wire
    # whose data stops short belongs elsewhere (e.g. a parallel line turning away).
    at_a = span.tower_a is not None and u.min() <= tower_reach
    at_b = span.tower_b is not None and u.max() >= end - tower_reach
    u_min = 0.0 if at_a else float(u.min())
    u_max = end if at_b else float(u.max())
    full = bool(at_a and at_b)
    bins = np.arange(np.floor(u_min), np.ceil(u_max) + 1)
    coverage = float(np.mean(np.histogram(u, bins=bins)[0] > 0)) if len(bins) > 1 else 0.0

    cond = Conductor(
        span.id, cid, idx, origin, axis, z0, u0, c, u_min, u_max,
        rmse_z=_rms(res),
        rmse_xy=float(np.std(lateral)),
        coverage=coverage,
        full_span=full,
        at_tower_a=bool(at_a),
        at_tower_b=bool(at_b),
        outliers=outliers,
    )
    if full:
        uu = np.linspace(u_min, u_max, 400)
        zz = cond.z(uu)
        chord = zz[0] + (zz[-1] - zz[0]) * (uu - uu[0]) / (uu[-1] - uu[0])
        cond.sag = float(np.max(chord - zz))
    if ground_tree is not None:
        s = cond.sample(1.0)
        near = ground_tree.query_ball_point(s[:, :2], r=1.0)
        clear = [z - ground_z[n].max() for (_, _, z), n in zip(s, near) if n]
        if clear:
            cond.min_ground_clearance = float(np.min(clear))
    return cond


def _near_towers(points, span: Span, margin: float) -> np.ndarray:
    """Points within `margin` of a tower end of the span, or past it.

    Clamps, insulator strings and jumpers hang there and bend away from the catenary.
    """
    d = (span.b - span.a) / max(span.length, 1e-9)
    u = (points[:, :2] - span.a) @ d
    out = np.zeros(len(points), dtype=bool)
    if span.tower_a is not None:
        out |= u < margin
    if span.tower_b is not None:
        out |= u > span.length - margin
    return out


def validate(
    c: Conductor, min_points: int, min_coverage: float, min_extent: float, max_rmse: float, span: Span | None = None,
) -> str | None:
    if not np.all(np.isfinite([c.z0, c.u0, c.c, c.rmse_z])):
        return "fit did not converge"
    # Short spans (15-30 m at junctions) carry fewer points: 3 per metre, capped.
    need = min(min_points, int(3 * (c.u_max - c.u_min)))
    if len(c.idx) < need:
        return f"only {len(c.idx)} points"
    if c.u_max - c.u_min < min_extent:
        return f"only {c.u_max - c.u_min:.0f} m long"
    if c.coverage < min_coverage:
        return f"covers {100 * c.coverage:.0f}% of its extent"
    if c.c <= 21.0 or c.c >= 0.99e5:
        return f"c = {c.c:.0f} m at the fit bound"
    if c.rmse_z > max_rmse:
        # Real conductors fit to a few cm; this is not a hanging wire.
        return f"rmse {100 * c.rmse_z:.0f} cm"
    if span is not None and span.tower_a is not None and not (c.at_tower_a or c.at_tower_b) and c.u_max - c.u_min < 30.0:
        # Wire leaving a tower starts at it. A short piece in a tower's span that does not
        # reach the tower was a mix of jumper, vegetation and unlabelled points (t1z6a).
        return "short piece detached from its tower"
    return None


def fit_spans(
    points: np.ndarray,
    wire_idx: np.ndarray,
    spans: list[Span],
    ground_points=None,
    tower_margin: float = 3.0,
    min_points: int = 100,
    min_coverage: float = 0.5,
    min_extent: float = 10.0,
    max_rmse: float = 0.25,
):
    """Fit every conductor of every span. `wire_idx` maps span.idx into `points`.

    Returns (conductors, rejected clusters); rejected ones carry the reason in `.rejected`.
    """
    ground_tree = ground_z = None
    if ground_points is not None and len(ground_points):
        ground_tree = cKDTree(ground_points[:, :2])
        ground_z = ground_points[:, 2]
    good, bad = [], []
    for span in spans:
        idx = wire_idx[span.idx]
        idx = idx[~_near_towers(points[idx], span, tower_margin)]
        local = Span(span.id, span.a, span.b, span.tower_a, span.tower_b, idx)
        def fit_group(g, eps):
            c = fit_conductor(points, g, local, 0, ground_tree, ground_z, tower_reach=tower_margin + 5.0)
            c.rejected = validate(c, min_points, min_coverage, min_extent, max_rmse, local)
            # Two conductors bridged by stray points fail on RMSE: split tighter and retry.
            if c.rejected and c.rejected.startswith("rmse") and len(g) >= 2 * min_points and eps > 0.15:
                sub = Span(local.id, local.a, local.b, local.tower_a, local.tower_b, g)
                parts = [p for p in split_conductors(points, sub, eps=eps / 2) if len(p) >= 10]
                if len(parts) > 1:
                    return [r for p in parts for r in fit_group(p, eps / 2)]
            return [c]

        groups = [g for g in split_conductors(points, local) if len(g) >= 10]
        fitted = [c for g in groups for c in fit_group(g, 0.5)]
        kept = [c for c in fitted if c.rejected is None]
        for cid, c in enumerate(kept):
            c.id = cid
        good += kept
        bad += [c for c in fitted if c.rejected is not None]
    return good, bad
