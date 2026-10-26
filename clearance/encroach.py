"""Vegetation-to-conductor clearance, tiers and encroachment zones."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .classify import _groups, components
from .scenarios import SCENARIOS, Curve

VIOLATION, URGENT, MONITOR, CLEAR = "violation", "urgent", "monitor", "clear"
TIERS = [VIOLATION, URGENT, MONITOR, CLEAR]


@dataclass
class Rules:
    """Clearance thresholds. Placeholders: take them from the utility's standard for the
    line's voltage (e.g. NERC FAC-003 MVCD, EN 50341 / national rules, utility trim specs)."""

    mvcd_m: float = 3.0  # minimum vegetation clearance distance: below this is a violation
    urgent_buffer_m: float = 1.5  # within mvcd + this: trim now
    growth_m_per_year: float = 1.0  # vegetation growth towards the line
    cycle_years: float = 3.0  # inspection / trimming cycle
    conductor_radius_m: float = 0.0  # bundle radius; distances are to the modelled centre
    min_zone_points: int = 5
    # Vegetation-labelled points this close (in plan) to a tower centre are left out: there
    # they are almost always tower steel that Phase 1 called vegetation (forest towers), and
    # the attachment points are not where a vegetation clearance is measured.
    tower_exclusion_m: float = 8.0

    @property
    def monitor_m(self) -> float:
        return self.mvcd_m + self.growth_m_per_year * self.cycle_years

    def tier(self, d: np.ndarray) -> np.ndarray:
        out = np.full(len(d), CLEAR, dtype=object)
        out[d < self.monitor_m] = MONITOR
        out[d < self.mvcd_m + self.urgent_buffer_m] = URGENT
        out[d < self.mvcd_m] = VIOLATION
        return out


@dataclass
class Clearance:
    """Per vegetation point: distance to the nearest conductor in every scenario."""

    dist: dict  # scenario -> (n,) float32
    nearest: dict  # scenario -> (n,) index into that scenario's sample array
    samples: dict  # scenario -> (m, 3) conductor samples
    owner: dict  # scenario -> (m,) index into `curves` for each sample
    curves: list

    def worst(self):
        names = list(self.dist)
        stack = np.vstack([self.dist[s] for s in names])
        k = np.argmin(stack, axis=0)
        return stack[k, np.arange(stack.shape[1])], np.array(names, dtype=object)[k]


def compute(veg: np.ndarray, curves: list[Curve], rules: Rules) -> Clearance:
    dist, nearest, samples, owner = {}, {}, {}, {}
    for s in SCENARIOS:
        idx = [i for i, c in enumerate(curves) if c.scenario == s]
        if not idx:
            continue
        xyz = np.vstack([curves[i].xyz for i in idx])
        own = np.concatenate([np.full(len(curves[i].xyz), i) for i in idx])
        d, j = cKDTree(xyz).query(veg, k=1, workers=-1)
        dist[s] = (d - rules.conductor_radius_m).astype(np.float32)
        nearest[s] = j
        samples[s] = xyz
        owner[s] = own
    return Clearance(dist, nearest, samples, owner, curves)


def floating(veg: np.ndarray, hag: np.ndarray, max_points: int = 200, min_hag: float = 3.0, voxel: float = 1.0):
    """Small vegetation-labelled clusters hanging in the air: not plants (Phase 1 errors,
    typically stray conductor or tower points). Vegetation grows from the ground: a crown,
    even one overhanging the line, is part of a large connected cluster, and a small cluster
    of real vegetation (a shrub) sits near the ground."""
    out = np.zeros(len(veg), dtype=bool)
    idx, comp = components(veg, np.ones(len(veg), dtype=bool), voxel)
    for _, pos in _groups(comp):
        sel = idx[pos]
        if len(sel) < max_points and hag[sel].min() > min_hag:
            out[sel] = True
    return out


def wire_fragments(veg: np.ndarray, clr: Clearance, radius: float = 0.5, share: float = 0.8, reach: float = 1.0):
    """Vegetation-labelled points that are really conductor (Phase 1 errors).

    A point within `radius` of the as-surveyed conductor model counts as wire if most of its
    vegetation neighbours within `reach` are on the wire too. A crown touching the wire keeps
    its contact points: the crown behind them (further from the wire) is in their neighbourhood.
    """
    from .scenarios import AS_SURVEYED

    d = clr.dist[AS_SURVEYED]
    out = np.zeros(len(veg), dtype=bool)
    cand = np.nonzero(d < radius)[0]
    if len(cand) == 0:
        return out
    tree = cKDTree(veg)
    for i, nb in zip(cand, tree.query_ball_point(veg[cand], r=reach, workers=-1)):
        if np.mean(d[nb] < radius) >= share:
            out[i] = True
    return out


def zones(veg: np.ndarray, clr: Clearance, rules: Rules, origin: np.ndarray, voxel: float = 1.0):
    """Cluster flagged vegetation into zones; one row per zone, worst first."""
    worst_d, worst_s = clr.worst()
    flagged = worst_d < rules.monitor_m
    if not flagged.any():
        return [], flagged
    idx, comp = components(veg, flagged, voxel)
    order = {t: i for i, t in enumerate(TIERS)}
    rows = []
    for _, pos in _groups(comp):
        sel = idx[pos]
        if len(sel) < rules.min_zone_points:
            continue
        k = sel[np.argmin(worst_d[sel])]
        scen = worst_s[k]
        curve = clr.curves[clr.owner[scen][clr.nearest[scen][k]]]
        wire_pt = clr.samples[scen][clr.nearest[scen][k]]
        d = float(worst_d[k])
        tier = rules.tier(np.array([d]))[0]
        try:
            area = float(ConvexHull(veg[sel, :2]).volume) if len(sel) >= 3 else 0.0
        except Exception:  # collinear points
            area = 0.0
        rows.append({
            "tier": tier,
            "min_clearance_m": round(d, 2),
            "worst_scenario": scen,
            "estimated": bool(curve.estimated),
            **{f"min_{s}_m": round(float(clr.dist[s][sel].min()), 2) for s in clr.dist},
            "span": curve.conductor.span_id,
            "conductor": curve.conductor.id,
            "horizontal_m": round(float(np.linalg.norm(veg[k, :2] - wire_pt[:2])), 2),
            "vertical_m": round(float(wire_pt[2] - veg[k, 2]), 2),  # > 0: vegetation below wire
            "x": round(float(veg[k, 0] + origin[0]), 2),
            "y": round(float(veg[k, 1] + origin[1]), 2),
            "z": round(float(veg[k, 2] + origin[2]), 2),
            "points": int(len(sel)),
            "plan_area_m2": round(area, 1),
            "years_to_violation": round(max(0.0, (d - rules.mvcd_m) / rules.growth_m_per_year), 1),
            "_points": sel,
        })
    rows.sort(key=lambda r: (order[r["tier"]], r["min_clearance_m"]))
    for i, r in enumerate(rows):
        r["zone"] = i + 1
    return rows, flagged
