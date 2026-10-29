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
    """Clearance thresholds, by default the Swiss LeV (SR 734.31) for a high-voltage line of
    `kv` nominal voltage:
    - sag and ice cases: Art. 35(4)b, vertical distance between trees and bare conductors at
      greatest sag, 1.5 m + 0.01 m/kV (fruit trees: 2.5 m + 0.01 m/kV). Applied here as a
      3D distance, which is stricter beside the line than a vertical-only reading.
    - wind cases: the LeV sets direct distances to trees case by case (Art. 35(1)); by
      analogy with Art. 34(2) (terrain beside the line in wind) 0.01 m/kV, at least 1.5 m.
    - as surveyed: the sag threshold (the survey is not a design state; shown for context).
    The growth rate and trimming cycle are assumptions; set them from the operator's plan.
    """

    kv: float = 380.0
    mvcd_m: float | None = None  # sag threshold; default from kv
    mvcd_wind_m: float | None = None  # wind threshold; default from kv
    urgent_buffer_m: float = 1.5  # within threshold + this: trim now
    growth_m_per_year: float = 1.0  # vegetation growth towards the line
    cycle_years: float = 3.0  # inspection / trimming cycle
    conductor_radius_m: float = 0.016  # distances are to the conductor axis; AAAC 600 mm2
    min_zone_points: int = 5
    # Vegetation-labelled points this close (in plan) to a tower centre are left out: there
    # they are almost always tower steel that Phase 1 called vegetation (forest towers), and
    # the attachment points are not where a vegetation clearance is measured.
    tower_exclusion_m: float = 8.0

    def __post_init__(self):
        if self.mvcd_m is None:
            self.mvcd_m = round(1.5 + 0.01 * self.kv, 2)
        if self.mvcd_wind_m is None:
            self.mvcd_wind_m = round(max(1.5, 0.01 * self.kv), 2)

    def threshold(self, scenario: str) -> float:
        from .scenarios import WIND_SCENARIOS

        return self.mvcd_wind_m if scenario in WIND_SCENARIOS else self.mvcd_m

    @property
    def monitor_m(self) -> float:
        return self.mvcd_m + self.growth_m_per_year * self.cycle_years

    def tier(self, d: np.ndarray) -> np.ndarray:
        """Tiers of clearances on the sag-threshold scale (Clearance.worst)."""
        out = np.full(len(d), CLEAR, dtype=object)
        out[d < self.monitor_m] = MONITOR
        out[d < self.mvcd_m + self.urgent_buffer_m] = URGENT
        out[d < self.mvcd_m] = VIOLATION
        return out


@dataclass
class Clearance:
    """Per vegetation point: distance to the nearest conductor in every scenario."""

    dist: dict  # scenario -> (n,) float32, real distance
    nearest: dict  # scenario -> (n,) index into that scenario's sample array
    samples: dict  # scenario -> (m, 3) conductor samples
    owner: dict  # scenario -> (m,) index into `curves` for each sample
    curves: list
    shift: dict  # scenario -> sag threshold - this scenario's threshold

    def worst(self):
        """(clearance, scenario) of the scenario with the smallest margin to its own
        threshold. The clearance is put on the sag-threshold scale (distance + shift), so one
        set of tiers applies: 0.4 m below the wind threshold ranks like 0.4 m below the sag
        threshold. `real_distance` gives the measured distance."""
        names = list(self.dist)
        stack = np.vstack([self.dist[s] + self.shift[s] for s in names])
        k = np.argmin(stack, axis=0)
        return stack[k, np.arange(stack.shape[1])], np.array(names, dtype=object)[k]

    def real_distance(self, scenario: str, i):
        return self.dist[scenario][i]


def compute(veg: np.ndarray, curves: list[Curve], rules: Rules) -> Clearance:
    dist, nearest, samples, owner, shift = {}, {}, {}, {}, {}
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
        shift[s] = float(rules.mvcd_m - rules.threshold(s))
    return Clearance(dist, nearest, samples, owner, curves, shift)


def floating(
    veg: np.ndarray,
    hag: np.ndarray,
    max_points: int = 200,
    min_hag: float = 3.0,
    voxel: float = 1.0,
):
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


def wire_fragments(
    veg: np.ndarray, clr: Clearance, radius: float = 0.5, share: float = 0.8, reach: float = 1.0
):
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


def unmodelled_wire(
    wire: np.ndarray,
    curves: list,
    veg: np.ndarray,
    rules: Rules,
    towers,
    origin,
    gap: float = 1.0,
    min_length: float = 10.0,
    min_aspect: float = 4.0,
) -> list[dict]:
    """Stretches of wire with no conductor model, and the vegetation next to them.

    Clearance is only measured to modelled conductors, so vegetation beside wire that Phase 2
    could not model (a sparse far stretch, a rejected cluster) is not assessed. This says
    where. Wire points more than `gap` from every as-surveyed conductor, grouped into
    line-shaped clusters >= `min_length` (not tower steel: towers' footprints are skipped).
    """
    from .classify import _xy_shape
    from .scenarios import AS_SURVEYED

    model = [c.xyz for c in curves if c.scenario == AS_SURVEYED]
    if not len(wire):
        return []
    far = np.ones(len(wire), dtype=bool)
    if model:
        far = cKDTree(np.vstack(model)).query(wire, k=1, workers=-1)[0] > gap
    for t in towers:
        far &= np.linalg.norm(wire[:, :2] - t.xy, axis=1) >= rules.tower_exclusion_m
    if not far.any():
        return []
    idx, comp = components(wire, far, 1.0)
    vtree = cKDTree(veg) if len(veg) else None
    rows = []
    for _, pos in _groups(comp):
        sel = idx[pos]
        if len(sel) < 20:
            continue
        length, width = _xy_shape(wire[sel, :2])
        if length < min_length or length < min_aspect * max(width, 0.5):
            continue
        near = 0
        if vtree is not None:
            hits = vtree.query_ball_point(wire[sel], r=rules.monitor_m)
            near = len(set(i for h in hits for i in h))
        c = wire[sel].mean(axis=0)
        rows.append(
            {
                "x": round(float(c[0] + origin[0]), 1),
                "y": round(float(c[1] + origin[1]), 1),
                "z": round(float(c[2] + origin[2]), 1),
                "length_m": round(length, 1),
                "wire_points": int(len(sel)),
                "vegetation_within_monitor_m": int(near),
            }
        )
    rows.sort(key=lambda r: -r["vegetation_within_monitor_m"])
    return rows


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
        tier = rules.tier(np.array([worst_d[k]]))[0]
        d = float(clr.dist[scen][k])  # real distance in the governing scenario
        required = rules.threshold(scen)
        try:
            area = float(ConvexHull(veg[sel, :2]).volume) if len(sel) >= 3 else 0.0
        except Exception:  # collinear points
            area = 0.0
        rows.append(
            {
                "tier": tier,
                "min_clearance_m": round(d, 2),
                "required_m": round(required, 2),
                "margin_m": round(d - required, 2),
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
                "years_to_violation": round(max(0.0, (d - required) / rules.growth_m_per_year), 1),
                "_points": sel,
            }
        )
    rows.sort(key=lambda r: (order[r["tier"]], r["margin_m"]))
    for i, r in enumerate(rows):
        r["zone"] = i + 1
    return rows, flagged
