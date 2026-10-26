"""Where each conductor can be: as surveyed, hot (max sag) and blown sideways (blowout).

Max sag. A hotter conductor is longer: S_hot = S (1 + alpha dT). The new catenary passes
through the same two attachment points with that arc length. This ignores the elastic
shortening that comes with the lower tension of a hotter wire, so it slightly overstates
the sag: the safe side for clearance. A sag-tension program (PLS-CADD, ruling span) is the
production replacement.

Blowout. Wind pushes the conductor sideways; it swings as a rigid curve about the chord
between its attachment points by theta = atan(wind load / weight per metre). Both sides
are checked.

Attachment points come from what each conductor's own data reaches (Conductor.at_tower_a/b):
- both towers: exact.
- one tower, and at least `min_estimate_m` of data: the far attachment is extrapolated
  along the fitted curve at an assumed span length; max sag and blowout are flagged
  `estimated`.
- otherwise (no tower, or a short piece): not assessed for sag and wind; only the
  as-surveyed position is checked. Extrapolating a 15 m piece over a 300 m span put the
  wind-swung wire metres off, into trees that were nowhere near the real wire (t2z5).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .catenary import Conductor
from .spans import Span

AS_SURVEYED = "as_surveyed"
MAX_SAG = "max_sag"
BLOWOUT_LEFT = "blowout_left"
BLOWOUT_RIGHT = "blowout_right"
SCENARIOS = [AS_SURVEYED, MAX_SAG, BLOWOUT_LEFT, BLOWOUT_RIGHT]


@dataclass
class LineParams:
    """Conductor and weather assumptions. Placeholders: set from the line's design data."""

    t_survey_c: float = 15.0  # conductor temperature during the LiDAR flight
    t_max_c: float = 80.0  # maximum operating temperature
    t_wind_c: float = 15.0  # conductor temperature in the blowout case
    alpha_per_c: float = 19.3e-6  # thermal expansion, ACSR
    diameter_m: float = 0.0286  # ACSR "Zebra"-like
    weight_n_per_m: float = 15.9
    wind_pressure_pa: float = 300.0
    assumed_span_m: float | None = None  # partial spans; default: median full span, else 300
    min_estimate_m: float = 30.0  # shortest one-tower piece whose sag and wind are estimated

    @property
    def swing_rad(self) -> float:
        return math.atan(self.wind_pressure_pa * self.diameter_m / self.weight_n_per_m)


@dataclass
class Curve:
    scenario: str
    conductor: Conductor
    xyz: np.ndarray  # (n, 3) samples
    estimated: bool  # partial span: one attachment point assumed


def arc_length(z0, u0, c, a, b) -> float:
    return c * (math.sinh((b - u0) / c) - math.sinh((a - u0) / c))


def through_points(a: float, za: float, b: float, zb: float, length: float):
    """Catenary (z0, u0, c) through (a, za) and (b, zb) with the given arc length."""
    L, h = b - a, zb - za
    k = math.sqrt(max(length**2 - h**2, 0.0))
    if k <= L * (1 + 1e-9):
        raise ValueError("arc length not longer than the chord")
    # 2 c sinh(L / 2c) falls from +inf to L as c grows: one root.
    c = brentq(lambda c: 2 * c * math.sinh(L / (2 * c)) - k, L / 1400.0, 1e8)
    u0 = a + L / 2 - c * math.asinh(h / (2 * c * math.sinh(L / (2 * c))))
    z0 = za - c * (math.cosh((a - u0) / c) - 1.0)
    return z0, u0, c


def attachments(cond: Conductor, span: Span, assumed_span: float, min_estimate: float = 30.0):
    """(u_a, u_b, estimated) attachment positions along the conductor, or None when they
    cannot be known (no tower reached, or too short a piece to extrapolate)."""
    if cond.at_tower_a and cond.at_tower_b:
        return cond.u_min, cond.u_max, False
    if not (cond.at_tower_a or cond.at_tower_b) or cond.u_max - cond.u_min < min_estimate:
        return None
    u_tower = 0.0 if cond.at_tower_a else float((span.b - cond.origin) @ cond.axis)
    if 0.5 * (cond.u_min + cond.u_max) > u_tower:
        far = u_tower + max(assumed_span, cond.u_max - u_tower + 1.0)
        return u_tower, far, True
    far = u_tower - max(assumed_span, u_tower - cond.u_min + 1.0)
    return far, u_tower, True


def _xyz(cond: Conductor, u: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.column_stack([cond.origin + u[:, None] * cond.axis, z])


def _cat(z0, u0, c, u):
    return z0 + c * (np.cosh((u - u0) / c) - 1.0)


def hot_params(cond: Conductor, ua: float, ub: float, dt: float, alpha: float):
    za, zb = float(cond.z(ua)), float(cond.z(ub))
    s = arc_length(cond.z0, cond.u0, cond.c, ua, ub) * (1 + alpha * dt)
    return through_points(ua, za, ub, zb, s)


def swing(cond: Conductor, params, ua: float, ub: float, u: np.ndarray, theta: float, side: int):
    """Rotate the curve about its chord by theta; side +1 = left of the wire direction."""
    z0, u0, c = params
    A = _xyz(cond, np.array([ua]), np.array([_cat(z0, u0, c, ua)]))[0]
    B = _xyz(cond, np.array([ub]), np.array([_cat(z0, u0, c, ub)]))[0]
    t = (B - A) / np.linalg.norm(B - A)
    P = _xyz(cond, u, _cat(z0, u0, c, u))
    C = A + ((u - ua) / (ub - ua))[:, None] * (B - A)  # chord point over the same xy
    v = P - C
    v_par = (v @ t)[:, None] * t
    v_perp = v - v_par
    # Rodrigues about t: the perpendicular part turns towards t x v_perp (sideways).
    turned = v_par + math.cos(theta) * v_perp + side * math.sin(theta) * np.cross(t, v_perp)
    return C + turned


def curves(conductors: list[Conductor], spans: list[Span], p: LineParams, step: float = 0.1):
    """All scenario curves for every conductor, sampled over each conductor's modelled extent."""
    full = [s.length for s in spans if s.full]
    assumed = p.assumed_span_m or (float(np.median(full)) if full else 300.0)
    by_id = {s.id: s for s in spans}
    out: list[Curve] = []
    theta = p.swing_rad
    for cond in conductors:
        span = by_id[cond.span_id]
        u = np.arange(cond.u_min, cond.u_max + step / 2, step)
        out.append(Curve(AS_SURVEYED, cond, _xyz(cond, u, cond.z(u)), False))
        att = attachments(cond, span, assumed, p.min_estimate_m)
        if att is None:
            continue  # not assessed for sag and wind
        ua, ub, est = att
        hot = hot_params(cond, ua, ub, p.t_max_c - p.t_survey_c, p.alpha_per_c)
        out.append(Curve(MAX_SAG, cond, _xyz(cond, u, _cat(*hot, u)), est))
        windy = hot_params(cond, ua, ub, p.t_wind_c - p.t_survey_c, p.alpha_per_c)
        out.append(Curve(BLOWOUT_LEFT, cond, swing(cond, windy, ua, ub, u, theta, +1), est))
        out.append(Curve(BLOWOUT_RIGHT, cond, swing(cond, windy, ua, ub, u, theta, -1), est))
    return out, assumed
