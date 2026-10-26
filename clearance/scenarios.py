"""Where each conductor can be: as surveyed, hot, iced and blown sideways.

Defaults follow the Swiss Leitungsverordnung (LeV, SR 734.31) for high-voltage overhead
lines ("Starkstromfreileitungen"):
- greatest sag (Art. 47): the worse of a conductor at 40 C, and a conductor at 0 C with a
  uniform additional load (snow/ice) of 20 N/m per conductor or sub-conductor, no wind;
- wind deflection (definitions, no. 37; Annex 4): conductor at 10 C, 500 N/m2 on wires.
Conductor data are typical for an AAAC/Aldrey (AlMgSi) 600 mm2 class conductor, the kind
used on Swiss lines; they are not the surveyed line's datasheet.

Sag-tension (`state_change`). Each conductor's fitted catenary gives its horizontal tension
at the survey: H1 = c * w. The change-of-state equation of a span of horizontal length L,

    H2^2 [H2 - H1 + EA w1^2 L^2 / (24 H1^2) + EA alpha (t2 - t1)] = EA w2^2 L^2 / 24,

gives the tension H2 at another temperature t2 and load w2, including the elastic relief
of a hotter or loaded wire. The new catenary (c2 = H2 / w2) passes through the same two
attachment points. Parabolic and per actual span (no ruling span, no creep); with a very
stiff conductor (EA -> infinity) it reduces to pure thermal elongation.

Blowout. Wind pushes the conductor sideways: the loaded catenary (c = H / sqrt(w^2 + q^2),
tension from the change of state at the wind temperature) swings as a rigid curve about the
chord between its attachment points by theta = atan(q / w), q = wind pressure * diameter.
Both sides are checked.

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
MAX_SAG = "max_sag"  # hot: t_max_c
MAX_SAG_ICE = "max_sag_ice"  # 0 C with snow/ice load
BLOWOUT_LEFT = "blowout_left"
BLOWOUT_RIGHT = "blowout_right"
SCENARIOS = [AS_SURVEYED, MAX_SAG, MAX_SAG_ICE, BLOWOUT_LEFT, BLOWOUT_RIGHT]
WIND_SCENARIOS = (BLOWOUT_LEFT, BLOWOUT_RIGHT)


@dataclass
class LineParams:
    """Conductor and weather assumptions; defaults per the Swiss LeV (see module docstring).
    The survey temperature and the conductor are assumptions: set them from flight records
    and the line's design data."""

    t_survey_c: float = 15.0  # conductor temperature during the LiDAR flight (assumed)
    t_max_c: float = 40.0  # LeV Art. 47 a (operators may rate lines hotter: raise it)
    t_ice_c: float = 0.0  # LeV Art. 47 b
    ice_n_per_m: float = 20.0  # LeV Art. 47 b, per conductor or sub-conductor
    t_wind_c: float = 10.0  # LeV definitions no. 37
    wind_pressure_pa: float = 500.0  # LeV Annex 4: wires and ropes
    alpha_per_c: float = 23.0e-6  # AAAC / AlMgSi
    diameter_m: float = 0.0319  # AAAC 600 mm2 class
    weight_n_per_m: float = 16.3
    area_mm2: float = 600.0
    e_modulus_gpa: float = 57.0
    assumed_span_m: float | None = None  # partial spans; default: median full span, else 300
    min_estimate_m: float = 30.0  # shortest one-tower piece whose sag and wind are estimated

    @property
    def swing_rad(self) -> float:
        return math.atan(self.wind_pressure_pa * self.diameter_m / self.weight_n_per_m)

    @property
    def ea_n(self) -> float:
        return self.e_modulus_gpa * 1e9 * self.area_mm2 * 1e-6


def state_change(h1: float, w1: float, t1: float, w2: float, t2: float, span: float, ea: float, alpha: float) -> float:
    """Horizontal tension H2 at temperature t2 and load w2 (N/m), from H1 at t1 and w1.
    Parabolic change-of-state equation (module docstring); one positive root."""
    k = h1 - ea * w1**2 * span**2 / (24 * h1**2) - ea * alpha * (t2 - t1)
    rhs = ea * w2**2 * span**2 / 24

    def f(h):
        return h * h * (h - k) - rhs

    lo = max(k, 0.0) + 1e-9
    hi = max(lo * 2, 1.0)
    while f(hi) < 0:
        hi *= 2
    return brentq(f, lo, hi)


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


def through_c(a: float, za: float, b: float, zb: float, c: float):
    """Catenary (z0, u0, c) with parameter c through (a, za) and (b, zb)."""
    L, h = b - a, zb - za
    u0 = a + L / 2 - c * math.asinh(h / (2 * c * math.sinh(L / (2 * c))))
    z0 = za - c * (math.cosh((a - u0) / c) - 1.0)
    return z0, u0, c


def loaded_params(cond: Conductor, ua: float, ub: float, w2: float, t2: float, p: LineParams):
    """The conductor's catenary at temperature t2 under load w2 (N/m), through the same
    attachment points: tension from the change of state, shape c = H2 / w2."""
    za, zb = float(cond.z(ua)), float(cond.z(ub))
    w1 = p.weight_n_per_m
    h2 = state_change(cond.c * w1, w1, p.t_survey_c, w2, t2, ub - ua, p.ea_n, p.alpha_per_c)
    return through_c(ua, za, ub, zb, h2 / w2)


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
        w = p.weight_n_per_m
        hot = loaded_params(cond, ua, ub, w, p.t_max_c, p)
        out.append(Curve(MAX_SAG, cond, _xyz(cond, u, _cat(*hot, u)), est))
        iced = loaded_params(cond, ua, ub, w + p.ice_n_per_m, p.t_ice_c, p)
        out.append(Curve(MAX_SAG_ICE, cond, _xyz(cond, u, _cat(*iced, u)), est))
        # In wind the conductor hangs in the plane of its resultant load.
        windy = loaded_params(cond, ua, ub, math.hypot(w, p.wind_pressure_pa * p.diameter_m), p.t_wind_c, p)
        out.append(Curve(BLOWOUT_LEFT, cond, swing(cond, windy, ua, ub, u, theta, +1), est))
        out.append(Curve(BLOWOUT_RIGHT, cond, swing(cond, windy, ua, ub, u, theta, -1), est))
    return out, assumed
