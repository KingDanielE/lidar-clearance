"""A conductor is modelled only where its data supports it (t2z5: a parallel line turning
away, and short pieces of a turning line)."""

import numpy as np

from clearance.catenary import catenary_z, fit_conductor
from clearance.scenarios import MAX_SAG, LineParams, attachments, curves
from clearance.spans import Span


def _wire(u0, u1, rng):
    u = rng.uniform(u0, u1, 2000)
    return np.column_stack([u, np.zeros_like(u), catenary_z(u, 40.0, 100.0, 1100.0)]).astype(np.float32)


def test_wire_stopping_short_of_a_tower_is_not_modelled_to_it():
    rng = np.random.default_rng(0)
    pts = _wire(4, 130, rng)  # span runs 0-200 between two towers; data stops at 130
    span = Span(0, np.array([0.0, 0.0]), np.array([200.0, 0.0]), 0, 1, np.arange(len(pts)))
    c = fit_conductor(pts, np.arange(len(pts)), span, 0)
    assert c.at_tower_a and not c.at_tower_b and not c.full_span
    assert c.u_min == 0.0 and abs(c.u_max - 130) < 1


def test_wire_reaching_both_towers_is_full():
    rng = np.random.default_rng(1)
    pts = _wire(4, 196, rng)
    span = Span(0, np.array([0.0, 0.0]), np.array([200.0, 0.0]), 0, 1, np.arange(len(pts)))
    c = fit_conductor(pts, np.arange(len(pts)), span, 0)
    assert c.full_span and c.u_min == 0.0 and abs(c.u_max - 200) < 1e-6
    assert attachments(c, span, 300.0)[2] is False  # exact, not estimated


def test_short_piece_gets_no_sag_or_wind_positions():
    rng = np.random.default_rng(2)
    pts = _wire(4, 20, rng)  # 16 m piece off one tower, span continues off the tile
    span = Span(0, np.array([0.0, 0.0]), np.array([95.0, 0.0]), 0, None, np.arange(len(pts)))
    c = fit_conductor(pts, np.arange(len(pts)), span, 0)
    assert attachments(c, span, 300.0) is None
    out, _ = curves([c], [span], LineParams())
    assert [k.scenario for k in out] == ["as_surveyed"]
    assert not any(k.scenario == MAX_SAG for k in out)


def test_long_piece_off_one_tower_is_estimated():
    rng = np.random.default_rng(3)
    pts = _wire(4, 90, rng)
    span = Span(0, np.array([0.0, 0.0]), np.array([95.0, 0.0]), 0, None, np.arange(len(pts)))
    c = fit_conductor(pts, np.arange(len(pts)), span, 0)
    ua, ub, est = attachments(c, span, 300.0)
    assert est and ua == 0.0 and ub == 300.0
