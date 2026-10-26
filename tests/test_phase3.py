import math

import numpy as np
import pytest

from clearance import encroach, hazard, linemodel, scenarios
from clearance.catenary import catenary_z
from clearance.classes import GROUND, VEGETATION, WIRE
from clearance.ground import pmf_dtm
from clearance.scenarios import arc_length, through_points
from clearance.synthetic import make_scene


@pytest.fixture(scope="module")
def corridor():
    pts, labels, info = make_scene(seed=4)
    model = linemodel.build(pts, labels, labels == WIRE)
    line = scenarios.LineParams()
    curves, _ = scenarios.curves(model.conductors, model.spans, line)
    return pts, labels, model, curves, line


def test_through_points_round_trip():
    z0, u0, c = 100.0, 40.0, 900.0
    a, b = 0.0, 130.0
    za, zb = catenary_z(a, z0, u0, c), catenary_z(b, z0, u0, c)
    got = through_points(a, za, b, zb, arc_length(z0, u0, c, a, b))
    assert got == pytest.approx((z0, u0, c), rel=1e-6)


def test_hot_conductor_sags_by_the_expected_amount(corridor):
    # Parabolic check: S - L = 8 s^2 / (3 L), so a longer wire gives s' = sqrt(3 L (S' - L) / 8).
    _, _, model, curves, line = corridor
    cond = model.conductors[0]
    span = next(s for s in model.spans if s.id == cond.span_id)
    L = cond.u_max - cond.u_min
    s0 = cond.sag
    S = L + 8 * s0**2 / (3 * L)
    S_hot = S * (1 + line.alpha_per_c * (line.t_max_c - line.t_survey_c))
    expected = math.sqrt(3 * L * (S_hot - L) / 8)
    surveyed = next(c for c in curves if c.conductor is cond and c.scenario == scenarios.AS_SURVEYED)
    hot = next(c for c in curves if c.conductor is cond and c.scenario == scenarios.MAX_SAG)
    drop = float(np.max(surveyed.xyz[:, 2] - hot.xyz[:, 2]))
    assert span.full
    assert drop == pytest.approx(expected - s0, abs=0.05)
    # Attachment points do not move.
    assert hot.xyz[0, 2] == pytest.approx(surveyed.xyz[0, 2], abs=0.01)
    assert hot.xyz[-1, 2] == pytest.approx(surveyed.xyz[-1, 2], abs=0.01)


def test_blowout_swings_by_the_wind_angle(corridor):
    _, _, model, curves, line = corridor
    cond = model.conductors[0]
    still = next(c for c in curves if c.conductor is cond and c.scenario == scenarios.AS_SURVEYED)
    left = next(c for c in curves if c.conductor is cond and c.scenario == scenarios.BLOWOUT_LEFT)
    right = next(c for c in curves if c.conductor is cond and c.scenario == scenarios.BLOWOUT_RIGHT)
    mid = len(still.xyz) // 2
    n = np.array([-cond.axis[1], cond.axis[0]])
    # Level span: sag s swings to s sin(theta) sideways, rises by s (1 - cos(theta)).
    s = cond.sag
    theta = line.swing_rad
    assert (left.xyz[mid, :2] - still.xyz[mid, :2]) @ n == pytest.approx(s * math.sin(theta), abs=0.1)
    assert (right.xyz[mid, :2] - still.xyz[mid, :2]) @ n == pytest.approx(-s * math.sin(theta), abs=0.1)
    assert left.xyz[mid, 2] - still.xyz[mid, 2] == pytest.approx(s * (1 - math.cos(theta)), abs=0.1)


def test_tree_under_the_line_is_a_violation_and_far_trees_are_clear(corridor):
    pts, labels, model, curves, _ = corridor
    rules = encroach.Rules()
    veg = pts[labels == VEGETATION]
    clr = encroach.compute(veg, curves, rules)
    rows, _ = encroach.zones(veg, clr, rules, np.zeros(3))
    assert rows[0]["tier"] == encroach.VIOLATION
    # The 22 m tree planted 4 m off the centre line at x = 85 (scene coordinates shift by
    # the cloud minimum, so compare relative to the tower at x = 20).
    tower_x = min(t.xy[0] for t in model.towers)
    assert rows[0]["x"] - tower_x == pytest.approx(85 - 20, abs=8)
    worst, _ = clr.worst()
    far = np.abs(veg[:, 1] - np.median([t.xy[1] for t in model.towers])) > 25
    assert np.all(rules.tier(worst[far]) == encroach.CLEAR)


def test_wire_fragment_dropped_but_contact_kept(corridor):
    pts, labels, model, curves, _ = corridor
    rules = encroach.Rules()
    wire_bits = pts[labels == WIRE][::50]  # stray wire points mislabelled as vegetation
    # A crown pushed into the conductor: a ball of points whose top touches the wire.
    w = curves[0].xyz[len(curves[0].xyz) // 2]
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    crown = w - np.array([0, 0, 2.6]) + u * 2.5 * rng.uniform(0.6, 1, (3000, 1))
    veg = np.vstack([wire_bits, crown])
    clr = encroach.compute(veg, curves, rules)
    frag = encroach.wire_fragments(veg, clr)
    assert frag[: len(wire_bits)].mean() > 0.9
    assert not frag[len(wire_bits):].any()


def test_hazard_tree_reach():
    dtm = pmf_dtm(np.array([[x, y, 0.0] for x in range(0, 60, 2) for y in range(0, 60, 2)], dtype=np.float32))
    wire = np.column_stack([np.linspace(0, 60, 601), np.full(601, 30.0), np.full(601, 15.0)])
    xy = np.array([[30.0, 50.0], [30.0, 40.0], [30.0, 55.0]])
    h = np.array([30.0, 18.5, 10.0])  # base-to-wire distances: 25, 18.03, 29.2
    _, d, margin, status = hazard.fall_in(xy, h, dtm, wire, mvcd=3.0)
    assert status.tolist() == [hazard.STRIKE, hazard.STRIKE, hazard.SAFE]
    assert margin[0] == pytest.approx(25.0 - 30.0, abs=0.05)


def test_tree_tops_found(corridor):
    pts, labels, *_ = corridor
    veg = pts[labels == VEGETATION]
    dtm = pmf_dtm(pts[labels == GROUND])
    hag = veg[:, 2] - dtm.sample(veg[:, :2])
    xy, h = hazard.tree_tops(veg, hag)
    assert len(xy) >= 20
    assert h.max() == pytest.approx(22.0, abs=1.5)  # tallest synthetic tree
