import numpy as np
import pytest

from clearance.catenary import catenary_z, fit, fit_spans
from clearance.classes import GROUND, PYLON, WIRE
from clearance.spans import build_spans, find_towers
from clearance.synthetic import make_scene

C_TRUE = 900.0  # synthetic.make_scene strings every wire with c = 900 m over 130 m spans
SAG_TRUE = C_TRUE * (np.cosh(65.0 / C_TRUE) - 1.0)


@pytest.fixture(scope="module")
def scene():
    # Fit on the true labels so this tests Phase 2 alone, not the classifier.
    pts, labels, info = make_scene(seed=2)
    towers = find_towers(pts, labels)
    wire_idx = np.nonzero(labels == WIRE)[0]
    spans = build_spans(pts[wire_idx, :2], towers)
    conds, rejected = fit_spans(pts, wire_idx, spans, ground_points=pts[labels == GROUND])
    assert rejected == []
    return pts, labels, info, towers, spans, conds


def test_towers_and_spans(scene):
    _, _, info, towers, spans, _ = scene
    assert sorted(round(float(t.xy[0])) for t in towers) == pytest.approx(info["towers"], abs=1)
    assert len(spans) == 2 and all(s.full for s in spans)


def test_four_conductors_per_span(scene):
    conds = scene[5]
    assert [sum(c.span_id == s for c in conds) for s in (0, 1)] == [4, 4]


def test_catenary_parameters_recovered(scene):
    for c in scene[5]:
        assert c.c == pytest.approx(C_TRUE, rel=0.03)
        assert c.sag == pytest.approx(SAG_TRUE, abs=0.05)
        assert c.rmse_z < 0.12
        assert c.min_ground_clearance > 15


def test_fit_handles_steep_partial_span():
    # Lowest point well outside the data, as on a span climbing a mountainside.
    rng = np.random.default_rng(0)
    u = np.sort(rng.uniform(0, 60, 400))
    z = catenary_z(u, 100.0, -300.0, 1100.0) + rng.normal(0, 0.03, len(u))
    z0, u0, c = fit(u, z)
    assert np.max(np.abs(catenary_z(u, z0, u0, c) - catenary_z(u, 100.0, -300.0, 1100.0))) < 0.05


def test_non_finite_fit_is_rejected():
    from clearance.catenary import Conductor, validate

    c = Conductor(0, 0, np.arange(500), np.zeros(2), np.array([1.0, 0.0]), 0.0, 0.0, 500.0,
                  0.0, 100.0, rmse_z=float("nan"), rmse_xy=0.0, coverage=1.0, full_span=True)
    assert validate(c, 100, 0.5, 10.0, 0.25) == "fit did not converge"


def test_no_towers_gives_one_span():
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(0, 100, 500), rng.normal(0, 0.05, 500)])
    spans = build_spans(xy, [])
    assert len(spans) == 1 and not spans[0].full and spans[0].length > 95


def test_pylon_fragments_merge():
    pts, labels, info = make_scene(seed=3)
    # Drop a 3 m band out of the middle of every tower so each splits into two pieces.
    keep = np.ones(len(pts), dtype=bool)
    for tx in info["towers"]:
        t = (labels == PYLON) & (np.abs(pts[:, 0] - pts[labels == PYLON, 0].min() - (tx - info["towers"][0])) < 10)
        base = pts[t, 2].min()
        keep &= ~(t & (pts[:, 2] > base + 12) & (pts[:, 2] < base + 15))
    assert len(find_towers(pts[keep], labels[keep], min_points=50, merge_distance=0)) > 3
    assert len(find_towers(pts[keep], labels[keep], min_points=50)) == 3
