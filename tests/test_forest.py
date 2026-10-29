"""Cases from the forest tile (t1z6a): towers fused with the canopy, bridged conductors."""

import numpy as np

from clearance.catenary import catenary_z, fit_spans
from clearance.classes import PYLON, VEGETATION, WIRE
from clearance.spans import Span, find_towers, towers_from_wires
from clearance.synthetic import make_scene


def test_towers_found_from_wires_when_pylons_are_labelled_vegetation():
    pts, labels, info = make_scene(seed=5)
    forest = labels.copy()
    forest[forest == PYLON] = VEGETATION  # what Phase 1 does to a tower standing in trees
    assert find_towers(pts, forest) == []
    found = towers_from_wires(pts, forest)
    tower_x = np.array(info["towers"]) - info["towers"][0] + pts[labels == PYLON, 0].min() + 3.0
    # The middle tower has wire on both sides, so its kink is measurable; end towers of a
    # tile are not (they become partial-span ends instead).
    assert len(found) >= 1
    for t in found:
        assert np.min(np.abs(tower_x - t.xy[0])) < 5.0


def test_no_towers_from_a_single_clean_span():
    pts, labels, info = make_scene(seed=6)
    one_span = (labels != WIRE) | (pts[:, 0] < pts[labels == PYLON, 0].min() + 125)
    assert towers_from_wires(pts[one_span], labels[one_span]) == []


def test_bridged_conductors_are_split():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 200, 3000)
    low = np.column_stack([u, np.zeros_like(u), catenary_z(u, 40.0, 100.0, 1200.0)])
    high = low + np.array([0.0, 0.0, 1.0])
    # Stray points bridging the two every 20 m (0.4 m apart: joined at eps 0.5, not 0.25).
    bridge = np.array(
        [
            [x, 0.0, catenary_z(x, 40.0, 100.0, 1200.0) + dz]
            for x in range(10, 200, 20)
            for dz in (0.4, 0.8)
        ]
    )
    pts = np.vstack([low, high, bridge]) + rng.normal(0, 0.02, (len(low) * 2 + len(bridge), 3))
    span = Span(0, np.array([0.0, 0.0]), np.array([200.0, 0.0]), 0, 1, np.arange(len(pts)))
    conds, rejected = fit_spans(pts.astype(np.float32), np.arange(len(pts)), [span])
    assert len(conds) == 2
    assert sorted(round(c.z(100.0) - 40.0, 1) for c in conds) == [0.0, 1.0]
