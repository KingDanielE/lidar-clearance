import numpy as np

from clearance import encroach, linemodel, scenarios
from clearance.classes import VEGETATION, WIRE
from clearance.synthetic import make_scene


def test_unmodelled_wire_is_reported_with_its_vegetation():
    pts, labels, _ = make_scene(seed=7)
    m = linemodel.build(pts, labels, labels == WIRE)
    curves, _ = scenarios.curves(m.conductors, m.spans, scenarios.LineParams())
    rules = encroach.Rules()
    veg = pts[labels == VEGETATION]
    wire = pts[labels == WIRE]
    # Every wire is modelled: nothing to report.
    assert encroach.unmodelled_wire(wire, curves, veg, rules, m.towers, np.zeros(3)) == []
    # A 40 m wire the model does not know about, with a bush 3 m below it.
    x = np.linspace(100, 140, 400)
    extra = np.column_stack([x, np.full_like(x, 5.0), np.full_like(x, 20.0)]).astype(np.float32)
    bush = (np.array([120.0, 5.0, 17.0]) + np.random.default_rng(0).normal(0, 0.5, (200, 3))).astype(np.float32)
    rows = encroach.unmodelled_wire(np.vstack([wire, extra]), curves, np.vstack([veg, bush]), rules, m.towers, np.zeros(3))
    assert len(rows) == 1
    assert abs(rows[0]["length_m"] - 40) < 1.5
    assert rows[0]["vegetation_within_monitor_m"] >= 200
