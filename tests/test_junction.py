"""Tower and line graph (graph.py): angle tower, branch line, towers 25 m apart, terminals."""

import numpy as np
import pytest

from clearance import linemodel
from clearance.classes import PYLON, VEGETATION, WIRE
from clearance.synthetic import make_junction_scene


def _graph(pts, labels, info):
    m = linemodel.build(pts, labels, labels == WIRE)
    names = {k: np.array(v) for k, v in info["towers"].items()}

    def name(xy):
        k = min(names, key=lambda k: np.linalg.norm(names[k] - xy))
        return k if np.linalg.norm(names[k] - xy) < 8 else "?"

    t = {x.id: name(x.xy) for x in m.towers}
    spans = sorted(
        ("".join(sorted(t.get(s.tower_a, "-") + t.get(s.tower_b, "-"))), sum(c.span_id == s.id for c in m.conductors))
        for s in m.spans
    )
    return sorted(t.values()), spans, m


@pytest.mark.parametrize("forest", [False, True])
def test_junction_graph(forest):
    pts, labels, info = make_junction_scene(seed=2)
    if forest:  # towers standing in trees: Phase 1 calls the steel vegetation
        labels = np.where(labels == PYLON, VEGETATION, labels)
    towers, spans, m = _graph(pts, labels, info)
    if not forest:
        assert towers == ["A", "B", "C", "D", "E"]
        assert spans == [("AB", 4), ("BC", 4), ("BD", 4), ("CE", 4)]  # CE: towers 25 m apart
        assert all(c.full_span for c in m.conductors)
    else:
        # Without pylon labels, terminal E (25 m from C) is not found: a terminal needs a
        # ray of >= 30 m (shorter rays past a tower are hardware). C-E is still modelled,
        # as a partial span from C with all 4 conductors.
        assert towers[:4] == ["A", "B", "C", "D"] and "?" not in towers
        assert {("AB", 4), ("BC", 4), ("BD", 4)} <= set(spans)
        assert ("CE", 4) in spans or ("-C", 4) in spans


def test_no_terminal_tower_at_the_tile_edge():
    pts, labels, info = make_junction_scene(seed=3)
    # Cut the tile through the middle of spans B-D and B-C: their wires run off the edge.
    keep = pts[:, 1] < 0.5 * (info["towers"]["B"][1] + info["towers"]["D"][1])
    towers, spans, _ = _graph(pts[keep], labels[keep], info)
    assert towers == ["A", "B"]  # no fake terminal towers where the wires leave the tile
    assert spans == [("-B", 4), ("-B", 4), ("AB", 4)]  # two partial spans from B
