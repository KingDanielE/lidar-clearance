"""Phase 2 in one call: towers, spans and fitted conductors from a classified cloud."""

from dataclasses import dataclass

import numpy as np

from . import graph
from .catenary import Conductor, fit_spans
from .classes import GROUND
from .spans import Span, Tower


@dataclass
class LineModel:
    towers: list[Tower]
    spans: list[Span]
    conductors: list[Conductor]
    rejected: list[Conductor]


def build(points: np.ndarray, labels: np.ndarray, wire_mask: np.ndarray) -> LineModel:
    wire_idx = np.nonzero(wire_mask)[0]
    if len(wire_idx) < 50:
        return LineModel([], [], [], [])
    # A tower and line graph (graph.py): several lines and junctions per tile.
    towers, spans = graph.build_line_graph(points, labels, wire_idx)
    ground = points[labels == GROUND]
    conds, rejected = fit_spans(points, wire_idx, spans, ground_points=ground)
    # A tower is confirmed by a conductor hanging from it. Canopy that Phase 1 called wire
    # can kink and branch like a line (t4z1: 4 towers 50-100 m off the line, in trees, none
    # of whose spans fitted a conductor). Drop such towers and their spans; their wire is
    # still reported as unmodelled in Phase 3. Ids are kept, so they stay traceable.
    by_id = {s.id: s for s in spans}
    with_wire = {t for c in conds for t in (by_id[c.span_id].tower_a, by_id[c.span_id].tower_b)}
    gone = {t.id for t in towers} - with_wire
    towers = [t for t in towers if t.id not in gone]
    spans = [s for s in spans if s.tower_a not in gone and s.tower_b not in gone]
    return LineModel(towers, spans, conds, rejected)
