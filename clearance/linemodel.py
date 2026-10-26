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
    return LineModel(towers, spans, conds, rejected)
