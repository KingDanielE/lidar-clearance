"""Phase 2 in one call: towers, spans and fitted conductors from a classified cloud."""

from dataclasses import dataclass

import numpy as np

from .catenary import Conductor, fit_spans
from .classes import GROUND
from .spans import Span, Tower, build_spans, find_towers, merge_towers, towers_from_wires


@dataclass
class LineModel:
    towers: list[Tower]
    spans: list[Span]
    conductors: list[Conductor]
    rejected: list[Conductor]


def build(points: np.ndarray, labels: np.ndarray, wire_mask: np.ndarray) -> LineModel:
    # Pylon clusters first; the wire geometry adds towers they missed (towers in forest).
    towers = merge_towers(find_towers(points, labels), towers_from_wires(points, labels))
    wire_idx = np.nonzero(wire_mask)[0]
    spans = build_spans(points[wire_idx, :2], towers)
    ground = points[labels == GROUND]
    conds, rejected = fit_spans(points, wire_idx, spans, ground_points=ground)
    return LineModel(towers, spans, conds, rejected)
