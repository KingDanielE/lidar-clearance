"""Joining neighbouring tiles: a span cut by the tile edge becomes whole."""

import numpy as np

from clearance import linemodel
from clearance.classes import WIRE
from clearance.io import read_many, write_las
from clearance.synthetic import make_scene


def test_joined_tiles_give_full_spans(tmp_path):
    pts, labels, info = make_scene(seed=8)
    origin = np.array([2_600_000.0, 1_200_000.0, 400.0])  # real-world-sized coordinates
    cut = pts[:, 0] < 85.0  # through the middle of the first span
    paths = []
    for k, m in enumerate((cut, ~cut)):
        # Each tile gets its own LAS origin, as real tiles do.
        shift = np.floor(pts[m].min(axis=0))
        p = tmp_path / f"tile{k}.las"
        write_las(p, pts[m] - shift, origin + shift, labels[m], truth=labels[m].astype(np.int16))
        paths.append(p)

    joined = read_many(paths, "truth")
    assert len(joined.points) == len(pts)
    real = np.sort((joined.points.astype(np.float64) + joined.origin)[:, 0])
    assert np.allclose(real, np.sort(pts[:, 0].astype(np.float64) + origin[0]), atol=2e-3)

    lab = joined.labels.astype(np.uint8)
    whole = linemodel.build(joined.points, lab, lab == WIRE)
    assert sum(s.full for s in whole.spans) == 2
    assert all(c.full_span for c in whole.conductors)

    first = read_many(paths[:1], "truth")
    lab1 = first.labels.astype(np.uint8)
    half = linemodel.build(first.points, lab1, lab1 == WIRE)
    assert not any(s.full for s in half.spans)  # the cut span is partial on its own
