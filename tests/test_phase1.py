import numpy as np
import pytest

from clearance import pipeline
from clearance.classes import GRIDNET, IGNORE, PYLON, VEGETATION, WIRE, remap
from clearance.evaluate import scores
from clearance.features import gap_below
from clearance.synthetic import make_scene


@pytest.fixture(scope="module")
def synthetic():
    pts, truth, _ = make_scene(seed=1)
    res = pipeline.run(pts, voxel=0.15, log=lambda *a: None)
    return scores(res.labels, truth)


def test_wires_found(synthetic):
    assert synthetic["wire"]["recall"] > 0.95
    assert synthetic["wire"]["precision"] > 0.85


def test_pylons_found(synthetic):
    assert synthetic["pylon"]["iou"] > 0.85


def test_ground_and_vegetation(synthetic):
    assert synthetic["ground"]["iou"] > 0.98
    assert synthetic["vegetation"]["iou"] > 0.95


def test_gap_below_ignores_points_in_the_same_run():
    # A sloping wire: three points stacked 0.2 m apart in one column, 20 m above ground.
    pts = np.array([[0.1, 0.1, 0.0], [0.1, 0.1, 20.0], [0.1, 0.1, 20.2], [0.1, 0.1, 20.4]], np.float32)
    gap = gap_below(pts)
    assert np.isinf(gap[0])
    assert np.allclose(gap[1:], 20.0)


def test_gridnet_mapping():
    raw = np.array([0, 5, 6, 9, 14, 16, 21, 255])
    out = remap(raw, GRIDNET)
    assert out.tolist() == [PYLON, WIRE, WIRE, PYLON, VEGETATION, 2, 6, IGNORE]
