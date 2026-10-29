import numpy as np

from clearance import hazard
from clearance.ground import pmf_dtm
from clearance.synthetic import tree


def _flat_dtm():
    return pmf_dtm(
        np.array(
            [[x, y, 0.0] for x in range(-40, 41, 2) for y in range(-40, 41, 2)], dtype=np.float32
        )
    )


def test_crown_radius_of_a_synthetic_tree():
    rng = np.random.default_rng(0)
    pts = tree(0.0, 0.0, 20.0, rng)  # crown radius 0.25 h + U(0, 1) = 5-6 m
    pts[:, 2] -= pts[:, 2].min()  # synthetic ground is not flat; stand it on z = 0
    xy, h = hazard.tree_tops(pts, pts[:, 2])
    assert len(xy) == 1 and abs(h[0] - 20) < 1.0
    radius, base = hazard.crowns(pts, pts[:, 2], xy, h)
    assert 4.0 <= radius[0] <= 6.5
    assert np.linalg.norm(base[0]) < 1.0  # vertical tree: base under the top


def test_leaning_tree_base_found():
    rng = np.random.default_rng(1)
    # Stem from (0, 0, 0) leaning to (6, 0, 15), a crown around the top.
    t = rng.uniform(0, 1, 400)[:, None]
    stem = np.array([0.0, 0, 0]) + t * np.array([6.0, 0, 15]) + rng.normal(0, 0.1, (400, 3))
    u = rng.normal(size=(3000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    crown = np.array([6.0, 0, 18]) + u * np.array([3.0, 3.0, 3.0]) * rng.uniform(0.6, 1, (3000, 1))
    pts = np.vstack([stem, crown]).astype(np.float32)
    xy, h = hazard.tree_tops(pts, pts[:, 2])
    radius, base = hazard.crowns(pts, pts[:, 2], xy, h)
    assert np.linalg.norm(base[0] - [0.0, 0.0]) < 1.5  # base at the foot of the stem
    assert np.linalg.norm(xy[0] - base[0]) > 4.0  # leaning ~6 m


def test_crown_extends_the_reach():
    dtm = _flat_dtm()
    wire = np.column_stack([np.linspace(-30, 30, 601), np.full(601, 20.5), np.zeros(601)])
    xy, h = np.array([[0.0, 0.0]]), np.array([20.0])
    _, _, m_stick, s_stick = hazard.fall_in(xy, h, dtm, wire, 3.8)
    _, _, m_crown, s_crown = hazard.fall_in(xy, h, dtm, wire, 3.8, xy, np.array([5.0]))
    assert s_stick[0] == hazard.FLASHOVER and m_stick[0] > 0  # 20 m stick, wire 20.5 m away
    assert s_crown[0] == hazard.STRIKE  # sqrt(20^2 + 5^2) = 20.6 m reaches it
