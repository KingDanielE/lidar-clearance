"""A synthetic power-line corridor with known labels, for tests and quick runs."""

import numpy as np

from .classes import BUILDING, GROUND, PYLON, VEGETATION, WIRE


def _segment(a, b, spacing, rng, noise=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / spacing))
    t = rng.random(n)[:, None]
    return a + t * (b - a) + rng.normal(0, noise, (n, 3))


def ground_z(x, y):
    return 0.02 * x + 1.5 * np.sin(y / 25.0) + 0.5 * np.sin(x / 40.0)


def pylon(x, y, rng, height=30.0, arm_z=26.0, arm=8.0):
    gz = ground_z(x, y)
    pts = []
    base, top = 3.0, 1.0
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for sx, sy in corners:
        pts.append(
            _segment(
                (x + sx * base, y + sy * base, gz),
                (x + sx * top, y + sy * top, gz + height),
                0.05,
                rng,
            )
        )
    for z in np.arange(gz + 3, gz + height, 3.0):
        half = base + (top - base) * (z - gz) / height
        for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
            pts.append(
                _segment(
                    (x + ax * half, y + ay * half, z), (x + bx * half, y + by * half, z), 0.08, rng
                )
            )
    for z in (arm_z, arm_z + 0.6):
        pts.append(_segment((x, y - arm, gz + z), (x, y + arm, gz + z), 0.05, rng))
    return np.vstack(pts)


def catenary(p0, p1, c, spacing, rng, noise=0.03, keep=0.8):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1[:2] - p0[:2])
    s = np.arange(0, L, spacing)
    s = s[rng.random(len(s)) < keep]
    # Lowest point placed so both ends hit the attachment heights.
    dz = p1[2] - p0[2]
    s0 = L / 2 - c * np.arcsinh(dz / (2 * c * np.sinh(L / (2 * c))))
    z = p0[2] + c * (np.cosh((s - s0) / c) - np.cosh(s0 / c))
    t = (s / L)[:, None]
    xy = p0[:2] + t * (p1[:2] - p0[:2])
    return np.column_stack([xy, z]) + rng.normal(0, noise, (len(s), 3))


def tree(x, y, height, rng, density=60.0):
    gz = ground_z(x, y)
    crown_r = 0.25 * height + rng.random()
    crown_h = 0.6 * height
    centre = np.array([x, y, gz + height - crown_h / 2])
    n = int(density * 4 / 3 * np.pi * crown_r**2 * crown_h / 2)
    # Mostly the outer shell (what LiDAR sees), some interior returns.
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = np.where(rng.random(n) < 0.75, rng.uniform(0.85, 1.0, n), rng.uniform(0, 0.85, n))
    crown = centre + u * r[:, None] * np.array([crown_r, crown_r, crown_h / 2])
    trunk = _segment((x, y, gz), (x, y, gz + height - crown_h), 0.1, rng, noise=0.15)
    return np.vstack([crown, trunk])


def building(x, y, w, d, h, rng, spacing=0.25):
    gz = ground_z(x, y)
    xs = np.arange(x, x + w, spacing)
    ys = np.arange(y, y + d, spacing)
    gx, gy = np.meshgrid(xs, ys)
    roof = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, gz + h)])
    roof += rng.normal(0, 0.02, roof.shape)
    return roof


JUNCTION_TOWERS = {
    "A": (20.0, 60.0),
    "B": (150.0, 60.0),
    "C": (280.0, 130.0),
    "D": (170.0, 190.0),
    "E": (300.0, 145.0),
}
JUNCTION_SPANS = [("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")]


def make_junction_scene(seed: int = 0):
    """A line with an angle tower (B), a branch line leaving B (junction) and two towers
    25 m apart (C, E). Returns (points, labels, info)."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []

    def add(pts, label):
        parts.append(pts)
        labels.append(np.full(len(pts), label, dtype=np.uint8))

    n = int(320 * 220 * 12)
    gx, gy = rng.uniform(0, 320, n), rng.uniform(0, 220, n)
    add(np.column_stack([gx, gy, ground_z(gx, gy) + rng.normal(0, 0.03, n)]), GROUND)
    for x, y in JUNCTION_TOWERS.values():
        add(pylon(x, y, rng), PYLON)
    attach = [(-6.0, 26.0), (0.0, 26.6), (6.0, 26.0), (0.0, 30.0)]
    for p, q in JUNCTION_SPANS:
        a, b = np.array(JUNCTION_TOWERS[p]), np.array(JUNCTION_TOWERS[q])
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        for off, h in attach:
            pa, pb = a + off * nrm, b + off * nrm
            add(
                catenary(
                    (*pa, ground_z(*a) + h), (*pb, ground_z(*b) + h), c=900.0, spacing=0.15, rng=rng
                ),
                WIRE,
            )
    for _ in range(60):
        tx, ty = rng.uniform(5, 315), rng.uniform(5, 215)
        P = np.array([tx, ty])
        near = min(
            np.linalg.norm(
                P
                - (
                    np.array(JUNCTION_TOWERS[p])
                    + np.clip(
                        (P - JUNCTION_TOWERS[p])
                        @ (np.subtract(JUNCTION_TOWERS[q], JUNCTION_TOWERS[p]))
                        / np.sum(np.subtract(JUNCTION_TOWERS[q], JUNCTION_TOWERS[p]) ** 2),
                        0,
                        1,
                    )
                    * np.subtract(JUNCTION_TOWERS[q], JUNCTION_TOWERS[p])
                )
            )
            for p, q in JUNCTION_SPANS
        )
        if near > 14:
            add(tree(tx, ty, rng.uniform(5, 16), rng), VEGETATION)
    pts = np.vstack(parts).astype(np.float32)
    lab = np.concatenate(labels)
    shift = pts.min(axis=0)
    towers = {k: (v[0] - shift[0], v[1] - shift[1]) for k, v in JUNCTION_TOWERS.items()}
    return pts - shift, lab, {"towers": towers, "spans": JUNCTION_SPANS}


def make_scene(seed: int = 0, length: float = 300.0, width: float = 80.0):
    """Return (points float32 (n, 3), labels uint8 (n,), info dict)."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []

    def add(pts, label):
        parts.append(pts)
        labels.append(np.full(len(pts), label, dtype=np.uint8))

    # Ground, about 20 points/m2.
    n = int(length * width * 20)
    gx, gy = rng.uniform(0, length, n), rng.uniform(0, width, n)
    add(np.column_stack([gx, gy, ground_z(gx, gy) + rng.normal(0, 0.03, n)]), GROUND)

    yc = width / 2
    towers = [20.0, 150.0, 280.0]
    for tx in towers:
        add(pylon(tx, yc, rng), PYLON)

    # Three phases on the cross-arm and an earth wire on top.
    attach = [(-6.0, 26.0), (0.0, 26.6), (6.0, 26.0), (0.0, 30.0)]
    for a, b in zip(towers, towers[1:]):
        for dy, h in attach:
            p0 = (a, yc + dy, ground_z(a, yc) + h)
            p1 = (b, yc + dy, ground_z(b, yc) + h)
            add(catenary(p0, p1, c=900.0, spacing=0.15, rng=rng), WIRE)

    # Trees away from the line, plus a few near or under it.
    for _ in range(40):
        tx, ty = rng.uniform(5, length - 5), rng.uniform(2, width - 2)
        if abs(ty - yc) < 12:
            continue
        add(tree(tx, ty, rng.uniform(5, 18), rng), VEGETATION)
    for tx, ty, h in [(85.0, yc + 4.0, 22.0), (210.0, yc - 9.0, 16.0), (110.0, yc + 10.0, 19.0)]:
        add(tree(tx, ty, h, rng), VEGETATION)

    add(building(230.0, 65.0, 14.0, 10.0, 6.0, rng), BUILDING)

    pts = np.vstack(parts).astype(np.float32)
    lab = np.concatenate(labels)
    pts -= pts.min(axis=0)
    return pts, lab, {"towers": towers}
