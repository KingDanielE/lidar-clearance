"""Plan view, along-line profile and 3D view coloured by class."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .classes import COLORS, GROUND, NAMES, WIRE  # noqa: E402

DRAW_ORDER = [GROUND, 1, 6, 5, 15, WIRE]  # wires drawn last so they stay visible


def _subsample(n: int, labels: np.ndarray, max_points: int, rng) -> np.ndarray:
    """Random subsample, but keep every wire point (they are few and the point of the plot)."""
    wire = np.nonzero(labels == WIRE)[0]
    other = np.nonzero(labels != WIRE)[0]
    budget = max(0, max_points - len(wire))
    if len(other) > budget:
        other = rng.choice(other, budget, replace=False)
    return np.concatenate([other, wire])


def _scatter(ax, x, y, labels, size):
    for c in DRAW_ORDER:
        m = labels == c
        if m.any():
            s = size * 2.5 if c == WIRE else size
            ax.scatter(x[m], y[m], s=s, c=COLORS[c], linewidths=0, rasterized=True)


def _legend(ax, labels):
    present = [c for c in DRAW_ORDER if np.any(labels == c)]
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[c], label=NAMES[c]) for c in present]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def line_frame(points: np.ndarray, labels: np.ndarray):
    """Rotation putting the wire direction along x. Falls back to the cloud's main axis."""
    src = points[labels == WIRE] if np.sum(labels == WIRE) > 50 else points
    xy = src[:, :2]
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd((xy - c)[:: max(1, len(xy) // 20000)], full_matrices=False)
    return c, vt[0], vt[1]


def plot_all(points, labels, out_prefix, title="", truth=None, max_points=400_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = _subsample(len(points), labels, max_points, rng)
    p, lab = points[idx], labels[idx]
    files = []

    # Plan view.
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter(ax, p[:, 0], p[:, 1], lab, 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title} plan view")
    _legend(ax, lab)
    fig.tight_layout()
    files.append(f"{out_prefix}_plan.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # Profile along the line, corridor of +-20 m.
    c, along, across = line_frame(points, labels)

    def profile(ax, lab_, name):
        s = (p[:, :2] - c) @ along
        o = (p[:, :2] - c) @ across
        m = np.abs(o) < 20
        _scatter(ax, s[m], p[m, 2], lab_[m], 0.8)
        ax.set_ylabel("z (m)")
        ax.set_title(name)
        _legend(ax, lab_[m])

    rows = 2 if truth is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(14, 4.5 * rows), sharex=True, squeeze=False)
    profile(axes[0, 0], lab, f"{title} profile along the line (+-20 m corridor), predicted")
    if truth is not None:
        profile(axes[1, 0], truth[idx], "ground truth")
    axes[-1, 0].set_xlabel("distance along line (m)")
    fig.tight_layout()
    files.append(f"{out_prefix}_profile.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    # 3D view.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    sub = _subsample(len(p), lab, 150_000, rng)
    for cl in DRAW_ORDER:
        m = lab[sub] == cl
        if m.any():
            q = p[sub][m]
            ax.scatter(
                q[:, 0], q[:, 1], q[:, 2], s=0.3 if cl != WIRE else 1.2, c=COLORS[cl], linewidths=0
            )
    ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), max(np.ptp(p[:, 2]) * 2, 1)))
    ax.view_init(elev=25, azim=-60)
    ax.set_title(f"{title} 3D view")
    _legend(ax, lab)
    fig.tight_layout()
    files.append(f"{out_prefix}_3d.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)
    return files
