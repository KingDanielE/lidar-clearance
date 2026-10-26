"""Phase 2: towers, spans and a catenary per conductor.

Reads the classified voxels written by Phase 1:

    python -m scripts.run_phase2 out/t5c2/t5c2_classified_voxels.las --out out/t5c2

With truth labels in the file it also fits the true wire points and reports how far the
model from predicted points is from the model from labelled points.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from clearance import linemodel  # noqa: E402
from clearance.catenary import Conductor  # noqa: E402
from clearance.classes import PYLON, WIRE  # noqa: E402
from clearance.io import read_classified, write_las  # noqa: E402
from clearance.spans import Tower  # noqa: E402


def model(points, labels, wire_mask):
    m = linemodel.build(points, labels, wire_mask)
    return m.towers, m.spans, m.conductors, m.rejected


def describe(c: Conductor) -> dict:
    mid = 0.5 * (c.u_min + c.u_max)
    return {
        "span": c.span_id,
        "conductor": c.id,
        "points": int(len(c.idx)),
        "full_span": c.full_span,
        "length_m": round(c.u_max - c.u_min, 1),
        "coverage": round(c.coverage, 3),
        "c_m": round(c.c, 1),
        "sag_m": None if c.sag is None else round(c.sag, 2),
        "lowest_point_u_m": round(c.u0, 1),
        "z_mid_m": round(float(c.z(mid)), 2),
        "rmse_z_m": round(c.rmse_z, 3),
        "rmse_xy_m": round(c.rmse_xy, 3),
        "outliers_dropped": c.outliers,
        "min_ground_clearance_m": None if c.min_ground_clearance is None else round(c.min_ground_clearance, 2),
    }


def compare(pred: list[Conductor], truth: list[Conductor]) -> list[dict]:
    """Match each predicted conductor to the nearest true one, report the model difference."""
    out = []
    for p in pred:
        s = p.sample(0.5)
        best = None
        for t in truth:
            u = (s[:, :2] - t.origin) @ t.axis
            inside = (u >= t.u_min) & (u <= t.u_max)
            if inside.sum() < 20:
                continue
            lateral = np.abs((s[inside, :2] - t.origin) @ np.array([-t.axis[1], t.axis[0]]))
            dz = s[inside, 2] - t.z(u[inside])
            score = np.median(lateral) + np.median(np.abs(dz))
            if best is None or score < best[0]:
                best = (score, t, dz, lateral)
        if best is None or best[0] > 2.0:
            out.append({"span": p.span_id, "conductor": p.id, "match": None})
            continue
        _, t, dz, lateral = best
        out.append({
            "span": p.span_id,
            "conductor": p.id,
            "match": [t.span_id, t.id],
            "max_abs_dz_m": round(float(np.max(np.abs(dz))), 3),
            "mean_abs_dz_m": round(float(np.mean(np.abs(dz))), 3),
            "max_lateral_m": round(float(np.max(lateral)), 3),
            "sag_diff_m": None if p.sag is None or t.sag is None else round(p.sag - t.sag, 3),
            "c_diff_pct": round(100 * (p.c - t.c) / t.c, 1),
        })
    return out


def plot(points, towers: list[Tower], spans, conds: list[Conductor], prefix: str, title: str):
    files = []
    by_span = {}
    for c in conds:
        by_span.setdefault(c.span_id, []).append(c)
    ids = sorted(by_span) or [s.id for s in spans[:1]]
    if not ids:
        return files
    fig, axes = plt.subplots(len(ids), 1, figsize=(13, 3.6 * len(ids)), squeeze=False)
    colors = plt.cm.tab10.colors
    for ax, sid in zip(axes[:, 0], ids):
        for c in by_span[sid]:
            p = points[c.idx]
            u = (p[:, :2] - c.origin) @ c.axis
            col = colors[c.id % 10]
            ax.scatter(u, p[:, 2], s=1, color=col, alpha=0.5, linewidths=0, rasterized=True)
            m = c.sample(0.5)
            uu = (m[:, :2] - c.origin) @ c.axis
            ax.plot(uu, m[:, 2], color="black", lw=0.8)
            ax.annotate(f"{c.id}: rmse {100 * c.rmse_z:.1f} cm", (uu[-1], m[-1, 2]), fontsize=7, xytext=(4, 0), textcoords="offset points")
        span = spans[sid]
        kind = "full span" if span.full else "partial span (runs off the tile)"
        ax.set_title(f"{title} span {sid}: {kind}, {span.length:.0f} m", fontsize=10)
        ax.set_ylabel("z (m)")
    axes[-1, 0].set_xlabel("distance along conductor (m)")
    fig.tight_layout()
    files.append(f"{prefix}_spans.png")
    fig.savefig(files[-1], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    for c in conds:
        m = c.sample(1.0)
        ax.plot(m[:, 0], m[:, 1], lw=0.8, color=colors[c.span_id % 10])
    for t in towers:
        ax.plot(*t.xy, marker="^", color="#2563eb", ms=10)
        ax.annotate(f"T{t.id}", t.xy, xytext=(6, 6), textcoords="offset points")
    for s in spans:
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], ls=":", color="gray", lw=0.8)
        mid = (s.a + s.b) / 2
        ax.annotate(f"span {s.id}", mid, xytext=(8, -8), textcoords="offset points", fontsize=8, color="gray")
    ax.set_aspect("equal")
    ax.set_title(f"{title}: towers, spans and fitted conductors (plan)")
    fig.tight_layout()
    files.append(f"{prefix}_route.png")
    fig.savefig(files[-1], dpi=140)
    plt.close(fig)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="classified voxels LAS from Phase 1")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem.replace("_classified_voxels", "")

    cloud, labels = read_classified(args.input)
    pts = cloud.points
    towers, spans, conds, rejected = model(pts, labels, labels == WIRE)
    print(f"{name}: {len(towers)} towers, {len(spans)} spans, {len(conds)} conductors, {len(rejected)} clusters rejected")
    for t in towers:
        print(f"  tower {t.id}: xy ({t.xy[0]:.1f}, {t.xy[1]:.1f}), top z {t.top_z:.1f}, {t.n_points} pts")
    rows = [describe(c) for c in conds]
    for r in rows:
        print("  ", r)

    report = {
        "tile": name,
        "towers": [{"id": t.id, "x": float(t.xy[0] + cloud.origin[0]), "y": float(t.xy[1] + cloud.origin[1]), "top_z": float(t.top_z + cloud.origin[2])} for t in towers],
        "spans": [
            {
                "id": s.id, "length_m": round(s.length, 1), "full": s.full,
                "tower_a": s.tower_a, "tower_b": s.tower_b, "wire_points": int(len(s.idx)),
                # Wire points left out of every conductor (too sparse to cluster or fit).
                "unmodelled_points": int(len(s.idx) - sum(len(c.idx) for c in conds if c.span_id == s.id)),
            }
            for s in spans
        ],
        "conductors": rows,
        "rejected_clusters": [
            {"span": c.span_id, "points": int(len(c.idx)), "reason": c.rejected} for c in rejected
        ],
    }

    if cloud.labels is not None:
        # Same towers and labels for ground; only the wire points change.
        truth_labels = labels.copy()
        truth_labels[cloud.labels == PYLON] = PYLON
        _, _, tconds, _ = model(pts, truth_labels, cloud.labels == WIRE)
        report["truth_conductors"] = [describe(c) for c in tconds]
        report["predicted_vs_truth"] = compare(conds, tconds)
        print("predicted vs truth-label models:")
        for r in report["predicted_vs_truth"]:
            print("  ", r)

    (out / f"{name}_phase2.json").write_text(json.dumps(report, indent=2))
    with open(out / f"{name}_conductors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["span"])
        w.writeheader()
        w.writerows(rows)

    # Densified model, one point every 0.25 m: the input to Phase 3's distance queries.
    samples = [c.sample(0.25) for c in conds]
    if samples:
        m = np.vstack(samples).astype(np.float32)
        write_las(out / f"{name}_conductor_model.las", m, cloud.origin, np.full(len(m), WIRE))
    files = plot(pts, towers, spans, conds, str(out / name), name)
    print("wrote", *files, sep="\n  ")


if __name__ == "__main__":
    main()
