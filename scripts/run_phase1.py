"""Phase 1: classify a LiDAR tile into ground / vegetation / wires / pylons / buildings.

    python -m scripts.run_phase1 --synthetic
    python -m scripts.run_phase1 data/t5c2.las --truth-field ground_truth --truth-map gridnet
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clearance import pipeline
from clearance.classes import MAPPINGS, NAMES, remap
from clearance.evaluate import confusion, format_report, scores
from clearance.io import read_las, write_las
from clearance.plots import plot_all
from clearance.synthetic import make_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="LAS/LAZ file")
    ap.add_argument("--synthetic", action="store_true", help="use a generated corridor")
    ap.add_argument("--truth-field", help="LAS field with ground-truth labels")
    ap.add_argument("--truth-map", choices=sorted(MAPPINGS), default="asprs")
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if args.synthetic:
        name = "synthetic"
        points, truth, _ = make_scene()
        origin, crs = np.zeros(3), None
    else:
        name = Path(args.input).stem
        print(f"reading {args.input}")
        cloud = read_las(args.input, args.truth_field)
        points, origin, crs = cloud.points, cloud.origin, cloud.crs
        truth = remap(cloud.labels, MAPPINGS[args.truth_map]) if cloud.labels is not None else None
        print(f"  read in {time.perf_counter() - t0:.1f} s")

    ext = np.ptp(points, axis=0)
    print(f"{name}: {len(points):,} points, extent {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} m")
    res = pipeline.run(points, voxel=args.voxel)

    counts = {NAMES[int(c)]: int(n) for c, n in zip(*np.unique(res.labels, return_counts=True))}
    report = {
        "tile": name,
        "points": int(len(points)),
        "voxels": int(len(res.vox_points)),
        "extent_m": [round(float(v), 1) for v in ext],
        "density_pts_per_m2": round(len(points) / float(ext[0] * ext[1]), 1),
        "predicted_counts": counts,
        "timings_s": res.timings,
    }
    md = [f"# Phase 1 classification: {name}\n", f"- points: {len(points):,}",
          f"- voxels ({args.voxel} m): {len(res.vox_points):,}",
          f"- timings (s): {res.timings}\n"]
    if truth is not None:
        s = scores(res.labels, truth)
        cm = confusion(res.labels, truth)
        report["scores"] = s
        report["confusion"] = cm.tolist()
        md.append(format_report(s, cm))
        print(format_report(s, cm))

    (out / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    (out / f"{name}_report.md").write_text("\n".join(md) + "\n")

    vox_truth = None
    if truth is not None:
        # Any one point's truth label stands in for its voxel (profile plot, Phase 2 checks).
        vox_truth = np.full(len(res.vox_points), -1, dtype=np.int16)
        vox_truth[res.inverse] = truth
    write_las(
        out / f"{name}_classified_voxels.las", res.vox_points, origin, res.vox_labels, crs, vox_truth
    )
    files = plot_all(res.vox_points, res.vox_labels, str(out / name), title=name, truth=vox_truth)
    print("wrote", *files, sep="\n  ")
    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
