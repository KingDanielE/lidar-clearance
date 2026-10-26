"""Find GridNet-HD zones where vegetation grows close to the line, from a ~2% sample of
each zone's LiDAR (hand labels), without downloading whole files.

    python -m scripts.screen_zones data/t5c2.las data/t5b6.las       # local files
    python -m scripts.screen_zones --remote t1z4 t1z5a ...           # Hugging Face, by range

Distance is from each sampled vegetation point to the nearest sampled wire point. Points
are not stored in a spatially uniform order (wire and near-line vegetation sit in a few
stretches of the file), so many small chunks are needed: on t5b6, 300 chunks put the
closest vegetation at 19.7 m, 3000 chunks at 9.2 m (full file: 8.4 m). A zone with only a
handful of near-line points (t5c2: 12 within 12 m, closest 5.3 m) screens as clear; that is
fine for ranking zones by how much vegetation is near the line. Screening numbers, not
Phase 3 clearances.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from clearance.classes import GRIDNET, VEGETATION, WIRE, remap
from clearance.sample import HttpSource, LocalSource, read_sample

URL = "https://huggingface.co/datasets/heig-vd-geo/GridNet-HD/resolve/main/{z}/lidar/{z}.las"
BANDS = (5.0, 8.0, 12.0)


def screen(src, name: str, fraction: float, chunks: int, workers: int = 8) -> dict:
    t0 = time.perf_counter()
    xyz, fields, header = read_sample(src, fraction, chunks, workers=workers)
    lab = remap(fields["ground_truth"].astype(np.int64), GRIDNET)
    wire, veg = xyz[lab == WIRE], xyz[lab == VEGETATION]
    row = {"zone": name, "points_M": round(header.point_count / 1e6, 1), "sampled": len(xyz),
           "wire_pts": len(wire), "veg_pts": len(veg)}
    if len(wire) and len(veg):
        d, j = cKDTree(wire).query(veg, k=1, workers=-1)
        k = int(np.argmin(d))
        row["min_veg_to_wire_m"] = round(float(d[k]), 2)
        for band in BANDS:
            row[f"veg_within_{band:g}m"] = int(np.sum(d < band))
        row["closest_xyz"] = [round(float(v), 1) for v in veg[k]]
        row["wire_above_m"] = round(float(wire[j[k], 2] - veg[k, 2]), 1)
    row["MB_read"] = round(getattr(src, "bytes_read", len(xyz) * header.point_format.size) / 1e6, 1)
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("zones", nargs="+", help="local LAS paths, or zone names with --remote")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--fraction", type=float, default=0.02)
    ap.add_argument("--chunks", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="out/screening")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for z in args.zones:
        name = z if args.remote else Path(z).stem
        try:
            src = HttpSource(URL.format(z=z)) if args.remote else LocalSource(z)
            row = screen(src, name, args.fraction, args.chunks, args.workers)
        except Exception as e:  # one bad zone must not stop the sweep
            row = {"zone": name, "error": repr(e)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        # Written after every zone, so a long sweep can be stopped and resumed.
        (out / "screening.json").write_text(json.dumps(rows, indent=1))

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "screening.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ranked = sorted((r for r in rows if "min_veg_to_wire_m" in r), key=lambda r: r["min_veg_to_wire_m"])
    print("\nclosest vegetation to wire (sample):")
    for r in ranked:
        print(f"  {r['zone']:7s} {r['min_veg_to_wire_m']:6.2f} m  within 8 m: {r['veg_within_8m']:5d}  within 12 m: {r['veg_within_12m']:5d}")


if __name__ == "__main__":
    main()
