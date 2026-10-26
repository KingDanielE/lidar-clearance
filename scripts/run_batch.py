"""Run all three phases on several tiles and write one summary row per tile.

    python -m scripts.run_batch data/t5c2.las data/t5b6.las --truth-field ground_truth --truth-map gridnet

Outputs go to out/<tile>/ as usual; the summary to out/batch_summary.csv and .md.
--skip-existing reuses a tile's outputs when its Phase 3 report is already there.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from scripts import run_phase1, run_phase2, run_phase3

PHASE3_EXTRA = ("kv", "mvcd", "t_survey", "t_max", "wind_pa")


def _r(v, n=3):
    return None if v is None else round(v, n)


def summarise(out: Path, name: str) -> dict:
    p1 = json.loads((out / f"{name}_report.json").read_text())
    p2 = json.loads((out / f"{name}_phase2.json").read_text())
    p3 = json.loads((out / f"{name}_phase3.json").read_text())
    s = p1.get("scores", {})
    row = {
        "tile": name,
        "points_M": round(p1["points"] / 1e6, 1),
        "wire_precision": _r(s.get("wire", {}).get("precision")),
        "wire_recall": _r(s.get("wire", {}).get("recall")),
        "wire_iou": _r(s.get("wire", {}).get("iou")),
        "pylon_iou": _r(s.get("pylon", {}).get("iou")),
        "towers": len(p2["towers"]),
        "spans": len(p2["spans"]),
        "conductors": len(p2["conductors"]),
        "max_rmse_cm": round(100 * max((c["rmse_z_m"] for c in p2["conductors"]), default=0), 1),
        "max_dz_vs_truth_cm": round(100 * max((x["max_abs_dz_m"] for x in p2.get("predicted_vs_truth", []) if x["match"]), default=0), 1),
    }
    clr = p3["min_clearance_by_scenario_m"]
    row["min_clear_surveyed_m"] = clr.get("as_surveyed")
    row["min_clear_max_sag_m"] = clr.get("max_sag")
    row["min_clear_blowout_m"] = min((v for k, v in clr.items() if k.startswith("blowout")), default=None)
    tiers = ("violation", "urgent", "monitor")
    row["zones"] = "/".join(str(sum(z["tier"] == t for z in p3["zones"])) for t in tiers)
    t = p3.get("with_truth_vegetation")
    if t:
        tc = t["min_clearance_by_scenario_m"]
        row["truth_min_clear_max_sag_m"] = tc.get("max_sag")
        row["truth_zones"] = "/".join(str(t["zones_by_tier"][k]) for k in tiers)
    row["hazard_strike"] = p3["hazard_trees"]["strike"]
    row["hazard_within_mvcd"] = p3["hazard_trees"]["within_mvcd"]
    v = p3.get("phase3_vs_truth")
    if v:
        row["truth_hazard"] = f"{t['hazard_strike']}/{t['hazard_within_mvcd']}"
        for key, short in (("flagged_points", "pts"), ("hazard_trees", "trees")):
            row[f"{short}_P"], row[f"{short}_R"] = v[key]["precision"], v[key]["recall"]
        row["zones_P"] = v["zones"]["predicted_zones"]["precision"]
        row["zones_R"] = v["zones"]["predicted_zones"]["recall"]
    e = p3.get("end_to_end_vs_truth")
    if e:
        ref = p3["reference"]
        row["ref_zones"] = "/".join(str(ref["zones_by_tier"][k]) for k in tiers)
        row["e2e_zones_P"] = e["zones"]["predicted_zones"]["precision"]
        row["e2e_zones_R"] = e["zones"]["predicted_zones"]["recall"]
        row["e2e_same_tier"] = f"{e['zones']['same_tier']}/{e['zones']['pairs']}"
        row["e2e_trees_P"] = e["hazard_trees"]["precision"]
        row["e2e_trees_R"] = e["hazard_trees"]["recall"]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", nargs="+", help="LAS files")
    ap.add_argument("--truth-field")
    ap.add_argument("--truth-map", default="asprs")
    ap.add_argument("--out", default="out")
    ap.add_argument("--skip-existing", action="store_true")
    for k in PHASE3_EXTRA:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float)
    args = ap.parse_args(argv)

    rows = []
    for tile in args.tiles:
        parts = tile.split("+")  # "a.las+b.las": neighbouring tiles joined into one
        name = "+".join(Path(p).stem for p in parts)
        out = Path(args.out) / name
        t0 = time.perf_counter()
        if not (args.skip_existing and (out / f"{name}_phase3.json").exists()):
            p1 = parts + ["--out", str(out)]
            if args.truth_field:
                p1 += ["--truth-field", args.truth_field, "--truth-map", args.truth_map]
            run_phase1.main(p1)
            vox = str(out / f"{name}_classified_voxels.las")
            run_phase2.main([vox, "--out", str(out)])
            p3 = [vox, "--out", str(out)]
            for k in PHASE3_EXTRA:
                v = getattr(args, k)
                if v is not None:
                    p3 += [f"--{k.replace('_', '-')}", str(v)]
            run_phase3.main(p3)
        row = summarise(out, name)
        row["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(row)
        print("SUMMARY", row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    base = Path(args.out)
    with open(base / "batch_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |" for r in rows]
    (base / "batch_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
