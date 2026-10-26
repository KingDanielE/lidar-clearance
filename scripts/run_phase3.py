"""Phase 3: vegetation clearance (as surveyed, max sag, blowout), encroachment zones and
hazard trees.

    python -m scripts.run_phase3 out/t5c2/t5c2_classified_voxels.las --out out/t5c2

Thresholds and line assumptions are placeholders (see clearance.encroach.Rules and
clearance.scenarios.LineParams); override the main ones on the command line.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from clearance import encroach, hazard, linemodel, scenarios, validate3  # noqa: E402
from clearance.classes import GROUND, PYLON, VEGETATION, WIRE  # noqa: E402
from clearance.ground import height_above_ground, pmf_dtm  # noqa: E402
from clearance.io import read_classified, write_las  # noqa: E402

TIER_COLORS = {
    encroach.VIOLATION: "#dc2626",
    encroach.URGENT: "#f97316",
    encroach.MONITOR: "#eab308",
    encroach.CLEAR: "#bbf7d0",
}
HAZARD_COLORS = {hazard.STRIKE: "#dc2626", hazard.FLASHOVER: "#f97316"}


def analyse(points, labels, veg_mask, rules, line, dtm, model=None):
    """Everything Phase 3 computes, for one vegetation mask.

    Returns (model, curves, assumed span, vegetation points, their HAG, clearance,
    {reason: points dropped}, indices of the vegetation points into `points`).
    """
    model = model or linemodel.build(points, labels, labels == WIRE)
    curves, assumed = scenarios.curves(model.conductors, model.spans, line)
    idx = np.nonzero(veg_mask)[0]
    near_tower = np.zeros(len(idx), dtype=bool)
    for t in model.towers:
        near_tower |= np.linalg.norm(points[idx, :2] - t.xy, axis=1) < rules.tower_exclusion_m
    idx = idx[~near_tower]
    veg = points[idx]
    hag = height_above_ground(veg, dtm)
    fl = encroach.floating(veg, hag)
    veg, hag, idx = veg[~fl], hag[~fl], idx[~fl]
    clr = encroach.compute(veg, curves, rules)
    frag = encroach.wire_fragments(veg, clr)
    if frag.any():
        veg, hag, idx = veg[~frag], hag[~frag], idx[~frag]
        clr = encroach.compute(veg, curves, rules)
    dropped = {"near_towers": int(near_tower.sum()), "floating_fragments": int(fl.sum()), "on_conductor": int(frag.sum())}
    return model, curves, assumed, veg, hag, clr, dropped, idx


def tier_counts(clr, rules):
    worst, _ = clr.worst()
    t = rules.tier(worst)
    return {k: int(np.sum(t == k)) for k in encroach.TIERS if k != encroach.CLEAR}


def plot_profile(points, labels, veg, worst, curves, rules, prefix, title):
    wire = np.vstack([c.xyz for c in curves if c.scenario == scenarios.AS_SURVEYED])
    c0 = wire[:, :2].mean(axis=0)
    _, _, vt = np.linalg.svd(wire[:, :2] - c0, full_matrices=False)
    along, across = vt[0], vt[1]

    def proj(p):
        return (p[:, :2] - c0) @ along, (p[:, :2] - c0) @ across

    fig, ax = plt.subplots(figsize=(14, 5))
    g = points[labels == GROUND]
    g = g[np.random.default_rng(0).choice(len(g), min(len(g), 150_000), replace=False)]
    s, o = proj(g)
    m = np.abs(o) < 30
    ax.scatter(s[m], g[m, 2], s=0.3, c="#d6c3a5", linewidths=0, rasterized=True)
    s, o = proj(veg)
    tiers = rules.tier(worst)
    m = np.abs(o) < 30
    for t in reversed(encroach.TIERS):
        k = m & (tiers == t)
        ax.scatter(s[k], veg[k, 2], s=0.8 if t == encroach.CLEAR else 3, c=TIER_COLORS[t], linewidths=0, rasterized=True)
    for c in curves:
        if c.scenario in (scenarios.AS_SURVEYED, scenarios.MAX_SAG):
            s, _ = proj(c.xyz)
            ls = "-" if c.scenario == scenarios.AS_SURVEYED else "--"
            ax.plot(s, c.xyz[:, 2], ls=ls, lw=0.6, color="black" if ls == "-" else "#dc2626")
    handles = [Line2D([], [], color="black", lw=1, label="conductor, as surveyed"),
               Line2D([], [], color="#dc2626", lw=1, ls="--", label="conductor, max sag")]
    handles += [Line2D([], [], marker="o", ls="", color=TIER_COLORS[t], label=f"vegetation: {t}") for t in encroach.TIERS]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_xlabel("distance along line (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{title}: vegetation clearance, +-30 m corridor (worst of all scenarios)")
    fig.tight_layout()
    fig.savefig(f"{prefix}_clearance_profile.png", dpi=140)
    plt.close(fig)
    return f"{prefix}_clearance_profile.png"


def plot_plan(points, labels, veg, worst, curves, zones, trees, towers, rules, prefix, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    v = veg[np.random.default_rng(0).choice(len(veg), min(len(veg), 200_000), replace=False)] if len(veg) else veg
    ax.scatter(v[:, 0], v[:, 1], s=0.3, c="#bbf7d0", linewidths=0, rasterized=True)
    tiers = rules.tier(worst)
    for t in (encroach.MONITOR, encroach.URGENT, encroach.VIOLATION):
        k = tiers == t
        ax.scatter(veg[k, 0], veg[k, 1], s=2, c=TIER_COLORS[t], linewidths=0, rasterized=True)
    for c in curves:
        if c.scenario == scenarios.AS_SURVEYED:
            ax.plot(c.xyz[:, 0], c.xyz[:, 1], color="black", lw=0.5)
        elif c.scenario in (scenarios.BLOWOUT_LEFT, scenarios.BLOWOUT_RIGHT):
            ax.plot(c.xyz[:, 0], c.xyz[:, 1], color="#2563eb", lw=0.4, alpha=0.6)
    for t in towers:
        ax.plot(*t.xy, marker="s", color="#1e3a8a", ms=7)
    for z in zones[:30]:
        ax.annotate(str(z["zone"]), (z["_x"], z["_y"]), fontsize=8, weight="bold", color=TIER_COLORS[z["tier"]])
    for status, color in HAZARD_COLORS.items():
        k = trees["status"] == status
        ax.scatter(trees["xy"][k, 0], trees["xy"][k, 1], marker="^", s=40, facecolors="none", edgecolors=color, label=f"hazard tree: {status}")
    handles = [Line2D([], [], color="black", lw=1, label="conductor, as surveyed"),
               Line2D([], [], color="#2563eb", lw=1, label="conductor, blowout")]
    handles += [Line2D([], [], marker="o", ls="", color=TIER_COLORS[t], label=t) for t in encroach.TIERS[:3]]
    handles += [Line2D([], [], marker="^", ls="", mfc="none", color=c, label=f"hazard tree: {s}") for s, c in HAZARD_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.set_title(f"{title}: encroachment zones (numbered) and hazard trees")
    fig.tight_layout()
    fig.savefig(f"{prefix}_clearance_plan.png", dpi=140)
    plt.close(fig)
    return f"{prefix}_clearance_plan.png"


def plot_section(veg, curves, at_xy, axis, prefix, title, rules):
    """Cross-section across the line at one station: every scenario's conductor positions."""
    n = np.array([-axis[1], axis[0]])

    def proj(p):
        d = p[:, :2] - at_xy
        return d @ axis, d @ n

    fig, ax = plt.subplots(figsize=(9, 6))
    a, o = proj(veg)
    m = (np.abs(a) < 2.0) & (np.abs(o) < 40)
    ax.scatter(o[m], veg[m, 2], s=3, c="#16a34a", linewidths=0, label="vegetation (+-2 m)")
    style = {
        scenarios.AS_SURVEYED: ("black", "o", "as surveyed"),
        scenarios.MAX_SAG: ("#dc2626", "v", "max sag"),
        scenarios.BLOWOUT_LEFT: ("#2563eb", "<", "blowout"),
        scenarios.BLOWOUT_RIGHT: ("#2563eb", ">", None),
    }
    for c in curves:
        a, o = proj(c.xyz)
        k = np.argmin(np.abs(a))
        if abs(a[k]) > 0.5:
            continue
        col, mk, lab = style[c.scenario]
        ax.plot(o[k], c.xyz[k, 2], marker=mk, color=col, ms=7, ls="")
        circ = plt.Circle((o[k], c.xyz[k, 2]), rules.mvcd_m, color=col, fill=False, lw=0.5, ls=":")
        ax.add_patch(circ)
    handles = [Line2D([], [], marker=s[1], ls="", color=s[0], label=s[2]) for s in style.values() if s[2]]
    handles.append(Line2D([], [], marker="o", ls="", color="#16a34a", label="vegetation (+-2 m)"))
    handles.append(Line2D([], [], ls=":", color="gray", label=f"MVCD {rules.mvcd_m} m"))
    ax.legend(handles=handles, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("offset across the line (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{prefix}_clearance_section.png", dpi=140)
    plt.close(fig)
    return f"{prefix}_clearance_section.png"


def write_geojson(path, zones, veg, origin):
    feats = []
    for z in zones:
        pts = veg[z["_points"], :2] + origin[:2]
        try:
            hull = pts[ConvexHull(pts).vertices] if len(pts) >= 3 else pts
        except Exception:
            hull = pts
        ring = [[round(float(x), 2), round(float(y), 2)] for x, y in hull]
        geom = {"type": "Polygon", "coordinates": [ring + ring[:1]]} if len(ring) >= 3 else {"type": "Point", "coordinates": ring[0]}
        props = {k: v for k, v in z.items() if not k.startswith("_")}
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    # Coordinates stay in the tile's projected CRS (not WGS84): assign it when loading in GIS.
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="classified voxels LAS from Phase 1")
    ap.add_argument("--out", default="out")
    ap.add_argument("--mvcd", type=float, default=encroach.Rules.mvcd_m)
    ap.add_argument("--t-survey", type=float, default=scenarios.LineParams.t_survey_c)
    ap.add_argument("--t-max", type=float, default=scenarios.LineParams.t_max_c)
    ap.add_argument("--wind-pa", type=float, default=scenarios.LineParams.wind_pressure_pa)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem.replace("_classified_voxels", "")
    prefix = str(out / name)

    rules = encroach.Rules(mvcd_m=args.mvcd)
    line = scenarios.LineParams(t_survey_c=args.t_survey, t_max_c=args.t_max, wind_pressure_pa=args.wind_pa)

    cloud, labels = read_classified(args.input)
    pts, origin = cloud.points, cloud.origin
    veg_mask = labels == VEGETATION
    dtm = pmf_dtm(pts[labels == GROUND])
    model = linemodel.build(pts, labels, labels == WIRE)
    if not model.conductors:
        # Nothing to measure clearance to; say so instead of reporting a clear line.
        report = {"tile": name, "conductors": 0, "error": "no conductor model (Phase 2 fitted no conductors)"}
        (out / f"{name}_phase3.json").write_text(json.dumps(report, indent=2))
        print(f"{name}: no conductors, Phase 3 skipped")
        return
    model, curves, assumed, veg, hag, clr, dropped, veg_idx = analyse(pts, labels, veg_mask, rules, line, dtm, model=model)
    worst, worst_s = clr.worst()
    zone_rows, _ = encroach.zones(veg, clr, rules, origin)
    for z in zone_rows:
        z["_x"], z["_y"] = z["x"] - origin[0], z["y"] - origin[1]

    # Hazard trees: treetops of the (cleaned) vegetation.
    xy, h = hazard.tree_tops(veg, hag)
    all_wire = np.vstack([c.xyz for c in curves])
    base_z, dist, margin, status = hazard.fall_in(xy, h, dtm, all_wire, rules.mvcd_m)
    trees = {"xy": xy, "status": status}
    tree_rows = [
        {
            "tree": i + 1, "status": status[i],
            "x": round(float(xy[i, 0] + origin[0]), 2), "y": round(float(xy[i, 1] + origin[1]), 2),
            "base_z": round(float(base_z[i] + origin[2]), 2), "height_m": round(float(h[i]), 1),
            "distance_to_conductor_m": round(float(dist[i]), 2), "margin_m": round(float(margin[i]), 2),
        }
        for i in np.argsort(margin) if status[i] != hazard.SAFE
    ]
    closest_trees = [
        {"height_m": round(float(h[i]), 1), "distance_to_conductor_m": round(float(dist[i]), 2), "margin_m": round(float(margin[i]), 2),
         "x": round(float(xy[i, 0] + origin[0]), 2), "y": round(float(xy[i, 1] + origin[1]), 2)}
        for i in np.argsort(margin)[:5]
    ]

    per_span = {}
    for s in model.spans:
        own = {s2: np.array([clr.curves[i].conductor.span_id == s.id for i in clr.owner[s2]]) for s2 in clr.dist}
        row = {}
        for s2 in clr.dist:
            k = own[s2][clr.nearest[s2]]
            row[s2] = round(float(clr.dist[s2][k].min()), 2) if k.any() else None
        per_span[s.id] = {"full": s.full, "length_m": round(s.length, 1), "min_clearance_m": row}

    report = {
        "tile": name,
        "rules": {**rules.__dict__, "monitor_m": rules.monitor_m},
        "line": {**line.__dict__, "swing_deg": round(math.degrees(line.swing_rad), 1), "assumed_span_m_used": round(assumed, 1)},
        "conductors": len(model.conductors),
        # Short pieces with no tower to pivot from: checked as surveyed only.
        "conductors_not_assessed_for_sag_wind": sum(
            not any(c.conductor is cond and c.scenario == scenarios.MAX_SAG for c in curves) for cond in model.conductors
        ),
        "vegetation_points": int(len(veg)),
        "vegetation_points_dropped": dropped,
        "points_by_tier": tier_counts(clr, rules),
        "min_clearance_by_scenario_m": {s: round(float(d.min()), 2) for s, d in clr.dist.items()} if len(veg) else {},
        "min_clearance_by_span": per_span,
        "zones": [{k: v for k, v in z.items() if not k.startswith("_")} for z in zone_rows],
        "hazard_trees": {"treetops_found": int(len(h)), "strike": int(np.sum(status == hazard.STRIKE)), "within_mvcd": int(np.sum(status == hazard.FLASHOVER)), "trees": tree_rows, "closest_to_reaching": closest_trees},
    }

    if cloud.labels is not None:
        # Same conductor model, hand-labelled vegetation: how much do Phase 1 errors matter?
        tv = cloud.labels == VEGETATION
        _, _, _, tveg, thag, tclr, _, tidx = analyse(pts, labels, tv, rules, line, dtm, model=model)
        tz, _ = encroach.zones(tveg, tclr, rules, origin)
        txy, th = hazard.tree_tops(tveg, thag)
        _, _, _, tstatus = hazard.fall_in(txy, th, dtm, all_wire, rules.mvcd_m)
        report["with_truth_vegetation"] = {
            "vegetation_points": int(tv.sum()),
            "points_by_tier": tier_counts(tclr, rules),
            "min_clearance_by_scenario_m": {s: round(float(d.min()), 2) for s, d in tclr.dist.items()} if tv.any() else {},
            "zones_by_tier": {t: sum(z["tier"] == t for z in tz) for t in encroach.TIERS[:3]},
            "hazard_strike": int(np.sum(tstatus == hazard.STRIKE)),
            "hazard_within_mvcd": int(np.sum(tstatus == hazard.FLASHOVER)),
        }
        pred_run = validate3.Run(veg_idx, worst, zone_rows, xy, status)
        report["phase3_vs_truth"] = validate3.compare(
            pred_run, validate3.Run(tidx, tclr.worst()[0], tz, txy, tstatus), rules,
        )
        # End to end: everything from the hand labels, including the conductor model (fitted
        # to hand-labelled wire). The run above shares our conductor model, so it cannot see
        # a wrong conductor; this one can.
        ref_labels = labels.copy()
        ref_labels[cloud.labels == PYLON] = PYLON
        ref_model = linemodel.build(pts, ref_labels, cloud.labels == WIRE)
        if ref_model.conductors:
            _, rcurves, _, rveg, rhag, rclr, _, ridx = analyse(pts, ref_labels, tv, rules, line, dtm, model=ref_model)
            rz, _ = encroach.zones(rveg, rclr, rules, origin)
            rxy, rh = hazard.tree_tops(rveg, rhag)
            _, _, _, rstatus = hazard.fall_in(rxy, rh, dtm, np.vstack([c.xyz for c in rcurves]), rules.mvcd_m)
            report["reference"] = {
                "conductors": len(ref_model.conductors),
                "towers": len(ref_model.towers),
                "min_clearance_by_scenario_m": {s: round(float(d.min()), 2) for s, d in rclr.dist.items()},
                "zones_by_tier": {t: sum(z["tier"] == t for z in rz) for t in encroach.TIERS[:3]},
                "hazard_strike": int(np.sum(rstatus == hazard.STRIKE)),
                "hazard_within_mvcd": int(np.sum(rstatus == hazard.FLASHOVER)),
            }
            report["end_to_end_vs_truth"] = validate3.compare(
                pred_run, validate3.Run(ridx, rclr.worst()[0], rz, rxy, rstatus), rules,
            )

    print(f"{name}: {len(model.conductors)} conductors, {len(veg):,} vegetation points (dropped: {dropped}), swing {math.degrees(line.swing_rad):.1f} deg")
    print("  min clearance by scenario:", report["min_clearance_by_scenario_m"])
    print("  points by tier:", report["points_by_tier"])
    print(f"  zones: {len(zone_rows)}", {t: sum(z['tier'] == t for z in zone_rows) for t in encroach.TIERS[:3]})
    for z in zone_rows[:10]:
        print("   ", {k: z[k] for k in ("zone", "tier", "min_clearance_m", "worst_scenario", "span", "horizontal_m", "vertical_m", "points", "years_to_violation", "estimated")})
    print(f"  hazard trees: {len(h)} treetops, {report['hazard_trees']['strike']} strike, {report['hazard_trees']['within_mvcd']} within MVCD")
    print("  trees closest to reaching a conductor:", [(t["height_m"], t["margin_m"]) for t in closest_trees])
    print("  min clearance by span:", {k: v["min_clearance_m"] for k, v in per_span.items()})
    if "with_truth_vegetation" in report:
        print("  with truth vegetation:", report["with_truth_vegetation"])
        v = report["phase3_vs_truth"]
        print("  vs truth (same model): flagged points", v["flagged_points"], "| zones", v["zones"], "| hazard trees", v["hazard_trees"])
        if "end_to_end_vs_truth" in report:
            print("  reference (all hand labels):", report["reference"])
            v = report["end_to_end_vs_truth"]
            print("  END TO END: flagged points", v["flagged_points"], "| zones", v["zones"], "| hazard trees", v["hazard_trees"])

    (out / f"{name}_phase3.json").write_text(json.dumps(report, indent=2, default=str))
    cols = [k for k in (zone_rows[0] if zone_rows else {}) if not k.startswith("_")]
    with open(out / f"{name}_encroachments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zone"] + [c for c in cols if c != "zone"])
        w.writeheader()
        w.writerows({k: v for k, v in z.items() if not k.startswith("_")} for z in zone_rows)
    write_geojson(out / f"{name}_encroachments.geojson", zone_rows, veg, origin)
    with open(out / f"{name}_hazard_trees.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tree_rows[0]) if tree_rows else ["tree"])
        w.writeheader()
        w.writerows(tree_rows)
    tier_code = {t: i for i, t in enumerate(encroach.TIERS)}
    write_las(
        out / f"{name}_vegetation_clearance.las", veg, origin, np.full(len(veg), VEGETATION),
        extra={"clearance": worst.astype(np.float32), "tier": np.array([tier_code[t] for t in rules.tier(worst)], dtype=np.uint8)},
    )

    files = [plot_profile(pts, labels, veg, worst, curves, rules, prefix, name),
             plot_plan(pts, labels, veg, worst, curves, zone_rows, trees, model.towers, rules, prefix, name)]
    if zone_rows:
        z = zone_rows[0]
        k = z["_points"][np.argmin(worst[z["_points"]])]
        cond = next(c.conductor for c in curves if c.conductor.span_id == z["span"] and c.conductor.id == z["conductor"])
        files.append(plot_section(veg, curves, veg[k, :2], cond.axis, prefix, f"{name}: cross-section at zone {z['zone']} ({z['tier']}, {z['min_clearance_m']} m, {z['worst_scenario']})", rules))
    elif model.conductors:
        span = next((s for s in model.spans if s.full), model.spans[0])
        cond = next(c for c in model.conductors if c.span_id == span.id)
        mid = cond.origin + 0.5 * (cond.u_min + cond.u_max) * cond.axis
        files.append(plot_section(veg, curves, mid, cond.axis, prefix, f"{name}: cross-section at mid-span {span.id}", rules))
    print("wrote", *files, sep="\n  ")


if __name__ == "__main__":
    main()
