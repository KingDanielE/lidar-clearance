"""Score Phase 3 run on predicted vegetation against Phase 3 run on hand-labelled vegetation.

Both runs use the same voxels and the same conductor model, so they can be compared point
by point: only which points count as vegetation (and so their clearance tier) differs.
This isolates what Phase 1's errors do to the clearance answer.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .encroach import CLEAR, TIERS, Rules
from .hazard import SAFE

NOT_VEG = "not_vegetation"
ROWS = TIERS + [NOT_VEG]


@dataclass
class Run:
    idx: np.ndarray  # vegetation points (indices into the shared voxel cloud)
    worst: np.ndarray  # worst-scenario clearance per vegetation point
    zones: list  # encroach.zones rows; "_points" index into idx
    tree_xy: np.ndarray  # treetops (n, 2)
    tree_status: np.ndarray


def _pr(tp: int, n_pred: int, n_truth: int) -> dict:
    return {
        "predicted": n_pred,
        "truth": n_truth,
        "matched": tp,
        "precision": round(tp / n_pred, 3) if n_pred else None,
        "recall": round(tp / n_truth, 3) if n_truth else None,
    }


def compare(pred: Run, truth: Run, rules: Rules, tree_match_m: float = 2.0) -> dict:
    pt = dict(zip(pred.idx.tolist(), rules.tier(pred.worst)))
    tt = dict(zip(truth.idx.tolist(), rules.tier(truth.worst)))
    p_flag = {i for i, t in pt.items() if t != CLEAR}
    t_flag = {i for i, t in tt.items() if t != CLEAR}

    # Tier confusion over every point flagged in either run.
    cm = {r: {c: 0 for c in ROWS} for r in ROWS}
    for i in p_flag | t_flag:
        cm[tt.get(i, NOT_VEG)][pt.get(i, NOT_VEG)] += 1
    out = {
        "flagged_points": _pr(len(p_flag & t_flag), len(p_flag), len(t_flag)),
        "tier_confusion_rows_truth_cols_pred": cm,
    }

    # Zones: a predicted zone matches a truth zone if they share a point.
    pz = [set(pred.idx[z["_points"]].tolist()) for z in pred.zones]
    tz = [set(truth.idx[z["_points"]].tolist()) for z in truth.zones]
    pairs = []
    for i, a in enumerate(pz):
        for j, b in enumerate(tz):
            if a & b:
                pairs.append((i, j))
    zp = {i for i, _ in pairs}
    zt = {j for _, j in pairs}
    out["zones"] = {
        "predicted_zones": _pr(len(zp), len(pz), len(tz))
        | {"recall": round(len(zt) / len(tz), 3) if tz else None},
        "same_tier": sum(pred.zones[i]["tier"] == truth.zones[j]["tier"] for i, j in pairs),
        "pairs": len(pairs),
        "max_abs_clearance_diff_m": round(
            max(
                (
                    abs(pred.zones[i]["min_clearance_m"] - truth.zones[j]["min_clearance_m"])
                    for i, j in pairs
                ),
                default=0.0,
            ),
            2,
        ),
    }

    # Hazard trees: non-safe treetops matched within tree_match_m in plan.
    p = pred.tree_xy[pred.tree_status != SAFE]
    t = truth.tree_xy[truth.tree_status != SAFE]
    if len(p) and len(t):
        dp, _ = cKDTree(t).query(p)
        dt, _ = cKDTree(p).query(t)
        out["hazard_trees"] = {
            **_pr(int(np.sum(dp <= tree_match_m)), len(p), len(t)),
            "recall": round(float(np.mean(dt <= tree_match_m)), 3),
        }
    else:
        out["hazard_trees"] = _pr(0, len(p), len(t))
    return out
