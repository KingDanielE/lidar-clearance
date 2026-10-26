import numpy as np

from clearance.encroach import Rules
from clearance.hazard import SAFE, STRIKE
from clearance.validate3 import Run, compare


def test_compare_counts_hits_misses_and_false_alarms():
    rules = Rules()  # violation < 3, urgent < 4.5, monitor < 6
    # Points 0-2 flagged in both; 3 only in truth (missed); 4 only predicted (false alarm,
    # e.g. a wire point Phase 1 called vegetation); 5 clear in both.
    truth = Run(np.array([0, 1, 2, 3, 5]), np.array([1.0, 4.0, 5.0, 2.0, 9.0]),
                [{"_points": np.array([0, 1]), "tier": "violation", "min_clearance_m": 1.0},
                 {"_points": np.array([3]), "tier": "violation", "min_clearance_m": 2.0}],
                np.array([[0.0, 0.0], [50.0, 50.0]]), np.array([STRIKE, STRIKE], dtype=object))
    pred = Run(np.array([0, 1, 2, 4, 5]), np.array([1.2, 4.0, 5.0, 0.5, 9.0]),
               [{"_points": np.array([0, 1]), "tier": "violation", "min_clearance_m": 1.2},
                {"_points": np.array([3]), "tier": "violation", "min_clearance_m": 0.5}],
               np.array([[0.5, 0.0], [90.0, 90.0]]), np.array([STRIKE, SAFE], dtype=object))

    r = compare(pred, truth, rules)
    assert r["flagged_points"] == {"predicted": 4, "truth": 4, "matched": 3, "precision": 0.75, "recall": 0.75}
    cm = r["tier_confusion_rows_truth_cols_pred"]
    assert cm["violation"]["violation"] == 1 and cm["violation"]["not_vegetation"] == 1
    assert cm["not_vegetation"]["violation"] == 1
    z = r["zones"]
    assert z["predicted_zones"]["precision"] == 0.5 and z["predicted_zones"]["recall"] == 0.5
    assert z["same_tier"] == 1 and z["max_abs_clearance_diff_m"] == 0.2
    assert r["hazard_trees"]["precision"] == 1.0 and r["hazard_trees"]["recall"] == 0.5
