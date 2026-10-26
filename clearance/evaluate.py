"""Per-class precision / recall / IoU against ground-truth labels."""

import numpy as np

from .classes import BUILDING, GROUND, IGNORE, NAMES, PYLON, VEGETATION, WIRE

EVAL_CLASSES = [GROUND, VEGETATION, BUILDING, PYLON, WIRE]


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    keep = truth != IGNORE
    pred, truth = pred[keep].astype(np.int64), truth[keep].astype(np.int64)
    out = {}
    for c in EVAL_CLASSES:
        p, t = pred == c, truth == c
        tp = int(np.sum(p & t))
        fp = int(np.sum(p & ~t))
        fn = int(np.sum(~p & t))
        out[NAMES[c]] = {
            "truth_points": int(t.sum()),
            "predicted_points": int(p.sum()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        }
    out["overall_accuracy"] = float(np.mean(pred == truth))
    return out


def confusion(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rows = truth, columns = prediction, in EVAL_CLASSES order."""
    keep = truth != IGNORE
    pos = {c: i for i, c in enumerate(EVAL_CLASSES)}
    lut = np.full(256, -1, dtype=np.int64)
    for c, i in pos.items():
        lut[c] = i
    t = lut[truth[keep].astype(np.int64)]
    p = lut[pred[keep].astype(np.int64)]
    ok = (t >= 0) & (p >= 0)
    n = len(EVAL_CLASSES)
    return np.bincount(t[ok] * n + p[ok], minlength=n * n).reshape(n, n)


def format_report(s: dict, cm: np.ndarray) -> str:
    def f(v):
        return "  -  " if v is None else f"{v:5.3f}"

    lines = ["| class | truth pts | predicted pts | precision | recall | IoU |", "|---|---:|---:|---:|---:|---:|"]
    for c in EVAL_CLASSES:
        r = s[NAMES[c]]
        lines.append(
            f"| {NAMES[c]} | {r['truth_points']:,} | {r['predicted_points']:,} | "
            f"{f(r['precision'])} | {f(r['recall'])} | {f(r['iou'])} |"
        )
    lines.append(f"\nOverall accuracy: {s['overall_accuracy']:.3f}\n")
    names = [NAMES[c] for c in EVAL_CLASSES]
    lines.append("Confusion matrix (rows = truth, columns = predicted):\n")
    lines.append("| truth \\ pred | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for i, n in enumerate(names):
        lines.append(f"| {n} | " + " | ".join(f"{v:,}" for v in cm[i]) + " |")
    return "\n".join(lines)
