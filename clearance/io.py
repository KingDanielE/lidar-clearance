"""Reading, voxel downsampling and writing point clouds."""

from dataclasses import dataclass

import laspy
import numpy as np


@dataclass
class Cloud:
    points: np.ndarray  # (n, 3) float32, metres, relative to `origin`
    origin: np.ndarray  # (3,) float64, real-world coordinates of (0, 0, 0)
    labels: np.ndarray | None = None  # raw ground-truth labels, if the file has them
    crs: object = None


def read_las(path, label_field: str | None = None) -> Cloud:
    las = laspy.read(path)
    h = las.header
    # Work relative to the tile's minimum so float32 keeps millimetre precision.
    origin = np.floor(np.array([h.mins[0], h.mins[1], h.mins[2]]))
    pts = np.empty((len(las.points), 3), dtype=np.float32)
    for i, dim in enumerate("XYZ"):
        raw = np.asarray(las[dim], dtype=np.float64)
        pts[:, i] = raw * h.scales[i] + (h.offsets[i] - origin[i])
    labels = None
    if label_field:
        labels = np.asarray(las[label_field]).astype(np.int16)
    crs = None
    try:
        crs = h.parse_crs()
    except Exception:
        pass
    return Cloud(pts, origin, labels, crs)


def read_many(paths, label_field: str | None = None) -> Cloud:
    """Several neighbouring tiles as one cloud (same CRS assumed), on a shared origin, so a
    span cut by a tile edge becomes whole again."""
    clouds = [read_las(p, label_field) for p in paths]
    if len(clouds) == 1:
        return clouds[0]
    origin = np.min([c.origin for c in clouds], axis=0)
    pts = np.vstack([(c.points.astype(np.float64) + (c.origin - origin)).astype(np.float32) for c in clouds])
    labels = None
    if all(c.labels is not None for c in clouds):
        labels = np.concatenate([c.labels for c in clouds])
    return Cloud(pts, origin, labels, clouds[0].crs)


def voxel_downsample(points: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """One point per occupied voxel (the voxel centroid).

    Returns (centroids, inverse) where inverse maps every input point to its voxel, so
    labels computed on the centroids can be copied back to the full cloud.
    """
    keys = np.floor(points / size).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    cent = np.empty((len(counts), 3), dtype=np.float32)
    for i in range(3):
        cent[:, i] = np.bincount(inverse, weights=points[:, i], minlength=len(counts)) / counts
    return cent, inverse


def write_las(
    path, points: np.ndarray, origin: np.ndarray, classification: np.ndarray, crs=None, truth=None,
    extra: dict[str, np.ndarray] | None = None,
):
    """Write a classified cloud; `truth` (our class codes, -1 = none) goes in an extra field,
    as does every array in `extra` (name -> per-point values)."""
    extra = dict(extra or {})
    if truth is not None:
        extra["truth"] = truth.astype(np.int16)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = origin
    header.scales = np.array([0.001, 0.001, 0.001])
    if crs is not None:
        header.add_crs(crs)
    for name, values in extra.items():
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=values.dtype))
    las = laspy.LasData(header)
    las.x = points[:, 0].astype(np.float64) + origin[0]
    las.y = points[:, 1].astype(np.float64) + origin[1]
    las.z = points[:, 2].astype(np.float64) + origin[2]
    las.classification = classification.astype(np.uint8)
    for name, values in extra.items():
        las[name] = values
    las.write(path)


def read_classified(path) -> tuple[Cloud, np.ndarray]:
    """Read a cloud written by write_las: (cloud with our truth codes or None, classification)."""
    las = laspy.read(path)
    origin = np.array(las.header.offsets, dtype=np.float64)
    pts = np.column_stack([np.asarray(las.x) - origin[0], np.asarray(las.y) - origin[1], np.asarray(las.z) - origin[2]]).astype(np.float32)
    truth = np.asarray(las.truth).astype(np.int16) if "truth" in las.point_format.dimension_names else None
    return Cloud(pts, origin, truth), np.asarray(las.classification).astype(np.uint8)
