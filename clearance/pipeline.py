"""Phase 1 pipeline: downsample -> DTM + HAG -> features -> classify -> back to full cloud."""

import time
from dataclasses import dataclass, field

import numpy as np

from . import features
from .classify import Params, classify
from .ground import Raster, height_above_ground, pmf_dtm
from .io import voxel_downsample


@dataclass
class Result:
    labels: np.ndarray  # per input point
    vox_points: np.ndarray
    vox_labels: np.ndarray
    vox_hag: np.ndarray
    inverse: np.ndarray  # input point -> voxel
    dtm: Raster
    timings: dict = field(default_factory=dict)


def run(points: np.ndarray, voxel: float = 0.15, params: Params | None = None, log=print) -> Result:
    t = {}

    def step(name, t0):
        t[name] = round(time.perf_counter() - t0, 1)
        log(f"  {name}: {t[name]} s")

    t0 = time.perf_counter()
    vox, inverse = voxel_downsample(points, voxel) if voxel else (points, np.arange(len(points)))
    step("downsample", t0)
    log(f"  {len(points):,} points -> {len(vox):,} voxels ({voxel} m)")

    t0 = time.perf_counter()
    dtm = pmf_dtm(vox)
    hag = height_above_ground(vox, dtm)
    step("ground", t0)

    t0 = time.perf_counter()
    f = features.compute(vox)
    step("features", t0)

    t0 = time.perf_counter()
    vlab = classify(vox, hag, f, params)
    step("classify", t0)

    return Result(vlab[inverse], vox, vlab, hag, inverse, dtm, t)
