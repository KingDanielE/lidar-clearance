"""Class codes (ASPRS LAS 1.4 where one exists) and dataset label mappings."""

import numpy as np

UNCLASSIFIED = 1
GROUND = 2
VEGETATION = 5
BUILDING = 6
WIRE = 14
PYLON = 15

NAMES = {
    UNCLASSIFIED: "unclassified",
    GROUND: "ground",
    VEGETATION: "vegetation",
    BUILDING: "building",
    WIRE: "wire",
    PYLON: "pylon",
}

# Colours used in every plot.
COLORS = {
    UNCLASSIFIED: "#9ca3af",
    GROUND: "#a16207",
    VEGETATION: "#16a34a",
    BUILDING: "#6b7280",
    WIRE: "#dc2626",
    PYLON: "#2563eb",
}

IGNORE = -1

# GridNet-HD `ground_truth` field (original ids in the .las files) -> our classes.
# 0-4 pylon, 5 conductor, 6-7 structural cable, 8-11 insulator, 12-13 unlabeled,
# 14 high veg, 15 low veg, 16 herbaceous, 17-18 rock/soil, 19 road, 20 water, 21 building.
GRIDNET = {
    **{i: PYLON for i in range(0, 5)},
    5: WIRE,
    6: WIRE,
    7: WIRE,
    **{i: PYLON for i in range(8, 12)},  # insulators hang on the pylon
    14: VEGETATION,
    15: VEGETATION,
    16: GROUND,  # grass: irrelevant for clearance, sits on the ground surface
    17: GROUND,
    18: GROUND,
    19: GROUND,
    20: GROUND,
    21: BUILDING,
}

# Standard ASPRS codes from a provider's classification -> our classes.
ASPRS = {
    2: GROUND,
    3: VEGETATION,
    4: VEGETATION,
    5: VEGETATION,
    6: BUILDING,
    9: GROUND,
    13: WIRE,
    14: WIRE,
    15: PYLON,
    16: PYLON,
}

MAPPINGS = {"gridnet": GRIDNET, "asprs": ASPRS}


def remap(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Map raw dataset labels to our classes; anything unmapped becomes IGNORE."""
    lut = np.full(max(256, int(labels.max()) + 1), IGNORE, dtype=np.int16)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut[labels.astype(np.int64)]
