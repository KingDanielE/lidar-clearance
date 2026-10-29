# Validation

Every number is scored against GridNet-HD's hand labels. Method: [method.md](method.md).

## Phase 1: classification
### Results on t5c2 (tuning tile)

Full resolution, every point scored against GridNet-HD labels (grass, rock, road -> ground;
insulators -> pylon; conductors and structural cables -> wire).

| class | precision | recall | IoU |
|---|---:|---:|---:|
| ground | 0.990 | 0.997 | 0.987 |
| vegetation | 0.420 | 0.738 | 0.365 |
| building | 1.000 | 0.582 | 0.582 |
| pylon | 0.991 | 0.952 | 0.944 |
| wire | 0.965 | 0.982 | 0.948 |

Overall accuracy 0.983; runtime about 1 minute on a laptop.

### Held-out check: t5b6 (thresholds unchanged)

Zone t5b6 (36.1 M points, 328 x 747 x 223 m, bigger towers, more trees) was never looked at
while tuning. Same parameters:

| class | precision | recall | IoU | (t5c2 IoU) |
|---|---:|---:|---:|---:|
| ground | 0.990 | 0.977 | 0.968 | 0.987 |
| vegetation | 0.632 | 0.781 | 0.537 | 0.365 |
| building | 0.000 | 0.000 | 0.000 | 0.582 |
| pylon | 0.939 | 0.918 | 0.867 | 0.944 |
| wire | 0.803 | 0.974 | 0.786 | 0.948 |

- Wire recall holds (97%); precision drops to 80%. The false wires are mostly pylon members
  (38k points): these towers are wider than the 15 m `pylon_radius`, so their outer parts are
  handed back as wires. 7k vegetation points also became wire.
- The tile has almost no buildings (9k points); the 62k "building" predictions are smooth
  ground and rock above the DTM.
- Phase 2 absorbs the wire false positives (see below), so the geometry is unaffected. For a
  Phase 1 fix: size `pylon_radius` from each tower's own footprint instead of a constant.

### Forest: t1z6a (thresholds unchanged)

| class | precision | recall | IoU |
|---|---:|---:|---:|
| ground | 0.949 | 0.887 | 0.847 |
| vegetation | 0.880 | 0.950 | 0.842 |
| building | 0.619 | 0.453 | 0.354 |
| pylon | - | 0.000 | 0.000 |
| wire | 0.689 | 0.992 | 0.685 |

- **No pylons found.** Each tower stands in trees as tall as it is (vegetation 95th
  percentile 21.5 m, towers 22 m), so on the 1 m connectivity grid a tower and hundreds of
  metres of canopy form one cluster (2.6-3.1 M points) that scatters like forest (0.28-0.31,
  vs 0.11 for tower steel). Tower points become vegetation (270k) or wire (97k).
  A per-point scattering filter before clustering was tried and rejected: it broke the
  alpine towers (recall 0.95 -> 0.69) and let tree fragments pass as towers here.
  Phase 2 finds these towers from the wires instead (below), so the clearance answer does
  not depend on Phase 1 getting them.
- Wires: 99% recall; the false wires are mostly that tower steel.

### Held-out forest: t1z8 (committed code, never tuned on)

| class | precision | recall | IoU |
|---|---:|---:|---:|
| ground | 0.957 | 0.876 | 0.843 |
| vegetation | 0.901 | 0.957 | 0.866 |
| building | 0.051 | 0.414 | 0.048 |
| pylon | 0.994 | 0.478 | 0.477 |
| wire | 0.825 | 0.995 | 0.822 |

Half of the tower steel is found here (the rest becomes vegetation or wire, as on t1z6a).
Buildings are poor: 458k points predicted against 57k true, mostly smooth ground and rock
above the terrain model; they play no part in the clearance results.

### Held-out forest: t4z1 (frozen commit `29cff99`, never looked at)

102.8 M points, 260 x 934 x 375 m, a line climbing through forest:

| class | precision | recall | IoU |
|---|---:|---:|---:|
| ground | 0.808 | 0.966 | 0.786 |
| vegetation | 0.992 | 0.963 | 0.956 |
| building | 0.698 | 0.125 | 0.119 |
| pylon | 0.948 | 0.379 | 0.372 |
| wire | 0.713 | 0.975 | 0.700 |

Wire recall holds (97.5%). The false wire is tower steel (80k points) and canopy (86k),
much of it in one patch of trees 50-100 m off the line; see Phases 2 and 3 for what it did
there.

Caveats:
- t5c2 numbers are tuned-on-the-data; t5b6 is the honest estimate.
- Vegetation here is tiny (0.7% of points, mostly shrubs under 1 m), so its precision is
  dominated by rough ground and building edges labelled vegetation. Low vegetation can't
  reach the wires; Phase 3 should only consider vegetation above a few metres.
- A tree that has grown into the wires could be absorbed into a pylon cluster only if it is
  also lattice-smooth and within 15 m of a tower; worth a dedicated check in Phase 3.

## Phase 2: conductor model

Alpine tiles: 2 towers, one full span and partial spans to the tile edges, 7 conductors per
span: a double circuit (2 x 3 phases) plus an earth wire. Forest tiles: 2-3 towers (found
from the wires), 5 conductors per span. Junction tile: 6 towers, 14 spans (5 full, from
15 m to 359 m).

| | t5c2 | t5b6 | t1z6a (forest) | t1z8 (forest) | t2z5 (junction) | t4z1 (forest) |
|---|---|---|---|---|---|---|
| towers found (true) | 2 (2) | 2 (2) | 3 (3) | 2 (2) | 6 (6) | 4 (4) |
| spans (full) | 3 (1) | 5 (1) | 5 (2) | 3 (1) | 14 (5) | 8 (3) |
| conductors (hand-label model) | 21 (21) | 22 (22) | 20 (21) | 15 (15) | 48 (46) | 34 (35) |
| conductors matching the hand-label model | 21 | 22 | 20 | 15 | 48 | 34 |
| vertical RMSE, all conductors | 2.8-5.0 cm | 3.0-6.3 cm | 2.7-4.0 cm | 2.3-3.0 cm | 2.3-4.4 cm | 3.0-6.6 cm |
| height difference to the hand-label model, median / max | 0.0 / 1.7 cm | 0.5 / 2.6 cm | 0.0 / 0.2 cm | 0.0 / 0.1 cm | 0.1 / 52 cm | 0.0 / 18 cm |

- Without the wire-found towers, the forest tile gave 0 towers, one 600 m "span" and no
  usable conductor; without the re-clustering, 18 of its 20 conductors.
- With the graph, t5b6's steep partial span is modelled as far as the hand-label model
  (before: 30 m short, see Phase 3).
- t2z5: the 52 cm difference is at the junction, where the hand-label model itself fails
  to model some conductors; our 2 extra conductors there are made entirely of
  hand-labelled wire. Span 1 used to hold a conductor and a wire diverging from it as one
  (lateral RMSE 3.8 m, 4,969 points silently dropped); split, it is 8 conductors, each
  2-3 cm wide in plan (the sideways split in [method.md](method.md)).
- t4z1 on the frozen commit `29cff99`: **Phase 2 crashed** (a diverging fit raised inside
  SciPy), so the held-out run gave no clearance result. With the crash fixed, canopy that
  Phase 1 called wire gave 4 false towers and a false conductor; tower confirmation and
  the stricter minimum for tower-less pieces removed them. The numbers here are after
  those fixes, so t4z1 is no longer held out.

**Classification errors barely move the geometry**: even on t5b6, where 20% of predicted
wire points are wrong, every conductor fitted from predicted points is within 3.8 cm of the
one fitted from hand-labelled points. On partial spans c itself can differ by a few percent
(a 40-130 m piece of a span constrains it weakly) while the curve over the covered stretch
still agrees to centimetres.

Note: the tower margin, residual trim and validation rules were added after the first
t5b6 run showed tower hardware and false wires spoiling fits, and the wire-found towers and
re-clustering after the first t1z6a run, so Phase 2 numbers on those tiles are no longer
strictly held out. Phase 1's t5b6 and t1z6a numbers are (Phase 1 was not changed).

Synthetic check (tests): c = 900 m and sag 2.35 m are recovered within 3% and 5 cm.

## Phase 3: vegetation clearance

Scoring (`clearance/validate3.py`), against the hand labels on the same voxels. Flagged
points are compared by index; a predicted zone matches a reference zone if they share a
point; hazard trees match within 2 m in plan.
- **End to end**: reference = conductor model fitted to the hand-labelled wire +
  hand-labelled vegetation. Covers classification and conductor modelling, not the sag
  and wind assumptions (both runs use the same ones).
- **Same model** (in the reports): our conductor model with hand-labelled vegetation.
  Isolates Phase 1's vegetation errors, but is blind to a wrong conductor model: on t2z5
  before the graph it showed 90% / 100% zones while end to end was 77% / 82%.

With the Swiss rules (380 kV):

| | t5c2 | t5b6 | t1z6a (forest) | t1z8 (forest) | t2z5 (junction) | t4z1 (forest) |
|---|---|---|---|---|---|---|
| ours: closest vegetation surveyed / hot / ice / wind | 9.24 / 8.56 / 8.94 / 9.82 m | 8.39 / 7.66 / 8.00 / 10.30 m | 1.17 / 0.82 / 0.98 / 0.45 m | 2.80 / 2.58 / 2.68 / 0.22 m | 1.21 / 1.18 / 1.23 / 0.78 m | 3.65 / 3.40 / 3.47 / 0.00 m |
| reference (all hand labels) | 12.05 / 11.67 / 11.87 / 12.64 m | the same | 1.17 / 0.82 / 0.98 / 0.44 m | 2.80 / 2.58 / 2.68 / 0.23 m | 1.21 / 1.99 / 2.01 / 2.00 m | 3.65 / 3.40 / 3.47 / 0.01 m |
| zones violation / urgent / monitor, ours (reference) | 0 (0) | 0 / 0 / 1 (0 / 0 / 1) | 14 / 9 / 20 (14 / 10 / 20) | 16 / 14 / 33 (16 / 14 / 33) | 29 / 18 / 28 (31 / 17 / 28) | 14 / 11 / 37 (14 / 11 / 39) |
| zones precision / recall | | 1.0 / 1.0 | 1.0 / 0.98 | **1.0 / 1.0** | 0.99 / 1.0 | 1.0 / 0.97 |
| same tier (matched pairs) | | 1 / 1 | 43 / 43 | **63 / 63** | 74 / 77 | 62 / 62 |
| flagged points precision / recall | | 1.0 / 1.0 | 0.97 / 0.97 | 1.0 / 1.0 | 0.97 / 0.97 | 1.0 / 1.0 |
| hazard trees precision / recall (reference count) | | | 0.88 / 0.97 (73) | 0.97 / 0.97 (74) | 0.89 / 0.96 (123) | 0.91 / 0.92 (271) |
| governing scenario of the zones | | hot | hot 38, wind 4, surveyed 1 | hot 44, wind 19 | hot 60, surveyed 8, wind 6, ice 1 | hot 36, wind 25, surveyed 1 |
| wire not modelled (vegetation beside it, not assessed) | none | none | 12 m (34,682 points) | none | 235 m (none) | 123 m (387,278 points) |
| conductors checked as surveyed only | 0 of 21 | 1 of 22 | 0 of 20 | 0 of 15 | 5 of 48 | 6 of 34 |

- **Held out (t1z8)**, on the committed code before these changes and with that version's
  3 m placeholder thresholds: 34 of 34 zones, all in the same tier, clearance within 2 cm;
  flagged points 99.9% / 100%; hazard trees 98% / 96% ([Results at a glance](../README.md#results-at-a-glance)).
- The hot case governs most zones; the wind case matters where trees stand beside the
  line (t1z8: a 44 degree swing brings a conductor to 0.22 m of a crown). The ice case is
  milder than the hot case on these spans: cooling from 15 C to 0 C shortens the wire,
  partly cancelling the 20 N/m (on a 130 m span: 2.35 m sag surveyed, 2.51 m iced, 2.93 m
  at 40 C; checked by hand calculation in the tests).
- **Survey temperature** (unknown, assumed 15 C) on t1z8: at 5 / 15 / 25 C the closest hot
  clearance is 2.47 / 2.58 / 2.66 m, the closest wind clearance 0.14 / 0.22 / 0.37 m, and
  violations 17 / 16 / 14. A colder assumption is the safe side.
- t1z6a: our closest point as surveyed was 0.69 m against the reference's 1.19 m; it was a
  13 m piece of jumper, vegetation and unlabelled points just past a tower, now rejected
  (Phase 2 validation). Both are now 1.17 m.
- t2z5: the reference fails to model some junction conductors that we model from
  hand-labelled wire; its larger hot and wind clearances (1.99 m and 2.00 m against our
  1.18 m and 0.78 m) come from conductors it does not have. Unmodelled wire went from
  896 m to 235 m with the sideways split; what is left (a straight cable through a
  tower-less span, rejected with c at the fit bound, and a 20 m piece) has no vegetation
  within the watch distance.
- t4z1: zones, tiers and clearances match the reference to 5 cm, after the Phase 2 fixes.
  Its 97 m of unmodelled wire is the patch of canopy Phase 1 called wire (31k points; the
  hand labels have no wire there). It is reported as *wire not modelled, vegetation
  beside it not assessed*, a cautious false alarm; before the fixes it was a false
  violation at 0.05 m. The 2 missed zones are monitor-tier.
- Hazard-tree precision fell on the forest tiles with crown width and lean (t1z6a 94% to
  88%); recall stays at 96-97%. On t1z6a the 10 trees we flag that the reference does not
  all come from Phase 1's vegetation (the same-model score is identical): 2 are the same
  trees with treetops 2.5-3.3 m apart (beyond the 2 m match), 6 sit on the 3.8 m line
  (margin 3.5-3.8 m against the reference's 3.8-4.3 m, from a slightly different base and
  so lean), and 2 are crowns of points the hand labels leave unlabelled.
- Alpine tiles: the line is well clear. Before the vegetation cleaning, t5c2 showed 3 zones
  and t5b6 8 zones plus 10 "strike" trees, all hand-labelled conductor or tower points that
  Phase 1 called vegetation.

## Joining neighbouring tiles
| | t1z6a alone | t1z6b alone | joined |
|---|---|---|---|
| towers | 3 | 3 | 5 |
| spans (full) | 5 (2) | 5 (2) | 11 (4) |
| partial-span conductors | 10 | 10 | 14 (20 separately) |
| zones relying on an estimated attachment | 7 of 43 | 25 of 79 | 19 of 111 (32 separately) |
| end to end: zones precision / recall | 1.0 / 0.98 | 1.0 / 1.0 | 0.98 / 1.0 |
| end to end: same tier (matched pairs) | 43 / 43 | 80 / 82 | 110 / 112 |
| conductors checked as surveyed only | 0 | 0 | 4 |

Joining cuts the zones that rely on an estimated attachment by about 40%, with the same
agreement with the hand labels. The joined run leaves 4 conductor pieces checked as
surveyed only (none separately): pieces 22-30 m long with no tower in either tile, so
there is nothing to hang a hot or wind curve from. The nearest vegetation is 7.8 m and
12.4 m from them as surveyed. The test splits the synthetic
corridor through a span and checks it is partial on its own and full when joined.

## Zone screening
- Only the 27 train/validation zones carry labels (test-zone labels are withheld). 16 were
  screened (about 40 MB and 2.5-6 minutes each). Vegetation within 5 m of a wire:

| zone | closest (sample) | | zone | closest (sample) |
|---|---:|---|---|---:|
| t1z6a | 1.7 m | | t4z1 | 3.7 m |
| t1z6b | 2.5 m | | t1z5b | 4.2 m |
| t1z8 | 3.0 m | | t6z2, t6z3 | 5.3 m |
| t2z5 | 3.3 m | | t5a4, t5b1, t3*, t6z4, t6z6 | 8-52 m |
