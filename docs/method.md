# Method

How each phase works, and why each rule is there. Scores are in [validation.md](validation.md).

## Phase 1: classification (`ground.py`, `features.py`, `classify.py`)

1. `io.voxel_downsample`: 0.15 m voxels (30.6 M -> 5.5 M points); labels are copied back
   to every original point at the end.
2. `ground.pmf_dtm`: progressive morphological filter on a 0.5 m min-z raster -> DTM ->
   height above ground (HAG). Points under 0.5 m HAG are ground.
3. `features`: eigenvalue features over 16 neighbours (linearity, planarity, scattering,
   main-direction and normal z) and `gap_below`, the empty space under each point's
   vertical run in its 0.5 m column.
4. `classify`:
   - **wires**: HAG > 3 m, linear, main direction < ~37 deg from horizontal, > 1 m of air
     below; clusters must be >= 15 m long and 4x longer than wide. Then short pieces aligned
     with nearby wires (sparse spans) and region growing into linear neighbours.
   - **pylons**: clusters >= 10 m tall, scattering < 0.14 (lattice, not crown), top within
     3 m of a wire and at least as high as the wires there. Points more than 15 m from the
     tower centre are handed back (wires the voxel grid joined to the tower).
   - **buildings**: smooth (scattering < 0.05) mostly planar clusters, >= 2 m wide.
   - **vegetation**: everything else above ground.


## Phase 2: conductor model (`graph.py`, `spans.py`, `catenary.py`)

A tile can hold several lines and junctions, so towers and spans form a graph
(`graph.build_line_graph`). The first version ordered towers along one axis and cut one
route into spans; at the t2z5 junction (four towers within 60 m, a second line leaving) it
found 3 of 6 towers and modelled conductors through the junction.

1. **Towers**, from three sources, merged within 8 m (earlier source wins):
   - *pylon clusters*: connected pylon points (>= 200 voxels), pieces within 10 m merged;
   - *kinks*, per direction group: wire points are grouped by their direction in plan
     (circular histogram peaks), and for each group the conductors' upper envelope (90th
     percentile of wire z per 1 m) must kink downwards (slope 15 m before minus 15 m after
     > 0.03; a catenary alone gives a negative value; both sides >= 80% covered, so not on
     the last metres before a tile edge, where a tree at wire height once passed as a tower)
     **and** the line's centre column
     (2.5 m radius, between the lowest conductor - 3 m and the top wire - 1 m, slid +-8 m)
     must hold >= 300 points that are neither ground nor wire. At mid-span that column is
     open air (vegetation must stay below the conductors); a tower body fills it. Kinks
     alone fire at tile edges and where the envelope switches conductor: true towers had
     2,300-7,300 centre-column points, false kinks 0-59.
   - *direction changes*: wires do not bend in mid-air. Where wire points within 6 m (3D)
     run in clearly different plan directions (doubled-angle mean resultant < 0.9, about
     26 degrees), there is an angle or junction tower. A candidate where every direction
     carries on past it is a **crossing** of two lines, not a tower, and is dropped (t2z5
     had one on its main span).
   - *terminals* (second pass): a partial span >= 30 m long that stops while the survey goes
     on (ground beyond) at structure at wire height ends on a terminal tower. Not at a tile
     edge (no ground beyond), nor within 25 m of a known tower (jumper pieces), and the
     structure must be steel-shaped: median scattering <= 0.2 (lattice ~0.11, crowns ~0.28).
     In t2z5 a line leaving the survey among trees as tall as the wires passed every other
     test; Phase 1 labels forest towers vegetation anyway, so labels cannot decide it.
   - *confirmation* (after the conductors are fitted): a tower stays only if a conductor
     hangs from one of its spans. On t4z1, canopy that Phase 1 called wire (86k points)
     kinked and branched like a line and gave 4 towers 50-100 m off the real line, in trees;
     none of their spans fitted a conductor. Their wire is still listed as unmodelled.
   On the seven tiles every true tower is found, within 0.0-2.6 m, and none is false. Moving
   wire-found towers onto the densest "steel" column was tried and pulled forest towers
   8-9 m off, onto the canopy.
2. **Spans** (graph edges):
   - two towers are joined if wire points running along the line between them (within
     12 m) cover >= 60% of it, in its direction: within 20 degrees, or atan(20 m / span
     length) on short spans. Conductors hang from cross-arm ends up to ~10 m either side of
     a tower centre, so on the 15 m span at the t2z5 junction they run at ~50 degrees to the
     line between the centres; with a fixed 20 degrees that span was missed;
   - an edge is a bypass, and dropped, if a third tower lies within its 12 m corridor, or
     forms a triangle of edges with it while lying within 15% of its length of the line
     (t1z6a's 389 m bypass passed the middle tower at 12.07 m);
   - wire points go to the nearest edge, only beside it (not past its towers), aligned or
     next to a tower;
   - wire leaving a tower that no edge explains becomes a partial span (a ray) to where it
     stops; wire near no tower becomes a tower-less span.
   Spans between two towers are *full*, the rest *partial*.
3. **Conductors**: in each span, cluster (DBSCAN, eps 0.5 m) on lateral offset and height
   above a common quadratic. Parallel wires keep a constant offset, so each collapses to a
   blob however patchy it is along the span. Bundled sub-conductors stay one conductor.
   Points within 3 m of a tower are left out first: clamps, insulator strings and jumpers
   hang there and bend away from the catenary.
   A conductor is modelled only where its data reaches: it counts as reaching a tower only if
   its points come within 8 m of it, otherwise the model stops where the data stops. In t2z5
   two wires of a parallel line stopped 53 m short of the span's tower; modelled to the
   tower, they ran through the trees and created false zones.
   A conductor wider than 0.3 m in plan (lateral RMSE; real ones are 2-5 cm) is two wires
   at the same height that the clustering joined: parallel phases bridged by stray points,
   or a wire turning off to another tower. RANSAC takes out the best-supported straight
   line in (along, lateral); the rest is clustered and fitted again. Points the vertical
   trim drops (below), if they are a fifth of the cluster or more, are fitted again too.
   On t2z5 one 10,416-point cluster held a conductor and a wire diverging from it; the
   fit kept 5,447 points to 6 cm vertically while 4,969 were silently lost.
4. **Catenary** per conductor, in its own vertical plane:
   `z(u) = z0 + c (cosh((u - u0)/c) - 1)`, robust least squares (soft-L1, 10 cm scale),
   started from a parabola fit, then up to two refits without points more than
   max(0.5 m, 3 robust sigma) off. `c = H/w` (horizontal tension / weight per metre).
   Reported: c, sag below the chord (full spans), fit RMSE (vertical and lateral), coverage,
   lowest ground clearance along the wire, outliers dropped. A fit that diverges (cosh
   overflow on a nearly straight cluster: the frozen code crashed on t4z1) comes back
   non-finite and is rejected, instead of stopping the run.
5. **Validation**: a cluster is rejected (listed with the reason in the report) if it has
   too few points (3 per metre, capped at 100; the full 100 for a piece with no tower at
   either end, since nothing anchors it: on t4z1 two scraps of canopy, 64 and 43 points,
   fitted c = 94 km and 48 m and made a false violation at 0.05 m), is < 10 m long, covers < 50% of its extent,
   has c at a fit bound, a vertical RMSE > 25 cm (real conductors fit to ~5 cm), did not
   converge, or is a piece < 30 m in a tower's span that does not reach the tower (wire
   leaving a tower starts at it; in t1z6a such a 13 m piece of jumper, vegetation and
   unlabelled points was 0.69 m from a tree, 8 m from any real wire). This is what removes
   the clusters made of Phase 1's false wire points. A large cluster rejected for
   RMSE is first re-clustered at half the radius (down to 0.125 m) and the pieces refitted:
   in the forest tile, stray tower points bridged two conductors into one 6,671-point
   cluster with a 2.65 m RMSE.
6. Outputs: `<name>_phase2.json`, `<name>_conductors.csv`, `<name>_conductor_model.las`
   (every conductor sampled at 0.25 m, the input for Phase 3), `<name>_spans.png`
   (points + fitted curves per span), `<name>_route.png` (plan).

### Limits

- Partial spans: the model is only valid over the stretch with points; c is weakly
  determined. Joining neighbouring tiles fixes this (see [Joining tiles](#joining-neighbouring-tiles)).
- Bundled conductors would be modelled at the bundle centre. The lateral and vertical fit
  RMSE (2-5 cm) on these lines shows single conductors, so Phase 3 subtracts only the
  conductor radius.
- Graph: a terminal tower is only found at the end of a partial span >= 30 m long, so a
  terminal within 25-30 m of the previous tower stays a partial span (tested). Several
  parallel lines sharing towers are not told apart.
- The fit is the line *as surveyed*. Max sag (hot conductor) and blowout (wind) are Phase 3.

## Phase 3: vegetation clearance (`scenarios.py`, `encroach.py`, `hazard.py`)

### Rules: Swiss Leitungsverordnung (LeV, SR 734.31)

The lines are in Switzerland, so the defaults follow the LeV for high-voltage overhead lines
(Starkstromfreileitungen), read from the official text (Fedlex):

| What | LeV | Used here |
|---|---|---|
| Vertical distance trees - bare conductors at greatest sag | Art. 35(4)b: 1.5 m + 0.01 m/kV (fruit trees 2.5 m + 0.01 m/kV) | **5.3 m** at 380 kV (3.7 m at 220 kV), as a 3D distance in the hot and ice cases |
| Greatest sag | Art. 47: the worse of 40 C, and 0 C with 20 N/m extra load (snow/ice) per conductor or sub-conductor, no wind | two scenarios: *hot* (40 C) and *ice* (0 C + 20 N/m) |
| Wind deflection | Definitions no. 37: conductor at 10 C; Annex 4: 500 N/m2 on wires | *blowout left/right* at 10 C, 500 N/m2: a 44 degree swing |
| Direct distance to trees in wind | Art. 35(1): case by case | **3.8 m** by analogy with Art. 34(2) (terrain beside the line in wind: 0.01 m/kV, at least 1.5 m) |

The nominal voltage is not in the data: 380 kV is assumed (`--kv`; 220 kV gives 3.7 m and
2.2 m). Operators may rate a line hotter than 40 C (`--t-max`). The conductor is a typical
AAAC/Aldrey (AlMgSi) 600 mm2 class conductor (d 31.9 mm, 16.3 N/m, EA 34 MN, alpha 23e-6/C),
the kind used on Swiss lines; not the line's datasheet. Growth (1 m/yr) and the trimming
cycle (3 years) are assumptions. All are parameters (`Rules`, `LineParams`).

### Method

1. **Scenarios** per conductor, sampled every 0.1 m over its modelled extent:
   - *as surveyed*: the Phase 2 catenary;
   - *hot*, *ice*, *wind*: the conductor's horizontal tension at the survey (H = c w, from
     its fitted catenary) is carried to the scenario's temperature and load with the
     change-of-state equation of its span, which includes the elastic relief of a hotter
     or loaded wire (a hotter wire also loses tension, so it stretches less):
     `H2^2 [H2 - H1 + EA w1^2 L^2 / (24 H1^2) + EA alpha (t2 - t1)] = EA w2^2 L^2 / 24`.
     The new catenary (c = H2 / w2) passes through the same attachment points. In wind the
     loaded curve (w2 = resultant of weight and wind) swings about its chord by
     atan(wind load / weight). Parabolic, per actual span, no creep, no ruling span; with an
     infinitely stiff conductor it reduces to pure thermal elongation (tested).
   - Attachment points come from what each conductor's data reaches: both towers -> exact;
     one tower and >= 30 m of data -> the far attachment is extrapolated along the fitted
     curve at the median full-span length, flagged `estimated`; otherwise (a short piece, or
     no tower) -> checked as surveyed only (`conductors_not_assessed_for_sag_wind`).
     Extrapolating 15 m pieces over a 270 m span swung them metres sideways in the wind
     case, into trees nowhere near the real wire.
2. **Vegetation cleaning** (Phase 1 errors would otherwise become false alarms):
   - *near towers*: vegetation-labelled points within 8 m (in plan) of a tower centre. In
     t1z6a the only 3 false zones were hand-labelled tower steel 3-6 m from a tower centre
     (every real zone was >= 30 m from a tower). The cost: a tree inside a tower footprint
     is not reported; tower surroundings need their own check.
   - *floating fragments*: vegetation clusters (1 m connectivity) with < 200 points and
     nothing below 3 m above ground. Plants grow from the ground; every false zone they
     caused on the alpine tiles was hand-labelled conductor.
   - *on-conductor points*: within 0.5 m of the conductor model with >= 80% of their 1 m
     neighbourhood also on it. A crown touching the wire keeps its contact points (the
     crown behind them is in their neighbourhood); tested.
3. **Clearance**: 3D distance from each vegetation point to the nearest conductor (axis
   minus conductor radius), per scenario, against that scenario's required distance. The
   scenario with the smallest margin governs; tiers by margin: violation < 0 <= urgent <
   1.5 m <= monitor < growth x cycle (3 m).
4. **Encroachment zones**: flagged points clustered (1 m connectivity); per zone the tier,
   governing scenario, clearance, required distance and margin, the minimum in every
   scenario, nearest span/conductor, horizontal and vertical offset, location, point count,
   plan area, years to violation. Sorted worst first.
5. **Unmodelled wire**: wire-labelled points more than 1 m from every conductor model,
   grouped into line-shaped stretches >= 10 m away from towers, with the number of
   vegetation points within the watch distance of each: listed as **not assessed**
   (`<name>_unmodelled_wire.csv`), instead of silently counting that vegetation as clear.
6. **Hazard trees**: treetops = local maxima (3 m window) of a 0.5 m canopy height model,
   >= 3 m tall. Each vegetation point goes to its nearest treetop (within 0.4 h + 2 m),
   giving the crown radius (spread of the upper half) and the base (where the low points
   stand, if >= 10; else vertical), so a leaning tree pivots at its real base. Reach =
   sqrt(stem^2 + crown radius^2) from the base, in any direction (a sphere, so downhill
   falls are included): `strike` if the reach touches a conductor in any scenario,
   `within_mvcd` if it comes within the wind distance (3.8 m). Which way a leaning tree
   will fall is not modelled.
7. Outputs: `<name>_phase3.json`, `<name>_encroachments.csv` / `.geojson` (zone hulls, in
   the tile's projected CRS: assign it in GIS), `<name>_hazard_trees.csv`,
   `<name>_unmodelled_wire.csv`, `<name>_vegetation_clearance.las` (fields `clearance`,
   `tier`), plots `_clearance_profile`, `_clearance_plan`, `_clearance_section`.


### Limits

- Thresholds: the voltage, the conductor and the operating temperature are assumed, not
  read from the line's records; growth and the trimming cycle are assumptions. Art. 35 sets
  a vertical distance; it is applied here as a 3D distance (stricter beside the line).
- Sag-tension is parabolic per actual span: no ruling span (suspension insulators let
  tension equalise between spans), no creep, no insulator swing.
- One survey temperature for the whole flight (sensitivity above).
- The end-to-end reference is our own Phase 2 on hand-labelled points; at junctions it is
  itself imperfect.

## Joining neighbouring tiles

A span cut by a tile edge has an estimated far attachment, so its hot, iced and wind
positions are estimates. `io.read_many` (Phase 1 with several inputs, or `a.las+b.las` in
`run_batch`) loads neighbouring tiles onto one origin and processes them as one cloud.
t1z6a and t1z6b (102 M points together) overlap and share two towers (results in
[validation.md](validation.md#joining-neighbouring-tiles)).


## Finding a zone with trees near the line

The first two tiles had no vegetation near the line, so Phase 3 could not be checked on real
encroachments. `scripts/screen_zones.py` finds candidate zones without downloading them:
`clearance/sample.py` reads the LAS header and then ~2% of the points as 2,000 evenly spaced
chunks, by HTTP byte range, and measures how close hand-labelled vegetation comes to
hand-labelled wire.

- Chunk count matters more than sample size: points are not stored in a spatially uniform
  order, so 300 chunks put t5b6's closest vegetation at 19.7 m, 2,000-3,000 chunks at
  8.9-9.2 m (full file: 8.4 m).
