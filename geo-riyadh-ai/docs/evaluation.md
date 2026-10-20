> **Note:** this file is a dated record of two evaluation passes made while the app covered a single 12 x 15 km study area. Statements about the "study area" describe that earlier state. The app now covers the whole downloaded city, divided by district; see [districts_and_map.md](districts_and_map.md), [capabilities.md](capabilities.md) and [agentic_analysis.md](agentic_analysis.md) for current results.

# Capability evaluation (2026-10-19, real central-Riyadh data incl. the Urbi parking layer)

Method: 11 varied questions through the full executor on the real area, independent brute-force re-checks, then a planner stress test with paraphrases and out-of-scope questions. Everything below was measured in that session.

## What works well
* **Numerical correctness.** The #1 suitability site's distance to the nearest four-way junction (81.0 m), competitors within 500 m (0), parking within 300 m (30) and distance to the nearest motorway/trunk (1,914 m) were recomputed with plain Shapely distances and matched exactly. District counts, proximity filters and the shop/highway constraint also match brute force (benchmarks).
* **Traceability.** Every number in every answer was found in a tool output (claim audit 0 unsupported in all 11 jobs).
* **Honest refusals.** Parcel questions on real data fail with the reason ("layer not available"); forecasts, quality rankings and parking "shortage" questions are refused with an explanation of what can be computed instead.
* **Speed.** Retrieval questions 0.1-3 s; supermarket ranking 11 s cold, 0.4 s on re-weighting; parking optimization 12-35 s.
* **A genuine finding.** With the Urbi layer loaded, 96% of shops are within 500 m (road distance) of a public mapped parking lot, so adding three more lots changes coverage by only 0.5-2 percentage points (the median reported capacity is 14 spaces; with a >=50-space baseline, 79% coverage and the three new sites add about 2 points). In this area and under this demand proxy, parking is not a bottleneck; the model does not say anything about occupancy.

## Defects the evaluation found (all fixed, with regression tests)
1. `compare_scenarios` crashed when parking was stored as polygons (`'Polygon' has no attribute 'x'`).
2. The optimizer counted residents-only, handicapped-only and taxi-only lots (and 5-space lots) as public supply. New arguments `existing_public_only` (default true) and `existing_min_capacity`.
3. "Which neighbourhood has the best schools?" was answered with an unranked list of 270 schools. Quality questions are now refused (no ratings exist).
4. A district comparison summary described all 31 districts instead of the two requested. Now only the compared units are reported.
5. "Predict where new shops will open" returned a static list. Forecasts are now refused.
6. Several regexes silently contained backspace characters (a shell-escaping slip of mine); fixed and the tests now cover the affected routes.

## Weaknesses that remain
* **The planner is rule-based and brittle on wording.** In-grammar gold set: 26/26. Natural paraphrases I wrote and then tuned on: 17/17 after tuning (meaningless as an estimate). **Fresh held-out paraphrases written after tuning: 8/14 (57%)**, e.g. "shop count by area", "restaurants beside a mosque", "explain how the ranking works" were not understood. Failures were mostly "unsupported" (safe) rather than wrong answers, but expect to rephrase. An LLM planner behind the same typed `ExecutionPlan` would be the natural fix; none is wired in.
* **Data gaps limit what can be answered:** no ratings, prices, population, traffic, occupancy. Parcels/zoning exist as Suhail data but were still downloading and are not yet loaded.
* **Suitability is a heuristic** with default weights chosen for the demo; stability (Spearman 0.95) is not accuracy. Zoning is UNVERIFIED until parcels are loaded.
* **Parking capacity is operator-reported;** large lots are not weighted by capacity in the access model (each lot is one facility).
* Study area only (12 x 15 km), single Overture snapshot; no temporal analysis.


---

# Full-project evaluation, second pass (all Overture + Suhail + Urbi data loaded)

**Tests:** GeoAcquirer 60 passed; Geo-RiyadhAI 51 passed. **Benchmarks re-run** (`benchmarks/results/`): GeoAcquirer results unchanged from the first run (junction topology exact on the synthetic city, constraint precision/recall 1.0 on geometric tasks, 20/20 query compilation, 737x index speed-up); Geo-RiyadhAI planner 26/26 on the gold set, 21/21 supported questions executed, 0 of 445 numbers untraceable, failure injection handled (jobs end as `partial`/`failed`/`awaiting_approval` with the reason).

## Coverage (the main limitation, now visible on the map)
The loaded data is the 188 km2 study area. The Urbi notebook found 939,056 buildings city-wide; the study area holds 110,291 of them (11.7%). The map now draws the data boundary and says so in its legend. Any "northern Riyadh", "all districts" or "study area" answer refers only to this box.

## Cross-source consistency (measured inside the box)
* Buildings: Urbi has 27% more than Overture (110,291 vs 87,092). 76% of Urbi building centroids fall inside an Overture footprint, so about a quarter of Urbi buildings have no Overture counterpart.
* 92% of Urbi buildings lie inside a Suhail parcel (8% do not).
* Zoning vs building purpose: 85% of Urbi *residential* buildings sit on Suhail residential parcels, but Urbi *commercial* buildings sit on residential (35%), mixed (28%), unknown/no-rule (26%) and only 0.8% on pure commercial parcels. Commercial use is mostly inside mixed-zoned parcels, which is why the zoning filter keeps "mixed" and why only 127 grid cells survive it.
* Data integrity: 0 invalid parcel geometries, 0 duplicate parcel ids.

## Optimization re-evaluated
With 6,633 public lots the parking problem is saturated (96% coverage before and after), so the old 1,159 -> 871 m improvement no longer applies. Against capacity-filtered baselines the solver now proves optimality (gap 0%) and optimized sites beat random placement, with modest gains (about 2 percentage points of coverage; see README). The benchmark's default rows are therefore weak discriminators; `benchmarks/results/optimization_capacity_filtered.json` holds the discriminating rows.

## Remaining weaknesses (unchanged or newly confirmed)
* Rule-based planner: 8/14 on fresh paraphrases.
* Heuristic suitability (not validated against outcomes); zoning is a keyword mapping of Suhail's Arabic text.
* Two sources disagree about which buildings exist (see above); which one is "right" cannot be decided without ground truth.
* The map and every analysis cover only the study area.
