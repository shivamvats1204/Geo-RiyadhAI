# What Geo-RiyadhAI can do

Everything below is exercised by `benchmarks/full_evaluation.py` on four areas of the full-city data (airport, central Riyadh, north Riyadh, east fringe). Results are in `benchmarks/results/full_evaluation.json`.

> **Update:** the app no longer has a fixed study area. You pick any district or neighbourhood of Riyadh (or the map view), and answers come back in plain language. See [districts_and_map.md](districts_and_map.md). The evaluation below was run on four prepared areas before that change.

## Evaluation summary (2026-10-20)

| Check | Result |
|---|---|
| Tool catalogue (37-40 tasks per area) | 152 OK, 5 refused with a stated reason, 0 crashes |
| Edge cases (62 per area, 248 total) | 248 / 248 handled (clean answer or clean refusal, never a crash) |
| HTTP API (28 requests) | 28 / 28 |
| Data integrity checks (14 per area) | 49 / 52; the 3 failures are real gaps in the sparse east fringe |
| Unit tests | GeoAcquirer 62 passed, Geo-RiyadhAI 57 passed |

The 5 refusals are all in the east fringe, which has no POIs, parcels or districts. Site suitability, facility optimization and scenario simulation correctly say there is nothing to rank, place or measure instead of inventing output.

## Work you can do

**Find and filter**
- Filter any layer (POIs, buildings, Urbi buildings, parking, junctions, districts, metro, attractions, parcels) by category, region (north/south/east/west/central or a drawn polygon) and proximity rules.
- Follow-up questions refine a previous result set.
- Natural-language constraint queries ("shops at a crossroads but not near a highway", "clinics within 2 km by road of a hospital", "pharmacies not within 200 m of a hospital", "banks at T-junctions"). Each result is verified as VERIFIED / PROBABLE / INSUFFICIENT_EVIDENCE / REJECTED.

**Measure and aggregate**
- Distance from features to the nearest object of another layer.
- Destinations reachable by road within a cutoff.
- Counts and density per district or grid cell.
- Genuine junction detection from road-connector topology: four-way, T, interchange. Grade-separated crossings are reported separately.

**Decide**
- Site suitability: hard filters, weighted criteria you can edit, sensitivity analysis, per-site explanations.
- Facility location: p-median and max-coverage with capacity constraints, compared against greedy and random baselines.
- What-if scenarios (add a facility, close a road or connector), all labelled as simulated.

**Learn and explain**
- Building land-use classification with spatially blocked cross-validation.
- Methodology and data-definition questions answered with citations from the bundled knowledge base.
- HTML evidence reports and CSV / GeoJSON / GeoParquet exports with lineage.

**Interfaces**: CLI, FastAPI (`/v1/...`) and a MapLibre web UI with an area selector, layer toggles, inspector, weight sliders, region drawing and scenario editing.

## Data cases handled

Dense city core, airport precinct, mid-density north, and a sparse fringe with no POIs. Also handled: empty layers, invalid or thin polygons after reprojection, roads with no junctions, disconnected road components, oversized features that cross map tiles, and all-zero weights.

## Edge cases covered

Over-long questions (5000 characters), unknown result ids, unknown scenario operations, a missing connector, a facility far outside the area, a tiny region, `p` larger than the number of candidates, no matching demand categories, empty scenario lists, and 5 simultaneous asks.

## What it refuses or flags

- It will not invent data for an empty layer. It says the layer is empty.
- Live data acquisition needs approval; the evaluation ran on cached data only.
- Every number in an answer carries a calculation id, and an audit pass flags any untraceable claim.

## Limits (see also `limitations.md`)

- Suhail parcel coverage and Urbi density vary by area; the east fringe has almost none.
- The Urbi and Suhail licence is the user's stated permission; no licence document was seen.
- Suhail land-use classes come from a keyword mapping of Arabic text, not an official classification.
- Accessibility uses road distance on the Overture network, with no traffic, turn restrictions or time of day.
- Scenario results are simulations, not forecasts.
