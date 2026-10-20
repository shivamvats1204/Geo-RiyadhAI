# City-wide Urbi + Suhail data: loading and verification (2026-10-19/20)

Raw downloads (outside OneDrive): `C:\Users\shiva\urbi_city\` (JSONL, 1.8 GB buildings), `C:\Users\shiva\suhail_city\` (parcels.parquet, rules.csv).
Converted/tiled store: `C:\Users\shiva\urbi_city\parquet\` (GeoParquet, centroid point + best polygon + all remaining fields as `extra_json`) and
`C:\Users\shiva\riyadh_city_store\` (0.05-degree tiles; `CityStore.query(layer, bbox)`).

## Counts (all measured)
| Layer | Source | Rows | Notes |
|---|---|---|---|
| urbi_building | Urbi | 1,146,585 | every scanned id has a record; 150 were recovered by re-fetch after silent transient failures |
| urbi_parking | Urbi | 40,977 | capacity missing 1.1%, name present only 1.3% |
| street | Urbi | 62,489 | polygons of street areas (485 are very long) |
| attraction | Urbi | 10,039 | 9,780 polygons are ~28 m2 map symbols, 259 points |
| metro | Urbi | 2,754 | polygons are ~78 m2 symbols: use the centroid |
| parcel | Suhail | 1,261,358 | 1,210,826 with a rule row, 50,532 "no rules"; valid polygons |

## Checks and results
* Completeness vs scan ids: 0 missing, 0 duplicates, 0 records outside the scan list (all five Urbi layers); 0 unparseable lines.
* Round trip: 1,500 random raw records per small layer re-parsed and compared field by field and geometry by geometry: 0 mismatches.
* Cross-run: 100% of the ids from the earlier 188 km2 download are in the city download (buildings 110,291/110,291).
* Geometry: 125 building, 86 street and 1 parking polygons were invalid and were repaired with `make_valid` (0 remain); 0 zero-area polygons; every centroid lies inside its polygon; 896 building centroids lie slightly outside the notebook bounds (scan overshoot).
* Building footprint area: median 200 m2, 99th percentile 2,265 m2, total 358 km2.
* Tiled store vs brute-force scan of the raw files, 6 layers x 5 boxes (airport, study area, north-east fringe, south, east of the Suhail box): 30/30 identical.

## Defects found while doing this (all fixed, with tests)
1. Silent loss of 150 buildings: a non-200 or short API reply was treated as "no data" while the run reported 0 failed batches. Non-200 replies and short batches now raise and are retried.
2. Tile lookup missed oversized features (long street polygons, huge parcels) stored under a different tile: first run found 7,213 streets vs 7,230 true. The store now indexes oversized features separately (485 streets, 113 parcels).
3. `CityStore.write` used label indexing and failed for batches with a non-zero-based index.
4. The Suhail HTTP pool held 10 connections for 64 threads, making the download CPU-bound; fixing it made rule chunks about 3x faster.

## Limits
* Parcels exist only inside the Suhail bounding box (46.40-47.10 E, 24.40-25.15 N): 201 tiles vs 977 tiles of Urbi buildings. Buildings outside it have no parcel/zoning data.
* "Complete" means complete relative to the scan grid; objects no probe reached cannot be detected without ground truth.
* Overture roads/POIs/buildings are still fetched per area (minutes per layer); the analyses run on one area at a time (e.g. `build_area_from_city`), not on the whole city at once. An 8 GB machine cannot hold 1.1 million polygons plus a road graph in memory.
