# Audit of the reference notebooks

Audited 2026-10-19 from `suhail_ai.ipynb` and `urbi.ipynb`. Findings below are read from the notebook source and saved cell outputs; nothing here was re-run against Suhail or 2GIS/Urbi (see "Authorization").

## suhail_ai.ipynb (parcel tiles + building-rule enrichment)

| Topic | Finding | Consequence / what GeoAcquirer does |
|---|---|---|
| **Is parcel polygon geometry saved?** | **No.** The tile cell computes `poly.centroid` and writes only `parcel_id, parcel_objectid, lon, lat` to CSV. The enrichment cell then adds zoning fields per `parcel_objectid`. | Any polygon-dependent feature (frontage, buildable area, parcel/building overlap) cannot be computed from the existing CSVs. `SuhailProvider` preserves polygons; `resolve_building_to_parcel` and `INSIDE_PARCEL` explicitly refuse centroid-only parcels. |
| Tile coordinate conversion | `lonlat_to_tile` is the standard slippy-map formula (correct). `tile_coords_to_lonlat` flips y with `(extent - py)/extent`, which is correct only because `mapbox_vector_tile.decode` defaults to a y-up origin. It is right but fragile. | Port decodes with `y_coord_down=True` and uses `py/extent`; covered by a pixel round-trip test (`tests/test_providers.py`). |
| Parcels cut by tile borders | Vector tiles clip features at the tile edge. The notebook takes the centroid of each clipped piece and `drop_duplicates(parcel_id)` keeps whichever came first, so the centroid of a border-spanning parcel is biased. | Port unions the pieces per parcel (`merge_parts`) and records `n_tile_parts`. |
| Feature-layer selection | `next(iter(tile))` uses only the first layer in each tile. | Port iterates all layers and keeps polygon features carrying parcel identifiers. |
| Error handling | `except: pass` at three levels; failed tiles and unparsable features vanish silently, so "3,290 parcels for Al Bahah" cannot be distinguished from "tiles failed". | Failures are counted (`CrawlStats`), retried with backoff, and surfaced as warnings in the provenance. |
| Rate limiting / politeness | 16 threads, 5 s timeout, no limiter, no retries on tile requests. | Bounded workers (default 4), `RateLimiter`, exponential backoff with jitter, abort on 401/403 and on repeated 429. No evasion logic. |
| Checkpointing | Tile phase: none. Enrichment: resumes by reading existing output files. | `Checkpoint` for tiles; enrichment keyed by `(parcel_objectid, rule_id)`. |
| **Resume bug** | Enrichment resume treats *every* `parcel_objectid` already present in an output file as done, including rows written as `RATE_LIMIT`, `HTTP_5xx`, `JSON_ERROR`. Those parcels are never retried. | Port retries transient statuses inside the crawler and only checkpoints a cell after success. |
| Source coverage | `REGION_CONFIG` boxes are city cores (Riyadh `25.15,46.40 -> 24.40,47.10`; most others ~15-25 km across), zoom fixed at 15. The saved run shows Al Bahah = 90 tiles / 3,290 parcels; the enrichment output shows the *Eastern Region* file (356,387 parcels, 19m44s). The two cells in the notebook therefore ran on different regions. | Coverage claims must be per region file. No evidence in the notebook that Riyadh was scraped. |
| Undocumented endpoints | `tiles.suhail.ai` and `api2.suhail.ai/parcel/buildingRules` are not documented public APIs in the notebook. | Provider disabled by default; see Authorization. |

## urbi.ipynb (2GIS/Urbi parking + buildings)

| Topic | Finding | Consequence |
|---|---|---|
| **Embedded credential** | An API key is hard-coded in three places (config, detail fetch, local-count check). The notebook also sends `User-Agent: 2GIS/5.0 (Android; Mobile)`. | **Rotate that key** and treat it as exposed. The repo's copy (`reference_notebooks/urbi.redacted.ipynb`) is redacted; a test fails the build if the key string appears in `src/`. The port reads `GEOACQUIRER_URBI_API_KEY` and uses an honest User-Agent. |
| Discovery method | Coarse scouting (0.01 deg, radius 1200 m) then fine drilling (0.002 deg, radius 300 m), dedup by id: a sound coarse-to-fine design. | Preserved as `acquisition.grid.coarse_to_fine` + `AdaptiveCrawler`. |
| Error handling | `fetch_parking_ids` breaks silently on non-200; `total` is unbound if the first request raises; bare `except: break/return None`. | Typed exceptions + counted failures. |
| Data content | Saved sample: 40,355 lots; 40,157 of them have schedule "24 Hours Daily", i.e. the schedule field carries almost no information. `Capacity` is operator-reported, not occupancy. | Exports carry `capacity_is_reported_not_live=True`. |
| Coordinate parsing | `extract_geo` regex `[\d\.]+` cannot parse negative or exponent coordinates (harmless in Riyadh, wrong elsewhere). | Fixed and unit-tested. |
| "Superior data" check | Cell 10 compares the scraped building count (15,549) against the API's reported count (5,924) in a 1.5 km radius and prints "SUPERIOR DATA". That is a count mismatch between differently-defined populations, not validation. | Not reproduced; the benchmark suite measures precision/recall against ground truth instead. |
| Incomplete flow | Cells 11-12 reference `riyadh_buildings_part*.csv`, which the notebook never creates (the duplicate check ends in `ValueError: No objects to concatenate`); a scan cell shows "1084/8640 ... 7.31 s/sector" (about 15 h projected). | The building-acquisition code is not in the supplied notebook, so GeoAcquirer gets buildings from Overture instead. |

## Authorization

Neither notebook records a licence or permission for bulk extraction, and the original code works around two things a provider would normally treat as access controls (an app-embedded key and a spoofed app User-Agent). GeoAcquirer therefore ships both adapters **disabled**. To enable one you must set `GEOACQUIRER_ENABLE_<NAME>=1` **and** `GEOACQUIRER_<NAME>_AUTHORIZATION_REF=<where your permission is recorded>`; the reference is copied into every manifest. Redistribution of derived data remains your responsibility.

## Traceability (notebook cell -> module)

| Notebook | Original logic | Module |
|---|---|---|
| suhail cell 5 | `lonlat_to_tile`, tile loop | `acquisition/grid.py::tile_xy, tiles_for_bbox` |
| suhail cell 5 | `tile_coords_to_lonlat`, `process_tile` | `providers/suhail.py::tile_px_to_lonlat, decode_tile_parcels` |
| suhail cell 7 | async enrichment, `FileHandler`, resume | `providers/suhail.py::fetch_details` + `acquisition/crawler.py` |
| urbi cell 0 | `fetch_parking_ids`, scout/drill | `providers/urbi.py::_ids_in, fetch_features` + `coarse_to_fine` |
| urbi cell 1 | detail fields / `byid` | `providers/urbi.py::_details, FIELDS` |
| urbi cell 2 | `extract_geo`, `parse_full_schedule`, CSV conversion | `providers/urbi.py::extract_geo, parse_full_schedule, item_to_row` |
