# Data sources, licences and limits

The machine-readable registry is `src/geoacquirer/providers/registry.json` (`geoacquirer sources` prints enablement).

| Source | Status | Licence / terms | Notes |
|---|---|---|---|
| Overture Maps (places, transportation, buildings) | enabled, no credentials | Places: CDLA-Permissive-2.0. Transportation & buildings: ODbL 1.0 (OSM-derived, share-alike) and per-record `sources.license`. Attribute "© Overture Maps Foundation; © OpenStreetMap contributors". | Release 2026-09-23.1 used in this repo's runs. First fetch of a 12 × 15 km bbox took 50 s (connectors), 54 s (POIs), 368 s (roads), minutes for buildings (S3 scans). |
| OpenStreetMap via Overpass | enabled | ODbL 1.0; public servers are fair-use | During development all public mirrors returned 502/504 for a tiny query (2026-10-19), so this adapter is covered by fixture tests only, not by a live run. |
| Local files (GeoJSON/GeoParquet/CSV lon,lat) | enabled | User responsibility | Entity type guessed from id columns (`poi_id`, `road_id`, `building_id`, `parcel_id`...). |
| Suhail (parcels, zoning rules) | **disabled** | Unestablished | Needs `GEOACQUIRER_ENABLE_SUHAIL=1` + `GEOACQUIRER_SUHAIL_AUTHORIZATION_REF`. |
| Urbi / 2GIS (parking) | **disabled** | Unestablished; needs your own key | Needs `GEOACQUIRER_ENABLE_URBI=1`, `..._AUTHORIZATION_REF`, `GEOACQUIRER_URBI_API_KEY`. |

## Coverage limitations that affect results

* Overture POIs are an aggregate of several sources; completeness for small shops is unknown. A shop missing from the output may simply be missing from the source.
* POIs are points: no entrance geometry. Distances use the stated positional uncertainty (default 10 m, `--` configurable in the IR).
* Overture has **no parcels**. Parcel-dependent constraints return `INSUFFICIENT_EVIDENCE`.
* Building footprints are not cadastral; POI↔building links are labelled `contained` or `nearest_within_20m`, and the latter is weak evidence (measured precision 20% on the synthetic benchmark, see `benchmarks/results/results.md`).
