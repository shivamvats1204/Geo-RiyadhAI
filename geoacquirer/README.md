# GeoAcquirer

A natural-language-driven geospatial data acquisition and verification agent. You ask a geographic question; it compiles the question to a typed query, works out which layers are missing, fetches them from authorized sources, evaluates the constraints with deterministic geometry, re-verifies the answer independently, and exports evidence-carrying results.

The language model (optional) only translates text into a validated query object. Every distance, junction and containment test is computed by Shapely/GeoPandas.

## Install and run

```bash
pip install -e ".[dev]"          # Python >= 3.11; optional extras: ".[suhail]"
pytest                            # offline tests, about 10 s
```

```bash
# flagship query on real Overture Maps data (first run downloads layers: several minutes; later runs use the cache)
geoacquirer run "Find shops in the central Riyadh study area that are located at crossroads but are not on highways. Return their names, categories, coordinates, associated buildings/parcels, nearby roads, and geographic evidence. Show the results on a map and export them to CSV and GeoJSON." --out runs
```

Outputs per run (`runs/<id>/`): `results.csv`, `results.geojson`, `map.html` (open in a browser), `report.html`, `manifest.json` (IR, defaults, sources, licences, retrieval dates, schema, verification, sha256 of each file, reproduction command), `audit.jsonl`, `state.json`.

```bash
geoacquirer plan "<question>"     # show the compiled IR and acquisition plan without fetching
geoacquirer sources               # registered sources and whether each is enabled
geoacquirer inventory <dir>       # what local datasets could satisfy a query
geoacquirer serve --out runs      # HTTP API: POST /v1/queries, GET /v1/queries/{id}, .../files/{name}, .../evidence/{id}
python benchmarks/run_benchmarks.py   # regenerates benchmarks/results/results.{json,md}
python scripts/prefetch_riyadh.py     # warm the cache for the Riyadh study area
```

Defaults for the flagship query (all shown in the manifest and report, all overridable in the query text or IR): four-way junction within **30 m**; motorway/trunk exclusion **60 m**; POI positional uncertainty **10 m** (no entrance geometry).

## What was measured

Everything below is reproducible; see `benchmarks/results/results.md` for the full tables.

* **Live Riyadh run** (Overture release 2026-09-23.1, study area 46.63-46.73 E, 24.65-24.77 N): 2,732 shop candidates → 55 `VERIFIED_MATCH`, 37 `PROBABLE_MATCH`, 61 `INSUFFICIENT_EVIDENCE`, 2,579 `REJECTED`; 4,027 four-way and 21,917 T junction clusters, 969 grade-separated crossings excluded; independent verifier re-checked 906 predicate evaluations with 0 disagreements (max distance difference 0.005 m). There is no ground-truth junction inventory for Riyadh, so this run demonstrates consistency, not accuracy.
* **Synthetic ground truth (12 × 12 street grid, brute-force oracle):** junction detection found exactly the 121 four-way and 44 T junctions and excluded the overpass; constraint precision/recall = 1.0 on the geometric tasks (shops at crossroads, parcel frontage, proximity, zone exclusion, road distance, multi-constraint); correct abstention (100%) when a required layer is missing.
* **Positional noise (σ = 8 m on shop positions):** with u = 10 m, `VERIFIED_MATCH` precision 0.94 (recall 0.38); treating raw distances as exact (u = 0) gives precision 0.75 / recall 0.80. The status bands trade recall for trustworthy "verified" results.
* **Entity resolution:** cross-source POI de-duplication precision/recall 1.0/1.0 with names present; 0.90/0.72 with 25% of names missing. POI→building: contained links 86% precise, `nearest_within_20m` links only 20% precise on the synthetic set, which is why they are labelled separately. Synthetic, not a manually curated Riyadh sample.
* **Query compilation:** 20/20 exact IR matches on the project's phrasing set (one case checks that an unsupported phrase is flagged rather than dropped). This is a small in-house set, not a general NL benchmark.
* **Index vs brute force:** nearest-road distance for 3,306 Riyadh shops against 45,796 segments: indexed 0.07 s; brute force extrapolated to 77 s (measured on a 400-shop subset); identical distances.

## Limitations (read before relying on results)

* POI coverage and positions come from the source; a shop absent from Overture is absent here. POIs are points, not entrances.
* Parcels: no authorized open source; parcel association is reported unavailable, never inferred. Suhail/Urbi adapters are **disabled** until you record authorization (see `docs/notebook_audit.md`). The Suhail/Urbi adapters have unit tests on their parsers but were never run against the live services.
* The OpenStreetMap/Overpass adapter has fixture tests only; public mirrors were unavailable during development.
* The LLM compiler path is implemented and tested with stub adapters only. The HTTP API and map are minimal.
* Docker/PostGIS were not available where this was built; `docker-compose.yml` is provided untested.
* **Security:** never commit keys. They come from environment variables or a git-ignored `.env` (see `.env.example`). The original exploration notebooks hard-coded a key; the copies in `reference_notebooks/` are redacted, and a test fails if a key-shaped string appears in the source.

Docs: `docs/architecture.md`, `docs/topology.md`, `docs/data_sources.md`, `docs/notebook_audit.md`.
