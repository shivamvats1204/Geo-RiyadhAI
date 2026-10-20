# Preparing the data

This repository contains **no map data**. The app is only as complete as the data you prepare, and some sources are licensed: read [`NOTICE.md`](../../NOTICE.md) first. Urbi/2GIS and Suhail are **disabled by default**; you may enable them only with your own access and permission.

Everything lives under one folder, `RIYADHAI_DATA_ROOT` (default: your home folder). Individual locations can be overridden with `RIYADHAI_CITYSTORE` and `RIYADHAI_PARTITION`. Set `RIYADHAI_DATA_DATE=YYYY-MM-DD` before building so provenance records carry the real retrieval date.

Disk needed for the whole city: about 3.5 GB of raw downloads (not needed to run the app afterwards) and about 1.5 GB of prepared data. The first full download took about 5.5 hours.

## Overview

```
raw downloads --> city store --> district registry + partition --> vector tiles --> analysis areas --> app
```

## 0. Authorization (only for Urbi and Suhail)

```bash
export GEOACQUIRER_ENABLE_URBI=1   GEOACQUIRER_URBI_AUTHORIZATION_REF="<where your permission is recorded>"   GEOACQUIRER_URBI_API_KEY=...
export GEOACQUIRER_ENABLE_SUHAIL=1 GEOACQUIRER_SUHAIL_AUTHORIZATION_REF="<where your permission is recorded>"
```

(Windows PowerShell: `$env:NAME = "value"`.) The scripts refuse to run without the reference. Keys belong in `.env` or the environment, never in code.

## 1. Raw downloads (hours; resumable)

```bash
python ../geoacquirer/scripts/run_urbi_city.py   "$RIYADHAI_DATA_ROOT/urbi_city"     # buildings, parking, streets, attractions, metro
python ../geoacquirer/scripts/run_suhail_city.py "$RIYADHAI_DATA_ROOT/suhail_city" 8 4  # parcel polygons, then zoning rules
```

Both write checkpoints, retry with backoff, and stop on authorization or rate-limit errors instead of pushing on.

## 2. City store

```bash
python scripts/build_city_store.py
```

Converts the Urbi JSONL to GeoParquet, checks and repairs geometry, and tiles every layer into 0.05 degree files with an index.

## 3. Administrative areas (three independent sources)

```bash
python scripts/fetch_overture_neighborhoods.py          # open data
python scripts/fetch_suhail_neighborhoods.py            # needs Suhail authorization
python scripts/prepare_urbi_divisions.py --scan         # needs Urbi authorization; --scan also finds areas no record references
python scripts/fetch_metro_network.py                   # metro lines and stations (Suhail)
```

## 4. District registry, partition and tiles

```bash
python scripts/build_partition.py        # registry (polygons, or centroids when a source has none) + every feature assigned to its districts, lossless
python scripts/pregen_tiles.py           # vector tiles for the map (resumable per layer)
```

## 5. Analysis areas

```bash
python scripts/build_areas.py            # the four default areas (live Overture download, a few minutes each, cached afterwards)
python scripts/build_areas.py my_area=46.60,24.78,46.76,24.92
```

The app also prepares an area on demand when you pick a district that no prepared area covers (it asks first, because it downloads open data).

## 6. Run and verify

```bash
python -m riyadhai.cli serve --data data_city --port 8080
python scripts/verify_city_spots.py 8     # random places vs the raw records and a fresh Suhail download
python benchmarks/full_evaluation.py --data data_city
```

## Known gaps to expect

Parcel coverage ends at the edge of the downloaded Suhail area, so outer districts can have buildings but no parcels. Features more than 300 m outside every district polygon stay unassigned and are drawn grey. The numbers in the docs (1.26 M parcels, 1.15 M buildings, 867 areas) describe one download of Riyadh in October 2026; yours will differ.
