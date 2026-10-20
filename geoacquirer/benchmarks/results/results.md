# GeoAcquirer benchmark results

All values measured by `benchmarks/run_benchmarks.py` (see its docstring for the oracle design).

Environment: {'python': '3.14.4', 'shapely': '2.2.0', 'geopandas': '1.2.0'}

## Synthetic city topology

`{'pois': 3168, 'parcels': 1296, 'segments': 314, 'four_way_found': 121, 'four_way_truth': 121, 't_junction_found': 44, 't_junction_truth': 44, 'grade_separated_crossings_excluded': 1}`

## Constraint precision / recall vs brute-force oracle

| Task | Precision | Recall | Predicted | Truth | Notes |
|---|---|---|---|---|---|
| 01_shops_fourway_not_highway / exact_geometry_u0 | 1.0 | 1.0 | 135 | 135 | coverage(non-insufficient, u=10)=0.954 |
| 01_shops_fourway_not_highway / with_u10_verified_only | 1.0 | 0.4222 | 57 | 135 | coverage(non-insufficient, u=10)=0.954 |
| 01_shops_fourway_not_highway / with_u10_verified_plus_probable | 1.0 | 1.0 | 135 | 135 | coverage(non-insufficient, u=10)=0.954 |
| 01b_positional_noise_sigma8m / u=0 / verified_only | 0.75 | 0.8 | 144 | 135 | insufficient fraction=0.0 |
| 01b_positional_noise_sigma8m / u=0 / verified_plus_probable | 0.75 | 0.8 | 144 | 135 | insufficient fraction=0.0 |
| 01b_positional_noise_sigma8m / u=10 / verified_only | 0.9444 | 0.3778 | 54 | 135 | insufficient fraction=0.04 |
| 01b_positional_noise_sigma8m / u=10 / verified_plus_probable | 0.75 | 0.8 | 144 | 135 | insufficient fraction=0.04 |
| 01b_positional_noise_sigma8m / u=20 / verified_only | 1.0 | 0.0741 | 10 | 135 | insufficient fraction=0.0972 |
| 01b_positional_noise_sigma8m / u=20 / verified_plus_probable | 0.75 | 0.8 | 144 | 135 | insufficient fraction=0.0972 |
| 02_commercial_parcels_two_roads | 1.0 | 1.0 | 228 | 228 |   |
| 03_parking_near_shops | 1.0 | 1.0 | 80 | 80 |   |
| 04_warehouses_outside_residential | 1.0 | 1.0 | 71 | 71 |   |
| 05_clinics_600m_by_road_of_hospital | 1.0 | 1.0 | 41 | 41 |   |
| 06_poi_building_association | 0.4939 | 0.9758 | 652 | 330 | {'none': 1848, 'nearest_within_20m': 366, 'contained': 286} {'contained': {'links': 286, 'correct': 247, 'precision': 0.8636}, 'nearest_within_20m': {'links': 366, 'correct': 75, 'precision': 0.2049}} |
| 07_multi_constraint | 1.0 | 1.0 | 70 | 70 |   |
| 08_multi_source_dedup (entity-resolution pairs) | 1.0 | 1.0 | 300 | 300 | rows 3768->3468; precision by match type {'proximity_and_name': {'pairs': 300, 'correct': 300, 'precision': 1.0}} |
| 09_missing_evidence_abstains | - | - | - | - | abstention=1.0, false_verified=0 of 2500 |
| 10_incomplete_conflicting_data (dedup, 25% names missing) | 0.9004 | 0.7233 | 241 | 300 | error-free=True, statuses valid=True; precision by match type {'proximity_and_category_only': {'pairs': 56, 'correct': 46, 'precision': 0.8214}, 'proximity_and_name': {'pairs': 185, 'correct': 171, 'precision': 0.9243}} |

## Query compilation (rule compiler)

20/20 exact IR matches (1.0).

Failures:

```json
[]
```

## Planner vs exhaustive retrieval

```json
{
  "measured_live_layer_seconds": {
    "connector": 48.4,
    "building": 869.8,
    "road": 366.1,
    "poi": 52.6
  },
  "tasks": [
    {
      "task": "flagship",
      "layers_planned": [
        "connector",
        "poi",
        "road"
      ],
      "layers_exhaustive": [
        "poi",
        "road",
        "connector",
        "building",
        "parcel"
      ],
      "est_live_seconds_planned": 467.1,
      "est_live_seconds_exhaustive": 1336.9
    },
    {
      "task": "parking_near_shops",
      "layers_planned": [
        "poi"
      ],
      "layers_exhaustive": [
        "poi",
        "road",
        "connector",
        "building",
        "parcel"
      ],
      "est_live_seconds_planned": 52.6,
      "est_live_seconds_exhaustive": 1336.9
    },
    {
      "task": "warehouses_outside_residential",
      "layers_planned": [
        "parcel",
        "poi"
      ],
      "layers_exhaustive": [
        "poi",
        "road",
        "connector",
        "building",
        "parcel"
      ],
      "est_live_seconds_planned": 52.6,
      "est_live_seconds_exhaustive": 1336.9
    }
  ],
  "note": "seconds are the measured first-fetch latency of each Overture layer for the 12x15 km Riyadh bbox; the 'planned' column sums only layers the IR needs"
}
```

## Indexed vs naive nearest-road distance (real Riyadh data)

```json
{
  "shops": 3306,
  "road_segments": 45796,
  "naive_subset": 400,
  "naive_seconds_subset": 7.163,
  "indexed_seconds_same_subset": 0.0097,
  "speedup_x": 737.3,
  "indexed_seconds_all_shops": 0.088,
  "naive_extrapolated_all_seconds": 59.2,
  "max_abs_distance_difference_m": 0.0
}
```

## Live flagship run (real Overture data)

```json
{
  "run": "20261019-141342-e9f951",
  "status_counts": {
    "REJECTED": 2579,
    "INSUFFICIENT_EVIDENCE": 61,
    "VERIFIED_MATCH": 55,
    "PROBABLE_MATCH": 37
  },
  "verification": {
    "ok": true,
    "checked_rows": 453,
    "checked_leaves": 906,
    "n_disagreements": 0,
    "max_abs_distance_error_m": 0.0049
  },
  "wall_seconds_this_run": 9.3,
  "budget": {
    "requests": 0,
    "features": 0,
    "elapsed_s": 9.26
  },
  "topology": [
    "45796 drivable segments, 28599 junction clusters (4027 four-way, 21917 T), 969 grade-separated crossings excluded"
  ],
  "overture_release": "2026-09-23.1"
}
```
