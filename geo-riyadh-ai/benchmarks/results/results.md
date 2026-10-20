# Geo-RiyadhAI benchmark results

Area `central_riyadh`, bbox [46.62, 24.64, 46.74, 24.78], Overture release 2026-09-23.1. All numbers measured by `benchmarks/run_benchmarks.py`.

## Orchestration: query-to-plan

- Gold questions: 26; exact tool-sequence match **26/26** (1.0); argument checks 14/14; follow-ups 3/3.
- Failures: `[]`

## Orchestration: execution on the synthetic area

- 21 supported questions run: done 21, failed 0, awaiting approval 0; task success rate **1.0**.
- Claim audit: 445 numbers in answers, 0 not traceable to a tool output (**hallucinated-claim rate 0.0**); facts with calculation id 94/94.
- Failure injection: `{"missing_district": {"status": "partial", "recorded_in_trace": true}, "layer_missing_parcels": {"status": "failed", "recorded_in_trace": true}, "unknown_tool_args": "ToolError raised cleanly", "approval_needed": {"status": "awaiting_approval", "recorded_in_trace": true}, "empty_result_followup": {"status": "done", "recorded_in_trace": true}}`
- Per-question errors: `[]`

## Spatial correctness (real data)

```json
{
  "utm_vs_geodesic_relative_error": {
    "pairs": 2000,
    "max": 6.7e-05,
    "median": 4.18e-05,
    "note": "UTM 38N scale error over a 12x15 km area"
  },
  "district_aggregation_vs_bruteforce": {
    "districts": 62,
    "mismatches": 0,
    "note": "shops counted inside district polygons; overlaps (macrohood vs neighborhood) count in each"
  },
  "topology_real_area": {
    "junction_clusters": 28599,
    "kinds": {
      "t_junction": 21917,
      "four_way": 4027,
      "other": 2135,
      "highway_interchange": 370,
      "complex": 150
    },
    "grade_separated_crossings_excluded": 969,
    "road_components": {
      "components": 193,
      "largest_component_nodes": 41035,
      "share_in_largest": 0.984
    },
    "note": "no ground-truth junction inventory exists for Riyadh; correctness is established on synthetic grids (tests, GeoAcquirer benchmarks)"
  },
  "filter_features_vs_bruteforce": {
    "engine_rows": 2726,
    "bruteforce_rows": 2726,
    "equal": true
  }
}
```

## Machine learning (building land use, spatial block CV)

```json
{
  "dataset": {
    "n_labelled": 30000,
    "commercial_share": 0.1025,
    "spatial_blocks_1km": 135,
    "label_source": "Overture building `subtype` (OSM-derived; noisy, not cadastral)",
    "release": "2026-09-23.1"
  },
  "spatial_cv": {
    "baseline_majority_class": {
      "balanced_accuracy": 0.5,
      "f1_commercial": 0.0,
      "pr_auc": 0.0987,
      "roc_auc": 0.4847,
      "ece_uncalibrated": 0.0
    },
    "baseline_logistic_shape_only": {
      "balanced_accuracy": 0.6302,
      "f1_commercial": 0.2781,
      "pr_auc": 0.2301,
      "roc_auc": 0.6807,
      "ece_uncalibrated": 0.3614
    },
    "baseline_rule_shopping_poi_within_100m": {
      "balanced_accuracy": 0.7108,
      "f1_commercial": 0.3723,
      "pr_auc": 0.2035
    },
    "lightgbm_all_features": {
      "balanced_accuracy": 0.8906,
      "f1_commercial": 0.7364,
      "pr_auc": 0.8047,
      "roc_auc": 0.9654,
      "ece_uncalibrated": 0.037
    },
    "lightgbm_random_split_for_contrast_only": {
      "pr_auc": 0.8921,
      "roc_auc": 0.9827,
      "balanced_accuracy": 0.9356
    }
  },
  "fold_pr_auc_lightgbm": [
    0.7681101142899694,
    0.8080722143017378,
    0.8329291267147317,
    0.8162060079292197,
    0.8086927179373413
  ],
  "ablations": {
    "without_shape": {
      "pr_auc": 0.7821,
      "roc_auc": 0.9572,
      "balanced_accuracy": 0.883,
      "f1_commercial": 0.7055
    },
    "without_neighbourhood": {
      "pr_auc": 0.7961,
      "roc_auc": 0.9603,
      "balanced_accuracy": 0.8927,
      "f1_commercial": 0.7333
    },
    "without_poi": {
      "pr_auc": 0.7584,
      "roc_auc": 0.9557,
      "balanced_accuracy": 0.8773,
      "f1_commercial": 0.6944
    },
    "without_road": {
      "pr_auc": 0.6681,
      "roc_auc": 0.9229,
      "balanced_accuracy": 0.8267,
      "f1_commercial": 0.5849
    }
  },
  "top_feature_gain": {
    "poi_total_100m": 74527.8,
    "road_class_30m": 74431.6,
    "dist_primary_m": 39721.7,
    "n_buildings_50m": 25952.4,
    "built_density_100m": 22907.7,
    "dist_secondary_m": 15756.2,
    "perimeter_m": 11912.2,
    "compactness": 11309.7,
    "poi_total_250m": 10645.6,
    "area_m2": 9903.7
  },
  "limitations": [
    "labels are noisy OSM-derived tags",
    "spatial CV is blocked at 1 km; performance in an unseen city is not measured",
    "probabilities are NOT calibrated (see ece_uncalibrated)"
  ],
  "training_seconds": 26.5
}
```

## Optimization (real data)

| region | p | objective | cand. | pairs | before m | greedy m | greedy+swap m | CP-SAT m | random m | before cov | greedy cov | swap cov | CP-SAT cov | random cov | status | gap % | solver s | highway ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| central | 3 | p_median | 400 | 175 | 124.8 | 118.3 | 118.3 | 118.3 | 124.6 | 0.9838 | 0.9923 | 0.9923 | 0.9923 | 0.9839 | OPTIMAL | 0.0 | 0.03 | True |
| central | 3 | max_coverage | 400 | 156 | 124.8 | 118.8 | 118.8 | 118.8 | 124.6 | 0.9838 | 0.9932 | 0.9932 | 0.9932 | 0.984 | OPTIMAL | 0.0 | 0.02 | True |
| north | 3 | p_median | 400 | 244 | 165.2 | 159.7 | 159.7 | 159.7 | 165.0 | 0.962 | 0.9705 | 0.9705 | 0.9705 | 0.9624 | OPTIMAL | 0.0 | 0.03 | True |
| central | 5 | p_median | 400 | 175 | 124.8 | 117.5 | 117.5 | 117.5 | 124.5 | 0.9838 | 0.9932 | 0.9932 | 0.9932 | 0.9841 | OPTIMAL | 0.0 | 0.04 | True |
| all | 3 | p_median | 400 | 447 | 179.6 | 176.0 | 176.0 | 176.0 | 179.4 | 0.9597 | 0.9648 | 0.9648 | 0.9648 | 0.9599 | OPTIMAL | 0.0 | 0.06 | True |
| all | 3 | max_coverage | 400 | 320 | 179.6 | 177.2 | 177.2 | 177.2 | 179.5 | 0.9597 | 0.9652 | 0.9652 | 0.9652 | 0.9599 | OPTIMAL | 0.0 | 0.02 | True |

## Performance

```json
{
  "filter_features": {
    "median_seconds": 0.26
  },
  "find_intersections": {
    "median_seconds": 0.205
  },
  "aggregate_spatial_statistics": {
    "median_seconds": 0.114
  },
  "calculate_network_accessibility": {
    "median_seconds": 8.506
  },
  "retrieve_reference_documents": {
    "median_seconds": 0.005
  },
  "python_heap_peak_mb_during_tool_calls": 24.7,
  "feature_scalability": [
    {
      "grid_step_m": 400,
      "eligible_cells": 687,
      "candidate_seconds": 1.84,
      "feature_seconds": 2.61
    },
    {
      "grid_step_m": 250,
      "eligible_cells": 1698,
      "candidate_seconds": 1.93,
      "feature_seconds": 1.78
    },
    {
      "grid_step_m": 150,
      "eligible_cells": 4726,
      "candidate_seconds": 5.86,
      "feature_seconds": 3.58
    },
    {
      "grid_step_m": 100,
      "eligible_cells": 10716,
      "candidate_seconds": 13.31,
      "feature_seconds": 6.65
    }
  ],
  "rank_site_suitability_first_call_seconds": 11.1,
  "rank_site_suitability_reweight_seconds_cached_features": 0.673,
  "map_layer_payloads_8x8km_view": {
    "poi": {
      "seconds": 0.282,
      "kb": 1358.6,
      "features": 6000
    },
    "road": {
      "seconds": 0.15,
      "kb": 546.2,
      "features": 2021
    },
    "district": {
      "seconds": 0.05,
      "kb": 45.8,
      "features": 27
    },
    "junction": {
      "seconds": 0.097,
      "kb": 369.7,
      "features": 1879
    }
  },
  "note": "map rendering FPS was not measured (no headless GPU profiling in this build); payload sizes and API latency are measured"
}
```
