# Tool registry

`GET /v1/tools` returns the live list with JSON schemas. Each tool: validates input (Pydantic, `extra="forbid"`), enforces bounds (row budget 50,000, acquisition bbox ≤ 0.1 deg², region within the selected area), returns a `ToolResult` (summary, optional table, optional result-set id, provenance, warnings, facts with calculation ids), and handles errors explicitly (`ToolError`, `NeedsApproval`).

| Tool | Purpose | Expensive / needs approval |
|---|---|---|
| `search_local_data` | areas, layers, row counts | no |
| `request_geoacquirer_data` | acquire missing layers (cache first) | live acquisition only |
| `inspect_layer_schema` | columns, null rates | no |
| `get_geographic_coverage` | coverage, topology stats, data gaps | no |
| `filter_features` | category + region + proximity filters, follow-ups via `base_result_id` | no |
| `run_geoquery` | natural-language constraint query via GeoAcquirer's evaluator (VERIFIED/PROBABLE/INSUFFICIENT/REJECTED) | no |
| `find_intersections` | genuine junctions from connector topology | no |
| `measure_proximity` | distance to nearest object of another layer | no |
| `calculate_network_accessibility` | destinations reachable by road within a cutoff | no |
| `aggregate_spatial_statistics` | counts and density per district or grid cell | no |
| `classify_land_use` | building land-use model: metrics / train / predict | training |
| `rank_site_suitability` | hard filters, weighted criteria, sensitivity, explanations | no (first call ~15 s) |
| `optimize_facility_locations` | p-median / max-coverage placement with baselines | no (time-limited) |
| `compare_scenarios` | simulated what-ifs (add facility, close road) | no |
| `retrieve_reference_documents` | cited methodology / data-definition passages | no |
| `render_map_layer`, `export_dataset` | GeoJSON for the map; CSV/GeoJSON/GeoParquet with lineage | no |
| `generate_evidence_report` | HTML report from the whole job (executor-level) | no |

`spatial_join` from the original list is covered by `filter_features(near=...)`, `measure_proximity` and `aggregate_spatial_statistics`.
