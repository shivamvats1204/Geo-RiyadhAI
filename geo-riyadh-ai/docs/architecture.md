# Geo-RiyadhAI architecture

```
 user question ─► planner (task classes + typed ExecutionPlan, deterministic) ─► executor (trace, checkpoints, approval gates)
                                                                                     │
        ┌─────────────── ToolRegistry (pydantic-validated, budgeted, provenance-returning) ──────────────┐
        │ search_local_data · request_geoacquirer_data · inspect_layer_schema · get_geographic_coverage  │
        │ filter_features · find_intersections · measure_proximity · aggregate_spatial_statistics        │
        │ calculate_network_accessibility · classify_land_use · rank_site_suitability                    │
        │ optimize_facility_locations · compare_scenarios · retrieve_reference_documents · run_geoquery  │
        │ render_map_layer · export_dataset · generate_evidence_report                                   │
        └───────────────┬───────────────────────────────────────────────────────────────┬────────────────┘
                        │ UrbanStore (canonical GeoParquet, provenance, derived, models) │ GeoAcquirer client (SDK; HTTP for queries)
                        ▼                                                                ▼
        features · suitability · optimization (CP-SAT) · simulation · ml (LightGBM, spatial CV) · rag (TF-IDF)   Overture / local files
                        │
                        ▼
        ResultStore (versioned result sets, lineage) ─► reporting (HTML evidence report) ─► FastAPI + MapLibre workspace
```

## Principles enforced in code

1. **The model never produces a number.** Tools compute; `Executor.compose_answer` returns only facts emitted by tools, each tagged `observed | model | inferred | reference` with a calculation id. `Executor.audit_claims` re-checks that every number in the answer exists in a tool output (the benchmark reports the count of failures).
2. **Typed everything.** Tool arguments are Pydantic models with bounds (`extra="forbid"`); invalid calls raise `ToolError` and are recorded in the trace instead of crashing the job.
3. **Missing data is a finding.** Layers that do not exist (parcels, population, occupancy) yield `UNVERIFIED`/`INSUFFICIENT_EVIDENCE` and an explicit gap statement; nothing is imputed.
4. **Heuristic is labelled heuristic.** Suitability scores carry the note "not validated predictions"; optimization carries "buildable land NOT verified"; scenarios carry `simulated: True`.
5. **Expensive steps need approval.** `request_geoacquirer_data` (live acquisition) and `classify_land_use` (training) raise `NeedsApproval`; the job pauses in `awaiting_approval` and resumes after `POST /v1/jobs/{id}/approve`.
6. **GeoAcquirer is consumed, not copied.** `geoacquirer_client.py` wraps the SDK; connectors, junction topology, the evaluator and the category ontology are imported from `geoacquirer`.

## Task classes

RETRIEVAL, SPATIAL_FILTERING, SPATIAL_JOIN, NETWORK_ANALYSIS, STATISTICAL_ANALYSIS, PREDICTION, OPTIMIZATION, SCENARIO_SIMULATION, TEMPORAL_ANALYSIS, REPORT_GENERATION. `TEMPORAL_ANALYSIS` is declared but unsupported in the planner: only one Overture snapshot is stored, and single-snapshot data cannot show change.

## Deviations from the brief (stated plainly)

* **PostgreSQL/PostGIS and Redis are not used.** No Docker/PostGIS was available. The "database" is GeoParquet files plus in-memory GeoPandas indexes (STRtree); `database/schemas/canonical.sql` documents the equivalent PostGIS schema (not executed). Background work is a thread behind a lock, not a worker queue.
* **Frontend is plain JavaScript + MapLibre GL (no React/TypeScript/Tailwind/deck.gl build).** It was kept build-free so it could be tested in the provided browser; large point layers are capped server-side (≤ 6,000 features per view).
* **LangGraph/LLM agent:** the planner is rule-based; an LLM could be added behind the same `ExecutionPlan` type but none is wired in.
* **Spatial RAG** is extractive TF-IDF over a small curated corpus (methods, data sources, zoning field notes), not an embedding index.
* **`spatial_join` tool:** delivered as `filter_features(near=...)`, `measure_proximity` and `aggregate_spatial_statistics` rather than a separate tool.
