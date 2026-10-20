# GeoAcquirer architecture

```
natural language ──► Compiler (rules | LLM→validated) ──► GeoQuery IR (Pydantic, strict, no SQL/code)
                                                           │
   Inventory (local files, cache) ◄── needs_for(IR) ───────┤
        │ gaps                                             ▼
        └──► Planner: deterministic source scoring ──► AcquisitionPlan (steps, est. cost, gaps, approval gate)
                                                           │
   Providers (Overture, OSM/Overpass, local files; Suhail/Urbi disabled) ── parallel per layer, budgeted, cached
                                                           ▼
   DataContext (metric UTM CRS): poi · road segments · connectors · junction clusters · buildings · parcels
                                                           ▼
   Evaluator: tri-state predicates (PASS / PASS_NEAR / UNKNOWN / FAIL_NEAR / FAIL) with AND/OR/NOT ──► status per entity
                                                           ▼
   Enrichment (entity resolution) ─► independent Verifier (brute force) ─► Exports + manifest + map + report
```

## Module map

| Package | Responsibility |
|---|---|
| `query_ir` | Typed IR: 13 operations, nested AND/OR/NOT, exact vs approximate, budgets, defaults, unparsed phrases |
| `agent` | `RuleCompiler`, `LLMCompiler` (validate-or-fallback), `GeoAcquirerAgent` state machine |
| `providers` | Interface (`discover_capabilities / estimate_coverage / fetch_features / fetch_details / normalize / report_provenance`), registry JSON, scoring, adapters |
| `acquisition` | Grid / quadtree / coarse-to-fine partitioning; resumable `AdaptiveCrawler` |
| `spatial` | Projection, junction clustering, grade-separated crossing report, connector road graph |
| `engine` | `DataContext`, tri-state `Evaluator` |
| `entity_resolution` | POI→building, building→parcel, POI→roads, cross-source POI matching and de-duplication |
| `verification` | Brute-force re-computation of distance predicates; disagreement = defect |
| `knowledge_graph` | Relational edge tables (SQLite) with params + provenance; `explain(node)` |
| `exports` | CSV, GeoJSON, GeoParquet, JSONL, manifest (sha256), HTML report, Leaflet map |
| `security`, `observability` | Allowlists, budgets, rate limits, safe identifiers, untrusted-text sanitizer, JSONL audit log |
| `api` | FastAPI `/v1/...` (submit, status, cancel, files, evidence, sources) |

## Status semantics (the contract with callers)

Each leaf predicate yields a code from the measured value and the entity's positional uncertainty `u` (default 10 m for point POIs without a mapped entrance; 0 for polygons):

* `PASS` – satisfied with margin ≥ u · `PASS_NEAR` – satisfied but inside the uncertainty band
* `FAIL` – violated with margin ≥ u · `FAIL_NEAR` – violated but inside the band
* `UNKNOWN` – the evidence layer is missing or unusable (never treated as false)

Combination: `AND` → FAIL < UNKNOWN < FAIL_NEAR < PASS_NEAR < PASS (minimum); `OR` → maximum over FAIL < FAIL_NEAR < UNKNOWN < PASS_NEAR < PASS; `NOT` swaps pass/fail. Final: PASS→`VERIFIED_MATCH`, PASS_NEAR→`PROBABLE_MATCH`, FAIL→`REJECTED`, FAIL_NEAR/UNKNOWN→`INSUFFICIENT_EVIDENCE`. No numeric confidence is invented; the benchmark reports empirical precision/recall per status instead.

## Request flow and checkpoints

`COMPILE → INVENTORY → PLAN → ACQUIRE → BUILD_CONTEXT → EVALUATE → ENRICH → VERIFY → EXPORT`, recorded in `<run>/state.json`; acquired layers are cached on disk (keyed by provider, release, entity, bbox, schema), so a re-run after interruption skips the network. `Cancelled` is raised between stages when the cancel event is set; the crawler stops mid-stage.

## Deviations from the brief (stated plainly)

* **PostGIS / Docker:** not available in the build environment (no Docker). Spatial work runs on GeoPandas/Shapely/DuckDB with GeoParquet; `docker-compose.yml` provides the API container and an *untested* optional PostGIS service. No code path depends on PostGIS.
* **LangGraph:** replaced by the explicit state machine above (the brief allows this).
* **UI:** a generated Leaflet map (`map.html`) per run, not a separate React app (the interactive studio is in the Geo-RiyadhAI project).
* **LLM compiler:** implemented and unit-tested with stub adapters; never exercised against a live model in this build.
* **Live POI→parcel:** no authorized open parcel source exists, so parcels stay "unavailable" unless the user supplies a parcel file or enables Suhail.
