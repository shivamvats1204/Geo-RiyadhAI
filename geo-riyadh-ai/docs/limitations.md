# Known limitations (read before using results)

## Data
* **Coverage:** the app covers the districts of Riyadh that were downloaded (one Urbi/Suhail download in October 2026; Overture release 2026-09-23.1). Analyses run on *prepared areas* (a bounding box with Overture layers cached); picking a district no area covers prepares one on demand and asks first. Parcels stop at the edge of the downloaded Suhail area, so outer districts can have buildings but no parcels.
* **No population, traffic, land prices, revenue or parking occupancy.** Questions that need them are refused. Parcels and zoning come from Suhail (about 1.26 M parcels city-wide) but the land-use class is a keyword mapping of Arabic text, and only 153 parcels are purely commercial (8,722 are mixed).
* **Parking** comes from Urbi plus Overture POIs. Capacities are operator-reported, not occupancy; the optimizer treats each lot as one facility and by default ignores residents-only, handicapped-only and taxi-only lots.
* POIs are an aggregate of several sources; absence of a shop is not evidence it does not exist. POIs are points without entrances.
* Building footprints are not cadastral. Only about a third carry a land-use subtype (mostly OSM-sourced); unlabelled footprints are mostly ML-detected houses.

## Methods
* **Suitability scores are heuristics** (weights are defaults chosen for the demo, adjustable in the UI). They are not validated against sales, footfall or business survival; no such outcome labels exist here. Rank-stability numbers describe sensitivity to weights, not accuracy.
* The **demand proxy** is residential-like built area reachable within 800 m by road (not population). Candidate sites are 150 m grid cells on streets and not inside existing footprints: availability, ownership and zoning are unchecked.
* **"North/south/east/west/central"** are geometric splits of the bounding box of the selected area, not administrative boundaries. District polygons are OSM-derived neighbourhood/macro-neighbourhood areas; some are municipal areas.
* **Network distance** is undirected and not a travel time; one-way roads, turn restrictions and speeds are not modelled. About 98% of road nodes lie in the largest connected component; disconnected pieces are handled by treating unreachable demand as capped at 2 km.
* **Optimization:** p-median uses CP-SAT with a time limit; when the status is `FEASIBLE` the reported optimality gap says how far the solution may be from optimal. Candidates are pruned to the best 400 by single-facility gain (heuristic), so the optimum is exact only over that set. Demand for parking is proxied by shop locations.
* **Land-use model:** noisy labels, 1 km block spatial CV on one city area; probabilities are uncalibrated; predictions for unlabelled buildings extrapolate to a different population than the one validated.
* **Scenarios** are simulations on a copy of the graph; capacity changes have no effect on the distance model.
* **RAG** is extractive over a small curated corpus; zoning-field definitions are inferred from field names and are explicitly marked non-official.

## Engineering
* No PostGIS/Redis/worker queue, no React/TypeScript build, no LLM wired into planning (see `docs/architecture.md`).
* The API runs analyses one at a time behind a lock; a long optimization blocks other requests.
* Map-rendering frame rate and large-scale (millions of features) behaviour were not measured. Only POIs (≤ 6,000), major roads and junctions are drawn per view; buildings are not rendered on the map.
* Live acquisition through GeoAcquirer takes several minutes per layer and depends on the public Overture S3 bucket.
