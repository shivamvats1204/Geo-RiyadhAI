# Methods used by Geo-RiyadhAI

## Evidence statuses
Every constraint result carries VERIFIED_MATCH, PROBABLE_MATCH, INSUFFICIENT_EVIDENCE or REJECTED. VERIFIED means all constraints passed with a margin larger than the positional uncertainty of the entity (10 m for POI points without an entrance). PROBABLE means they passed but at least one is inside the uncertainty band. INSUFFICIENT_EVIDENCE means a required layer was missing or a threshold was crossed inside the uncertainty band. REJECTED means a hard constraint failed.

## Junction detection
Junctions are built from explicit road connectors, never from line crossings. Overpasses and tunnels that share no connector are excluded. Divided carriageways within 25 m are merged into one junction; ramps are not counted as approaches. Four-way means four distinct approaches.

## Suitability scoring
Candidate locations are grid cells (150 m spacing) that pass hard filters: not within 60 m of a motorway or trunk road, a street within 80 m, not inside an existing building footprint, and not within 100 m of a competitor. Each criterion is normalized with a 5th-95th percentile min-max, direction-adjusted, and combined with weights. The result is a heuristic ranking. It is not a validated prediction of sales or business success.

## Demand proxy
Demand is the residential-like built area reachable within 800 m along the road network. Buildings not positively labelled as non-residential are counted as residential-like. This is a proxy for nearby population, not a population estimate.

## Sensitivity analysis
Rank stability is measured two ways: each weight multiplied by 0.5 and 1.5 in turn, and 400 Monte-Carlo weight vectors drawn from a Dirichlet distribution centred on the base weights. The top-k frequency of a cell is the share of weight vectors in which it stays in the top k.

## Facility location
The p-median model minimizes demand-weighted road distance to the nearest facility; the maximum-coverage model maximizes demand within a service radius. A greedy heuristic, an exact CP-SAT solution and a random baseline are always reported together. Distances are undirected network distances, not travel times.

## Parking analysis
Overture lists few parking facilities for Riyadh. Capacity in the Urbi data is operator-reported and there is no live occupancy. Few mapped parking facilities near a shop cluster do not prove a parking shortage. The platform reports mapped supply, not deficiency, unless demand and occupancy data are provided.

## Machine learning
The land-use classifier predicts residential versus commercial for buildings from footprint shape, neighbourhood density, nearby POIs and roads. Labels come from OSM-derived tags. Evaluation uses 1 km spatial blocks. Scores are not calibrated probabilities.

## Scenarios
Scenario results are simulations. Adding a facility or closing a road changes only a copy of the graph; the canonical data is never modified.
