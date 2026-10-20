# Junction detection (topology)

Implemented in `src/geoacquirer/spatial/topology.py`; tests in `tests/test_topology.py` and the synthetic benchmark.

1. **Connectivity comes from connectors, not geometry.** Two road lines that cross without sharing a connector (bridge, tunnel, data gap) are *not* joined. `find_grade_separated_crossings` lists them (live Riyadh study area: 969 excluded crossings) so the exclusion is visible.
2. **Arms, not degree.** Each segment contributes one arm per connector end (two if the connector is mid-segment). Arm bearing is measured 15 m along the segment.
3. **Divided carriageways.** Connectors linked by segments ≤ 25 m are unioned into one *cluster*; segments entirely inside the cluster are internal and ignored; remaining arms are merged into *approaches* when their bearings differ < 25° and the road names agree (or are missing).
4. **Ramps.** `subclass == link` / `is_link` arms are never counted as approaches. A cluster that contains a motorway arm, or ramps plus a trunk, is `highway_interchange`, never `four_way`.
5. **Kind** from the approach count: 4 → `four_way`, 3 → `t_junction`, ≥5 → `complex`, else `other`.

Known limitations: staggered crossroads (offset > 25 m) appear as two T-junctions; service roads and driveways are excluded by default (`exclude_minor_service`), which can hide mall entrances; `at` is assumed to be a fraction of the longest line part for MultiLineStrings. Topology is only as good as the source's connectors; there is no ground-truth junction inventory for Riyadh, so correctness is established on synthetic cases (including divided roads, overpasses, ramps) and by the independent distance verifier, not by a Riyadh-wide accuracy figure.
