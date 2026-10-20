# How the SOTA references shaped this build (and what they did not)

Full reference list: [`../../SOTA_references.md`](../../SOTA_references.md). This page says, paper by paper, what was actually used. Items marked **not implemented** are candidate upgrades, not features.

| Reference | Use here | Status |
|---|---|---|
| Wang et al., *Cross-validation for geospatial data* (TMLR 2023); Ratsakatika et al. (arXiv:2609.28194) | Spatial 1 km block CV for the land-use model; a random-split run is reported only as a contrast to show leakage inflation (PR-AUC 0.89 random vs 0.80 spatial); tests assert no block appears in both train and test | **implemented** |
| PANGAEA (arXiv:2412.04204): GeoFMs do not reliably beat supervised baselines | Every model result is reported next to simple baselines (majority class, logistic on shape only, a one-line POI rule) and feature-group ablations | **implemented** (as methodology; PANGAEA itself was not run) |
| GeoBenchX (arXiv:2503.18129), GeoAnalystBench (arXiv:2509.05881) | Informed the tool-calling design: typed tools, tasks that must be *refused* when data is missing, unsolvable-task cases in the gold set | **design influence only**; those benchmarks were not run, our gold set is in-house |
| Bartik, Gupta & Milo (NBER 2026) LLM reading of zoning codes | RAG layer cites passages and refuses numeric answers from documents; zoning-field notes flagged non-official | **partial** (no LLM extraction of zoning rules) |
| CaLLiPer (arXiv:2411.06229), SatCLIP (arXiv:2311.17179), AlphaEarth Foundations (arXiv:2507.22291) | Candidate feature sources for the land-use and suitability models (location/POI/EO embeddings) | **not implemented**; the current model uses hand-built neighbourhood features. A natural next experiment: add AlphaEarth/SatCLIP embeddings and compare under the same spatial CV |
| GlobalBuildingAtlas (arXiv:2506.04106) | Alternative open footprint/height source to replace scraped data | **not integrated** (Overture buildings used) |
| UrbanGPT (arXiv:2403.00813), UrbanDiT (arXiv:2411.12164), UrbanFM (arXiv:2602.20677) | Would apply if occupancy/traffic time series existed | **not implemented**: no spatio-temporal target data is available (parking has no occupancy) |
| SHARE (arXiv:1911.10516) and parking-availability GNNs | Same: needs availability time series | **not implemented** |
| Knowledge-driven site selection / GCN site selection (arXiv:2111.00787; Lan et al. 2022) | Motivated using road-network accessibility (reachable demand within 800 m) rather than straight-line counts | **implemented as a heuristic**, no learned GNN |
| Law, Paige & Russell (arXiv:1807.07155) | Hedonic/visual-feature valuation | **not implemented** (no price data) |

Honest summary: the build adopts the *evaluation discipline* of the literature (spatial validation, baselines, ablations, refusal on missing evidence, calibration reporting) rather than its newest models. The newest models need labels or time series this data set does not have.
