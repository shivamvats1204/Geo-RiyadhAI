# GeoAI — SOTA reference list for the project

Compiled 2026-10-19. Every paper below was checked against its arXiv/venue page (title, authors, date) unless marked **[unverified]**. Claims are paraphrased from abstracts; I did not read full texts.

## Context: what the folder contains

- `prompt.txt` is **empty (0 bytes)**, so the actual task is not known. This list is therefore built around what the two notebooks imply: Saudi parcel/zoning data (Suhail), Riyadh parking lots and buildings (2GIS), i.e. **parcel-, building- and POI-level urban analytics in Saudi Arabia**.
- Update this list once the prompt is available.

---

## 1. Orientation: surveys of the field

| Paper | Link | Why it matters |
|---|---|---|
| Hao, Jiang, Zou et al. — *Geospatial Representation Learning: A Survey from Deep Learning to The LLM Era* (2025, v2 Feb 2026) | [arXiv:2505.09651](https://arxiv.org/abs/2505.09651) | Best single taxonomy of data → method → application, including region/urban embeddings. Read first. |
| Yang, Plaza et al. — *Survey of Multimodal Geospatial Foundation Models* (Oct 2025) | [arXiv:2510.22964](https://arxiv.org/abs/2510.22964) | Modality-driven map of GeoFMs (optical, SAR, vision-language). Authors per search snippet. |
| *On the Opportunities and Challenges of Foundation Models for Geospatial AI* (2023) | [arXiv:2304.06798](https://arxiv.org/abs/2304.06798) | The framing paper for "why GeoAI needs multimodal FMs". Authors **[unverified]**. |
| Janowicz et al. — *GeoFM: how will geo-foundation models reshape spatial data science and GeoAI?* (IJGIS 2025) | [Bristol record](https://research-information.bris.ac.uk/en/publications/geofm-how-will-geo-foundation-models-reshape-spatial-data-science/) | Open challenges: scaling laws, truly multimodal GeoFMs. |
| Young, Gabtraupov, Soror et al. — *How Usable Are Geospatial Foundation Models? A Systematic Evaluation of 89 Models* (Aug 2026) | [arXiv:2608.03804](https://arxiv.org/abs/2608.03804) | Practical: nearly a third of GeoFMs ship nothing beyond source code. Use it to pick models you can actually run. |

## 2. Geospatial foundation models and embeddings (Earth observation)

| Paper | Link | Takeaway for the project |
|---|---|---|
| Brown, Kazmierski, Pasquarella et al. — *AlphaEarth Foundations* (Google DeepMind, Jul 2025) | [arXiv:2507.22291](https://arxiv.org/abs/2507.22291) | Free annual 2017–2024 global embedding layers (via Earth Engine). Cheapest strong per-pixel/per-parcel feature for Saudi cities. |
| Jakubik, Yang, Blumenstiel et al. — *TerraMind* (Apr 2025) | [arXiv:2504.11171](https://arxiv.org/abs/2504.11171) | Any-to-any generative multimodal EO model; reports SOTA on PANGAEA. |
| Szwarcman, Roy, Fraccaro et al. — *Prithvi-EO-2.0* (Dec 2024) | [arXiv:2412.02732](https://arxiv.org/abs/2412.02732) | Open 300M/600M multi-temporal model (HLS); +8% over v1 on GEO-Bench. |
| Tseng, Fuller, Reil et al. — *Galileo* (Feb 2025) | [arXiv:2502.09356](https://arxiv.org/abs/2502.09356) | Multimodal (optical, SAR, elevation, weather) with global + local objectives; good for small objects such as buildings. |
| Wang, Xiong, Liu et al. — *Towards a Unified Copernicus Foundation Model for Earth Vision* (Mar 2025) | [arXiv:2503.11849](https://arxiv.org/abs/2503.11849) | Copernicus-FM + 18.7M-image pretraining set + 15-task bench. |
| Zhu, Chen, Zhang et al. — *GlobalBuildingAtlas* (Jun 2025) | [arXiv:2506.04106](https://arxiv.org/abs/2506.04106) | Open global footprints + 3 m heights + LoD1; a licence-clean substitute for scraped building data. |

## 3. Benchmarks — read these before claiming "SOTA"

| Paper | Link | Key finding |
|---|---|---|
| Marsocci, Jia, Le Bellier et al. — *PANGAEA* (Dec 2024) | [arXiv:2412.04204](https://arxiv.org/abs/2412.04204) | Open GeoFMs **do not consistently beat supervised UNet / ViT baselines**, especially with little labelled data. Always include a plain baseline. |
| Simumba, Lehmann, Fraccaro et al. — *GEO-Bench-2* (Nov 2025) | [arXiv:2511.15658](https://arxiv.org/abs/2511.15658) | 19 openly licensed datasets grouped by capability; use to choose a backbone. |
| Krechetova & Kochedykov — *GeoBenchX* (Mar 2025) | [arXiv:2503.18129](https://arxiv.org/abs/2503.18129) | Multistep GIS tool-calling for LLM agents, incl. unsolvable tasks to test refusal. |
| Zhang, Gao, Wei et al. — *GeoAnalystBench* (Sep 2025, Transactions in GIS) | [arXiv:2509.05881](https://arxiv.org/abs/2509.05881) | 50 spatial-analysis workflow/code tasks; site selection and spatial relations are the hardest. |

## 4. Location encoders and urban / POI embeddings

| Paper | Link | Takeaway |
|---|---|---|
| Klemmer, Rolf, Robinson et al. — *SatCLIP* (2023) | [arXiv:2311.17179](https://arxiv.org/abs/2311.17179) | Pretrained (lat, lon) → embedding, trained against Sentinel-2. Drop-in feature for any point/parcel. |
| Vivanco Cepeda, Nayak, Shah — *GeoCLIP* (NeurIPS 2023) | [arXiv:2309.16020](https://arxiv.org/abs/2309.16020) | Random-Fourier-feature GPS encoder aligned to images. |
| Wu, Cao, Wang et al. — *TorchSpatial* (NeurIPS 2024 D&B) | [arXiv:2406.15658](https://arxiv.org/abs/2406.15658) | 15 location encoders in one library + LocBench + geo-bias score. Use to ablate encoders. |
| Wang, Cheng, Law et al. — *CaLLiPer: Multimodal Contrastive Learning of Urban Space Representations from POI Data* (Nov 2024; CEUS 2025) | [arXiv:2411.06229](https://arxiv.org/abs/2411.06229) | Aligns location embeddings with POI text; +5–15% on land-use and socioeconomic mapping (London). The closest match to your POI/parcel data. |
| Holvoet, Blier-Wong, Antonio — *A multi-view contrastive learning framework for spatial embeddings in risk modelling* (Nov 2025) | [arXiv:2511.17954](https://arxiv.org/abs/2511.17954) | Satellite + OSM → embeddings from lat/lon alone; improves GLM/GAM/boosting on real-estate pricing. |
| *Urban Region Embedding via Multi-View Contrastive Prediction* (ReCP, AAAI 2024) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28718) | POI + mobility fusion baseline. |

## 5. Urban spatio-temporal foundation models and urban LLMs

| Paper | Link | Takeaway |
|---|---|---|
| Yuan, Han, Ding et al. — *UrbanDiT: Diffusion Transformers as Open-World Spatiotemporal Foundation Models* (NeurIPS 2025) | [arXiv:2411.12164](https://arxiv.org/abs/2411.12164) | One model for forecasting / imputation / interpolation across cities, with prompts. |
| Chen, Wu, Chen et al. — *UrbanFM* (Feb 2026) | [arXiv:2602.20677](https://arxiv.org/abs/2602.20677) | WorldST (100+ cities) + zero-shot to unseen cities. Directly relevant if you forecast demand in several Saudi cities. |
| Li, Xia, Tang et al. — *UrbanGPT* (KDD 2024) | [arXiv:2403.00813](https://arxiv.org/abs/2403.00813) | ST encoder + instruction tuning; strong zero-shot when labels are scarce. |
| Feng, Liu, Du et al. — *CityGPT* (KDD 2025) | [arXiv:2406.13948](https://arxiv.org/abs/2406.13948) | Urban spatial knowledge injected into smaller LLMs. |
| Manvi, Khanna, Mai et al. — *GeoLLM* (ICLR 2024) | [arXiv:2310.06213](https://arxiv.org/abs/2310.06213) | Prompting LLMs with OSM context extracts usable location knowledge (population, wealth). |

## 6. Parcels, land use, buildings, zoning (Saudi-relevant)

| Paper | Link | Takeaway |
|---|---|---|
| *TerraVision* — vacant ("White Land") detection in Riyadh, *Land* 15(7), 15 Jul 2026 | [MDPI](https://www.mdpi.com/2073-445X/15/7/1271) | Closest direct precedent: semantic segmentation on 0.31–0.34 m imagery, bilingual dashboard. A LinkedIn post by the team says SegFormer won (secondary source). Page returned 403 for me; **metrics unverified**. |
| Bartik, Gupta, Milo — "generative regulatory measurement" of zoning codes (NBER conference paper, Apr 2026) | [PDF](https://conference.nber.org/conf_papers/f233049.pdf) | LLM + retrieval to read zoning codes: 96% binary accuracy, 0.87 correlation on continuous items. Template for turning Suhail-style `maxBuildingHeight` / setback rules into structured features. Exact title and series number **[unverified]**. |
| *Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale* (KDD 2017) | [arXiv:1704.02965](https://arxiv.org/abs/1704.02965) | Classic land-use-from-imagery baseline (Urban Atlas labels). Authors **[unverified]**. |

## 7. Parking and mobility prediction

| Paper | Link | Takeaway |
|---|---|---|
| Zhang, Liu, Liu et al. — *SHARE* (AAAI 2020) | [arXiv:1911.10516](https://arxiv.org/abs/1911.10516) | City-wide availability via hierarchical GCN + RNN; handles lots with no sensor, which matches static 2GIS data. |
| *Predicting vacant parking space availability zone-wisely* (ST-GBGRU) | [arXiv:2205.02113](https://arxiv.org/abs/2205.02113) | Graph-based GRU, zone-level forecasts. |
| *Predicting Parking Lot Availability by Graph-to-Sequence Model* | [arXiv:2206.10160](https://arxiv.org/abs/2206.10160) | Deployed in an app (Santander). |
| *Graph Neural Networks in Intelligent Transportation Systems* (survey) | [arXiv:2401.00713](https://arxiv.org/abs/2401.00713) | Entry point to GNN-for-mobility literature. |

Gap: I found **no Riyadh-specific parking study**, so a static-capacity parking dataset is a genuine novelty, but note that all of the above need *occupancy time series*, which your 2GIS pull does not contain (only capacity, schedule, access).

## 8. Real-estate valuation and site selection

| Paper | Link | Takeaway |
|---|---|---|
| Law, Paige, Russell — *Take a Look Around: Street View and Satellite Images to Estimate House Prices* | [arXiv:1807.07155](https://arxiv.org/abs/1807.07155) | Visual-appeal features in a hedonic model. |
| Nordstrom, Nordstrom, Webb — *Using Images as Covariates: Measuring Curb Appeal with Deep Learning* (2024) | [arXiv:2403.19915](https://arxiv.org/abs/2403.19915) | Image embeddings as hedonic covariates. |
| *Knowledge-driven Site Selection via Urban Knowledge Graph* | [arXiv:2111.00787](https://arxiv.org/abs/2111.00787) | KG-based retail site selection. Authors **[unverified]**. |
| Lan et al. — *Site Selection via Learning Graph Convolutional Neural Networks: Singapore* (Remote Sensing 2022) | [MDPI](https://www.mdpi.com/2072-4292/14/15/3579) | GCN over transit network beats local-feature-only models. |

## 9. Evaluation methodology (do not skip)

| Paper | Link | Takeaway |
|---|---|---|
| Wang, Hopkins, Hallman, Robinson, Hutchinson — *Cross-validation for geospatial data* (TMLR 2023) | [MLAnthology](https://mlanthology.org/tmlr/2023/wang2023tmlr-crossvalidation) | Random CV is biased under spatial autocorrelation and covariate shift; gives a criterion and algorithm. |
| Ratsakatika, Zotta, Keshav et al. — *Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage* (Sep 2026) | [arXiv:2609.28194](https://arxiv.org/abs/2609.28194) | With a 10 km buffer, embedding gains over plain Sentinel features shrink to noise. Directly relevant if you use AlphaEarth/SatCLIP features. |

---

## What this suggests for the build (provisional, pending the real prompt)

1. **Features:** AlphaEarth embeddings (+ SatCLIP/CaLLiPer-style location/POI embeddings) per parcel centroid, joined to building footprints/heights (GlobalBuildingAtlas / Overture / OSM).
2. **Targets:** zoning/land-use class, building coefficient/height, vacancy ("White Land"), parking capacity or demand.
3. **Baselines first:** plain gradient boosting on tabular features and a UNet/SegFormer baseline. PANGAEA shows GeoFMs often fail to beat these.
4. **Validation:** spatial block or buffered CV by neighbourhood, never random splits; parcels next to each other leak.
5. **LLM layer (optional):** a retrieval-augmented zoning-rule reader (Bartik et al. pattern) and a tool-calling GIS agent evaluated on GeoBenchX/GeoAnalystBench-style tasks.

## Data-source caution

The notebooks pull from `tiles.suhail.ai` / `api2.suhail.ai` with no authentication, and from the 2GIS catalog API using a hard-coded key and a spoofed `2GIS/5.0 (Android; Mobile)` user agent. I cannot tell whether you hold a licence for either. Before building anything you plan to publish or ship, check their terms; open alternatives that cover the same ground are OpenStreetMap, Overture Maps, GlobalBuildingAtlas, Google/Microsoft building footprints (check each licence), and the Saudi open-data portals. Also do not commit that API key to a public repo.

## Not found / limits of this search

- No paper found for **UrbanCLIP**, **UrbanVLP** or **UrbanLLaVA** (searches returned nothing); left out rather than guessed.
- **Clay** (open EO model) has no paper I could find; it appears in GEO-Bench-2 rankings only.
- "SOTA" moves quickly; several items above are from 2026. Re-check before citing in a paper.
