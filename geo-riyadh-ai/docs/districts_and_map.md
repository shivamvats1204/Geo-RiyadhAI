# All-Riyadh district map and the merged analysis app

`http://127.0.0.1:8080/` is one workspace: the district map and the analyst together. (Use `127.0.0.1`; on this Windows machine `localhost` adds a 2 s IPv6 delay to every request.)

## How the city is divided

| Source | What | Count |
|---|---|---|
| Urbi/2GIS catalogue (`adm_div`) | districts / living areas / settlements / district areas, each with polygon, centroid and English + Arabic name | 325 / 13 / 80 / 18 |
| Suhail vector tiles (`neighborhoods` layer) | neighbourhoods with Arabic names (248 polygons + 1 centroid-only) | 249 |
| Overture `division_area` | macrohood / neighbourhood / microhood polygons | 182 |

Total 867 areas. 18 of the Urbi areas are not referenced by any downloaded record; they were found by scanning the catalogue on a 2.5 km grid. A polygon is drawn when the source has one; otherwise a centroid dot (1 area).

Every feature of every layer (1,146,585 buildings, 1,261,358 parcels, 40,977 parking, 62,489 streets, 10,039 attractions, 2,754 metro) is assigned to an Urbi district, a Suhail neighbourhood and an Overture neighbourhood by point-in-polygon (smallest polygon wins). A feature up to 300 m outside every polygon is attached to the nearest; further away it stays **unassigned** and is shown grey (162,449 buildings, mostly the outskirts, are outside all Urbi district polygons). Nothing is dropped: row counts in equal row counts out for every layer.

Cross-check: Urbi's own recorded district name equals the spatially assigned district for 99.8 % of buildings (970,742 / 972,887), 99.99 % of attractions and 98.5 % of metro records.

## Map data
Vector tiles are pre-generated (`scripts/pregen_tiles.py`) at zoom 14 (buildings, parcels, plus detail for streets/parking/attractions/metro) and zoom 12-13 for the overview, then served as static files. A polygon smaller than 4 tile units² is drawn as a centroid dot instead of disappearing. Tile accounting (input = polygons + points + centroid dots + edge-touching replicas clipped away): buildings 1,240,802 = 1,238,911 + 1,819 + 72; parcels 1,387,283 = 1,384,750 + 1,900 + 633.

## Verification at random places (`scripts/verify_city_spots.py`, results in `benchmarks/results/city_spot_verification.json`)
Eight random 300 m boxes (4 random buildings, 4 random points inside random districts):
* **Buildings vs the raw downloaded records**: every raw record is in the store at every spot; every stored building is present in the served tiles (0 missing).
* **Parcels vs a fresh download of the Suhail tile**: identical where the store has parcels (76/76, 127/127, 75/75). **Gap found:** at 2 of 8 spots (far east edge, Al Uthmaniyah) Suhail has parcels (1 and 15) that the store does not: the Suhail crawl did not cover the outer edge of the city.
* **Overture footprints (a different dataset)**: 81/83, 119/120 and 55/67 Urbi buildings in dense districts overlap an Overture building (about 80-99 %); lower in sparse outskirts. These are different datasets, so equality is not expected.
* In the browser, "Check this view" compares what MapLibre drew with what the store holds for the viewport (Olaya example: 66/66 buildings, 104/104 parcels, 8/8 parking, 15/15 streets, 0 missing, 0 extra).

## Merged app
* **Where** replaces the old "Area" box: choose a district or neighbourhood (search, English + Arabic names), or use the map view. All answers are limited to that polygon. There is no fixed study area.
* If the chosen place is inside already-prepared data, it is ready instantly. Otherwise the app downloads open map data for it in the background (first time only, roughly 5-10 minutes; not exercised end to end in the evaluation). Areas larger than 0.1 deg² must be narrowed.
* **Layers** are a compact bar above the map (chips) that expands into toggles for boundaries and data layers.
* **Answers** are in plain language: a one-sentence headline, big-number cards, short bullets, "how sure are we" (from how stable the ranking is across 400 what-if weightings), "good to know" limits, how to read the table, and follow-up buttons. The technical facts with calculation ids are under "Technical details". The wording is generated only from computed numbers (`src/riyadhai/plain.py`).
* Question types with tailored plain summaries: best site for a facility, counts by district, where to add parking, rule-based place search, explain a term, what-if. Others get a generic summary.
