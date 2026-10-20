"""Synthetic study area with planted structure (used by tests and benchmarks; everything is labelled synthetic).

An 8 x 8 block grid (200 m blocks). Planted facts:
* residential demand hot spot around node (6,6): dense residential buildings, no competitors  -> best supermarket site
* a competitor cluster (3 supermarkets) around node (2,2)
* a motorway 100 m south of the grid (no connectors) and a trunk 100 m east
* two parking facilities, far apart; shops concentrated along column x=3
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, box
import shapely

from geoacquirer.testing import METRIC, RoadBuilder

from .geoacquirer_client import AcquisitionResult
from .store import UrbanStore

ORIGIN = (46.675, 24.71)
N = 8
BLOCK = 200.0
EXT = N * BLOCK


def synthetic_acquisition(seed: int = 3, label_buildings: bool = True) -> tuple[AcquisitionResult, dict]:
    rng = np.random.default_rng(seed)
    tf_to_m = Transformer.from_crs(4326, METRIC, always_xy=True)
    ox, oy = tf_to_m.transform(*ORIGIN)
    tf_ll = Transformer.from_crs(METRIC, 4326, always_xy=True)

    def ll_geom(g):
        def f(a):
            xs, ys = tf_ll.transform(a[:, 0] + ox, a[:, 1] + oy)
            return np.column_stack([xs, ys])
        return shapely.transform(g, f)

    b = RoadBuilder()
    for i in range(N + 1):
        for j in range(N + 1):
            b.node(f"n{i}_{j}", i * BLOCK, j * BLOCK)
    for i in range(N + 1):
        for j in range(N):
            b.road(f"v{i}_{j}", f"n{i}_{j}", f"n{i}_{j+1}", "primary" if i % 4 == 0 else "residential", f"V{i}")
    for j in range(N + 1):
        for i in range(N):
            b.road(f"h{i}_{j}", f"n{i}_{j}", f"n{i+1}_{j}", "primary" if j % 4 == 0 else "residential", f"H{j}")
    b.node("m0", -300, -100).node("m1", EXT + 300, -100)
    b.road("motorway", "m0", "m1", "motorway", "South Motorway")
    b.node("t0", EXT + 100, -300).node("t1", EXT + 100, EXT + 300)
    b.road("trunk", "t0", "t1", "trunk", "East Trunk")
    seg, conn = b.frames()
    roads = gpd.GeoDataFrame({"road_id": seg["id"], "class": seg["class"], "subclass": seg["subclass"], "subtype": "road",
                              "road_name": seg["names"].apply(lambda n: n["primary"] if n else None),
                              "connectors_json": seg["connectors"].apply(lambda c: __import__("json").dumps(c)), "road_flags_json": None},
                             geometry=[ll_geom(g) for g in seg.geometry], crs=4326)
    connectors = gpd.GeoDataFrame({"connector_id": conn["id"]}, geometry=[ll_geom(g) for g in conn.geometry], crs=4326)

    # buildings: residential 20x20 boxes, dense near (6,6)
    blds = []
    def add_b(x, y, sub):
        blds.append({"building_id": f"b{len(blds)}", "name": None, "subtype": sub if label_buildings else None, "class": None, "height": None, "num_floors": None,
                     "source_dataset": "synthetic", "geometry": box(x, y, x + 20, y + 20)})
    for _ in range(900):
        # lanes between streets: keep clear of street centrelines (offset 30..170 inside a block)
        bi, bj = rng.integers(0, N), rng.integers(0, N)
        w = 6 if (bi in (5, 6) and bj in (5, 6)) else 1  # hot spot weighting
        if rng.random() > w / 6 and w == 1 and rng.random() > 0.35:
            continue
        add_b(bi * BLOCK + rng.uniform(30, 150), bj * BLOCK + rng.uniform(30, 150), "residential")
    for _ in range(60):
        bi, bj = rng.integers(0, N), rng.integers(0, N)
        add_b(bi * BLOCK + rng.uniform(30, 150), bj * BLOCK + rng.uniform(30, 150), "commercial")
    buildings = gpd.GeoDataFrame(blds, geometry="geometry", crs=METRIC)
    buildings = gpd.GeoDataFrame(buildings.drop(columns="geometry"), geometry=[ll_geom(g) for g in buildings.geometry], crs=4326)

    # POIs
    rows = []
    def add(kind, path, x, y):
        rows.append({"poi_id": f"p{len(rows)}", "name": f"{kind}-{len(rows)}", "category": path[-1], "category_path": path, "basic_category": None,
                     "confidence": 0.9, "operating_status": None, "address": None, "source_dataset": "synthetic", "geometry": Point(x, y)})
    for cx, cy in [(2 * BLOCK + 40, 2 * BLOCK + 30), (2 * BLOCK - 60, 2 * BLOCK + 80), (2 * BLOCK + 90, 2 * BLOCK - 70)]:
        add("market", ["shopping", "food_and_beverage_store", "grocery_store"], cx, cy)
    for _ in range(180):
        # shops concentrated along the street x = 3 * BLOCK
        add("shop", ["shopping", "specialty_store"], 3 * BLOCK + rng.normal(0, 40), rng.uniform(0, EXT))
    for _ in range(60):
        add("shop", ["shopping", "specialty_store"], rng.uniform(0, EXT), rng.uniform(0, EXT))
    add("parking", ["travel_and_transportation", "parking"], 1 * BLOCK + 20, 1 * BLOCK + 20)
    add("parking", ["travel_and_transportation", "parking"], 7 * BLOCK - 20, 7 * BLOCK - 20)
    add("hospital", ["health_care", "hospital"], 4 * BLOCK + 30, 4 * BLOCK + 30)
    poi = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC)
    poi = gpd.GeoDataFrame(poi.drop(columns="geometry"), geometry=[ll_geom(g) for g in poi.geometry], crs=4326)

    districts = gpd.GeoDataFrame({"district_id": ["d_west", "d_east"], "name": ["West Synthetic", "East Synthetic"], "subtype": ["macrohood", "macrohood"],
                                  "class": "land", "admin_level": 9}, geometry=[ll_geom(box(-50, -50, EXT / 2, EXT + 50)), ll_geom(box(EXT / 2, -50, EXT + 50, EXT + 50))], crs=4326)
    minx, miny = tf_ll.transform(ox - 400, oy - 400)
    maxx, maxy = tf_ll.transform(ox + EXT + 500, oy + EXT + 500)
    bbox = (minx, miny, maxx, maxy)
    prov = {k: {"provider_id": "synthetic", "dataset_version": "synthetic-v1", "license": "n/a (synthetic)", "retrieved_at": "2026-01-01T00:00:00", "from_cache": True}
            for k in ("poi", "road", "connector", "building", "district")}
    acq = AcquisitionResult({"poi": poi, "road": roads, "connector": connectors, "building": buildings, "district": districts}, prov)
    truth = {"hotspot_xy": (6 * BLOCK, 6 * BLOCK), "competitor_xy": (2 * BLOCK, 2 * BLOCK), "bbox": bbox, "origin_metric": (ox, oy)}
    return acq, truth


def synthetic_area(root, name: str = "synthetic", seed: int = 3):
    acq, truth = synthetic_acquisition(seed)
    store = UrbanStore(root)
    ds = store._assemble(name, truth["bbox"], acq)
    return store, ds, truth
