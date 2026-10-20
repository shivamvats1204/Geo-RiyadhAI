"""Seeded synthetic city (12x12 street grid, 200 m blocks) with analytically known ground truth."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box

from geoacquirer.engine import DataContext
from geoacquirer.testing import METRIC, RoadBuilder

N = 12           # blocks per side
BLOCK = 200.0
EXTENT = N * BLOCK


def build_city(seed: int = 7, n_shops: int = 2500):
    rng = np.random.default_rng(seed)
    b = RoadBuilder()
    for i in range(N + 1):
        for j in range(N + 1):
            b.node(f"n{i}_{j}", i * BLOCK, j * BLOCK)
    for i in range(N + 1):
        for j in range(N):
            b.road(f"v{i}_{j}", f"n{i}_{j}", f"n{i}_{j+1}", "primary" if i % 4 == 0 else "residential", f"Street V{i}")
    for j in range(N + 1):
        for i in range(N):
            b.road(f"h{i}_{j}", f"n{i}_{j}", f"n{i+1}_{j}", "primary" if j % 4 == 0 else "residential", f"Street H{j}")
    # trunk road 100 m north of the grid and a motorway 100 m west; neither connects to the grid
    b.node("t0", -300, EXTENT + 100).node("t1", EXTENT + 300, EXTENT + 100)
    b.road("trunk", "t0", "t1", "trunk", "North Trunk")
    b.node("m0", -100, -300).node("m1", -100, EXTENT + 300)
    b.road("motorway", "m0", "m1", "motorway", "West Motorway")
    seg, conn = b.frames()

    # ---------------- POIs
    rows = []
    cats = {"shop": ["shopping", "specialty_store"], "grocery": ["shopping", "food_and_beverage_store", "grocery_store"],
            "restaurant": ["food_and_drink", "restaurant"], "clinic": ["health_care", "clinic"], "hospital": ["health_care", "hospital"],
            "warehouse": ["services_and_business", "warehouse"], "parking": ["travel_and_transportation", "parking"]}
    pid = 0

    def add(kind, x, y, **kw):
        nonlocal pid
        pid += 1
        rows.append({"poi_id": f"p{pid}", "name": f"{kind}-{pid}", "category": cats[kind][-1], "category_path": cats[kind], "kind": kind,
                     "confidence": 0.9, "source_dataset": "synthetic", "geometry": Point(x, y), **kw})

    for _ in range(n_shops):
        add("shop" if rng.random() < 0.8 else "grocery", rng.uniform(-50, EXTENT + 50), rng.uniform(-50, EXTENT + 50))
    for _ in range(400):
        add("restaurant", rng.uniform(0, EXTENT), rng.uniform(0, EXTENT))
    # clinics & hospitals sit exactly on intersections -> analytic road distances
    for _ in range(60):
        add("clinic", BLOCK * rng.integers(0, N + 1), BLOCK * rng.integers(0, N + 1))
    for _ in range(8):
        add("hospital", BLOCK * rng.integers(0, N + 1), BLOCK * rng.integers(0, N + 1))
    for _ in range(80):
        add("parking", rng.uniform(0, EXTENT), rng.uniform(0, EXTENT))
    for _ in range(120):
        add("warehouse", rng.uniform(0, EXTENT), rng.uniform(0, EXTENT))
    poi = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC)

    # ---------------- parcels: each block is 3x3 lots (corner lots touch two streets, edge lots one, centre none)
    lots = []
    zone_of_block = {}
    for bi in range(N):
        for bj in range(N):
            zone = rng.choice(["residential", "commercial", "industrial"], p=[0.5, 0.35, 0.15])
            zone_of_block[(bi, bj)] = zone
            x0, y0 = bi * BLOCK, bj * BLOCK
            inset, lot = 4.0, (BLOCK - 8.0) / 3
            for a in range(3):
                for c in range(3):
                    lx0, ly0 = x0 + inset + a * lot, y0 + inset + c * lot
                    lots.append({"parcel_id": f"L{bi}_{bj}_{a}{c}", "landuse": zone, "block": f"{bi}_{bj}",
                                 "n_outer_sides": int(a in (0, 2)) + int(c in (0, 2)), "geometry": box(lx0, ly0, lx0 + lot, ly0 + lot)})
    parcels = gpd.GeoDataFrame(lots, geometry="geometry", crs=METRIC)
    return seg, conn, poi, parcels


def make_ctx(seg, conn, poi, parcels=None, buildings=None) -> DataContext:
    ctx = DataContext(crs=METRIC)
    ctx.add_roads(seg, conn)
    ctx.layers["poi"] = poi
    if parcels is not None:
        ctx.layers["parcel"] = parcels
    if buildings is not None:
        ctx.layers["building"] = buildings
    return ctx


def interior_nodes():
    return [Point(i * BLOCK, j * BLOCK) for i in range(1, N) for j in range(1, N)]


def highway_lines():
    from shapely.geometry import LineString
    return [LineString([(-300, EXTENT + 100), (EXTENT + 300, EXTENT + 100)]), LineString([(-100, -300), (-100, EXTENT + 300)])]
