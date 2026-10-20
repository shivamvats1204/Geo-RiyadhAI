"""Synthetic road/POI fixtures with known ground truth (used by tests and benchmarks, offline)."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Point, box

METRIC = 32638  # UTM 38N (Riyadh)


class RoadBuilder:
    """Builds Overture-style segments + connectors from named nodes in metric coordinates."""

    def __init__(self):
        self.nodes: dict[str, tuple[float, float]] = {}
        self.segs: list[dict] = []

    def node(self, nid: str, x: float, y: float):
        self.nodes[nid] = (x, y)
        return self

    def road(self, sid, a, b, cls="residential", name=None, subclass=None, flags=None, via=None, mid_connectors=()):
        pts = [self.nodes[a]] + list(via or []) + [self.nodes[b]]
        line = LineString(pts)
        conns = [{"connector_id": a, "at": 0.0}]
        for cid, frac in mid_connectors:
            conns.append({"connector_id": cid, "at": frac})
        conns.append({"connector_id": b, "at": 1.0})
        self.segs.append(
            {
                "id": sid, "subtype": "road", "class": cls, "subclass": subclass,
                "connectors": conns, "road_flags": flags,
                "names": {"primary": name} if name else None, "geometry": line,
            }
        )
        return self

    def frames(self):
        seg = gpd.GeoDataFrame(self.segs, geometry="geometry", crs=METRIC)
        used = {c["connector_id"] for s in self.segs for c in s["connectors"]}
        conn = gpd.GeoDataFrame(
            [{"id": n, "geometry": Point(*xy)} for n, xy in self.nodes.items() if n in used],
            geometry="geometry", crs=METRIC,
        )
        return seg, conn


def grid_city() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """A 3x3 street grid, 200 m blocks, plus a motorway (north) and an overpass with no connector.

    Ground truth: node C (centre, 4 roads) is a four-way; B and D are T-junctions (3 arms);
    the overpass crossing at (400, 100) is grade-separated and must NOT be a junction.
    """
    b = RoadBuilder()
    # vertical street x=0,200,400 ; horizontal street y=0,200,400
    for i, x in enumerate((0, 200, 400)):
        for j, y in enumerate((0, 200, 400)):
            b.node(f"n{i}{j}", x, y)
    names_v = {0: "West St", 1: "Centre Ave", 2: "East St"}
    names_h = {0: "South St", 1: "Main St", 2: "North St"}
    for i in range(3):
        for j in range(2):
            b.road(f"v{i}{j}", f"n{i}{j}", f"n{i}{j+1}", "residential", names_v[i])
    for j in range(3):
        for i in range(2):
            b.road(f"h{i}{j}", f"n{i}{j}", f"n{i+1}{j}", "residential", names_h[j])
    # motorway running east-west at y=900 (no connectors to the grid)
    b.node("m0", -200, 900).node("m1", 800, 900)
    b.road("mw", "m0", "m1", "motorway", "King Fahd Rd")
    # overpass: a tertiary road crossing a residential street without sharing a connector
    b.node("o0", 300, -100).node("o1", 300, 100)  # crosses h10? no: crosses nothing; add explicit cross
    b.node("p0", 100, 100).node("p1", 500, 100)
    b.road("ovp", "p0", "p1", "tertiary", "Overpass Rd", flags=[{"values": ["is_bridge"], "between": None}])
    # ensure ovp geometrically crosses the vertical street v10 (x=200, y 0..200)
    return b.frames()


def _fixture_roads() -> RoadBuilder:
    """grid_city plus a second four-way (600,840) sitting 60 m south of the motorway at y=900."""
    b = RoadBuilder()
    for i, x in enumerate((0, 200, 400)):
        for j, y in enumerate((0, 200, 400)):
            b.node(f"n{i}{j}", x, y)
    nv = {0: "West St", 1: "Centre Ave", 2: "East St"}
    nh = {0: "South St", 1: "Main St", 2: "North St"}
    for i in range(3):
        for j in range(2):
            b.road(f"v{i}{j}", f"n{i}{j}", f"n{i}{j+1}", "residential", nv[i])
    for j in range(3):
        for i in range(2):
            b.road(f"h{i}{j}", f"n{i}{j}", f"n{i+1}{j}", "residential", nh[j])
    b.node("m0", -200, 900).node("m1", 800, 900)
    b.road("mw", "m0", "m1", "motorway", "King Fahd Rd")
    b.node("p0", 100, 100).node("p1", 500, 100)
    b.road("ovp", "p0", "p1", "tertiary", "Overpass Rd", flags=[{"values": ["is_bridge"], "between": None}])
    b.node("x0", 500, 840).node("x1", 700, 840).node("x2", 600, 740).node("x3", 600, 940).node("xc", 600, 840)
    b.road("e1", "x0", "xc", "secondary", "Service Rd").road("e2", "xc", "x1", "secondary", "Service Rd")
    b.road("s1", "x2", "xc", "secondary", "Cross St").road("s2", "xc", "x3", "secondary", "Cross St")
    return b


# (id, x, y, category_path, expected_status) in metric offsets from the fixture origin
FIXTURE_SHOPS = [
    ("A", 210, 205, ["shopping", "specialty_store"], "VERIFIED_MATCH"),
    ("B", 225, 225, ["shopping", "specialty_store"], "INSUFFICIENT_EVIDENCE"),
    ("C", 216, 216, ["shopping", "specialty_store"], "PROBABLE_MATCH"),
    ("D", 300, 300, ["shopping", "specialty_store"], "REJECTED"),
    ("E", 405, 203, ["shopping", "specialty_store"], "REJECTED"),  # at a T-junction only
    ("F", 610, 845, ["shopping", "food_and_beverage_store", "grocery_store"], "INSUFFICIENT_EVIDENCE"),  # highway 55 m: inside uncertainty band
    ("G", 606, 858, ["shopping", "specialty_store"], "REJECTED"),  # crossroad ok, highway 42 m -> excluded
    ("R", 211, 206, ["food_and_drink", "restaurant"], "NOT_A_SHOP"),
]
FIXTURE_ORIGIN = (46.675, 24.71)  # lon, lat -> (0,0) of the metric fixture


def fixture_layers() -> dict:
    """Synthetic Overture-schema layers (WGS84) with known ground truth, for offline pipeline tests."""
    import json as _json
    from pyproj import Transformer

    tf_to_m = Transformer.from_crs(4326, METRIC, always_xy=True)
    ox, oy = tf_to_m.transform(*FIXTURE_ORIGIN)
    tf_to_ll = Transformer.from_crs(METRIC, 4326, always_xy=True)

    def ll(x, y):
        return Point(*tf_to_ll.transform(ox + x, oy + y))

    import shapely
    import numpy as np

    def geom_ll(g):
        def f(arr):
            xs, ys = tf_to_ll.transform(arr[:, 0] + ox, arr[:, 1] + oy)
            return np.column_stack([xs, ys])
        return shapely.transform(g, f)

    b = _fixture_roads()
    seg, conn = b.frames()
    roads = gpd.GeoDataFrame({
        "road_id": seg["id"], "class": seg["class"], "subclass": seg["subclass"], "subtype": "road",
        "road_name": seg["names"].apply(lambda n: n["primary"] if n else None),
        "connectors_json": seg["connectors"].apply(_json.dumps),
        "road_flags_json": seg["road_flags"].apply(lambda f: _json.dumps(f) if f else None),
    }, geometry=[geom_ll(g) for g in seg.geometry], crs=4326)
    connectors = gpd.GeoDataFrame({"connector_id": conn["id"]}, geometry=[geom_ll(g) for g in conn.geometry], crs=4326)
    poi = gpd.GeoDataFrame({
        "poi_id": [s[0] for s in FIXTURE_SHOPS], "name": [f"Shop {s[0]}" for s in FIXTURE_SHOPS],
        "category": [s[3][-1] for s in FIXTURE_SHOPS], "category_path": [s[3] for s in FIXTURE_SHOPS],
        "basic_category": "x", "confidence": 0.9, "operating_status": None, "address": None, "source_dataset": "synthetic",
    }, geometry=[ll(s[1], s[2]) for s in FIXTURE_SHOPS], crs=4326)
    buildings = gpd.GeoDataFrame({"building_id": ["BLD1", "BLD2"], "name": None, "subtype": None, "class": None,
                                  "height": None, "num_floors": None, "source_dataset": "synthetic"},
                                 geometry=[geom_ll(box(205, 200, 220, 214)), geom_ll(box(590, 835, 640, 880))], crs=4326)
    return {"poi": poi, "road": roads, "connector": connectors, "building": buildings,
            "truth": {s[0]: s[4] for s in FIXTURE_SHOPS if s[4] != "NOT_A_SHOP"}}


def write_fixture_files(directory) -> dict:
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    L = fixture_layers()
    for k in ("poi", "road", "connector", "building"):
        L[k].to_parquet(d / f"{k}.parquet")
    return L


def divided_crossroad() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Two divided roads crossing: 4 raw 3/4-way nodes ~14 m apart that are ONE crossroad."""
    b = RoadBuilder()
    # East-west dual carriageway at y=+-7, north-south dual carriageway at x=+-7
    b.node("W1", -300, 7).node("W2", -300, -7).node("E1", 300, 7).node("E2", 300, -7)
    b.node("N1", -7, 300).node("N2", 7, 300).node("S1", -7, -300).node("S2", 7, -300)
    b.node("a", -7, 7).node("b", 7, 7).node("c", 7, -7).node("d", -7, -7)
    # eastbound (y=-7?) keep simple: segments from outer node to inner nodes and between inner nodes
    b.road("w_in", "W1", "a", "primary", "Olaya St").road("w_in2", "W2", "d", "primary", "Olaya St")
    b.road("e_in", "b", "E1", "primary", "Olaya St").road("e_in2", "c", "E2", "primary", "Olaya St")
    b.road("n_in", "a", "N1", "primary", "Prince St").road("n_in2", "b", "N2", "primary", "Prince St")
    b.road("s_in", "d", "S1", "primary", "Prince St").road("s_in2", "c", "S2", "primary", "Prince St")
    # short internal links between the four inner nodes (14 m)
    b.road("ab", "a", "b", "primary", "Olaya St").road("bc", "b", "c", "primary", "Prince St")
    b.road("cd", "c", "d", "primary", "Olaya St").road("da", "d", "a", "primary", "Prince St")
    return b.frames()


def interchange() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """A motorway with an off-ramp merging into a local road: must be a highway_interchange."""
    b = RoadBuilder()
    b.node("m0", -400, 0).node("jx", 0, 0).node("m1", 400, 0)
    b.node("r0", 0, 100).node("l0", -200, 100).node("l1", 200, 100)
    b.road("mw1", "m0", "jx", "motorway", "Ring Rd").road("mw2", "jx", "m1", "motorway", "Ring Rd")
    b.road("ramp", "jx", "r0", "motorway", subclass="link", name=None)
    b.road("loc1", "l0", "r0", "secondary", "Exit St").road("loc2", "r0", "l1", "secondary", "Exit St")
    return b.frames()
