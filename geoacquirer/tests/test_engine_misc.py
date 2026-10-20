import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point, box

from geoacquirer.engine import DataContext, Evaluator
from geoacquirer.engine.evaluator import Code, and_combine, near_codes, not_codes, or_combine
from geoacquirer.entity_resolution import (match_pois, name_similarity, normalize_name, resolve_building_to_parcel,
                                           resolve_poi_to_building)
from geoacquirer.knowledge_graph import SpatialKG
from geoacquirer.query_ir import (And, AreaSpec, EntitySpec, EntityType, GeoQuery, Not, ObjectSpec, Op, Or, Predicate)
from geoacquirer.temporal import classify_change, coverage_caveat
from geoacquirer.testing import METRIC, grid_city


def test_kleene_style_combination_never_turns_unknown_into_truth():
    P, PN, U, FN, F = (np.array([c], dtype=np.int8) for c in (Code.PASS, Code.PASS_NEAR, Code.UNKNOWN, Code.FAIL_NEAR, Code.FAIL))
    assert and_combine([P, U])[0] == Code.UNKNOWN
    assert and_combine([P, PN])[0] == Code.PASS_NEAR
    assert and_combine([F, U])[0] == Code.FAIL
    assert and_combine([P, FN])[0] == Code.FAIL_NEAR
    assert or_combine([F, U])[0] == Code.UNKNOWN
    assert or_combine([F, P])[0] == Code.PASS
    assert not_codes(U)[0] == Code.UNKNOWN and not_codes(PN)[0] == Code.FAIL_NEAR and not_codes(F)[0] == Code.PASS


def test_near_codes_bands():
    d = np.array([5.0, 25.0, 30.0, 35.0, 45.0, np.inf])
    c = near_codes(d, 30.0, np.full(6, 10.0))
    assert c.tolist() == [Code.PASS, Code.PASS_NEAR, Code.PASS_NEAR, Code.FAIL_NEAR, Code.FAIL, Code.FAIL]
    assert near_codes(d, 30.0, np.zeros(6)).tolist() == [Code.PASS, Code.PASS, Code.PASS, Code.FAIL, Code.FAIL, Code.FAIL]


def _ctx_with_pois(points, cats):
    seg, conn = grid_city()
    ctx = DataContext(crs=METRIC)
    ctx.add_roads(seg, conn)
    ctx.layers["poi"] = gpd.GeoDataFrame({"poi_id": list("abcdefgh")[:len(points)], "name": list("ABCDEFGH")[:len(points)],
                                          "category_path": cats}, geometry=[Point(p) for p in points], crs=METRIC)
    return ctx


def _q(expr, cats=("shop",)):
    return GeoQuery(raw_text="t", targets=[EntitySpec(entity_type=EntityType.POI, categories=list(cats))], area=AreaSpec(name="x"),
                    constraint=expr, position_uncertainty_m=0.0)


def test_or_and_not_over_real_geometry():
    ctx = _ctx_with_pois([(205, 205), (405, 205), (300, 300)], [["shopping"]] * 3)
    four = Predicate(op=Op.AT_INTERSECTION, distance_m=30, junction_kind="four_way")
    tj = Predicate(op=Op.AT_INTERSECTION, distance_m=30, junction_kind="t_junction")
    res = Evaluator(ctx, _q(Or(children=[four, tj]))).run()
    assert res.status.tolist() == ["VERIFIED_MATCH", "VERIFIED_MATCH", "REJECTED"]
    res2 = Evaluator(ctx, _q(Not(child=four))).run()
    assert res2.status.tolist() == ["REJECTED", "VERIFIED_MATCH", "VERIFIED_MATCH"]


def test_missing_layers_make_constraints_unverifiable_not_false():
    ctx = _ctx_with_pois([(205, 205)], [["shopping"]])
    inside_parcel = Predicate(op=Op.INSIDE_PARCEL, object=ObjectSpec(entity_type=EntityType.PARCEL))
    res = Evaluator(ctx, _q(inside_parcel)).run()
    assert res.status.iloc[0] == "INSUFFICIENT_EVIDENCE"
    assert "parcel" in json.loads(res.evidence_json.iloc[0])[0]["reason"]
    # a failed AND member still rejects even if another member is unverifiable
    far = Predicate(op=Op.AT_INTERSECTION, distance_m=1, junction_kind="four_way")
    res2 = Evaluator(ctx, _q(And(children=[inside_parcel, far]))).run()
    assert res2.status.iloc[0] == "REJECTED"
    # centroid-only parcels cannot support polygon containment
    ctx.layers["parcel"] = gpd.GeoDataFrame({"parcel_id": ["p"]}, geometry=[Point(205, 205)], crs=METRIC)
    assert Evaluator(ctx, _q(inside_parcel)).run().status.iloc[0] == "INSUFFICIENT_EVIDENCE"


def test_changed_between_requires_two_snapshots():
    ctx = _ctx_with_pois([(205, 205)], [["shopping"]])
    p = Predicate(op=Op.CHANGED_BETWEEN, snapshots=("2026-01", "2026-09"))
    assert Evaluator(ctx, _q(p)).run().status.iloc[0] == "INSUFFICIENT_EVIDENCE"
    old = ctx.layers["poi"].iloc[:0].copy()
    ctx.snapshots = {"2026-01": old, "2026-09": ctx.layers["poi"]}
    assert Evaluator(ctx, _q(p)).run().status.iloc[0] == "VERIFIED_MATCH"  # POI absent in the old snapshot -> added
    assert coverage_caveat(old, ctx.layers["poi"])["old_count"] == 0


def test_frontage_counts_distinct_roads_and_distance_by_road():
    seg, conn = grid_city()
    ctx = DataContext(crs=METRIC)
    ctx.add_roads(seg, conn)
    ctx.layers["building"] = gpd.GeoDataFrame({"building_id": ["corner", "mid"]},
                                              geometry=[box(190, 190, 210, 210), box(100, 20, 120, 40)], crs=METRIC)
    ctx.layers["poi"] = gpd.GeoDataFrame({"poi_id": ["x"], "name": ["X"], "category_path": [["shopping"]]}, geometry=[Point(300, 200)], crs=METRIC)
    q = GeoQuery(raw_text="t", targets=[EntitySpec(entity_type=EntityType.BUILDING)], area=AreaSpec(name="x"),
                 constraint=Predicate(op=Op.HAS_FRONTAGE_ON, distance_m=15, min_count=2, object=ObjectSpec(entity_type=EntityType.ROAD)))
    res = Evaluator(ctx, q).run()
    assert dict(zip(res.building_id, res.status)) == {"corner": "VERIFIED_MATCH", "mid": "REJECTED"}
    byroad = Predicate(op=Op.DISTANCE_BY_ROAD, distance_m=150, object=ObjectSpec(entity_type=EntityType.POI, categories=["shop"]))
    q2 = GeoQuery(raw_text="t", targets=[EntitySpec(entity_type=EntityType.BUILDING)], area=AreaSpec(name="x"), constraint=byroad, position_uncertainty_m=0)
    r2 = Evaluator(ctx, q2).run()
    # corner building snaps to node (200,200); shop snaps to (200,200) via the nearest node at 100 m -> within 150 m by road
    assert r2.set_index("building_id").loc["corner", "status"] == "VERIFIED_MATCH"


def test_entity_resolution_names_arabic_and_matching():
    assert normalize_name("شركة  أحمد") == normalize_name("احمد")
    assert name_similarity("Panda Supermarket", "panda super market") > 0.6
    assert name_similarity("Panda", "Danube") < 0.5
    a = gpd.GeoDataFrame({"poi_id": ["a1", "a2", "a3"], "name": ["Panda", "Danube", "Zed"], "category_path": [["shopping"]] * 3},
                         geometry=[Point(0, 0), Point(100, 0), Point(500, 500)], crs=METRIC)
    b = gpd.GeoDataFrame({"poi_id": ["b1", "b2"], "name": ["panda", "Danube Mart"], "category_path": [["shopping"]] * 2},
                         geometry=[Point(8, 3), Point(104, 2)], crs=METRIC)
    m = match_pois(a, b)
    assert dict(zip(m.a_id, m.b_id)) == {"a1": "b1", "a2": "b2"}
    assert set(m.match_type) == {"proximity_and_name"}


def test_nameless_pois_are_never_name_matches():
    assert normalize_name(float("nan")) == "" and normalize_name(None) == "" and normalize_name(pd.NA) == ""
    assert name_similarity(float("nan"), float("nan")) == 0.0
    a = gpd.GeoDataFrame({"poi_id": ["a"], "name": [np.nan], "category_path": [["shopping", "x"]]}, geometry=[Point(0, 0)], crs=METRIC)
    b = gpd.GeoDataFrame({"poi_id": ["b"], "name": [np.nan], "category_path": [["shopping", "y"]]}, geometry=[Point(5, 0)], crs=METRIC)
    assert match_pois(a, b).empty  # no shared name, no shared specific category


def test_poi_to_building_and_parcel_centroid_vs_polygon():
    poi = gpd.GeoDataFrame({"x": [1, 2, 3]}, geometry=[Point(5, 5), Point(25, 5), Point(500, 5)], crs=METRIC)
    bld = gpd.GeoDataFrame({"building_id": ["B"]}, geometry=[box(0, 0, 10, 10)], crs=METRIC)
    r = resolve_poi_to_building(poi, bld)
    assert r.building_match.tolist() == ["contained", "nearest_within_20m", "none"]
    assert r.building_distance_m.iloc[1] == pytest.approx(15.0)
    assert resolve_poi_to_building(poi, None).building_match.unique().tolist() == ["layer_unavailable"]
    # parcels: polygon overlap is strong evidence, centroid-in-building is explicitly weaker
    parcels = gpd.GeoDataFrame({"parcel_id": ["P1", "P2"]}, geometry=[box(-5, -5, 12, 12), box(100, 100, 110, 110)], crs=METRIC)
    pr = resolve_building_to_parcel(bld, parcels)
    assert pr.parcel_id.iloc[0] == "P1" and pr.parcel_match.iloc[0] == "polygon_overlap" and pr.parcel_overlap_fraction.iloc[0] == pytest.approx(1.0)
    cent = gpd.GeoDataFrame({"parcel_id": ["C1"]}, geometry=[Point(5, 5)], crs=METRIC)
    assert resolve_building_to_parcel(bld, cent).parcel_match.iloc[0] == "parcel_centroid_in_building"
    assert resolve_building_to_parcel(bld, None).parcel_match.iloc[0] == "layer_unavailable"


def test_temporal_change_classification():
    old = gpd.GeoDataFrame({"poi_id": ["a", "b"], "name": ["A", "B"], "category": ["x", "x"], "operating_status": [None, None]},
                           geometry=[Point(0, 0), Point(1, 1)], crs=METRIC)
    new = gpd.GeoDataFrame({"poi_id": ["a", "b", "c"], "name": ["A", "B2", "C"], "category": ["x", "x", "y"], "operating_status": [None] * 3},
                           geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs=METRIC)
    ch = classify_change(old, new, new)
    assert ch.change_type.tolist() == ["unchanged", "attribute_changed:name", "added"]
    assert coverage_caveat(old, new)["suspicious_coverage_shift"] is False  # 2 -> 3 is exactly 1.5x: not above the threshold
    assert coverage_caveat(old, new.iloc[:0])["suspicious_coverage_shift"] is True  # everything vanished: coverage, not demolition


def test_knowledge_graph_explains_edges():
    kg = SpatialKG()
    kg.add_node("s1", "poi", "Shop")
    kg.add_edge("s1", "LOCATED_IN", "b1", {"match": "contained"}, {"source": "overture"})
    kg.add_edge("s1", "AT_JUNCTION", "j1", {"distance_m": 12.0})
    assert kg.explain("s1") == ["s1 -[LOCATED_IN match=contained]-> b1", "s1 -[AT_JUNCTION distance_m=12.0]-> j1"]
    assert kg.find("AT_JUNCTION", "poi") == ["s1"]
    assert kg.neighbors("b1")[0]["provenance"] == {"source": "overture"}


def test_api_end_to_end_with_local_fixture(tmp_path):
    import time
    from geoacquirer.agent import GeoAcquirerAgent
    from geoacquirer.api import create_app
    from geoacquirer.storage import LayerCache
    from geoacquirer.testing import FIXTURE_ORIGIN, write_fixture_files

    write_fixture_files(tmp_path / "data")
    app = create_app(tmp_path / "runs", agent_factory=lambda: GeoAcquirerAgent(tmp_path / "runs", local_dir=tmp_path / "data", cache=LayerCache(tmp_path / "c")))
    c = TestClient(app)
    assert c.get("/v1/health").json()["ok"] is True
    srcs = c.get("/v1/sources").json()
    assert srcs["suhail"]["enabled"] is False and srcs["overture"]["enabled"] is True
    bbox = [FIXTURE_ORIGIN[0] - 0.004, FIXTURE_ORIGIN[1] - 0.004, FIXTURE_ORIGIN[0] + 0.012, FIXTURE_ORIGIN[1] + 0.012]
    r = c.post("/v1/queries", json={"text": "Find shops at crossroads but not on highways in Riyadh", "bbox": bbox})
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(100):
        st = c.get(f"/v1/queries/{jid}").json()
        if st["state"] != "running":
            break
        time.sleep(0.2)
    assert st["state"] == "done", st
    assert st["counts"]["VERIFIED_MATCH"] == 1 and "EXPORT" in st["stages_completed"]
    gj = c.get(f"/v1/queries/{jid}/files/results.geojson")
    assert gj.status_code == 200 and gj.json()["features"]
    assert c.get(f"/v1/queries/{jid}/files/state.json").status_code == 404  # only exported files are served
    ev = c.get(f"/v1/queries/{jid}/evidence/A").json()
    assert ev["status"] == "VERIFIED_MATCH" and any(e["predicate"].startswith("AT_INTERSECTION") for e in ev["evidence"])
    assert c.get("/v1/queries/nope").status_code == 404
    assert c.post("/v1/queries", json={"text": "x"}).status_code == 422
