import geopandas as gpd
from shapely.geometry import Point, box

from geoacquirer.agent import RuleCompiler
from geoacquirer.agent.pipeline import needs_for, normalize_parking
from geoacquirer.query_ir import EntityType
from geoacquirer.verification import anchor_check, check_layer, cross_source


def test_airport_query_compiles_to_parking_target_and_airport_bbox():
    q = RuleCompiler().compile("Find parking facilities at King Khalid International Airport. Return names, capacity and access type")
    assert q.primary().entity_type == EntityType.PARKING and q.area.name == "King Khalid International Airport"
    assert q.area.bbox == (46.675, 24.935, 46.735, 24.985)
    assert [n.layer for n in needs_for(q) if n.required] == ["parking"]


def test_urbi_parking_normalization_keeps_attributes_and_names_unnamed_lots():
    g = gpd.GeoDataFrame({"parking_id": ["1", "2"], "address": ["Riyadh, Park International T1", ""], "subtype": ["ground", None], "capacity": [1000, 10],
                          "access": ["public", "residents_only"]}, geometry=[Point(46.7, 24.95), Point(46.71, 24.96)], crs=4326)
    n = normalize_parking(g, "urbi")
    assert list(n.poi_id) == ["1", "2"] and n.name[0].startswith("Riyadh") and n.name[1] == "Parking 2"
    assert n.category_path[0] == ["travel_and_transportation", "parking"] and set(n.source_dataset) == {"urbi"}
    assert list(n.access) == ["public", "residents_only"]


def test_data_checks_flag_outside_invalid_and_duplicates():
    g = gpd.GeoDataFrame({"poi_id": ["a", "a", "b", "c"], "capacity": [1, 2, None, 4]},
                         geometry=[Point(0.5, 0.5), Point(0.6, 0.6), Point(5, 5), box(0, 0, 1, 1).buffer(0)], crs=4326)
    r = check_layer("parking", g, (0, 0, 1, 1), "poi_id", ["capacity"])
    assert r["outside_requested_area"] == 1 and r["outside_ids"] == ["b"] and r["duplicate_ids"] == 2
    assert r["attribute_completeness"]["capacity"] == 0.75 and any("outside" in p for p in r["problems"])


def test_cross_source_and_anchor_checks():
    crs = 32638
    a = gpd.GeoDataFrame({"poi_id": ["u1", "u2"]}, geometry=[Point(46.70, 24.95), Point(46.72, 24.97)], crs=4326)
    b = gpd.GeoDataFrame({"poi_id": ["o1"]}, geometry=[Point(46.7002, 24.9501)], crs=4326)
    r = cross_source(a, b, crs, 100, "urbi", "overture")
    assert r["urbi_with_overture_within_100m"] == 1 and r["share_urbi_confirmed_by_overture"] == 0.5
    an = anchor_check(a, (46.70, 24.95), crs, 1000)
    assert an["within"] == 1 and an["beyond"] == 1
