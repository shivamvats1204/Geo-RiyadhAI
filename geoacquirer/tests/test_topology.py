
from geoacquirer.spatial import build_junctions, find_grade_separated_crossings
from geoacquirer.testing import divided_crossroad, grid_city, interchange


def test_grid_four_way_and_t_junctions():
    seg, conn = grid_city()
    j = build_junctions(seg, conn)
    kinds = j.groupby("kind").size().to_dict()
    assert kinds.get("four_way") == 1
    assert kinds.get("t_junction") == 4
    centre = j[j.kind == "four_way"].iloc[0].geometry
    assert (centre.x, centre.y) == (200.0, 200.0)


def test_overpass_is_not_a_junction_but_is_reported_as_excluded():
    seg, conn = grid_city()
    j = build_junctions(seg, conn)
    # no junction anywhere near the overpass crossings at (200,100) and (400,100)
    for pt in [(200, 100), (400, 100)]:
        assert not any(abs(g.x - pt[0]) < 5 and abs(g.y - pt[1]) < 5 for g in j.geometry)
    gs = find_grade_separated_crossings(seg)
    pts = {(round(p.x), round(p.y)) for p in gs.geometry}
    assert (200, 100) in pts and (400, 100) in pts
    assert gs["flagged_bridge_or_tunnel"].all()


def test_divided_carriageways_merge_into_one_crossroad():
    seg, conn = divided_crossroad()
    j = build_junctions(seg, conn)
    four = j[j.kind == "four_way"]
    assert len(four) == 1, j[["kind", "n_approaches", "n_connectors"]]
    assert four.iloc[0].n_connectors == 4
    assert abs(four.iloc[0].geometry.x) < 1 and abs(four.iloc[0].geometry.y) < 1


def test_ramp_and_motorway_make_highway_interchange_not_crossroad():
    seg, conn = interchange()
    j = build_junctions(seg, conn)
    assert set(j.kind) == {"highway_interchange"}
    assert j.has_ramp.any()
    assert not (j.kind == "four_way").any()


def test_mid_segment_connector_creates_two_arms():
    from geoacquirer.testing import RoadBuilder

    b = RoadBuilder()
    b.node("a", 0, 0).node("b", 200, 0).node("c", 100, 100)
    # a through road with a connector in the middle; side road ends at that connector
    b.road("thru", "a", "b", "residential", "Thru", mid_connectors=[("c", 0.5)])
    b.nodes["c"] = (100, 0)
    b.node("d", 100, 150)
    b.road("side", "c", "d", "residential", "Side")
    seg, conn = b.frames()
    j = build_junctions(seg, conn)
    assert len(j) == 1 and j.iloc[0].kind == "t_junction"


def test_networks_without_junctions_or_crossings_yield_empty_typed_frames():
    from geoacquirer.testing import RoadBuilder
    b = RoadBuilder()
    b.node("a", 0, 0).node("b", 500, 0)
    b.road("only", "a", "b", "residential", "Lonely Rd")
    seg, conn = b.frames()
    j = build_junctions(seg, conn)
    assert len(j) == 0 and {"kind", "n_approaches", "geometry"} <= set(j.columns) and j.crs == seg.crs
    x = find_grade_separated_crossings(seg)
    assert len(x) == 0 and "geometry" in x.columns
    from geoacquirer.engine import DataContext
    from geoacquirer.testing import METRIC
    ctx = DataContext(crs=METRIC)
    ctx.add_roads(seg, conn)  # used to raise ValueError: Unknown column geometry
    assert len(ctx.layers["junction"]) == 0
