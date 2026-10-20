import numpy as np
import pytest
from shapely.geometry import Point

from riyadhai import features as F
from riyadhai import optimization as OPT
from riyadhai import simulation as SIM
from riyadhai import suitability as SUIT
from riyadhai.testing import BLOCK
from geoacquirer.ontology import category_mask


def test_store_roundtrip_and_topology(world, tmp_path):
    store, ds, truth, ctx = world
    cov = ds.coverage()
    assert cov["layers"]["poi"]["rows"] > 200 and cov["junction_kinds"]["four_way"] == 49  # 7x7 interior nodes
    assert cov["road_network"]["components"] >= 3  # grid + motorway + trunk are disconnected
    store2 = type(store)(store.root)
    ds2 = store2.load_area("synthetic")
    assert ds2.coverage()["junction_kinds"] == cov["junction_kinds"]
    assert set(ds2.provenance) >= {"poi", "road", "building"}


def test_candidates_respect_hard_filters(world):
    _, ds, truth, ctx = world
    cfg = F.FeatureConfig(grid_step_m=100)
    cells, counts = F.candidate_cells(ds, cfg)
    road = ds.layers["road"]
    hw = road[road["class"].isin(["motorway", "trunk"])]
    d = np.array([hw.geometry.distance(g).min() for g in cells.geometry])
    assert (d > 60).all() and counts["removed_near_highway"] > 0 and counts["eligible_cells"] == len(cells)
    occ = ds.layers["building"]
    assert not any(occ.geometry.contains(g).any() for g in cells.geometry[:200])


def test_features_match_independent_computation(world):
    _, ds, truth, ctx = world
    cfg = F.FeatureConfig(grid_step_m=200)
    cells, _ = F.candidate_cells(ds, cfg)
    f = F.compute_features(ds, cells, cfg)
    poi = ds.layers["poi"]
    comp = poi[category_mask(poi["category_path"], ["supermarket"])]
    i = 17
    g = cells.geometry.iloc[i]
    assert f.competitors_500m.iloc[i] == int((comp.geometry.distance(g) <= 500).sum())
    four = ds.layers["junction"].query("kind == 'four_way'")
    assert f.dist_to_four_way_m.iloc[i] == pytest.approx(four.geometry.distance(g).min(), abs=1e-6)
    assert f.network_demand_800m.notna().mean() > 0.95


def test_suitability_finds_planted_hotspot_and_avoids_competitors(world):
    _, ds, truth, ctx = world
    cfg = F.FeatureConfig(grid_step_m=100)
    cells, _ = F.candidate_cells(ds, cfg)
    f = F.compute_features(ds, cells, cfg)
    xy = cells.assign(x=cells.geometry.x, y=cells.geometry.y)[["x", "y"]].set_index(f.index)
    res = SUIT.rank_sites(f, xy, k=3, n_mc=60)
    hx, hy = truth["hotspot_xy"]
    cx, cy = truth["competitor_xy"]
    ox, oy = truth["origin_metric"]
    best = res.top.iloc[0]
    bxy = xy.loc[res.top.index[0]]
    assert np.hypot(bxy.x - (ox + hx), bxy.y - (oy + hy)) < 450, "top site should be near the planted demand hot spot"
    # competitor cluster is excluded or ranked far below
    near_comp = [i for i in res.top.index if np.hypot(xy.at[i, "x"] - (ox + cx), xy.at[i, "y"] - (oy + cy)) < 300]
    assert not near_comp
    assert best["score"] == res.ranked["score"].max()
    assert 0 <= res.top["top_k_frequency"].min() <= res.top["top_k_frequency"].max() <= 1
    assert "contributes" in res.top.iloc[0]["explanation"] and res.sensitivity["monte_carlo"]["n"] == 60
    # spatial separation between recommendations
    pts = [Point(xy.at[i, "x"], xy.at[i, "y"]) for i in res.top.index]
    assert all(a.distance(b) >= 800 for k, a in enumerate(pts) for b in pts[k + 1:])
    assert any("not validated" in n for n in res.notes) and any("UNVERIFIED" in n for n in res.notes)


def test_filters_report_and_unverified_zoning(world):
    _, ds, truth, ctx = world
    cfg = F.FeatureConfig(grid_step_m=200)
    cells, _ = F.candidate_cells(ds, cfg)
    f = F.compute_features(ds, cells, cfg)
    ok, rep = SUIT.apply_filters(f, SUIT.HardFilters(allowed_landuse=["commercial"]))
    assert "NOT APPLIED" in rep["landuse_filter"] and rep["eligible"] == int(ok.sum())
    assert (f["landuse_zoning"] == "UNVERIFIED").all()
    ok2, rep2 = SUIT.apply_filters(f, SUIT.HardFilters(min_competitor_distance_m=500))
    assert rep2["eligible"] <= rep["eligible"]


def test_weights_validated():
    with pytest.raises(ValueError):
        SUIT.score(__import__("pandas").DataFrame({c[0]: [1.0, 2.0] for c in SUIT.CRITERIA.values()}), {k: 0 for k in SUIT.CRITERIA})


def _problem(ds, p_demand_centers):
    import geopandas as gpd
    pts = []
    rng = np.random.default_rng(0)
    for (cx, cy), n in p_demand_centers:
        for _ in range(n):
            pts.append(Point(cx + rng.normal(0, 25), cy + rng.normal(0, 25)))
    demand = gpd.GeoDataFrame(geometry=pts, crs=ds.crs)
    cfg = F.FeatureConfig(grid_step_m=100, exclude_occupied=False)
    cells, _ = F.candidate_cells(ds, cfg)
    return demand, cells


def test_optimization_beats_baselines_and_matches_known_optimum(world):
    _, ds, truth, ctx = world
    ox, oy = truth["origin_metric"]
    demand, cells = _problem(ds, [((ox + 2 * BLOCK, oy + 6 * BLOCK), 60), ((ox + 6 * BLOCK, oy + 2 * BLOCK), 60)])
    prob = OPT.build_problem(ds, demand, None, cells, None, radius_m=300)
    for obj in ("p_median", "max_coverage"):
        res = OPT.optimize(prob, 2, obj, time_limit_s=20, n_random=60)
        sel = cells[cells.cell_id.isin(res["cpsat"]["selected"])]
        centers = [Point(ox + 2 * BLOCK, oy + 6 * BLOCK), Point(ox + 6 * BLOCK, oy + 2 * BLOCK)]
        for c in centers:  # one chosen site within 250 m of each demand cluster
            assert sel.geometry.distance(c).min() < 250, (obj, res["cpsat"])
        assert res["cpsat"]["mean_distance_m"] <= res["greedy"]["mean_distance_m"] + 1e-6
        assert res["cpsat"]["mean_distance_m"] < res["random_baseline"]["mean_distance_m_mean"]
        assert res["cpsat"]["mean_distance_m"] < res["before"]["mean_distance_m"]
        assert res["cpsat"]["coverage_within_radius"] >= res["before"]["coverage_within_radius"]
        assert len(res["cpsat"]["selected"]) <= 2 and res["assumptions"]


def test_cpsat_optimal_objective_equals_bruteforce_on_small_problem(world):
    import itertools
    _, ds, truth, ctx = world
    ox, oy = truth["origin_metric"]
    demand, cells = _problem(ds, [((ox + 2 * BLOCK, oy + 6 * BLOCK), 25), ((ox + 6 * BLOCK, oy + 2 * BLOCK), 25), ((ox + 4 * BLOCK, oy + 4 * BLOCK), 20)])
    prob = OPT.prune_candidates(OPT.build_problem(ds, demand, None, cells, None, radius_m=300), 40)
    sol, info = OPT.solve_cpsat(prob, 2, "p_median", 30)
    best = min(prob.objective(list(c)) for c in itertools.combinations(range(len(prob.cand_ids)), 2))
    assert info["status"] == "OPTIMAL"
    assert prob.objective(sol) == pytest.approx(best, abs=0.5)


def test_scenarios_are_simulated_and_directionally_correct(world):
    _, ds, truth, ctx = world
    poi = ds.layers["poi"]
    demand = poi[category_mask(poi["category_path"], ["shop"])]
    from pyproj import Transformer
    ox, oy = truth["origin_metric"]
    to_ll = Transformer.from_crs(ds.crs, 4326, always_xy=True)
    lon, lat = to_ll.transform(ox + 3 * BLOCK, oy + 4 * BLOCK)
    s_add = SIM.Scenario("add near shop street", [{"type": "add_facility", "lon": lon, "lat": lat}])
    r = SIM.run_scenario(ds, s_add, demand, radius_m=400)
    assert r["simulated"] is True and r["delta"]["mean_distance_m"] < 0 and r["delta"]["coverage_within_radius"] >= 0
    assert r["after"]["facilities"] == r["before"]["facilities"] + 1
    # closing a central connector must not improve access
    con = [c for c in ds.network().graph.nodes if ds.network().graph.degree[c] == 4][len(ds.network().graph) // 40]
    r2 = SIM.run_scenario(ds, SIM.Scenario("close", [{"type": "remove_connector", "connector_id": con}]), demand, radius_m=400)
    assert r2["delta"]["mean_distance_m"] >= 0 and "closed connector" in r2["applied"][0]
    # baseline untouched afterwards
    assert ds.network().graph.has_node(con)
    with pytest.raises(ValueError):
        SIM.Scenario("bad", [{"type": "teleport"}]).validate()
    rows = SIM.compare_scenarios([r, r2])
    assert rows[0]["scenario"] == "add near shop street" and all(x["simulated"] for x in rows)


def test_swap_local_search_never_worse_than_greedy_and_cpsat_never_worse_than_swap(world):
    _, ds, truth, ctx = world
    ox, oy = truth["origin_metric"]
    demand, cells = _problem(ds, [((ox + 2 * BLOCK, oy + 6 * BLOCK), 40), ((ox + 6 * BLOCK, oy + 2 * BLOCK), 40), ((ox + 4 * BLOCK, oy + 4 * BLOCK), 30)])
    prob = OPT.prune_candidates(OPT.build_problem(ds, demand, None, cells, None, radius_m=300), 60)
    for obj in ("p_median", "max_coverage"):
        g = OPT.greedy(prob, 3, obj)
        sw = OPT.swap_local_search(prob, g, obj)
        key = (lambda c: prob.objective(c)) if obj == "p_median" else (lambda c: -prob.coverage(c))
        assert key(sw) <= key(g) + 1e-9 and len(set(sw)) == 3
        res = OPT.optimize(prob, 3, obj, time_limit_s=10, n_random=20)
        assert res["cpsat"]["mean_distance_m"] <= res["greedy_swap"]["mean_distance_m"] + 1e-6 or obj == "max_coverage"


def test_scenario_with_no_demand_is_refused_not_a_division_by_zero(world):
    _, ds, _, _ = world
    poi = ds.layers["poi"]
    empty = poi.iloc[:0]
    with pytest.raises(ValueError, match="no demand points"):
        SIM.run_scenario(ds, SIM.Scenario("nothing", []), empty, radius_m=400)


def test_selected_scope_limits_the_analysis_region(world):
    from shapely.geometry import box
    from riyadhai.tools import RegionSpec, resolve_region
    _, ds, _, _ = world
    full, label_full = resolve_region(ds, RegionSpec())
    minx, miny, maxx, maxy = full.bounds
    ds.scope = {"name": "Test District", "uid": "t", "polygon": box(minx, miny, minx + (maxx - minx) / 3, maxy)}
    try:
        part, label = resolve_region(ds, RegionSpec())
        assert part.area < full.area * 0.5 and label == "Test District"
        assert ds.full_polygon().area == pytest.approx(full.area)  # the prepared extent itself is unchanged
    finally:
        ds.scope = None


def test_plain_summary_for_site_ranking_uses_only_computed_numbers():
    from types import SimpleNamespace
    from riyadhai import plain
    ds = SimpleNamespace(layers={}, crs=32638, scope=None, scope_label=lambda: "Test District")
    ctx = SimpleNamespace(area=ds, results=None)
    job = {"status": "done", "plan": {"steps": [{"tool": "rank_site_suitability", "args": {"facility": "pharmacy"}}]},
           "results": {"s1": {"tool": "rank_site_suitability", "result_id": "x", "summary": {}}},
           "tables": {"s1": [{"rank": 1, "cell": "c1", "score": 0.81, "top5_frequency": 0.9, "competitors_500m": 0, "dist_four_way_m": 20.0, "retail_500m": 12, "parking_300m": 3, "dist_highway_m": 900.0}]},
           "answer": [{"type": "observed", "text": "Data gaps: no population"}, {"type": "observed", "text": "Candidate screening: 100 grid cells, 10 scored."}]}
    jobobj = SimpleNamespace(results={"s1": SimpleNamespace(result_id="x")})
    p = plain.summarize(job, jobobj, ctx)
    assert "pharmacy" in p["headline"] and "81 out of 100" in p["headline"]
    assert p["confidence"]["level"] == "high" and p["cards"][1]["value"] == "10"
    assert any("no other pharmacy" in t for t in p["points"]) and p["limits"]
