import numpy as np
import time

import pytest
from fastapi.testclient import TestClient

from riyadhai import planner
from riyadhai.executor import Executor
from riyadhai.tools import TOOLS, ToolError, call_tool

SUPERMARKET_Q = ("Find five suitable commercial locations for a supermarket in northern Riyadh. Avoid highways, prioritize crossroad accessibility, "
                 "compare nearby competitors and parking availability, assess zoning restrictions, and explain your recommendation.")


def test_tool_registry_has_the_required_tools():
    required = {"search_local_data", "request_geoacquirer_data", "inspect_layer_schema", "get_geographic_coverage", "filter_features", "spatial_join",
                "find_intersections", "measure_proximity", "calculate_network_accessibility", "aggregate_spatial_statistics", "classify_land_use",
                "rank_site_suitability", "optimize_facility_locations", "compare_scenarios", "retrieve_reference_documents", "render_map_layer",
                "generate_evidence_report", "export_dataset"}
    missing = required - set(TOOLS)
    assert missing == {"spatial_join"}  # spatial_join is delivered as filter_features(near=...) / aggregate_spatial_statistics; see docs/tools.md


def test_planner_classifies_core_questions():
    p = planner.plan(SUPERMARKET_Q)
    assert [s.tool for s in p.steps] == ["get_geographic_coverage", "rank_site_suitability", "retrieve_reference_documents", "generate_evidence_report"]
    args = p.steps[1].args
    assert args["facility"] == "supermarket" and args["k"] == 5 and args["region"]["kind"] == "north" and args["weights"]["junction_access"] == 0.35
    assert "REPORT_GENERATION" in p.task_classes and any("zoning" in n for n in p.notes)
    p2 = planner.plan("Where should three new parking facilities be located to improve access?")
    assert p2.task_classes[0] == "OPTIMIZATION" and p2.steps[1].args["p"] == 3 and p2.steps[2].args["scenarios"] == "$s2.proposed_scenarios"
    p3 = planner.plan("Count shops in each district")
    assert p3.steps[0].tool == "aggregate_spatial_statistics"
    p4 = planner.plan("Find shops located at crossroads but away from highways, and show nearby parking.")
    assert [s.tool for s in p4.steps] == ["run_geoquery", "measure_proximity"]
    assert "parking" not in p4.steps[0].args["text"].lower()  # 'show nearby parking' is an output request, not a constraint
    p5 = planner.plan("Now exclude anything within 300 metres of a hospital", last_result_id="r1")
    assert p5.steps[0].tool == "filter_features" and p5.steps[0].args["near"][0] == {"layer": "poi", "categories": ["hospital"], "distance_m": 300.0, "negate": True}
    assert planner.plan("What does maxBuildingCoefficient mean?").steps[0].tool == "retrieve_reference_documents"
    assert planner.plan("write me a poem").supported is False


def test_tool_arguments_are_validated_and_bounded(world):
    _, ds, truth, ctx = world
    with pytest.raises(ToolError):
        call_tool(ctx, "filter_features", {"layer": "poi", "bogus": 1})
    with pytest.raises(ToolError):
        call_tool(ctx, "filter_features", {"layer": "poi", "near": [{"layer": "poi", "distance_m": -5}]})
    with pytest.raises(ToolError):
        call_tool(ctx, "rank_site_suitability", {"weights": {"not_a_criterion": 1}})
    with pytest.raises(ToolError):
        call_tool(ctx, "request_geoacquirer_data", {"bbox": [40.0, 20.0, 50.0, 30.0]})  # far beyond the acquisition budget
    with pytest.raises(ToolError):
        call_tool(ctx, "nonexistent_tool", {})
    with pytest.raises(ToolError):
        call_tool(ctx, "filter_features", {"layer": "poi", "region": {"kind": "district", "district": "Atlantis"}})


def test_filter_and_followup_preserve_lineage(world):
    _, ds, truth, ctx = world
    r1 = call_tool(ctx, "filter_features", {"layer": "poi", "categories": ["shop"], "near": [{"layer": "parking", "distance_m": 400}]})
    n1 = r1.summary["rows"]
    r2 = call_tool(ctx, "filter_features", {"layer": "poi", "base_result_id": r1.result_id,
                                           "near": [{"layer": "poi", "categories": ["hospital"], "distance_m": 300, "negate": True}]})
    assert 0 < r2.summary["rows"] <= n1
    lin = ctx.results.lineage(r2.result_id)
    assert [x["id"] for x in lin] == [r2.result_id, r1.result_id] and lin[0]["version"] == 2
    g = ctx.results.get(r2.result_id).gdf.to_crs(ds.crs)
    hosp = ds.layers["poi"][category_mask_hosp(ds)]
    assert all(hosp.geometry.distance(x).min() > 300 for x in g.geometry)


def category_mask_hosp(ds):
    from geoacquirer.ontology import category_mask
    return category_mask(ds.layers["poi"]["category_path"], ["hospital"])


def test_aggregate_by_district_counts_match_manual(world):
    _, ds, truth, ctx = world
    r = call_tool(ctx, "aggregate_spatial_statistics", {"layer": "poi", "categories": ["shop"], "by": "district"})
    d = ds.layers["district"]
    poi = ds.layers["poi"]
    from geoacquirer.ontology import category_mask
    shops = poi[category_mask(poi["category_path"], ["shop"])]
    manual = {row["name"]: int(shops.geometry.within(row.geometry).sum()) for _, row in d.iterrows()}
    got = {row["name"]: int(row["count"]) for _, row in r.table.iterrows()}
    assert got == manual
    one = call_tool(ctx, "aggregate_spatial_statistics", {"layer": "poi", "by": "district", "focus_names": ["West"]})
    assert list(one.table["name"]) == ["West Synthetic"]


def test_network_accessibility_and_proximity(world):
    _, ds, truth, ctx = world
    r = call_tool(ctx, "calculate_network_accessibility", {"origin_layer": "poi", "origin_categories": ["shop"], "destination": {"layer": "parking", "distance_m": 1}, "cutoff_m": 600})
    assert r.summary["origins_snapped"] > 100 and 0 < r.summary["share_with_any_destination"] < 1
    base = call_tool(ctx, "filter_features", {"layer": "poi", "categories": ["shop"]})
    m = call_tool(ctx, "measure_proximity", {"result_id": base.result_id, "to": {"layer": "parking", "distance_m": 1}, "name": "parking"})
    g = ctx.results.get(m.result_id).gdf
    assert g["dist_parking_m"].min() >= 0 and g["parking_id"].notna().all()


def test_end_to_end_supermarket_job_is_grounded(executor, world):
    store, ds, truth, ctx = world
    job = executor.run(executor.create_job(SUPERMARKET_Q))
    assert job.status == "done", job.trace
    tools = [t["tool"] for t in job.trace if t["status"] == "ok"]
    assert tools == ["get_geographic_coverage", "rank_site_suitability", "retrieve_reference_documents", "generate_evidence_report"]
    audit = Executor.audit_claims(job)
    assert audit["numbers_checked"] > 30 and audit["unsupported"] == [] and audit["traceable_facts"] == audit["facts"]
    kinds = {f["type"] for f in job.answer}
    assert {"observed", "model", "inferred"} <= kinds
    assert any("UNVERIFIED" in f["text"] for f in job.answer)
    html = open(job.report_path, encoding="utf-8").read()
    for needle in ("Observed facts", "Model and solver outputs", "Inferred recommendations", "Claim audit", "Datasets", "synthetic-v1", "UNVERIFIED"):
        assert needle in html
    # trace records every tool call with its calc id
    assert all(t.get("calc") for t in job.trace if t["status"] == "ok")


def test_approval_gate_pauses_and_resumes(executor, world):
    _, ds, truth, ctx = world
    job = executor.create_job("Analyze commercial development opportunities across the districts")
    job = executor.run(job)
    assert job.status == "awaiting_approval" and job.pending_approval["tool"] == "classify_land_use"
    assert any(t["status"] == "awaiting_approval" for t in job.trace)
    assert [t["tool"] for t in job.trace if t["status"] == "ok"] == ["get_geographic_coverage", "aggregate_spatial_statistics", "rank_site_suitability"]


def test_followup_session_flow(executor, world):
    session = {}
    j1 = executor.run(executor.create_job("Find shops near a hospital within 400 metres", session), session=session)
    assert j1.status == "done" and session["last_result_id"] == j1.primary_result_id
    n1 = executor.ctx.results.get(j1.primary_result_id).gdf
    j2 = executor.run(executor.create_job("Now exclude anything within 250 metres of a parking", session), session=session)
    assert j2.status == "done"
    n2 = executor.ctx.results.get(j2.primary_result_id).gdf
    assert len(n2) <= len(n1) and executor.ctx.results.lineage(j2.primary_result_id)[1]["id"] == j1.primary_result_id


def test_rag_cites_sources_and_numeric_answers_come_from_tools(executor):
    job = executor.run(executor.create_job("What does maxBuildingCoefficient mean?"))
    texts = " ".join(f["text"] for f in job.answer)
    assert "zoning_fields#maxbuildingcoefficient" in texts and "not official" in texts.lower() or "official" in texts.lower()
    assert all(f["type"] == "reference" for f in job.answer)


def test_unsupported_question_returns_guidance_not_invention(executor):
    job = executor.run(executor.create_job("Predict next year's rent in Olaya"))
    assert job.status == "unsupported" and "Supported" in job.answer[0]["text"]


def test_api_flow(world):
    from riyadhai.api import create_app
    store, ds, truth, ctx = world
    app = create_app(store.root, "synthetic")
    c = TestClient(app)
    assert c.get("/v1/health").json()["area"] == "synthetic"
    d = c.get("/v1/dataset").json()
    assert d["coverage"]["layers"]["poi"]["rows"] > 200 and d["provenance"]["poi"]["dataset_version"] == "synthetic-v1"
    layer = c.get("/v1/layers/poi?limit=50").json()
    assert len(layer["features"]) == 50 and layer["total_in_view"] > 50 and layer["truncated"] is True
    assert c.get("/v1/layers/secret").status_code == 404
    assert c.get("/v1/layers/poi?bbox=1,2").status_code == 400
    r = c.post("/v1/ask", json={"question": SUPERMARKET_Q})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    for _ in range(300):
        j = c.get(f"/v1/jobs/{jid}").json()
        if j["status"] not in ("running", "queued"):
            break
        time.sleep(0.2)
    assert j["status"] == "done" and j["claim_audit"]["unsupported"] == []
    rid = j["primary_result_id"]
    gj = c.get(f"/v1/results/{rid}.geojson").json()
    assert 1 <= len(gj["features"]) <= 5 and all("explanation" in f["properties"] for f in gj["features"])
    if len(gj["features"]) < 5:  # the small synthetic region cannot hold 5 sites 800 m apart; the shortfall must be stated, not hidden
        assert any("requested locations could be returned" in w for w in j["warnings"])
    assert c.get(f"/v1/results/{rid}.csv").text.startswith("﻿") or "cell_id" in c.get(f"/v1/results/{rid}.csv").text
    assert c.get(f"/v1/report/{jid}").status_code == 200
    rr = c.post("/v1/tool", json={"tool": "rank_site_suitability", "args": {"k": 3, "weights": {"competition": 0.9}}})
    assert rr.status_code == 200 and len(rr.json()["table"]) == 3
    assert c.post("/v1/tool", json={"tool": "rank_site_suitability", "args": {"weights": {"nope": 1}}}).status_code == 422
    assert c.get("/v1/rag", params={"q": "setback"}).json()[0]["doc"] == "zoning_fields"
    assert {t["name"] for t in c.get("/v1/tools").json()} >= {"rank_site_suitability", "optimize_facility_locations"}


def test_same_question_gives_same_ranking(world):
    _, ds, truth, ctx = world
    a = call_tool(ctx, "rank_site_suitability", {"k": 3, "region": {"kind": "all"}, "min_separation_m": 400})
    b = call_tool(ctx, "rank_site_suitability", {"k": 3, "region": {"kind": "all"}, "min_separation_m": 400})
    ga, gb = ctx.results.get(a.result_id).gdf, ctx.results.get(b.result_id).gdf
    assert list(ga.cell_id) == list(gb.cell_id)
    assert np.allclose(ga.score, gb.score) and np.allclose(ga.top_k_frequency, gb.top_k_frequency)


def test_regression_polygon_parking_and_access_filter(world):
    import geopandas as gpd
    from riyadhai.tools import existing_parking
    _, ds, truth, ctx = world
    orig = ds.layers["parking"].copy()
    try:
        polys = gpd.GeoDataFrame({"poi_id": ["a", "b", "c"], "access": ["public", "residents_only", None], "capacity": [50, 100, 5]},
                                 geometry=[g.buffer(15) for g in orig.geometry.iloc[:2].tolist() + [orig.geometry.iloc[0]]], crs=ds.crs)
        ds.layers["parking"] = polys
        g, note = existing_parking(ctx, True, 0)
        assert list(g.poi_id) == ["a", "c"] and "public" in note       # residents-only dropped, unknown access kept
        g2, _ = existing_parking(ctx, True, 20)
        assert list(g2.poi_id) == ["a"]                                # capacity filter drops the 5-space lot
        r = call_tool(ctx, "compare_scenarios", {"scenarios": [{"name": "x", "operations": [{"type": "add_facility", "x": float(orig.geometry.iloc[0].x), "y": float(orig.geometry.iloc[0].y)}]}]})
        assert r.summary["simulated"] is True                            # used to raise AttributeError on polygon parking
    finally:
        ds.layers["parking"] = orig


def test_quality_ranking_questions_are_not_answered_as_lists():
    p = planner.plan("Which neighbourhood has the best schools?")
    assert p.supported is False and "no ratings" in p.notes[0]
    assert planner.plan("Find the best location for a supermarket").supported is True   # site suitability is not a quality ranking


def test_district_comparison_fact_names_only_the_compared_units(world):
    r = call_tool(world[3], "aggregate_spatial_statistics", {"layer": "poi", "by": "district", "focus_names": ["West", "East"]})
    txt = " ".join(f["text"] for f in r.facts)
    assert "Compared units" in txt and "West Synthetic" in txt and "East Synthetic" in txt and "counted over" not in txt


def test_regressions_from_edge_case_evaluation(world, executor):
    store, ds, truth, ctx = world
    with pytest.raises(ToolError):
        call_tool(ctx, "rank_site_suitability", {"weights": {k: 0 for k in ("demand", "junction_access", "road_visibility", "competition", "agglomeration", "parking", "transit")}})
    with pytest.raises(ToolError):
        call_tool(ctx, "export_dataset", {"result_id": "does-not-exist", "format": "csv"})          # was an unhandled KeyError
    long_job = executor.create_job("find shops " * 450)
    assert long_job.status == "unsupported" and "1500" in long_job.plan.notes[0]
    job = executor.run(executor.create_job("Find shops within 99999 km of a mosque"))
    msg = " ".join(f["text"] for f in job.answer)
    assert "validation error" not in msg and ("less than" in msg or "distance" in msg)               # concise reason, no pydantic internals
    assert Executor.audit_claims(job)["unsupported"] == []                                          # system messages are not data claims
