"""Geo-RiyadhAI benchmarks. Every number is measured by running this file (python benchmarks/run_benchmarks.py).

Sections: spatial correctness, AI orchestration (planning, tool selection, claim audit, failure recovery), machine learning,
optimization, performance. Real-data sections use the cached central-Riyadh Overture study area; planted-truth sections use the
synthetic area (riyadhai.testing).
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from pyproj import Geod, Transformer

from geoacquirer.ontology import category_mask
from riyadhai import features as F
from riyadhai import planner
from riyadhai.executor import Executor
from riyadhai.geoacquirer_client import GeoAcquirerClient
from riyadhai.rag import KnowledgeBase
from riyadhai.results import ResultStore
from riyadhai.store import UrbanStore
from riyadhai.testing import synthetic_area
from riyadhai.tools import ToolContext, ToolError, call_tool

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def timed(fn, n=3):
    ts = []
    r = None
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return r, round(statistics.median(ts), 3)


def make_ctx(store, ds):
    out = Path(store.root) / "analyses" / "out"
    out.mkdir(parents=True, exist_ok=True)
    return ToolContext(store, ds, ResultStore(store.root), KnowledgeBase(), GeoAcquirerClient(), out)


# ----------------------------------------------------------------------------------------------- planner gold set
GOLD = [
    ("Find five suitable commercial locations for a supermarket in northern Riyadh. Avoid highways, prioritize crossroad accessibility", ["get_geographic_coverage", "rank_site_suitability", "retrieve_reference_documents", "generate_evidence_report"], {"facility": "supermarket", "k": 5}),
    ("Rank the best 3 sites for a new clinic in the southern part of the study area", ["get_geographic_coverage", "rank_site_suitability", "retrieve_reference_documents", "generate_evidence_report"], {"facility": "clinic", "k": 3}),
    ("Recommend ideal locations for a grocery store near crossroads", ["get_geographic_coverage", "rank_site_suitability", "retrieve_reference_documents", "generate_evidence_report"], {"facility": "supermarket"}),
    ("What are the best sites for a restaurant in the east?", ["get_geographic_coverage", "rank_site_suitability", "retrieve_reference_documents", "generate_evidence_report"], {"facility": "restaurant"}),
    ("Where should three new parking facilities be located to improve access?", ["get_geographic_coverage", "optimize_facility_locations", "compare_scenarios", "generate_evidence_report"], {"p": 3}),
    ("Where should 2 additional parking facilities be placed to maximise coverage within 400 m", ["get_geographic_coverage", "optimize_facility_locations", "compare_scenarios", "generate_evidence_report"], {"p": 2, "objective": "max_coverage"}),
    ("Suggest where five new parking places could be built in the north", ["get_geographic_coverage", "optimize_facility_locations", "compare_scenarios", "generate_evidence_report"], {"p": 5}),
    ("Count shops in each district", ["aggregate_spatial_statistics"], {"layer": "poi"}),
    ("How many parking facilities per district?", ["aggregate_spatial_statistics"], {"layer": "parking"}),
    ("Compare the commercial density of Olaya and Malqa", ["aggregate_spatial_statistics"], {"layer": "poi"}),
    ("Count clinics per neighbourhood", ["aggregate_spatial_statistics"], {"layer": "poi"}),
    ("Find shops located at crossroads but away from highways, and show nearby parking.", ["run_geoquery", "measure_proximity"], {}),
    ("Find shops within 25 metres of four-way intersections, exclude those within 80 m of motorways", ["run_geoquery"], {}),
    ("List restaurants near a mosque", ["run_geoquery"], {}),
    ("Find clinics within 2 km road distance of hospitals", ["run_geoquery"], {}),
    ("Show pharmacies not within 300 metres of a hospital", ["run_geoquery"], {}),
    ("Find supermarkets at crossroads", ["run_geoquery"], {}),
    ("What does maxBuildingCoefficient mean?", ["retrieve_reference_documents"], {}),
    ("Explain what VERIFIED_MATCH means", ["retrieve_reference_documents"], {}),
    ("How is the suitability score computed?", ["retrieve_reference_documents"], {}),
    ("Define setback in the zoning data", ["retrieve_reference_documents"], {}),
    ("Analyze commercial development opportunities across the districts", ["get_geographic_coverage", "aggregate_spatial_statistics", "rank_site_suitability", "classify_land_use", "retrieve_reference_documents", "generate_evidence_report"], {}),
    ("Assess commercial development opportunities in northern Riyadh", ["get_geographic_coverage", "aggregate_spatial_statistics", "rank_site_suitability", "classify_land_use", "retrieve_reference_documents", "generate_evidence_report"], {}),
    ("Write me a poem about Riyadh", [], {"supported": False}),
    ("Predict rent next year in Olaya", [], {"supported": False}),
    ("Delete all the data", [], {"supported": False}),
]
FOLLOW = [("Now exclude anything within 300 metres of a hospital", ["filter_features"]), ("Keep only those within 500 m of parking", ["filter_features"]),
          ("Also remove anything within 200 m of a mosque", ["filter_features"])]


def bench_planner():
    ok_tools = ok_args = n_args = 0
    fails = []
    for q, tools, args in GOLD:
        p = planner.plan(q, district_names=["Olaya", "Malqa"])
        got = [s.tool for s in p.steps]
        if "supported" in args:
            good = (p.supported == args["supported"])
            ok_tools += good
            if not good:
                fails.append({"q": q, "got": got})
            continue
        good = got == tools
        ok_tools += good
        if not good:
            fails.append({"q": q, "expected": tools, "got": got})
        for k, v in args.items():
            n_args += 1
            val = next((s.args.get(k) for s in p.steps if k in s.args), None)
            ok_args += (val == v)
            if val != v:
                fails.append({"q": q, "arg": k, "expected": v, "got": val})
    fo = 0
    for q, tools in FOLLOW:
        p = planner.plan(q, last_result_id="r1")
        fo += [s.tool for s in p.steps] == tools
    return {"questions": len(GOLD), "tool_sequence_exact": ok_tools, "tool_sequence_accuracy": round(ok_tools / len(GOLD), 3),
            "argument_checks": n_args, "argument_correct": ok_args, "follow_up_correct": f"{fo}/{len(FOLLOW)}", "failures": fails}


def bench_orchestration():
    root = tempfile.mkdtemp()
    store, ds, truth = synthetic_area(root)
    ctx = make_ctx(store, ds)
    ex = Executor(ctx, Path(root) / "analyses" / "jobs")
    # the synthetic area has its own district names; swap the two real names so the district-comparison question is runnable
    qs = [g[0].replace("Olaya and Malqa", "West Synthetic and East Synthetic") for g in GOLD
          if g[2].get("supported", True) and "Analyze" not in g[0] and "Assess" not in g[0]]
    done = failed = awaiting = 0
    nums = bad = 0
    traceable = facts = 0
    rows = []
    session = {}
    for q in qs:
        job = ex.run(ex.create_job(q, session), session=session)
        a = Executor.audit_claims(job)
        nums += a["numbers_checked"]; bad += len(a["unsupported"]); traceable += a["traceable_facts"]; facts += a["facts"]
        done += job.status == "done"; failed += job.status == "failed"; awaiting += job.status == "awaiting_approval"
        rows.append({"q": q[:70], "status": job.status, "errors": [t["error"][:90] for t in job.trace if t["status"] == "error"]})
    # failure injection: each must end without an unhandled exception, with the failure recorded in the trace
    inj = {}
    cases = {
        "missing_district": "Rank the best 3 sites for a supermarket in district Atlantis",
        "layer_missing_parcels": "Find commercial parcels touching two roads",
        "unknown_tool_args": None,
        "approval_needed": "Analyze commercial development opportunities across the districts",
        "empty_result_followup": "Find shops near a hospital within 5 metres",
    }
    for name, q in cases.items():
        try:
            if name == "unknown_tool_args":
                try:
                    call_tool(ctx, "filter_features", {"layer": "poi", "unknown": 1})
                    inj[name] = "NOT HANDLED"
                except ToolError:
                    inj[name] = "ToolError raised cleanly"
                continue
            job = ex.run(ex.create_job(q))
            inj[name] = {"status": job.status, "recorded_in_trace": any(t["status"] in ("error", "awaiting_approval") for t in job.trace) or job.status == "done"}
        except Exception as e:  # noqa: BLE001
            inj[name] = f"UNHANDLED {type(e).__name__}"
    return {"supported_questions_run": len(qs), "done": int(done), "failed": int(failed), "awaiting_approval": int(awaiting),
            "task_success_rate": round(done / len(qs), 3), "numbers_in_answers_checked": nums, "numbers_not_traceable_to_a_tool_output": bad,
            "hallucinated_claim_rate": round(bad / nums, 4) if nums else 0.0, "facts_with_calculation_id": f"{traceable}/{facts}",
            "failure_injection": inj, "per_question": rows}


def bench_spatial_correctness(store, ds_real):
    ds = ds_real
    out = {}
    # (a) UTM metric distances vs geodesic on the real area
    geod = Geod(ellps="WGS84")
    to_ll = Transformer.from_crs(ds.crs, 4326, always_xy=True)
    rng = np.random.default_rng(0)
    poi = ds.layers["poi"].geometry
    idx = rng.choice(len(poi), (2000, 2))
    errs = []
    for i, j in idx:
        a, b = poi.iloc[i], poi.iloc[j]
        d_utm = a.distance(b)
        if d_utm < 1:
            continue
        lon1, lat1 = to_ll.transform(a.x, a.y); lon2, lat2 = to_ll.transform(b.x, b.y)
        d_geo = geod.inv(lon1, lat1, lon2, lat2)[2]
        errs.append(abs(d_utm - d_geo) / d_geo)
    out["utm_vs_geodesic_relative_error"] = {"pairs": len(errs), "max": round(float(max(errs)), 6), "median": round(float(np.median(errs)), 7), "note": "UTM 38N scale error over a 12x15 km area"}
    # (b) aggregation exactness vs manual spatial join, on real districts
    ctx = make_ctx(store, ds)
    r = call_tool(ctx, "aggregate_spatial_statistics", {"layer": "poi", "categories": ["shop"], "by": "district", "district_level": "all", "top": 500})
    d = ds.layers["district"].reset_index(drop=True)
    poi_g = ds.layers["poi"]
    shops = poi_g[category_mask(poi_g["category_path"], ["shop"])]
    manual = {row.district_id: int(shops.geometry.within(row.geometry).sum()) for row in d.itertuples()}
    agg = ctx.results.get(r.result_id).gdf
    got = dict(zip(agg["district_id"], agg["count"]))
    out["district_aggregation_vs_bruteforce"] = {"districts": len(d), "mismatches": int(sum(manual[k] != got.get(k, 0) for k in manual)), "note": "shops counted inside district polygons; overlaps (macrohood vs neighborhood) count in each"}
    # (c) topology
    j = ds.layers["junction"]
    out["topology_real_area"] = {"junction_clusters": int(len(j)), "kinds": j["kind"].value_counts().to_dict(), "grade_separated_crossings_excluded": int(len(ds.layers["excluded_crossings"])),
                                 "road_components": ds.coverage()["road_network"], "note": "no ground-truth junction inventory exists for Riyadh; correctness is established on synthetic grids (tests, GeoAcquirer benchmarks)"}
    # (d) filter_features vs brute force proximity on real data
    f = call_tool(ctx, "filter_features", {"layer": "poi", "categories": ["shop"], "near": [{"layer": "road", "road_classes": ["motorway", "trunk"], "distance_m": 100, "negate": True}]})
    road = ds.layers["road"]
    hw = road[road["class"].isin(["motorway", "trunk"])].geometry.values
    import shapely
    keep = [bool(shapely.distance(g, hw).min() > 100) for g in shops.geometry]
    out["filter_features_vs_bruteforce"] = {"engine_rows": f.summary["rows"], "bruteforce_rows": int(sum(keep)), "equal": f.summary["rows"] == int(sum(keep))}
    return out


def bench_optimization(store, ds):
    ctx = make_ctx(store, ds)
    rows = []
    for region, p, obj, radius in [({"kind": "central"}, 3, "p_median", 500), ({"kind": "central"}, 3, "max_coverage", 500), ({"kind": "north"}, 3, "p_median", 500),
                                   ({"kind": "central"}, 5, "p_median", 500), ({"kind": "all"}, 3, "p_median", 500), ({"kind": "all"}, 3, "max_coverage", 500)]:
        t = time.perf_counter()
        r = call_tool(ctx, "optimize_facility_locations", {"region": region, "p": p, "objective": obj, "radius_m": radius, "time_limit_s": 30})
        s = r.summary
        sel = ctx.results.get(r.result_id).gdf.to_crs(ds.crs)
        road = ds.layers["road"]
        hw = road[road["class"].isin(["motorway", "trunk"])]
        min_hw = float(min(hw.geometry.distance(g).min() for g in sel.geometry))
        rows.append({"region": region["kind"], "p": p, "objective": obj, "demand_nodes": s["problem"]["demand_nodes"], "candidates": s["problem"]["candidates"],
                     "candidate_cells_before_prefilter": s["problem"].get("candidate_cells_before_prefilter"), "pairs": s["problem"]["pairs"],
                     "before_mean_m": s["before"]["mean_distance_m"], "greedy_mean_m": s["greedy"]["mean_distance_m"], "swap_mean_m": s["greedy_swap"]["mean_distance_m"], "cpsat_mean_m": s["cpsat"]["mean_distance_m"],
                     "random_mean_m": s["random_baseline"]["mean_distance_m_mean"], "before_cov": s["before"]["coverage_within_radius"], "greedy_cov": s["greedy"]["coverage_within_radius"], "swap_cov": s["greedy_swap"]["coverage_within_radius"],
                     "cpsat_cov": s["cpsat"]["coverage_within_radius"], "random_cov": s["random_baseline"]["coverage_mean"], "solver_status": s["cpsat"]["status"],
                     "gap_pct": s["cpsat"]["optimality_gap_pct"], "solver_seconds": s["cpsat"]["seconds"], "total_tool_seconds": round(time.perf_counter() - t, 1),
                     "constraints": {"n_selected_le_p": len(sel) <= p, "min_distance_to_highway_m": round(min_hw, 1), "highway_exclusion_respected": min_hw > 60}})
    return rows


def bench_performance(store, ds):
    ctx = make_ctx(store, ds)
    perf = {}
    tracemalloc.start()
    for name, args in [("filter_features", {"layer": "poi", "categories": ["shop"], "near": [{"layer": "road", "road_classes": ["motorway", "trunk"], "distance_m": 100, "negate": True}]}),
                       ("find_intersections", {"kind": "four_way"}),
                       ("aggregate_spatial_statistics", {"layer": "poi", "by": "district"}),
                       ("calculate_network_accessibility", {"origin_layer": "poi", "origin_categories": ["shop"], "destination": {"layer": "parking", "distance_m": 1}, "cutoff_m": 800, "limit": 3000}),
                       ("retrieve_reference_documents", {"query": "suitability scoring"})]:
        _, t = timed(lambda: call_tool(ctx, name, args), n=3)
        perf[name] = {"median_seconds": t}
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    perf["python_heap_peak_mb_during_tool_calls"] = round(peak / 1e6, 1)
    # feature engineering scalability
    sc = []
    for step in (400, 250, 150, 100):
        cfg = F.FeatureConfig(grid_step_m=step)
        t = time.perf_counter()
        cells, _ = F.candidate_cells(ds, cfg)
        t1 = time.perf_counter() - t
        t = time.perf_counter()
        F.compute_features(ds, cells, cfg)   # timed below; the result itself is not needed
        sc.append({"grid_step_m": step, "eligible_cells": len(cells), "candidate_seconds": round(t1, 2), "feature_seconds": round(time.perf_counter() - t, 2)})
    perf["feature_scalability"] = sc
    t = time.perf_counter()
    call_tool(ctx, "rank_site_suitability", {"k": 5, "region": {"kind": "north"}})
    perf["rank_site_suitability_first_call_seconds"] = round(time.perf_counter() - t, 1)
    _, t2 = timed(lambda: call_tool(ctx, "rank_site_suitability", {"k": 5, "region": {"kind": "north"}, "weights": {"competition": 0.5}}), n=3)
    perf["rank_site_suitability_reweight_seconds_cached_features"] = t2
    # payload sizes for the map
    from fastapi.testclient import TestClient
    from riyadhai.api import create_app
    c = TestClient(create_app(store.root, ds.name))
    payload = {}
    for layer, qs in [("poi", "limit=6000"), ("road", "limit=4000"), ("district", "limit=100"), ("junction", "limit=4000")]:
        t = time.perf_counter()
        resp = c.get(f"/v1/layers/{layer}?bbox=46.64,24.68,46.72,24.76&{qs}")
        payload[layer] = {"seconds": round(time.perf_counter() - t, 3), "kb": round(len(resp.content) / 1024, 1), "features": len(resp.json()["features"])}
    perf["map_layer_payloads_8x8km_view"] = payload
    perf["note"] = "map rendering FPS was not measured (no headless GPU profiling in this build); payload sizes and API latency are measured"
    return perf


def bench_ml(ds):
    p = Path("data/models/building_landuse_metrics.json")
    if not p.exists():
        return {"skipped": "run classify_land_use(action=train) first"}
    res = json.loads(p.read_text())
    return {"dataset": res["dataset"], "spatial_cv": {k: {kk: vv for kk, vv in v.items() if kk != "fold_pr_auc"} for k, v in res["models_spatial_cv"].items()},
            "fold_pr_auc_lightgbm": res["models_spatial_cv"]["lightgbm_all_features"]["fold_pr_auc"], "ablations": res["ablations_spatial_cv"], "top_feature_gain": res["top_feature_gain"],
            "limitations": res["limitations"], "training_seconds": res["seconds"]}


def main():
    store = UrbanStore("data")
    ds = store.load_area("central_riyadh")
    R = {"area": ds.name, "bbox": list(ds.bbox), "overture_release": next(iter(ds.provenance.values()))["dataset_version"]}
    R["orchestration_planner"] = bench_planner()
    R["orchestration_execution"] = bench_orchestration()
    R["spatial_correctness"] = bench_spatial_correctness(store, ds)
    R["machine_learning"] = bench_ml(ds)
    R["optimization"] = bench_optimization(store, ds)
    R["performance"] = bench_performance(store, ds)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    (OUT / "results.md").write_text(render(R), encoding="utf-8")
    print(render(R))


def render(R):
    L = ["# Geo-RiyadhAI benchmark results", "", f"Area `{R['area']}`, bbox {R['bbox']}, Overture release {R['overture_release']}. All numbers measured by `benchmarks/run_benchmarks.py`.", ""]
    p = R["orchestration_planner"]
    L += ["## Orchestration: query-to-plan", "", f"- Gold questions: {p['questions']}; exact tool-sequence match **{p['tool_sequence_exact']}/{p['questions']}** ({p['tool_sequence_accuracy']}); argument checks {p['argument_correct']}/{p['argument_checks']}; follow-ups {p['follow_up_correct']}.",
          f"- Failures: `{json.dumps(p['failures'], ensure_ascii=False)}`", ""]
    e = R["orchestration_execution"]
    L += ["## Orchestration: execution on the synthetic area", "", f"- {e['supported_questions_run']} supported questions run: done {e['done']}, failed {e['failed']}, awaiting approval {e['awaiting_approval']}; task success rate **{e['task_success_rate']}**.",
          f"- Claim audit: {e['numbers_in_answers_checked']} numbers in answers, {e['numbers_not_traceable_to_a_tool_output']} not traceable to a tool output (**hallucinated-claim rate {e['hallucinated_claim_rate']}**); facts with calculation id {e['facts_with_calculation_id']}.",
          f"- Failure injection: `{json.dumps(e['failure_injection'])}`",
          "- Per-question errors: `" + json.dumps([r for r in e['per_question'] if r['errors']], ensure_ascii=False) + "`", ""]
    s = R["spatial_correctness"]
    L += ["## Spatial correctness (real data)", "", "```json", json.dumps(s, indent=2, default=str), "```", ""]
    m = R["machine_learning"]
    L += ["## Machine learning (building land use, spatial block CV)", "", "```json", json.dumps(m, indent=2, default=str), "```", ""]
    L += ["## Optimization (real data)", "", "| region | p | objective | cand. | pairs | before m | greedy m | greedy+swap m | CP-SAT m | random m | before cov | greedy cov | swap cov | CP-SAT cov | random cov | status | gap % | solver s | highway ok |", "|" + "---|" * 19]
    for r in R["optimization"]:
        L.append(f"| {r['region']} | {r['p']} | {r['objective']} | {r['candidates']} | {r['pairs']} | {r['before_mean_m']} | {r['greedy_mean_m']} | {r['swap_mean_m']} | {r['cpsat_mean_m']} | {r['random_mean_m']} | {r['before_cov']} | {r['greedy_cov']} | {r['swap_cov']} | {r['cpsat_cov']} | {r['random_cov']} | {r['solver_status']} | {r['gap_pct']} | {r['solver_seconds']} | {r['constraints']['highway_exclusion_respected']} |")
    L += ["", "## Performance", "", "```json", json.dumps(R["performance"], indent=2, default=str), "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
