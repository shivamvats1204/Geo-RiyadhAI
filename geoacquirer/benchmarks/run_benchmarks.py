"""GeoAcquirer benchmark suite. Every number printed here is measured by running this file.

    python benchmarks/run_benchmarks.py            # writes benchmarks/results/results.json and results.md

Ground truth comes from the synthetic city's construction plus a brute-force oracle that shares no code with the engine
(plain numpy/shapely loops, no spatial index, no tri-state logic).
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_city import EXTENT, N, build_city, highway_lines, interior_nodes, make_ctx  # noqa: E402

from geoacquirer.agent import RuleCompiler  # noqa: E402
from geoacquirer.agent.pipeline import needs_for  # noqa: E402
from geoacquirer.engine import Evaluator  # noqa: E402
from geoacquirer.entity_resolution import dedupe_pois, resolve_poi_to_building  # noqa: E402
from geoacquirer.query_ir import (And, AreaSpec, EntitySpec, EntityType, GeoQuery, Not, ObjectSpec, Op, Predicate)  # noqa: E402
from geoacquirer.testing import METRIC  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

def q(target_type, cats, expr, u=0.0):
    return GeoQuery(raw_text="bench", targets=[EntitySpec(entity_type=target_type, categories=cats)], area=AreaSpec(name="synthetic"),
                    constraint=expr, position_uncertainty_m=u)


def prf(pred: set, truth: set):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(truth) if truth else 1.0
    return {"precision": round(p, 4), "recall": round(r, 4), "n_pred": len(pred), "n_truth": len(truth), "tp": tp}


def ids(res, statuses):
    key = "poi_id" if "poi_id" in res.columns else "parcel_id"
    return set(res.loc[res.status.isin(statuses), key])


def run_eval(ctx, query):
    t = time.perf_counter()
    res = Evaluator(ctx, query).run()
    return res, time.perf_counter() - t


FOUR = Predicate(op=Op.AT_INTERSECTION, distance_m=30, junction_kind="four_way")
NOT_HW = Not(child=Predicate(op=Op.NEAR, distance_m=60, object=ObjectSpec(entity_type=EntityType.ROAD, road_classes=["motorway", "trunk"])))


def oracle_flagship(poi, nodes_xy, hw, tol_int=30, tol_hw=60):
    shops = poi[poi.kind.isin(["shop", "grocery"])]
    pts = np.array([[g.x, g.y] for g in shops.geometry])
    d_int = np.min(np.hypot(pts[:, None, 0] - nodes_xy[None, :, 0], pts[:, None, 1] - nodes_xy[None, :, 1]), axis=1)
    d_hw = np.min(np.stack([shapely.distance(shops.geometry.values, h) for h in hw]), axis=0)
    ok = (d_int <= tol_int) & (d_hw > tol_hw)
    return set(shops.poi_id[ok])


def main():
    results: dict = {"environment": {"python": sys.version.split()[0], "shapely": shapely.__version__, "geopandas": gpd.__version__}}
    seg, conn, poi, parcels = build_city()
    ctx = make_ctx(seg, conn, poi, parcels)
    nodes_xy = np.array([[p.x, p.y] for p in interior_nodes()])
    hw = highway_lines()
    results["city"] = {"pois": len(poi), "parcels": len(parcels), "segments": len(ctx.layers["road"]),
                       "four_way_found": int((ctx.layers["junction"].kind == "four_way").sum()), "four_way_truth": (N - 1) ** 2,
                       "t_junction_found": int((ctx.layers["junction"].kind == "t_junction").sum()), "t_junction_truth": 4 * (N - 1),
                       "grade_separated_crossings_excluded": len(ctx.layers["excluded_crossings"])}
    tasks = {}

    # 1 ---- shops at four-way, away from highways (u = 0: exact-geometry semantics)
    truth1 = oracle_flagship(poi, nodes_xy, hw)
    res, dt = run_eval(ctx, q(EntityType.POI, ["shop"], And(children=[FOUR, NOT_HW]), 0.0))
    tasks["01_shops_fourway_not_highway"] = {"exact_geometry_u0": prf(ids(res, ["VERIFIED_MATCH", "PROBABLE_MATCH"]), truth1), "seconds": round(dt, 3)}
    res10, dt10 = run_eval(ctx, q(EntityType.POI, ["shop"], And(children=[FOUR, NOT_HW]), 10.0))
    tasks["01_shops_fourway_not_highway"]["with_u10_verified_only"] = prf(ids(res10, ["VERIFIED_MATCH"]), truth1)
    tasks["01_shops_fourway_not_highway"]["with_u10_verified_plus_probable"] = prf(ids(res10, ["VERIFIED_MATCH", "PROBABLE_MATCH"]), truth1)
    tasks["01_shops_fourway_not_highway"]["status_counts_u10"] = res10.status.value_counts().to_dict()
    tasks["01_shops_fourway_not_highway"]["coverage_non_insufficient_u10"] = round(float((res10.status != "INSUFFICIENT_EVIDENCE").mean()), 4)
    # 1b ---- robustness to positional noise: observe shops with N(0, 8 m) error, judge against TRUE positions
    rng = np.random.default_rng(11)
    noisy = poi.copy()
    sh = noisy.kind.isin(["shop", "grocery"])
    dx, dy = rng.normal(0, 8, sh.sum()), rng.normal(0, 8, sh.sum())
    noisy.loc[sh, "geometry"] = [Point(g.x + a, g.y + b) for g, a, b in zip(noisy.loc[sh, "geometry"], dx, dy)]
    ctx_n = make_ctx(seg, conn, noisy, parcels)
    noise = {}
    for u in (0.0, 10.0, 20.0):
        r, _ = run_eval(ctx_n, q(EntityType.POI, ["shop"], And(children=[FOUR, NOT_HW]), u))
        noise[f"u={u:g}"] = {"verified_only": prf(ids(r, ["VERIFIED_MATCH"]), truth1),
                             "verified_plus_probable": prf(ids(r, ["VERIFIED_MATCH", "PROBABLE_MATCH"]), truth1),
                             "insufficient_fraction": round(float((r.status == "INSUFFICIENT_EVIDENCE").mean()), 4)}
    tasks["01b_positional_noise_sigma8m"] = noise

    # 2 ---- commercial lots touching two roads
    lots = parcels[parcels.landuse == "commercial"]
    truth2 = set(lots.parcel_id[lots.n_outer_sides >= 2])
    expr2 = Predicate(op=Op.HAS_FRONTAGE_ON, distance_m=10, min_count=2, object=ObjectSpec(entity_type=EntityType.ROAD))
    r2, dt = run_eval(ctx, q(EntityType.PARCEL, ["commercial"], expr2))
    tasks["02_commercial_parcels_two_roads"] = {"metrics": prf(ids(r2, ["VERIFIED_MATCH"]), truth2), "seconds": round(dt, 3)}

    # 3 ---- parking within 100 m of shops
    shops = poi[poi.kind.isin(["shop", "grocery"])]
    park = poi[poi.kind == "parking"]
    d = np.array([shapely.distance(g, shops.geometry.values).min() for g in park.geometry])
    truth3 = set(park.poi_id[d <= 100])
    r3, dt = run_eval(ctx, q(EntityType.POI, ["parking"], Predicate(op=Op.NEAR, distance_m=100, object=ObjectSpec(entity_type=EntityType.POI, categories=["shop"]))))
    tasks["03_parking_near_shops"] = {"metrics": prf(ids(r3, ["VERIFIED_MATCH"]), truth3), "seconds": round(dt, 3)}

    # 4 ---- warehouses outside residential zones
    res_lots = parcels[parcels.landuse == "residential"].geometry.values
    wh = poi[poi.kind == "warehouse"]
    inside = np.array([any(shapely.contains(res_lots, g)) for g in wh.geometry])
    truth4 = set(wh.poi_id[~inside])
    r4, dt = run_eval(ctx, q(EntityType.POI, ["warehouse"], Predicate(op=Op.NOT_WITHIN, object=ObjectSpec(entity_type=EntityType.PARCEL, categories=["residential"]))))
    tasks["04_warehouses_outside_residential"] = {"metrics": prf(ids(r4, ["VERIFIED_MATCH"]), truth4), "seconds": round(dt, 3)}

    # 5 ---- clinics within 600 m by road of hospitals (all on intersections -> Manhattan distance on the lattice)
    cl, ho = poi[poi.kind == "clinic"], poi[poi.kind == "hospital"]
    def manhattan(a, b): return abs(a.x - b.x) + abs(a.y - b.y)
    truth5 = {r.poi_id for r in cl.itertuples() if min(manhattan(r.geometry, h) for h in ho.geometry) <= 600}
    r5, dt = run_eval(ctx, q(EntityType.POI, ["clinic"], Predicate(op=Op.DISTANCE_BY_ROAD, distance_m=600, exactness="approximate",
                                                                   object=ObjectSpec(entity_type=EntityType.POI, categories=["hospital"]))))
    tasks["05_clinics_600m_by_road_of_hospital"] = {"metrics": prf(ids(r5, ["VERIFIED_MATCH"]), truth5), "seconds": round(dt, 3)}

    # 6 ---- POI -> building association precision/recall
    rng = np.random.default_rng(3)
    sample = shops.sample(400, random_state=3)
    blds, truth6 = [], {}
    for k, r in enumerate(sample.itertuples()):
        if k < 250:   # contained
            x0, y0 = r.geometry.x - rng.uniform(2, 14), r.geometry.y - rng.uniform(2, 14)
            blds.append({"building_id": f"B{k}", "geometry": box(x0, y0, x0 + 18, y0 + 18)}); truth6[r.poi_id] = f"B{k}"
        elif k < 330:  # adjacent: building 8 m away
            blds.append({"building_id": f"B{k}", "geometry": box(r.geometry.x + 8, r.geometry.y - 5, r.geometry.x + 26, r.geometry.y + 13)}); truth6[r.poi_id] = f"B{k}"
    bgdf = gpd.GeoDataFrame(blds, geometry="geometry", crs=METRIC)
    link = resolve_poi_to_building(shops.reset_index(drop=True), bgdf)
    pred = {(pid, b) for pid, b in zip(shops.poi_id, link.building_id) if b is not None}
    tru = set(truth6.items())
    by_type = {}
    for mt in sorted(link.building_match.unique()):
        sel = {(pid, b) for pid, b, t in zip(shops.poi_id, link.building_id, link.building_match) if t == mt and b is not None}
        if sel:
            by_type[mt] = {"links": len(sel), "correct": len(sel & tru), "precision": round(len(sel & tru) / len(sel), 4)}
    tasks["06_poi_building_association"] = {"metrics": prf(pred, tru), "match_types": link.building_match.value_counts().to_dict(),
                                            "precision_by_match_type": by_type,
                                            "note": "shops with no ground-truth building that happen to lie within 20 m of another shop's building count as false links"}

    # 7 ---- multi positive/negative constraints
    near_park = Predicate(op=Op.NEAR, distance_m=150, object=ObjectSpec(entity_type=EntityType.PARKING))
    not_hosp = Not(child=Predicate(op=Op.NEAR, distance_m=100, object=ObjectSpec(entity_type=EntityType.POI, categories=["hospital"])))
    dpark = np.array([shapely.distance(g, park.geometry.values).min() for g in shops.geometry])
    dhosp = np.array([shapely.distance(g, ho.geometry.values).min() for g in shops.geometry])
    t1set = oracle_flagship(poi, nodes_xy, hw)
    truth7 = {pid for pid, a, b in zip(shops.poi_id, dpark, dhosp) if pid in t1set and a <= 150 and b > 100}
    r7, dt = run_eval(ctx, q(EntityType.POI, ["shop"], And(children=[FOUR, NOT_HW, near_park, not_hosp])))
    tasks["07_multi_constraint"] = {"metrics": prf(ids(r7, ["VERIFIED_MATCH"]), truth7), "seconds": round(dt, 3)}

    # 8 ---- two sources with duplicates (cross-source entity resolution)
    rng = np.random.default_rng(5)
    dup_src = shops.sample(300, random_state=5)
    b_rows, true_pairs = [], set()
    for k, r in enumerate(dup_src.itertuples()):
        j = rng.normal(0, 12, 2)
        nm = r.name.replace("-", " ") if k % 2 else r.name
        b_rows.append({"poi_id": f"b{k}", "name": nm, "category": r.category, "category_path": r.category_path, "kind": r.kind, "confidence": 0.6,
                       "source_dataset": "source_b", "geometry": Point(r.geometry.x + j[0], r.geometry.y + j[1])}); true_pairs.add(frozenset({r.poi_id, f"b{k}"}))
    for k in range(300, 600):  # B-only POIs far from A POIs (not duplicates)
        b_rows.append({"poi_id": f"b{k}", "name": f"unique-{k}", "category": "x", "category_path": ["shopping"], "kind": "shop", "confidence": 0.6,
                       "source_dataset": "source_b", "geometry": Point(rng.uniform(0, EXTENT), rng.uniform(0, EXTENT))})
    both = pd.concat([poi.assign(source_dataset="source_a"), gpd.GeoDataFrame(b_rows, geometry="geometry", crs=METRIC)], ignore_index=True)
    merged, mlog = dedupe_pois(gpd.GeoDataFrame(both, geometry="geometry", crs=METRIC))
    found = {frozenset({r.kept, r.dropped}) for r in mlog.itertuples()}
    # a B-only POI might land within 60 m of an A shop with a matching category by chance; those are counted as precision errors, honestly
    def by_mt(lg):
        out = {}
        for mt in sorted(lg.match_type.unique()) if len(lg) else []:
            sel = {frozenset({r.kept, r.dropped}) for r in lg.itertuples() if r.match_type == mt}
            out[mt] = {"pairs": len(sel), "correct": len(sel & true_pairs), "precision": round(len(sel & true_pairs) / len(sel), 4)}
        return out
    tasks["08_multi_source_dedup"] = {"pair_metrics": prf(found, true_pairs), "rows_before": len(both), "rows_after": len(merged),
                                      "match_types": mlog.match_type.value_counts().to_dict() if len(mlog) else {},
                                      "precision_by_match_type": by_mt(mlog)}
    # 9 ---- necessary evidence unavailable -> must abstain
    ctx_np = make_ctx(seg, conn, poi, None)
    r9, dt = run_eval(ctx_np, q(EntityType.POI, ["shop"], Predicate(op=Op.INSIDE_PARCEL, object=ObjectSpec(entity_type=EntityType.PARCEL))))
    tasks["09_missing_evidence_abstains"] = {"abstention_rate": round(float((r9.status == "INSUFFICIENT_EVIDENCE").mean()), 4),
                                             "false_verified": int((r9.status == "VERIFIED_MATCH").sum()), "n": len(r9)}

    # 10 ---- incomplete / conflicting source data: missing categories & names, 25% corrupted duplicate names
    dirty = both.copy()
    idx = dirty.sample(frac=0.10, random_state=1).index
    dirty.loc[idx, "category_path"] = None
    dirty.loc[dirty.sample(frac=0.25, random_state=2).index, "name"] = None
    ctx_d = make_ctx(seg, conn, gpd.GeoDataFrame(dirty, geometry="geometry", crs=METRIC), parcels)
    ok = True
    try:
        r10, dt = run_eval(ctx_d, q(EntityType.POI, ["shop"], And(children=[FOUR, NOT_HW])))
        statuses_valid = set(r10.status) <= {"VERIFIED_MATCH", "PROBABLE_MATCH", "INSUFFICIENT_EVIDENCE", "REJECTED"}
    except Exception:  # noqa: BLE001
        ok, statuses_valid, dt = False, False, 0
    merged_d, mlog_d = dedupe_pois(gpd.GeoDataFrame(dirty, geometry="geometry", crs=METRIC))
    found_d = {frozenset({r.kept, r.dropped}) for r in mlog_d.itertuples()}
    tasks["10_incomplete_conflicting_data"] = {"ran_without_error": ok, "all_statuses_valid": statuses_valid,
                                               "dedup_with_25pct_names_missing": prf(found_d, true_pairs),
                                               "dedup_match_types": mlog_d.match_type.value_counts().to_dict() if len(mlog_d) else {},
                                               "dedup_precision_by_match_type": by_mt(mlog_d)}
    results["tasks"] = tasks

    # ---------------- query compilation success
    cases = [
        ("Find shops in Riyadh that are located at crossroads but are not on highways", dict(ent="poi", ops=["AT_INTERSECTION", "NEAR"], dists=[30, 60])),
        ("Find shops within 25 metres of four-way intersections, exclude those within 80 m of motorways, in Riyadh", dict(ent="poi", ops=["AT_INTERSECTION", "NEAR"], dists=[25, 80])),
        ("List supermarkets at a crossroad in Olaya", dict(ent="poi", ops=["AT_INTERSECTION"], dists=[30])),
        ("Find parking facilities within 150 metres of shops in Riyadh", dict(ent="parking", ops=["NEAR"], dists=[150])),
        ("Find clinics within 2 km road distance of hospitals in Riyadh", dict(ent="poi", ops=["DISTANCE_BY_ROAD"], dists=[2000])),
        ("Find commercial parcels touching two roads in Riyadh", dict(ent="parcel", ops=["HAS_FRONTAGE_ON"], dists=[10])),
        ("Find warehouses outside residential zones in Riyadh", dict(ent="poi", ops=["NOT_WITHIN"], dists=[None])),
        ("Find shops in Riyadh that are not within 300 metres of a hospital", dict(ent="poi", ops=["NEAR"], dists=[300])),
        ("show restaurants near a mosque in North Riyadh", dict(ent="poi", ops=["NEAR"], dists=[200])),
        ("Find pharmacies at T-junctions in Riyadh", dict(ent="poi", ops=["AT_INTERSECTION"], dists=[30])),
        ("find banks away from highways in Riyadh", dict(ent="poi", ops=["NEAR"], dists=[60])),
        ("Find cafes near parking in Riyadh", dict(ent="poi", ops=["NEAR"], dists=[200])),
        ("Find shops at four-way intersections within 40 metres in Riyadh", dict(ent="poi", ops=["AT_INTERSECTION"], dists=[40])),
        ("Find gas stations within 500 m of schools in Riyadh", dict(ent="poi", ops=["NEAR"], dists=[500])),
        ("Find mosques not within 100 m of a bank in Riyadh", dict(ent="poi", ops=["NEAR"], dists=[100])),
        ("find متاجر at crossroads in Riyadh", dict(ent="poi", ops=["AT_INTERSECTION"], dists=[30])),
        ("Find hospitals within 1 km road distance of clinics in Riyadh", dict(ent="poi", ops=["DISTANCE_BY_ROAD"], dists=[1000])),
        ("Find shops near four-way intersections but not near motorway, Riyadh", dict(ent="poi", ops=["AT_INTERSECTION", "NEAR"], dists=[30, 60])),
        ("Find industrial parcels touching two roads in Riyadh", dict(ent="parcel", ops=["HAS_FRONTAGE_ON"], dists=[10])),
        ("Find warehouses near industrial roads", dict(ent="poi", ops=[], dists=[], unparsed=True)),  # unsupported semantics must be FLAGGED, not silently dropped
    ]
    comp = RuleCompiler()
    from geoacquirer.query_ir import iter_predicates
    ok_n, fails = 0, []
    for text, exp in cases:
        try:
            qq = comp.compile(text)
            preds = list(iter_predicates(qq.constraint))
            got = dict(ent=qq.primary().entity_type.value, ops=[p.op.value for p in preds], dists=[p.distance_m for p in preds])
            if exp.get("unparsed"):
                got["unparsed"] = bool(qq.unparsed_phrases)
            if got == exp:
                ok_n += 1
            else:
                fails.append({"query": text, "expected": exp, "got": got})
        except Exception as e:  # noqa: BLE001
            fails.append({"query": text, "error": str(e)})
    results["query_compilation"] = {"n": len(cases), "exact_ir_match": ok_n, "success_rate": round(ok_n / len(cases), 3), "failures": fails}

    # ---------------- planner vs exhaustive retrieval (layers needed), with measured live per-layer latency from the cache
    lat = {}
    for mp in glob.glob(str(Path.home() / ".geoacquirer" / "cache" / "*.json")):
        m = json.loads(Path(mp).read_text(encoding="utf-8"))
        if m.get("entity") and m.get("latency_s") and m.get("rows", 0) > 10000:
            lat[m["entity"]] = round(m["latency_s"], 1)
    plan_rows = []
    task_irs = {"flagship": q(EntityType.POI, ["shop"], And(children=[FOUR, NOT_HW])),
                "parking_near_shops": q(EntityType.POI, ["parking"], Predicate(op=Op.NEAR, distance_m=100, object=ObjectSpec(entity_type=EntityType.POI, categories=["shop"]))),
                "warehouses_outside_residential": q(EntityType.POI, ["warehouse"], Predicate(op=Op.NOT_WITHIN, object=ObjectSpec(entity_type=EntityType.PARCEL, categories=["residential"])))}
    for name, qq in task_irs.items():
        qq.output.requested_attributes = []  # isolate what the predicates themselves need
        need = {n.layer for n in needs_for(qq)}
        plan_rows.append({"task": name, "layers_planned": sorted(need), "layers_exhaustive": ["poi", "road", "connector", "building", "parcel"],
                          "est_live_seconds_planned": round(sum(lat.get(l, 0) for l in need), 1),
                          "est_live_seconds_exhaustive": round(sum(lat.get(l, 0) for l in ["poi", "road", "connector", "building"]), 1)})
    results["planner_vs_exhaustive"] = {"measured_live_layer_seconds": lat, "tasks": plan_rows,
                                        "note": "seconds are the measured first-fetch latency of each Overture layer for the 12x15 km Riyadh bbox; the 'planned' column sums only layers the IR needs"}

    # ---------------- indexed vs naive on real Riyadh data (skipped if the cache is empty)
    results["index_vs_naive"] = bench_index_vs_naive()

    # ---------------- live flagship run summary (latest run directory, if any)
    runs = sorted(glob.glob(str(Path(__file__).parents[1] / "runs" / "*" / "manifest.json")))
    if runs:
        m = json.loads(Path(runs[-1]).read_text(encoding="utf-8"))
        results["live_flagship_run"] = {"run": Path(runs[-1]).parent.name, "status_counts": m["status_counts_all_candidates"],
                                        "verification": {k: m["verification"][k] for k in ("ok", "checked_rows", "checked_leaves", "n_disagreements", "max_abs_distance_error_m")},
                                        "wall_seconds_this_run": m["wall_seconds"], "budget": m["budget"], "topology": m["topology"]["notes"],
                                        "overture_release": next(iter(m["provenance"].values()))["dataset_version"]}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (OUT / "results.md").write_text(render_md(results), encoding="utf-8")
    print(render_md(results))


def bench_index_vs_naive(n_naive: int = 400):
    from geoacquirer.providers.overture import OvertureProvider, segments_frame
    from geoacquirer.spatial import to_metric, utm_crs_for_bbox, prepare_segments
    from geoacquirer.storage import LayerCache
    bb = (46.62, 24.64, 46.74, 24.78)
    prov = OvertureProvider(cache=LayerCache())
    try:
        # cache-only: never trigger a multi-minute download from a benchmark
        key_poi = LayerCache.key(p="overture", r=prov._release(), t="poi", b=tuple(round(v, 6) for v in bb), v=3)
        key_road = LayerCache.key(p="overture", r=prov._release(), t="road", b=tuple(round(v, 6) for v in bb), v=3)
        pc, rc = prov.cache.get(key_poi), prov.cache.get(key_road)
        if not pc or not rc:
            return {"skipped": "live cache not populated (run scripts/prefetch_riyadh.py)"}
    except Exception as e:  # noqa: BLE001
        return {"skipped": str(e)}
    crs = utm_crs_for_bbox(bb)
    shops = to_metric(pc[0], crs)
    shops = shops[shops.category_path.apply(lambda p: p is not None and "shopping" in list(p))].reset_index(drop=True)
    roads = prepare_segments(segments_frame(to_metric(rc[0], crs)))
    t = time.perf_counter()
    pairs, d_idx = roads.sindex.nearest(shops.geometry.values, return_all=False, return_distance=True)
    t_idx_all = time.perf_counter() - t
    sub = shops.geometry.values[:n_naive]
    geoms = roads.geometry.values
    t = time.perf_counter()
    d_naive = np.array([shapely.distance(g, geoms).min() for g in sub])
    t_naive = time.perf_counter() - t
    t = time.perf_counter()
    roads.sindex.nearest(sub, return_all=False, return_distance=True)
    t_idx_sub = time.perf_counter() - t
    dmap = dict(zip(pairs[0], d_idx))
    max_err = float(max(abs(dmap[i] - d_naive[i]) for i in range(len(sub))))
    return {"shops": len(shops), "road_segments": len(roads), "naive_subset": n_naive, "naive_seconds_subset": round(t_naive, 3),
            "indexed_seconds_same_subset": round(t_idx_sub, 4), "speedup_x": round(t_naive / max(t_idx_sub, 1e-9), 1),
            "indexed_seconds_all_shops": round(t_idx_all, 3), "naive_extrapolated_all_seconds": round(t_naive / n_naive * len(shops), 1),
            "max_abs_distance_difference_m": max_err}


def render_md(r: dict) -> str:
    L = ["# GeoAcquirer benchmark results", "", "All values measured by `benchmarks/run_benchmarks.py` (see its docstring for the oracle design).", "",
         f"Environment: {r['environment']}", "", "## Synthetic city topology", "", f"`{r['city']}`", "", "## Constraint precision / recall vs brute-force oracle", ""]
    L.append("| Task | Precision | Recall | Predicted | Truth | Notes |")
    L.append("|---|---|---|---|---|---|")
    for name, t in r["tasks"].items():
        if name == "01_shops_fourway_not_highway":
            for k in ("exact_geometry_u0", "with_u10_verified_only", "with_u10_verified_plus_probable"):
                m = t[k]; L.append(f"| {name} / {k} | {m['precision']} | {m['recall']} | {m['n_pred']} | {m['n_truth']} | coverage(non-insufficient, u=10)={t['coverage_non_insufficient_u10']} |")
        elif name.startswith("01b"):
            for u, v in t.items():
                for kk in ("verified_only", "verified_plus_probable"):
                    m = v[kk]; L.append(f"| {name} / {u} / {kk} | {m['precision']} | {m['recall']} | {m['n_pred']} | {m['n_truth']} | insufficient fraction={v['insufficient_fraction']} |")
        elif "metrics" in t:
            m = t["metrics"]; L.append(f"| {name} | {m['precision']} | {m['recall']} | {m['n_pred']} | {m['n_truth']} | {t.get('match_types', '')} {t.get('precision_by_match_type', '')} |")
        elif "pair_metrics" in t:
            m = t["pair_metrics"]; L.append(f"| {name} (entity-resolution pairs) | {m['precision']} | {m['recall']} | {m['n_pred']} | {m['n_truth']} | rows {t['rows_before']}->{t['rows_after']}; precision by match type {t['precision_by_match_type']} |")
        elif "abstention_rate" in t:
            L.append(f"| {name} | - | - | - | - | abstention={t['abstention_rate']}, false_verified={t['false_verified']} of {t['n']} |")
        elif "dedup_with_25pct_names_missing" in t:
            m = t["dedup_with_25pct_names_missing"]; L.append(f"| {name} (dedup, 25% names missing) | {m['precision']} | {m['recall']} | {m['n_pred']} | {m['n_truth']} | error-free={t['ran_without_error']}, statuses valid={t['all_statuses_valid']}; precision by match type {t['dedup_precision_by_match_type']} |")
    qc = r["query_compilation"]
    L += ["", "## Query compilation (rule compiler)", "", f"{qc['exact_ir_match']}/{qc['n']} exact IR matches ({qc['success_rate']}).", "", "Failures:", "", "```json", json.dumps(qc["failures"], indent=2, ensure_ascii=False), "```", "",
          "## Planner vs exhaustive retrieval", "", "```json", json.dumps(r["planner_vs_exhaustive"], indent=2), "```", "",
          "## Indexed vs naive nearest-road distance (real Riyadh data)", "", "```json", json.dumps(r["index_vs_naive"], indent=2), "```", ""]
    if "live_flagship_run" in r:
        L += ["## Live flagship run (real Overture data)", "", "```json", json.dumps(r["live_flagship_run"], indent=2), "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
