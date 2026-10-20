"""Run the five flagship demonstrations on the real cached Riyadh study area and save every output.

    python examples/run_demos.py [--only demo3]

Outputs go to examples/outputs/<demo>/ : answer.json (facts with calc ids), trace.json, tables, GeoJSON/CSV exports, report.html.
Demo 2 (parcel intelligence) needs a parcel polygon layer that is NOT available for real Riyadh data in this repo, so it runs
(a) on the real area to show the honest refusal and (b) on a clearly-labelled SYNTHETIC parcel layer to show the mechanism.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from riyadhai.executor import Executor
from riyadhai.geoacquirer_client import GeoAcquirerClient
from riyadhai.rag import KnowledgeBase
from riyadhai.results import ResultStore
from riyadhai.store import UrbanStore
from riyadhai.tools import ToolContext, call_tool

OUT = Path(__file__).parent / "outputs"


def run_job(ctx, ex, question, name, approve=()):
    t = time.time()
    job = ex.run(ex.create_job(question), approved=set(approve))
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    audit = Executor.audit_claims(job)
    (d / "answer.json").write_text(json.dumps({"question": question, "status": job.status, "answer": job.answer, "warnings": list(dict.fromkeys(job.warnings)),
                                               "claim_audit": audit, "seconds": round(time.time() - t, 1)}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (d / "trace.json").write_text(json.dumps({"plan": job.plan.to_dict(), "trace": job.trace}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for sid, r in job.results.items():
        if r.table is not None and r.tool != "retrieve_reference_documents":
            r.table.to_csv(d / f"table_{sid}_{r.tool}.csv", index=False, encoding="utf-8-sig")
        if r.result_id and r.tool in ("rank_site_suitability", "optimize_facility_locations", "run_geoquery", "aggregate_spatial_statistics", "measure_proximity"):
            e = call_tool(ctx, "export_dataset", {"result_id": r.result_id, "format": "geojson"})
            shutil.copy(e.summary["path"], d / f"{r.tool}_{r.result_id}.geojson")
            e = call_tool(ctx, "export_dataset", {"result_id": r.result_id, "format": "csv"})
            shutil.copy(e.summary["path"], d / f"{r.tool}_{r.result_id}.csv")
    if job.report_path:
        shutil.copy(job.report_path, d / "report.html")
    print(f"[{name}] status={job.status} seconds={round(time.time() - t, 1)} facts={len(job.answer)} claim_audit_unsupported={len(audit['unsupported'])}")
    for f in job.answer[:4]:
        print("   -", f["type"], f["text"][:160])
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    store = UrbanStore(a.data)
    ds = store.load_area("central_riyadh")
    out = Path(a.data) / "analyses" / "out"
    out.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(store, ds, ResultStore(store.root), KnowledgeBase(), GeoAcquirerClient(), out)
    ex = Executor(ctx, store.root / "analyses" / "jobs")
    want = lambda n: not a.only or a.only in n  # noqa: E731

    if want("demo1"):
        run_job(ctx, ex, "Find shops located at crossroads but away from highways, and show nearby parking.", "demo1_complex_search")
    if want("demo2"):
        run_job(ctx, ex, "Find commercial parcels touching two roads", "demo2a_parcels_REAL_suhail")
        demo2_synthetic()
    if want("demo3"):
        run_job(ctx, ex, "Find five suitable commercial locations for a supermarket in northern Riyadh. Avoid highways, prioritize crossroad accessibility, "
                         "compare nearby competitors and parking availability, assess zoning restrictions, and explain your recommendation.", "demo3_site_suitability")
    if want("demo4"):
        run_job(ctx, ex, "Where should three new parking facilities be located to improve access across the study area?", "demo4_parking_optimization")
    if want("demo5"):
        agg = call_tool(ctx, "aggregate_spatial_statistics", {"layer": "poi", "categories": ["shop"], "by": "district", "top": 3})
        names = [r["name"] for r in agg.table.to_dict("records")][:2]
        print("selected districts (top shop counts):", names)
        run_job(ctx, ex, f"Analyze commercial development opportunities across {names[0]} and {names[1]}", "demo5_research_report", approve=["classify_land_use"])


def demo2_synthetic():
    from riyadhai.testing import synthetic_area, BLOCK
    import tempfile
    root = Path(tempfile.mkdtemp())
    st, ds, truth = synthetic_area(root)
    ox, oy = truth["origin_metric"]
    lots, rng = [], np.random.default_rng(1)
    for bi in range(8):
        for bj in range(8):
            zone = rng.choice(["residential", "commercial", "industrial"], p=[.5, .35, .15])
            lot = (BLOCK - 8) / 3
            for a_ in range(3):
                for c in range(3):
                    x0, y0 = ox + bi * BLOCK + 4 + a_ * lot, oy + bj * BLOCK + 4 + c * lot
                    lots.append({"parcel_id": f"SYN-{bi}{bj}{a_}{c}", "landuse": zone, "geometry": box(x0, y0, x0 + lot, y0 + lot)})
    g = gpd.GeoDataFrame(lots, geometry="geometry", crs=ds.crs).to_crs(4326)
    p = root / "synthetic_parcels.geojson"
    g.to_file(p, driver="GeoJSON")
    st.ingest_local(ds, p, "parcel", licence="SYNTHETIC demonstration data")
    ctx = ToolContext(st, ds, ResultStore(st.root), KnowledgeBase(), GeoAcquirerClient(), root / "out")
    (root / "out").mkdir(exist_ok=True)
    ex = Executor(ctx, st.root / "analyses" / "jobs")
    job = run_job(ctx, ex, "Find commercial parcels touching two roads", "demo2b_parcels_SYNTHETIC_mechanism")
    n = job.results["s1"].summary if "s1" in job.results else {}
    print("   SYNTHETIC parcel layer: counts", n.get("status_counts"))


if __name__ == "__main__":
    sys.exit(main())
