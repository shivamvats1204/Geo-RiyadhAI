"""HTTP API + static map UI. Single-process; analyses run one at a time behind a lock (the area data is shared in memory)."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .executor import Executor, Job
from .geoacquirer_client import GeoAcquirerClient
from .paths import city_store
from .rag import KnowledgeBase
from .results import ResultStore
from .store import UrbanStore
from .tools import TOOLS, ToolContext, ToolError, call_tool

MAJOR = ["motorway", "trunk", "primary", "secondary"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    session_id: Optional[str] = None
    mode: str = Field(default="auto", pattern="^(auto|tools|code)$")   # auto: tools when the planner understands, code agent for recipes / anything else; code: always the code agent
    approve: list[str] = Field(default_factory=list)


class ApproveRequest(BaseModel):
    tools: list[str] = Field(min_length=1)


class ToolCall(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


def create_app(data_root: Path | str = "data", area_name: str = "central_riyadh", frontend_dir: Optional[Path] = None,
               client: Optional[GeoAcquirerClient] = None) -> FastAPI:
    from .envfile import load_dotenv
    load_dotenv()
    app = FastAPI(title="Geo-RiyadhAI", version=__version__)
    store = UrbanStore(data_root)
    ds = store.load_area(area_name)
    out = Path(data_root) / "analyses" / "out"
    out.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(store, ds, ResultStore(store.root), KnowledgeBase(), client or GeoAcquirerClient(), out)
    ex = Executor(ctx, store.root / "analyses" / "jobs")
    jobs: dict[str, Job] = {}
    sessions: dict[str, dict] = {}
    lock = threading.Lock()

    def _run(job: Job, session: dict, approved: set):
        with lock:
            try:
                ex.run(job, approved=approved, session=session)
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"

    @app.get("/v1/areas")
    def areas():
        out = []
        for n in store.areas():
            meta = json.loads((store.root / "canonical" / n / "meta.json").read_text())
            rows = {p.stem: int(__import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(p).metadata.num_rows) for p in (store.root / "canonical" / n).glob("*.parquet")
                    if p.stem in ("poi", "building", "urbi_building", "parcel", "parking", "street", "attraction", "metro", "district")}
            out.append({"name": n, "bbox": meta["bbox"], "rows": rows, "active": n == ctx.area.name})
        return out

    @app.post("/v1/area")
    def set_area(req: dict):
        name = str(req.get("name", ""))
        with lock:
            try:
                ds_new = store.load_area(name)
            except FileNotFoundError:
                raise HTTPException(404, f"area {name!r} is not in the store; available: {store.areas()}")
            ctx.area = ds_new
            ctx.cache.clear()
            ctx.approved.clear()
        return {"active": name, "bbox": list(ds_new.bbox)}

    @app.get("/v1/health")
    def health():
        ok, why = ctx.client.available()
        return {"ok": True, "version": __version__, "area": ctx.area.name, "geoacquirer": {"available": ok, "detail": why}, "llm": getattr(app.state, "analyst", None) is not None and app.state.analyst.llm is not None}

    @app.get("/v1/dataset")
    def dataset():
        cov = ctx.area.coverage()
        prov = {k: {kk: v.get(kk) for kk in ("provider_id", "dataset_version", "license", "retrieved_at", "from_cache")} for k, v in ctx.area.provenance.items()}
        ok, why = ctx.client.available()
        return {"coverage": cov, "provenance": prov, "bbox": list(ctx.area.bbox), "notes": ctx.area.notes,
                "geoacquirer": {"available": ok, "detail": why, "sources": ctx.client.sources() if ok else {}},
                "offline_mode": not ok}

    @app.get("/v1/tools")
    def tools():
        return [{"name": n, "description": d, "expensive": e, "schema": m.model_json_schema()} for n, (m, _, e, d) in TOOLS.items()]

    @app.get("/v1/layers/{name}")
    def layer(name: str, bbox: Optional[str] = None, limit: int = Query(4000, ge=1, le=20000), classes: Optional[str] = None):
        if name not in ("poi", "road", "building", "district", "junction", "parking", "metro", "attraction", "parcel") or name not in ctx.area.layers:
            raise HTTPException(404, "unknown layer")
        g = ctx.area.layers[name]
        if name == "road":
            wanted = classes.split(",") if classes else MAJOR
            g = g[g["class"].isin(wanted)]
        if name == "attraction":
            g = g[~g["subtype"].isin(["light_pole", "bench", "lift", "stairs", "waiting_place", "transport_3d_stop"])]
        if name == "junction":
            g = g[g["kind"].isin(["four_way", "highway_interchange"])]
        if bbox:
            try:
                w, s, e, n = (float(x) for x in bbox.split(","))
            except ValueError:
                raise HTTPException(400, "bbox must be w,s,e,n")
            from shapely.geometry import box
            import geopandas as gpd
            poly = gpd.GeoSeries([box(w, s, e, n)], crs=4326).to_crs(ctx.area.crs).iloc[0]
            g = g[g.geometry.intersects(poly)]
        total = len(g)
        g = g.head(limit)
        keep = {"poi": ["name", "category", "poi_id"], "road": ["road_name", "class"], "building": ["building_id", "subtype", "class", "height"],
                "district": ["name", "subtype"], "junction": ["kind", "road_names"], "parking": ["name", "source", "capacity", "access"],
                "metro": ["name", "kind"], "attraction": ["name", "subtype_name"], "parcel": ["parcel_id", "landuse", "landuse_ar", "maxBuildingHeight"]}[name]
        keep = [c for c in keep if c in g.columns]
        if name == "building":  # buildings as polygons only when few; the client asks for a small bbox at high zoom
            pass
        gj = json.loads(g[keep + ["geometry"]].to_crs(4326).to_json())
        gj["total_in_view"] = total
        gj["truncated"] = total > len(g)
        return JSONResponse(gj)

    from .city_api import add_city_routes  # city-wide district map (only active when the partitioned store exists)
    city = add_city_routes(app)
    city_root = city_store()
    analysis_jobs: dict = {}
    from .codeagent import AnalysisJob, CodeAnalyst, env_llm
    from . import recipes as _recipes
    from .georag import GeoRAG, load_or_build_entities
    rag = GeoRAG(example_path=store.root / "analyses" / "examples.jsonl", kb=ctx.kb)
    app.state.rag = rag
    if city is not None:
        def _build_entities():  # place-name index for the whole city (cached after the first build); the analysis works without it until it is ready
            try:
                rag.set_entities(load_or_build_entities(city.root / "entities.parquet", city.root, city.reg, city_root / "metro_network"))
            except Exception as e:  # noqa: BLE001
                rag.entity_error = f"{type(e).__name__}: {e}"
        threading.Thread(target=_build_entities, daemon=True).start()
    analyst = CodeAnalyst(ctx, env_llm(), rag=rag)
    app.state.analyst = analyst

    def _start_code(question: str) -> dict:
        job = AnalysisJob(id=uuid.uuid4().hex[:10], question=question)
        analysis_jobs[job.id] = job
        ok, why = analyst.available(question)
        if not ok:
            job.status, job.error = "unsupported", why
            return {"job_id": job.id, "session_id": None, "status": job.status, "mode": "code"}
        threading.Thread(target=analyst.run, args=(job,), daemon=True).start()
        return {"job_id": job.id, "session_id": None, "status": "running", "mode": "code"}

    @app.post("/v1/ask", status_code=202)
    def ask(req: AskRequest):
        if req.mode == "code" or (req.mode == "auto" and _recipes.match(req.question)):
            return _start_code(req.question)
        sid = req.session_id or uuid.uuid4().hex[:8]
        session = sessions.setdefault(sid, {})
        job = ex.create_job(req.question, session)
        if not job.plan.supported and req.mode == "auto" and analyst.llm is not None:
            return _start_code(req.question)   # the fixed tools do not cover it: let the model write the analysis
        jobs[job.id] = job
        if not job.plan.supported:
            ex.run(job)
            return {"job_id": job.id, "session_id": sid, "status": job.status, "mode": "tools"}
        threading.Thread(target=_run, args=(job, session, set(req.approve)), daemon=True).start()
        return {"job_id": job.id, "session_id": sid, "status": "running", "mode": "tools"}

    @app.get("/v1/analysis/{job_id}")
    def get_analysis(job_id: str):
        j = analysis_jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown analysis")
        return j.to_dict()

    @app.post("/v1/analysis/{job_id}/approve", status_code=202)
    def approve_analysis(job_id: str):
        j = analysis_jobs.get(job_id)
        if j is None or j.status != "needs_approval":
            raise HTTPException(409, "this analysis is not waiting for approval")
        j.approved = True
        j.status = "running"
        threading.Thread(target=analyst.run, args=(j,), daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/v1/figures/{name}")
    def figure(name: str):
        import re as _re
        if not _re.fullmatch(r"fig_[0-9a-f]{10}\.png", name):
            raise HTTPException(404, "unknown figure")
        f = analyst.fig_dir / name
        if not f.exists():
            raise HTTPException(404, "unknown figure")
        return FileResponse(f, media_type="image/png")

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            p = store.root / "analyses" / "jobs" / f"{job_id}.json"
            if p.exists():
                return json.loads(p.read_text())
            raise HTTPException(404, "unknown job")
        d = job.to_dict()
        if job.status in ("done", "failed", "partial"):
            d["claim_audit"] = Executor.audit_claims(job)
            d["tables"] = {sid: json.loads(r.table.head(40).to_json(orient="records")) for sid, r in job.results.items() if r.table is not None}
            d["results"] = {sid: {"tool": r.tool, "result_id": r.result_id, "calc_id": r.calc_id, "summary": json.loads(json.dumps(r.summary, default=str)) if r.tool != "classify_land_use" else {}}
                            for sid, r in job.results.items()}
            try:
                from . import plain
                d["plain"] = plain.summarize(d, job, ctx) if job.status != "failed" else None
            except Exception as e:  # noqa: BLE001  the plain summary must never hide the technical result
                d["plain"] = None
                d["plain_error"] = f"{type(e).__name__}: {e}"
        return d

    @app.post("/v1/jobs/{job_id}/approve", status_code=202)
    def approve(job_id: str, req: ApproveRequest):
        job = jobs.get(job_id)
        if job is None or job.status != "awaiting_approval":
            raise HTTPException(409, "job is not awaiting approval")
        sid = next((k for k, v in sessions.items()), None)
        threading.Thread(target=_run, args=(job, sessions.get(sid, {}), set(req.tools)), daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/v1/results/{rid}.geojson")
    def result_geojson(rid: str, limit: int = 20000):
        try:
            g = ctx.results.get(rid).gdf
        except KeyError:
            raise HTTPException(404, "unknown result")
        g = g.head(limit)
        for c in g.columns:
            if c != "geometry" and g[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                g = g.drop(columns=c)
        return Response(g.to_json(), media_type="application/geo+json")

    @app.get("/v1/results/{rid}.csv")
    def result_csv(rid: str):
        try:
            g = ctx.results.get(rid).gdf.copy()
        except KeyError:
            raise HTTPException(404, "unknown result")
        cen = g.to_crs(ctx.area.crs).geometry.centroid.to_crs(4326)
        g["lon"], g["lat"] = cen.x.to_numpy(), cen.y.to_numpy()
        d = g.drop(columns="geometry")
        for c in d.columns:
            if d[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                d[c] = d[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict, np.ndarray)) else v)
        return Response(d.to_csv(index=False, encoding="utf-8-sig"), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={rid}.csv"})

    @app.get("/v1/results/{rid}/lineage")
    def lineage(rid: str):
        try:
            return ctx.results.lineage(rid)
        except KeyError:
            raise HTTPException(404, "unknown result")

    @app.post("/v1/tool")
    def run_tool(call: ToolCall):
        """Direct, synchronous tool call (used by the UI for weight re-ranking and scenarios)."""
        with lock:
            try:
                r = call_tool(ctx, call.tool, call.args)
            except ToolError as e:
                raise HTTPException(422, str(e))
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "NeedsApproval":
                    raise HTTPException(409, str(e))
                raise
        return {"tool": r.tool, "calc_id": r.calc_id, "result_id": r.result_id, "summary": json.loads(json.dumps(r.summary, default=str)),
                "table": json.loads(r.table.head(50).to_json(orient="records")) if r.table is not None else None, "warnings": r.warnings, "facts": r.facts, "seconds": r.seconds}

    @app.get("/v1/report/{job_id}", response_class=HTMLResponse)
    def report(job_id: str):
        p = out / f"report_{job_id}.html"
        if not p.exists():
            raise HTTPException(404, "no report for this job")
        return HTMLResponse(p.read_text(encoding="utf-8"))

    @app.get("/v1/rag")
    def rag(q: str = Query(min_length=2, max_length=300), k: int = 3):
        return ctx.kb.search(q, k)

    # ------------------------------------------------------------------ scope (replaces the fixed "study area")
    build_jobs: dict[str, dict] = {}

    def _urbi_districts_into(ds):
        """Make the area's district layer the Urbi districts shown on the map (English + Arabic names), once per area."""
        if getattr(ds, "_urbi_districts", False) or city is None:
            return
        from shapely.geometry import box as _box
        reg = city.reg
        ud = reg[(reg.source == "urbi") & (reg.kind == "district") & reg.has_polygon]
        ud = ud[ud.geometry.intersects(_box(*ds.bbox))]
        import geopandas as gpd
        g = gpd.GeoDataFrame({"district_id": ud.uid.values, "name": ud["name"].values, "name_ar": ud.name_ar.values, "subtype": "district"}, geometry=ud.geometry.values, crs=4326)
        ds.layers["district"] = g.to_crs(ds.crs).reset_index(drop=True)
        ds._urbi_districts = True

    def _apply_scope(area_name: str, uid, name: str, geom) -> dict:
        import geopandas as gpd
        with lock:
            ds_new = store.load_area(area_name)
            _urbi_districts_into(ds_new)
            poly = gpd.GeoSeries([geom], crs=4326).to_crs(ds_new.crs).iloc[0]
            for a in store._areas.values():
                a.scope = None
            ds_new.scope = {"name": name, "uid": uid, "polygon": poly}
            ds_new._ctx = None
            ctx.area = ds_new
            ctx.cache.clear()
            ctx.approved.clear()
        inside = ds_new.full_polygon().intersection(poly).area / poly.area if poly.area else 0.0
        return {"state": "ready", "scope": name, "uid": uid, "area": area_name, "coverage_share": round(float(inside), 3),
                "layers": {k: int(len(v)) for k, v in ds_new.layers.items() if k in ("poi", "urbi_building", "building", "parcel", "parking", "district") and len(v)}}

    def _best_area(geom):
        from shapely.geometry import box as _box
        best, share = None, 0.0
        for n in store.areas():
            meta = json.loads((store.root / "canonical" / n / "meta.json").read_text())
            frac = geom.intersection(_box(*meta["bbox"])).area / geom.area if geom.area else 0.0
            if frac > share + 1e-9:
                best, share = n, frac
        return best, share

    @app.post("/v1/scope")
    def set_scope(req: dict):
        from shapely.geometry import Point, box as _box
        uid, bb = req.get("uid"), req.get("bbox")
        if uid:
            if city is None:
                raise HTTPException(503, "district registry is not built")
            row = city.reg[city.reg.uid == uid]
            if row.empty:
                raise HTTPException(404, f"unknown area {uid!r}")
            r = row.iloc[0]
            geom = r.geometry if r.has_polygon else Point(r.centroid_lon, r.centroid_lat).buffer(0.01)
            name = r["name"] or r["name_ar"] or uid
        elif bb and len(bb) == 4:
            w, so, e, n = (float(v) for v in bb)
            geom, name, uid = _box(w, so, e, n), "the area on the map", None
        else:
            raise HTTPException(400, "give a district uid or a bbox")
        best, share = _best_area(geom)
        if best is not None and share >= 0.05:
            return _apply_scope(best, uid, name, geom)
        w, so, e, n = geom.bounds
        pad = 0.003
        bbox = (w - pad, so - pad, e + pad, n + pad)
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0.1:
            raise HTTPException(400, f"{name} is too large to prepare in one go; pick a smaller district or zoom the map in.")
        bid = "b" + uuid.uuid4().hex[:6]
        build_jobs[bid] = {"state": "running", "message": f"Downloading open map data for {name} (first time only, about 5-10 minutes)…"}

        def _build():
            try:
                from .citystore import CityStore
                from .store import build_area_from_city
                aname = f"scope_{abs(hash(str(bbox))) % 10**8}"
                build_area_from_city(store, aname, bbox, ctx.client, CityStore(city_root), allow_live=True)
                build_jobs[bid] = {"state": "ready", "result": _apply_scope(aname, uid, name, geom)}
            except Exception as e:  # noqa: BLE001
                build_jobs[bid] = {"state": "failed", "message": f"{type(e).__name__}: {str(e)[:200]}"}
        threading.Thread(target=_build, daemon=True).start()
        return {"state": "building", "build_id": bid, "message": build_jobs[bid]["message"]}

    @app.get("/v1/scope/build/{bid}")
    def scope_build(bid: str):
        if bid not in build_jobs:
            raise HTTPException(404, "unknown build")
        return build_jobs[bid]

    @app.get("/v1/scope")
    def get_scope():
        sc = ctx.area.scope
        return {"scope": sc["name"] if sc else None, "uid": sc["uid"] if sc else None, "area": ctx.area.name}

    if city is not None:
        _urbi_districts_into(ctx.area)  # district analyses use the same Urbi districts the map shows
    app.state.city = city

    fe = Path(frontend_dir) if frontend_dir else Path(__file__).resolve().parents[2] / "frontend"
    if fe.exists():
        app.mount("/", StaticFiles(directory=fe, html=True), name="ui")
    app.state.ctx, app.state.executor, app.state.jobs = ctx, ex, jobs
    return app
