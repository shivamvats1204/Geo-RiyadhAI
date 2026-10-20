"""Versioned HTTP API: submit a geographic question, poll progress, fetch results and evidence."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..agent import Cancelled, GeoAcquirerAgent
from ..providers.registry import is_authorized, load_registry

API_VERSION = "v1"


class QueryRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)
    bbox: Optional[tuple[float, float, float, float]] = None
    include_statuses: Optional[list[str]] = None


def create_app(work_dir: Path = Path("runs"), local_dir: Optional[Path] = None, agent_factory=None) -> FastAPI:
    app = FastAPI(title="GeoAcquirer", version=__version__)
    jobs: dict[str, dict] = {}
    work_dir = Path(work_dir)

    def make_agent() -> GeoAcquirerAgent:
        return agent_factory() if agent_factory else GeoAcquirerAgent(work_dir, local_dir=local_dir)

    def _worker(job_id: str, req: QueryRequest):
        job = jobs[job_id]
        try:
            agent = make_agent()
            res = agent.run(req.text, area_bbox=req.bbox, run_id=job_id, cancel=job["cancel"], include_statuses=req.include_statuses)
            job.update(state="done", counts=res.counts, summary=res.summary, files={k: Path(v).name for k, v in res.files.items()})
        except Cancelled:
            job["state"] = "cancelled"
        except Exception as e:  # noqa: BLE001
            job.update(state="failed", error=f"{type(e).__name__}: {str(e)[:500]}")

    @app.get(f"/{API_VERSION}/health")
    def health():
        return {"ok": True, "version": __version__, "api": API_VERSION}

    @app.get(f"/{API_VERSION}/sources")
    def sources():
        reg = load_registry()
        out = {}
        for pid, e in reg.items():
            ok, why = is_authorized(pid, e)
            out[pid] = {**{k: e[k] for k in ("official_docs", "allowed_usage", "entity_types", "coverage", "freshness", "limits")}, "enabled": ok, "enablement": why if not ok else "open"}
        return out

    @app.post(f"/{API_VERSION}/queries", status_code=202)
    def submit(req: QueryRequest):
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"state": "running", "cancel": threading.Event(), "text": req.text}
        threading.Thread(target=_worker, args=(job_id, req), daemon=True).start()
        return {"id": job_id, "state": "running", "status_url": f"/{API_VERSION}/queries/{job_id}"}

    @app.get(f"/{API_VERSION}/queries/{{job_id}}")
    def status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown query id")
        st_path = work_dir / job_id / "state.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        return {"id": job_id, "state": job["state"], "stages_completed": st.get("completed", []), "plan": st.get("plan"),
                **{k: job[k] for k in ("counts", "summary", "files", "error") if k in job}}

    @app.post(f"/{API_VERSION}/queries/{{job_id}}/cancel")
    def cancel(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown query id")
        job["cancel"].set()
        return {"id": job_id, "cancel_requested": True}

    @app.get(f"/{API_VERSION}/queries/{{job_id}}/files/{{name}}")
    def get_file(job_id: str, name: str):
        job = jobs.get(job_id)
        if job is None or job["state"] != "done":
            raise HTTPException(404, "results not available")
        if name not in set(job.get("files", {}).values()):
            raise HTTPException(404, "no such file")  # allowlist: only exported files are served
        return FileResponse(work_dir / job_id / name)

    @app.get(f"/{API_VERSION}/queries/{{job_id}}/evidence/{{entity_id}}")
    def evidence(job_id: str, entity_id: str):
        job = jobs.get(job_id)
        if job is None or job["state"] != "done":
            raise HTTPException(404, "results not available")
        import geopandas as gpd
        g = gpd.read_file(work_dir / job_id / "results.geojson") if (work_dir / job_id / "results.geojson").exists() else None
        if g is None:
            raise HTTPException(404, "no geojson export")
        row = g[g["id"].astype(str) == entity_id]
        if row.empty:
            raise HTTPException(404, "entity not in exported results")
        r = row.iloc[0]
        ev = r["evidence_json"]
        ev = json.loads(ev) if isinstance(ev, str) else ev  # some GeoJSON readers pre-parse JSON-looking strings
        return JSONResponse({"id": entity_id, "name": r["name"], "status": r["status"], "evidence": ev})

    return app
