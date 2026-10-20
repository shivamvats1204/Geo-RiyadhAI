"""Plan executor: dependency-ordered tool calls, trace, checkpointing, approval gates, answer composition with claim audit."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .planner import ExecutionPlan, plan as make_plan
from .tools import NeedsApproval, ToolContext, ToolError, ToolResult, call_tool

NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


@dataclass
class Job:
    id: str
    question: str
    plan: ExecutionPlan
    status: str = "queued"          # queued | running | awaiting_approval | done | failed | unsupported
    trace: list[dict] = field(default_factory=list)
    results: dict = field(default_factory=dict)         # step id -> ToolResult (in memory)
    facts: list[dict] = field(default_factory=list)
    answer: list[dict] = field(default_factory=list)    # [{"type", "text", "calc"}]
    warnings: list[str] = field(default_factory=list)
    pending_approval: Optional[dict] = None
    primary_result_id: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"id": self.id, "question": self.question, "status": self.status, "plan": self.plan.to_dict(), "trace": self.trace, "answer": self.answer,
                "warnings": self.warnings, "pending_approval": self.pending_approval, "primary_result_id": self.primary_result_id,
                "report": self.report_path, "error": self.error}


def _resolve(value, results: dict):
    if isinstance(value, str) and value.startswith("$"):
        sid, _, path = value[1:].partition(".")
        r = results.get(sid)
        if r is None:
            raise ToolError(f"dependency {sid} did not produce a result")
        cur = r.result_id if path == "result_id" else r.summary
        if path and path != "result_id":
            for part in path.split("."):
                cur = cur[part]
        return cur
    if isinstance(value, dict):
        return {k: _resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, results) for v in value]
    return value


class Executor:
    def __init__(self, ctx: ToolContext, jobs_dir: Path):
        self.ctx = ctx
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, question: str, session: Optional[dict] = None) -> Job:
        s = session or {}
        if len(question) > 1500:
            from .planner import ExecutionPlan
            p = ExecutionPlan(question[:60] + "...", [], [], ["The question is longer than 1500 characters; please shorten it to one or two sentences."], supported=False)
            return Job(uuid.uuid4().hex[:10], question[:200], p, "unsupported")
        names = [n for n in (self.ctx.area.layers["district"]["name"].dropna().unique().tolist() if self.ctx.area.has("district") else [])]
        p = make_plan(question, district_names=names, last_result_id=s.get("last_result_id"), last_result_layer=s.get("last_result_layer", "poi"))
        return Job(uuid.uuid4().hex[:10], question, p, "queued" if p.supported else "unsupported")

    # ------------------------------------------------------------------ run
    def run(self, job: Job, approved: Optional[set] = None, session: Optional[dict] = None) -> Job:
        if not job.plan.supported:
            job.answer = [{"type": "inferred", "text": job.plan.notes[0], "calc": None}]
            self._save(job)
            return job
        self.ctx.approved |= set(approved or [])
        job.status = "running"
        job.pending_approval = None
        done = {t["step"] for t in job.trace if t["status"] == "ok"}
        for st in job.plan.steps:
            if st.id in done and st.id in job.results:
                continue
            if any(d not in job.results for d in st.depends_on):
                job.trace.append({"step": st.id, "tool": st.tool, "status": "skipped", "reason": "a dependency failed"})
                continue
            t0 = time.perf_counter()
            try:
                if st.tool == "generate_evidence_report":
                    res = self._report(job)
                else:
                    args = _resolve(st.args, job.results)
                    res = call_tool(self.ctx, st.tool, args)
                job.results[st.id] = res
                job.facts += res.facts
                job.warnings += res.warnings
                job.trace = [t for t in job.trace if t["step"] != st.id] + [{
                    "step": st.id, "tool": st.tool, "status": "ok", "why": st.why, "seconds": round(time.perf_counter() - t0, 3), "calc": res.calc_id,
                    "args": args if st.tool != "generate_evidence_report" else {}, "result_id": res.result_id, "summary_keys": list(res.summary)[:12],
                    "provenance_layers": sorted(res.provenance)}]
            except NeedsApproval as e:
                job.status = "awaiting_approval"
                job.pending_approval = {"step": st.id, "tool": e.tool, "why": e.why}
                job.trace.append({"step": st.id, "tool": st.tool, "status": "awaiting_approval", "why": e.why})
                self._save(job)
                return job
            except ToolError as e:
                job.trace.append({"step": st.id, "tool": st.tool, "status": "error", "error": str(e)})
                job.warnings.append(f"{st.tool}: {e}")
            except Exception as e:  # noqa: BLE001
                job.trace.append({"step": st.id, "tool": st.tool, "status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"})
                job.warnings.append(f"{st.tool} failed unexpectedly: {type(e).__name__}")
            self._save(job)
        ok = [t for t in job.trace if t["status"] == "ok"]
        has_error = any(t["status"] == "error" for t in job.trace)
        job.status = "failed" if not ok else ("partial" if has_error else "done")  # partial: some steps succeeded, at least one failed
        if not ok:
            job.error = "; ".join(t.get("error", "") for t in job.trace if t["status"] == "error")[:500]
        primary = [r.result_id for r in job.results.values() if r.result_id and r.tool in ("rank_site_suitability", "optimize_facility_locations", "run_geoquery", "filter_features", "aggregate_spatial_statistics", "find_intersections")]
        job.primary_result_id = primary[-1] if primary else None
        if session is not None and job.primary_result_id:
            rs = self.ctx.results.get(job.primary_result_id)
            session["last_result_id"] = job.primary_result_id
            session["last_result_layer"] = "poi" if "poi_id" in rs.gdf.columns else "building" if "building_id" in rs.gdf.columns else "poi"
        job.answer = self.compose_answer(job)
        if not ok:
            job.answer = [{"type": "observed", "text": "No result: the analysis could not be completed. " + (job.error or "no step succeeded"), "calc": None}]
        self._save(job)
        return job

    # --------------------------------------------------------------- answer
    @staticmethod
    def compose_answer(job: Job) -> list[dict]:
        """The answer is the ordered list of computed facts (each carries its calculation id) plus warnings. No free text is added."""
        seen, out = set(), []
        order = {"observed": 0, "model": 1, "inferred": 2, "reference": 3}
        for f in sorted(job.facts, key=lambda f: order.get(f["type"], 3)):
            if f["text"] in seen:
                continue
            seen.add(f["text"])
            out.append(f)
        return out

    @staticmethod
    def audit_claims(job: Job) -> dict:
        """Every number in the composed answer must appear in a tool summary, table or fact; report any that does not."""
        haystack = json.dumps([{k: r.summary for k, r in job.results.items()}], default=str)
        for r in job.results.values():
            if r.table is not None:
                haystack += r.table.to_csv(index=False)
        for f in job.facts:
            haystack += f["text"]
        norm = lambda s: s.replace(",", "").rstrip("%")  # noqa: E731
        hay_nums = {norm(m.group(0)) for m in NUM_RE.finditer(haystack)}
        unsupported = []
        total = 0
        for f in job.answer:
            if not f.get("calc"):
                continue  # system messages (refusals, guidance, failures) carry no data-derived numbers; only tool facts are audited
            for m in NUM_RE.finditer(f["text"]):
                total += 1
                n = norm(m.group(0))
                if n not in hay_nums and f"{float(n):g}" not in hay_nums:
                    unsupported.append({"claim": f["text"][:120], "number": m.group(0)})
        return {"numbers_checked": total, "unsupported": unsupported, "rate": round(len(unsupported) / total, 4) if total else 0.0,
                "traceable_facts": sum(1 for f in job.answer if f.get("calc")), "facts": len(job.answer)}

    # --------------------------------------------------------------- report
    def _report(self, job: Job) -> ToolResult:
        from .reporting import build_report
        path = build_report(job, self.ctx)
        job.report_path = str(path)
        return ToolResult("generate_evidence_report", {"path": str(path)}).fact("observed", f"Evidence report written to {path.name}.")

    def _save(self, job: Job) -> None:
        d = self.jobs_dir
        (d / f"{job.id}.json").write_text(json.dumps(job.to_dict(), indent=2, default=str), encoding="utf-8")
