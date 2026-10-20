"""CLI: build the area from GeoAcquirer, ask a question, serve the UI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BBOX = (46.62, 24.64, 46.74, 24.78)


def main(argv=None) -> int:
    from .envfile import load_dotenv
    load_dotenv()  # LLM settings etc. from .env (real environment variables take precedence)
    ap = argparse.ArgumentParser(prog="riyadhai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="acquire the study area through GeoAcquirer and build the canonical store")
    b.add_argument("--name", default="central_riyadh")
    b.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX))
    b.add_argument("--data", default="data")
    b.add_argument("--allow-live", action="store_true", help="permit live acquisition (several minutes) when layers are not cached")
    a = sub.add_parser("ask", help="ask one question and print the evidence")
    a.add_argument("question")
    a.add_argument("--data", default="data")
    a.add_argument("--area", default="central_riyadh")
    a.add_argument("--approve", nargs="*", default=[])
    s = sub.add_parser("serve", help="run the API and the map UI")
    s.add_argument("--data", default="data")
    s.add_argument("--area", default="central_riyadh")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    ns = ap.parse_args(argv)

    if ns.cmd == "build":
        from .geoacquirer_client import GeoAcquirerClient
        from .store import UrbanStore
        ds = UrbanStore(ns.data).build_area(ns.name, tuple(ns.bbox), GeoAcquirerClient(), allow_live=ns.allow_live)
        print(json.dumps(ds.coverage(), indent=2, default=str))
        return 0
    if ns.cmd == "ask":
        from .executor import Executor
        from .geoacquirer_client import GeoAcquirerClient
        from .rag import KnowledgeBase
        from .results import ResultStore
        from .store import UrbanStore
        from .tools import ToolContext
        st = UrbanStore(ns.data)
        out = Path(ns.data) / "analyses" / "out"
        out.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(st, st.load_area(ns.area), ResultStore(st.root), KnowledgeBase(), GeoAcquirerClient(), out)
        ex = Executor(ctx, st.root / "analyses" / "jobs")
        job = ex.run(ex.create_job(ns.question), approved=set(ns.approve))
        print(f"status: {job.status}")
        if job.pending_approval:
            print("needs approval:", job.pending_approval, "(re-run with --approve", job.pending_approval["tool"] + ")")
        for f in job.answer:
            print(f"[{f['type']}] {f['text']}  ({f['calc']})")
        for w in dict.fromkeys(job.warnings):
            print("WARNING:", w)
        print("claim audit:", json.dumps(Executor.audit_claims(job)))
        if job.report_path:
            print("report:", job.report_path)
        return 0
    if ns.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(ns.data, ns.area), host=ns.host, port=ns.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
