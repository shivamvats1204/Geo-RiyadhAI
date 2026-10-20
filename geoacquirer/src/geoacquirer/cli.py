"""Command line: geoacquirer run|plan|serve|sources|inventory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import GeoAcquirerAgent, RuleCompiler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="geoacquirer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="answer a natural-language geographic query")
    r.add_argument("query")
    r.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    r.add_argument("--out", default="runs")
    r.add_argument("--local-dir")
    r.add_argument("--statuses", nargs="*", help="statuses to export (default: VERIFIED_MATCH PROBABLE_MATCH)")
    r.add_argument("--yes", action="store_true", help="approve expensive acquisitions")
    r.add_argument("--max-requests", type=int, help="source-request budget (default 200; Urbi scouting of a few km needs thousands)")

    p = sub.add_parser("plan", help="compile and show the acquisition plan without fetching")
    p.add_argument("query")
    p.add_argument("--bbox", nargs=4, type=float)
    p.add_argument("--local-dir")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--out", default="runs")
    s.add_argument("--local-dir")

    sub.add_parser("sources", help="list registered sources and enablement")

    i = sub.add_parser("inventory", help="scan a directory of local datasets")
    i.add_argument("path")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        agent = GeoAcquirerAgent(Path(a.out), local_dir=a.local_dir, approve=(lambda plan: True) if a.yes else None)
        res = agent.run(a.query, area_bbox=tuple(a.bbox) if a.bbox else None, include_statuses=a.statuses, max_requests=a.max_requests)
        print(json.dumps({"run_dir": str(res.run_dir), "counts": res.counts, "files": {k: str(v) for k, v in res.files.items()}}, indent=2))
        print("\n".join(res.summary))
        return 0
    if a.cmd == "plan":
        q = RuleCompiler().compile(a.query, tuple(a.bbox) if a.bbox else None)
        agent = GeoAcquirerAgent(Path("runs"), local_dir=a.local_dir)
        print(q.model_dump_json(indent=2))
        print(json.dumps(agent.plan(q).to_dict(), indent=2, default=str))
        return 0
    if a.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(Path(a.out), Path(a.local_dir) if a.local_dir else None), host=a.host, port=a.port)
        return 0
    if a.cmd == "sources":
        from .providers.registry import is_authorized, load_registry
        for pid, e in load_registry().items():
            ok, why = is_authorized(pid, e)
            print(f"{pid:14s} {'ENABLED ' if ok else 'DISABLED'}  {e['entity_types']}  {'' if ok else why}")
        return 0
    if a.cmd == "inventory":
        from .providers.local import scan_inventory
        for it in scan_inventory(Path(a.path)):
            print(json.dumps(it.__dict__, default=str))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
