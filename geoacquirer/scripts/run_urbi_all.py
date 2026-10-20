"""Urbi/2GIS multi-type acquisition (port of urbi_master.ipynb) - needs GEOACQUIRER_ENABLE_URBI=1, ..._AUTHORIZATION_REF, ..._API_KEY.

Differences from the notebook: honest User-Agent, bounded concurrency (default 8) with rate limiting and backoff, stop on 401/403 or a 429 storm,
resumable ID and JSONL files. Task configs (step/radius) are copied from the notebook.
Usage: python scripts/run_urbi_all.py out_dir [west south east north] [--tasks BUILDINGS,PARKING,...]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import requests

from geoacquirer.acquisition import AdaptiveCrawler, FatalSourceError, RateLimitedError, TransientError, Checkpoint
from geoacquirer.providers.registry import require_authorized
from geoacquirer.security import RateLimiter

FIELDS = (
    "items.locale,items.flags,items.search_attributes.detection_type,"
    "items.search_attributes.relevance,items.adm_div,items.city_alias,"
    "items.region_id,items.segment_id,items.reviews,items.point,request_type,"
    "context_rubrics,query_context,items.links,items.name_ex,items.name_back,"
    "items.org,items.group,items.dates,items.external_content,"
    "items.contact_groups,items.comment,items.ads.options,"
    "items.email_for_sending.allowed,items.stat,items.stop_factors,"
    "items.description,items.geometry.centroid,"
    "items.geometry.selection,items.geometry.hover,"
    "items.geometry.style,items.timezone_offset,items.context,"
    "items.level_count,items.address,items.is_paid,items.access,"
    "items.access_comment,items.for_trucks,items.is_incentive,"
    "items.paving_type,items.capacity,items.schedule,items.schedule_special,"
    "items.floors,items.floor_id,items.floor_plans,ad,items.rubrics,"
    "items.routes,items.platforms,items.directions,items.barrier,"
    "items.reply_rate,items.purpose,items.purpose_code,items.attribute_groups,"
    "items.route_logo,items.has_goods,items.has_apartments_info,"
    "items.has_pinned_goods,items.has_realty,items.has_otello_stories,"
    "items.has_exchange,items.has_payments,items.has_dynamic_congestion,"
    "items.is_promoted,items.congestion,items.delivery,items.order_with_cart,"
    "search_type,items.has_discount,items.metarubrics,items.detailed_subtype,"
    "items.temporary_unavailable_atm_services,items.poi_category,"
    "items.has_ads_model,items.vacancies,items.structure_info.material,"
    "items.structure_info.floor_type,items.structure_info.gas_type,"
    "items.structure_info.year_of_construction,"
    "items.structure_info.elevators_count,"
    "items.structure_info.is_in_emergency_state,"
    "items.structure_info.project_type,items.has_otello_hotels,"
    "items.subtype"  # Added for Metro/Streets identification
)

BASE = "https://catalog.api.2gis.ru/3.0"
TASKS = [
    {"name": "BUILDINGS", "api_type": "building", "step": 0.002, "radius": 250},
    {"name": "PARKING", "api_type": "parking", "step": 0.005, "radius": 600},
    {"name": "ATTRACTIONS", "api_type": "attraction", "step": 0.01, "radius": 1200},
    {"name": "METRO", "api_type": ["station.metro", "station_entrance", "station_platform"], "step": 0.015, "radius": 1500},
    {"name": "STREETS", "api_type": "street", "step": 0.01, "radius": 1200},
]
ref = require_authorized("urbi")
KEY = os.environ["GEOACQUIRER_URBI_API_KEY"]
WORKERS = int(os.environ.get("URBI_WORKERS", "8"))
limiter = RateLimiter(float(os.environ.get("URBI_INTERVAL", "0.01")))
session = requests.Session()
session.headers.update({"User-Agent": "geoacquirer/0.1 (+authorized-research)", "Accept": "application/json"})
session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=WORKERS * 2, pool_maxsize=WORKERS * 2))


def get(path, params, timeout=20):
    r = session.get(f"{BASE}/{path}", params={**params, "key": KEY}, timeout=timeout)
    if r.status_code in (401, 403):
        raise FatalSourceError(f"HTTP {r.status_code}")
    if r.status_code == 429:
        raise RateLimitedError("429")
    if r.status_code >= 500:
        raise TransientError(str(r.status_code))
    if r.status_code == 200:
        return r.json().get("result", {})
    if r.status_code == 404:
        return {}
    raise TransientError(f"HTTP {r.status_code}")  # never treat an unexpected status as "no data": it must be retried and counted


def scan(cell, task):
    lat, lon = cell
    ids = set()
    for t in (task["api_type"] if isinstance(task["api_type"], list) else [task["api_type"]]):
        page = 1
        while page <= 50:
            res = get("items", {"point": f"{lon},{lat}", "radius": task["radius"], "type": t, "fields": "items.id", "page_size": 50, "page": page}, 10)
            items = res.get("items", [])
            ids.update(str(i["id"]).split("_")[0] for i in items)
            if len(items) < 50:
                break
            page += 1
    return [{"id": i} for i in ids]


def run_scan(task, bounds, out):
    f = out / f"{task['name'].lower()}_ids.txt"
    lat0, lat1, lon0, lon1 = bounds
    cells = [(float(a), float(b)) for a in np.arange(lat0, lat1, task["step"]) for b in np.arange(lon0, lon1, task["step"])]
    ck = Checkpoint(out / f"{task['name'].lower()}_scan.checkpoint.json")
    known = set(f.read_text().split()) if f.exists() else set()
    fh = open(f, "a")

    def on_items(new):
        for it in new:
            if it["id"] not in known:
                known.add(it["id"]); fh.write(it["id"] + chr(10))
        fh.flush()

    c = AdaptiveCrawler(lambda cell: scan(cell, task), key=lambda it: it["id"], cell_id=lambda c: f"{c[0]:.5f},{c[1]:.5f}", max_workers=WORKERS,
                        limiter=limiter, checkpoint=ck, on_items=on_items, max_rate_limited=200)
    st = c.run(cells)
    fh.close()
    print(f"  scan {task['name']}: sectors={st.cells_total} done={st.cells_done} failed={st.cells_failed} ids={len(known)} rate_limited={st.rate_limited} aborted={st.aborted}", flush=True)
    return known, st


def fetch_batch(batch):
    res = get("items/byid", {"id": ",".join(batch), "fields": FIELDS, "locale": "en_SA"}, 40)
    items = res.get("items", [])
    if len(items) < len(batch):  # a short answer is a failure to retry, not a result
        raise TransientError(f"batch returned {len(items)} of {len(batch)} items")
    return items


def run_details(task, ids, out):
    jf = out / f"{task['name'].lower()}_data.jsonl"
    done = set()
    if jf.exists():
        for line in open(jf, encoding="utf-8"):
            try:
                done.add(str(json.loads(line)["id"]).split("_")[0])
            except Exception:
                pass
    todo = [i for i in ids if i not in done]
    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    fh = open(jf, "a", encoding="utf-8")

    def on_items(new):
        for it in new:
            fh.write(json.dumps(it, ensure_ascii=False) + chr(10))
        fh.flush()

    c = AdaptiveCrawler(fetch_batch, key=lambda it: it["id"], cell_id=lambda b: b[0], max_workers=WORKERS, limiter=limiter, on_items=on_items, max_rate_limited=200)
    st = c.run(batches)
    fh.close()
    got = len(done) + st.items
    print(f"  details {task['name']}: ids={len(ids)} previously={len(done)} new={st.items} batches_failed={st.cells_failed} aborted={st.aborted}", flush=True)
    return got, st


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--tasks=")), None)
    out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
    w, s, e, n = (float(x) for x in args[1:5]) if len(args) >= 5 else (46.62, 24.64, 46.74, 24.78)
    bounds = (s, n, w, e)  # (lat_min, lat_max, lon_min, lon_max) as in the notebook
    summary = {"authorization_ref": ref, "bbox_wsen": [w, s, e, n], "tasks": {}}
    for task in TASKS:
        if only and task["name"] not in only:
            continue
        t0 = time.time()
        print(f">>> {task['name']}", flush=True)
        ids, st = run_scan(task, bounds, out)
        if st.aborted and st.aborted != "cancelled":
            print("ABORTED:", st.aborted); summary["tasks"][task["name"]] = {"aborted": st.aborted}; break
        got, st2 = run_details(task, sorted(ids), out)
        summary["tasks"][task["name"]] = {"ids": len(ids), "records": got, "seconds": round(time.time() - t0), "scan_failed": st.cells_failed, "detail_batches_failed": st2.cells_failed}
        (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print("ALL DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
