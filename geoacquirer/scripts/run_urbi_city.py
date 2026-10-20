"""City-wide Urbi acquisition with coarse scouting (skips empty desert), all five types, resumable.
Env: GEOACQUIRER_ENABLE_URBI=1, ..._AUTHORIZATION_REF, ..._API_KEY, URBI_WORKERS. Arg: out_dir
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_urbi_all as U  # noqa: E402
from geoacquirer.acquisition import AdaptiveCrawler, Checkpoint  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
LAT0, LAT1, LON0, LON1 = 23.6740, 25.2848, 45.9930, 48.1804   # BOUND from urbi_master.ipynb
COARSE, COARSE_R = 0.02, 1700


def probe(cell, task):
    lat, lon = cell
    n = 0
    for t in (task["api_type"] if isinstance(task["api_type"], list) else [task["api_type"]]):
        res = U.get("items", {"point": f"{lon},{lat}", "radius": COARSE_R, "type": t, "fields": "items.id", "page_size": 1}, 10)
        n += int(res.get("total", 0) or 0)
    return [{"id": f"{lat:.4f},{lon:.4f}", "n": n}]


def positive_cells(task):
    f = out / f"{task['name'].lower()}_coarse.json"
    if f.exists():
        return [tuple(c) for c in json.loads(f.read_text())]
    cells = [(float(a), float(b)) for a in np.arange(LAT0 + COARSE / 2, LAT1, COARSE) for b in np.arange(LON0 + COARSE / 2, LON1, COARSE)]
    c = AdaptiveCrawler(lambda cell: probe(cell, task), key=lambda it: it["id"], cell_id=lambda c: f"{c[0]:.4f},{c[1]:.4f}", max_workers=U.WORKERS,
                        limiter=U.limiter, checkpoint=Checkpoint(out / f"{task['name'].lower()}_coarse.checkpoint.json"), max_rate_limited=300)
    st = c.run(cells)
    pos = [tuple(map(float, k.split(","))) for k, v in c.items.items() if v["n"] > 0]
    print(f"  coarse {task['name']}: probes={len(cells)} nonempty={len(pos)} failed={st.cells_failed} aborted={st.aborted}", flush=True)
    if st.aborted or st.cells_failed:
        sys.exit("coarse scouting incomplete")
    f.write_text(json.dumps(pos))
    return pos


summary = {"authorization_ref": U.ref, "tasks": {}}
for name in ["PARKING", "METRO", "ATTRACTIONS", "STREETS", "BUILDINGS"]:
    task = next(t for t in U.TASKS if t["name"] == name)
    t0 = time.time()
    print(f">>> {name}", flush=True)
    pos = positive_cells(task)
    h = COARSE / 2
    fine = set()
    for lat, lon in pos:
        for a in np.arange(lat - h, lat + h, task["step"]):
            for b in np.arange(lon - h, lon + h, task["step"]):
                fine.add((round(float(a), 5), round(float(b), 5)))
    fine = sorted(fine)
    print(f"  fine sectors: {len(fine)}", flush=True)
    f = out / f"{name.lower()}_ids.txt"
    known = set(f.read_text().split()) if f.exists() else set()
    fh = open(f, "a")

    def on_items(new):
        for it in new:
            if it["id"] not in known:
                known.add(it["id"])
                fh.write(it["id"] + chr(10))
        fh.flush()

    c = AdaptiveCrawler(lambda cell: U.scan(cell, task), key=lambda it: it["id"], cell_id=lambda c: f"{c[0]:.5f},{c[1]:.5f}", max_workers=U.WORKERS,
                        limiter=U.limiter, checkpoint=Checkpoint(out / f"{name.lower()}_scan.checkpoint.json"), on_items=on_items, max_rate_limited=500)
    st = c.run(fine)
    fh.close()
    print(f"  scan {name}: sectors={st.cells_total} done={st.cells_done} failed={st.cells_failed} ids={len(known)} 429={st.rate_limited} aborted={st.aborted} {time.time() - t0:.0f}s", flush=True)
    if st.aborted and st.aborted != "cancelled":
        summary["tasks"][name] = {"aborted": st.aborted}
        break
    got, st2 = U.run_details(task, sorted(known), out)
    summary["tasks"][name] = {"ids": len(known), "records": got, "scan_failed_sectors": st.cells_failed, "detail_batches_failed": st2.cells_failed,
                              "seconds": round(time.time() - t0)}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
print("ALL DONE", json.dumps(summary), flush=True)
