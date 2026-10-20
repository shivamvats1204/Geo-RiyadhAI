"""Collect every Urbi/2GIS administrative area (district, living area, settlement, district area) with its polygon and centroid.

1. stream the downloaded JSONL dumps and collect the `adm_div` entries the records point to
2. fetch each area's polygon and centroid by id from the catalogue API
3. optional (--scan): sweep the whole city extent on a grid for areas that NO downloaded record references (found 18 extra in Riyadh)
Output: <data root>/urbi_districts/divisions_raw.json (+ scan_result.json with --scan), read by scripts/build_partition.py

Needs: GEOACQUIRER_ENABLE_URBI=1, GEOACQUIRER_URBI_AUTHORIZATION_REF and GEOACQUIRER_URBI_API_KEY in the environment (see .env.example). The key is never logged.
Usage: python scripts/prepare_urbi_divisions.py [--scan]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoacquirer.providers.registry import require_authorized  # noqa: E402
from riyadhai.citystore import CityStore  # noqa: E402
from riyadhai.paths import city_store, raw_dir  # noqa: E402

BASE = "https://catalog.api.2gis.ru/3.0"
FIELDS = "items.geometry.centroid,items.geometry.selection,items.geometry.hover,items.name,items.full_name,items.type,items.subtype,items.subtype_name,items.adm_div,items.id"
KINDS = ("district", "living_area", "settlement", "district_area")
SKIP_TYPES = ("country", "region", "city")


def api_key() -> str:
    key = os.environ.get("GEOACQUIRER_URBI_API_KEY")
    if not key:
        sys.exit("set GEOACQUIRER_URBI_API_KEY (see .env.example)")
    return key


def referenced_divisions(urbi_dir: Path) -> dict:
    found = {}
    for f in sorted(urbi_dir.glob("*_data.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                for a in d.get("adm_div") or []:
                    if a.get("type") not in SKIP_TYPES:
                        found[a["id"]] = {"type": a.get("type"), "id": a["id"], "name": a.get("name")}
        print("scanned", f.name, len(found), "areas so far", flush=True)
    return found


def fetch_by_id(ids: list, key: str) -> dict:
    out = {}
    for i in range(0, len(ids), 25):
        batch = ids[i:i + 25]
        for attempt in range(5):
            r = requests.get(f"{BASE}/items/byid", params={"id": ",".join(batch), "locale": "en_SA", "fields": FIELDS, "key": key}, timeout=40)
            if r.status_code == 200 and r.json().get("meta", {}).get("code") == 200:
                for it in r.json()["result"].get("items", []):
                    out[it["id"].split("_")[0]] = it
                break
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"catalogue lookup failed for ids starting {batch[0]}")
    return out


def grid_scan(bbox: tuple, key: str, known: set, step: float = 0.025, workers: int = 8) -> dict:
    w, s, e, n = bbox
    pts = [(x, y) for x in [w + i * step for i in range(int((e - w) / step) + 2)] for y in [s + j * step for j in range(int((n - s) / step) + 2)]]
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2))
    found = {}

    def query(arg):
        (x, y), kind = arg
        for page in (1, 2, 3):
            for attempt in range(5):
                try:
                    r = sess.get(f"{BASE}/items", params={"type": f"adm_div.{kind}", "point": f"{x:.5f},{y:.5f}", "radius": 2000, "page": page, "page_size": 50, "locale": "en_SA",
                                                          "fields": "items.name,items.subtype,items.id", "key": key}, timeout=30)
                    j = r.json()
                    if r.status_code == 404 or j["meta"]["code"] == 404:
                        return
                    if r.status_code == 200 and j["meta"]["code"] == 200:
                        break
                except Exception:  # noqa: BLE001  network hiccup: retry with backoff
                    pass
                time.sleep(1 + attempt)
            else:
                return
            items = j.get("result", {}).get("items", [])
            for it in items:
                found[it["id"].split("_")[0]] = (it.get("subtype"), it["name"])
            if len(items) < 50:
                return

    tasks = [(p, k) for p in pts for k in ("district", "living_area", "settlement")]
    print(len(tasks), "catalogue queries", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(query, tasks))
    return {i: v for i, v in found.items() if i not in known}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urbi-dir", type=Path, default=raw_dir("urbi_city"))
    ap.add_argument("--out-dir", type=Path, default=raw_dir("urbi_districts"))
    ap.add_argument("--scan", action="store_true", help="also sweep the city on a grid for areas no record references")
    ns = ap.parse_args()
    require_authorized("urbi")
    key = api_key()
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    seen = referenced_divisions(ns.urbi_dir)
    items = fetch_by_id([i for i, v in seen.items() if v["type"] in KINDS], key)
    (ns.out_dir / "divisions_raw.json").write_text(json.dumps(list(items.values()), ensure_ascii=False), encoding="utf-8")
    print("referenced areas fetched:", len(items), flush=True)
    if ns.scan:
        tiles = CityStore(city_store()).index("urbi_building")["tiles"]
        xs = [int(t[1:].split("_")[0]) * 0.05 for t in tiles]
        ys = [int(t[1:].split("_")[1]) * 0.05 for t in tiles]
        extra = grid_scan((min(xs), min(ys), max(xs) + 0.05, max(ys) + 0.05), key, set(items))
        (ns.out_dir / "scan_result.json").write_text(json.dumps({"new": extra}, ensure_ascii=False), encoding="utf-8")
        print("areas found by the sweep that no record references:", len(extra))


if __name__ == "__main__":
    main()
