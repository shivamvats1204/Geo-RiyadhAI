"""Build the district registry and the district-partitioned city store (run once; ~10-20 minutes).

Env: GEOACQUIRER_URBI_API_KEY (only needed to fetch Arabic names / any extra districts found by the catalogue scan).
"""
import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riyadhai import districts as D, partition as P  # noqa: E402
from riyadhai.paths import city_store, partition_root, raw_dir  # noqa: E402

SRC = city_store()
DST = partition_root()
UD = raw_dir("urbi_districts")
BASE = "https://catalog.api.2gis.ru/3.0"
FIELDS = "items.geometry.centroid,items.geometry.selection,items.geometry.hover,items.name,items.full_name,items.type,items.subtype,items.subtype_name,items.adm_div,items.id"
KEY = os.environ.get("GEOACQUIRER_URBI_API_KEY")


def byid(ids, locale, fields):
    out = {}
    for i in range(0, len(ids), 25):
        b = ids[i:i + 25]
        for k in range(5):
            r = requests.get(f"{BASE}/items/byid", params={"id": ",".join(b), "locale": locale, "fields": fields, "key": KEY}, timeout=40)
            if r.status_code == 200 and r.json().get("meta", {}).get("code") == 200:
                for it in r.json()["result"].get("items", []):
                    out[it["id"].split("_")[0]] = it
                break
            time.sleep(2 * (k + 1))
        else:
            raise RuntimeError(f"byid failed for {b[:2]}…")
    return out


def main():
    DST.mkdir(parents=True, exist_ok=True)
    raw = json.loads((UD / "divisions_raw.json").read_text(encoding="utf-8"))
    sc = UD / "scan_result.json"
    new = []
    if sc.exists():
        new = list(json.loads(sc.read_text(encoding="utf-8"))["new"])
    if new:
        print("scan found", len(new), "districts not referenced by any record; fetching their geometry")
        raw += list(byid(new, "en_SA", FIELDS).values())
        (UD / "divisions_raw_plus_scan.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    ids = [r["id"].split("_")[0] for r in raw]
    ar = byid(ids, "ar_SA", "items.name,items.id")
    sp = gpd.read_parquet(raw_dir("suhail_neighborhoods") / "neighborhoods.parquet")
    sc_ = gpd.read_parquet(raw_dir("suhail_neighborhoods") / "neighborhood_centroids.parquet")
    ov = gpd.read_parquet(raw_dir("overture_districts") / "overture_neighborhoods.parquet")
    reg = D.build_registry(raw, ar, sp, sc_, ov)
    cw = D.crosswalk(reg)
    reg = reg.merge(cw, on="uid", how="left")
    reg["best_urbi_name"] = reg.best_urbi.map(dict(zip(reg.uid, reg.name)))
    reg.to_parquet(DST / "registry.parquet")
    print("registry", len(reg), reg.groupby(["source", "kind"]).size().to_dict(), "no polygon:", int((~reg.has_polygon).sum()))
    report = P.build(SRC, DST, reg, ["urbi_building", "parcel", "urbi_parking", "street", "attraction", "metro"])
    flat = pd.read_parquet(DST / "assignments_flat.parquet")
    counts: dict = {}
    for col in ("urbi_district", "suhail_nbh", "overture_nbh"):
        for (uid, layer), n in flat.groupby([col, "layer"]).size().items():
            counts.setdefault(uid, {})[layer] = int(n)
    layers = {k: v["rows_out"] for k, v in report.items()}
    summary = {"layers": layers, "partition": report, "district_counts": counts, "registry_rows": len(reg),
               "registry_by_source_kind": {f"{a}:{b}": int(n) for (a, b), n in reg.groupby(["source", "kind"]).size().items()},
               "registry_no_polygon": reg.loc[~reg.has_polygon, "uid"].tolist(),
               "crosswalk_overlap_median": float(reg.overlap_share.median(skipna=True)), "scan_extra_districts": len(new)}
    (DST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", {k: v["rows_out"] for k, v in report.items()})


if __name__ == "__main__":
    main()
