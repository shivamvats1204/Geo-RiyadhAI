"""Authorized Suhail acquisition: parcel polygons (tiles) then zoning rules, chunked and resumable.

Requires GEOACQUIRER_ENABLE_SUHAIL=1 and GEOACQUIRER_SUHAIL_AUTHORIZATION_REF (recorded in the output manifest).
Usage: python scripts/run_suhail.py [out_dir] [west south east north]
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from geoacquirer.providers.suhail import SuhailProvider

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_suhail")
bbox = tuple(float(x) for x in sys.argv[2:6]) if len(sys.argv) >= 6 else (46.62, 24.64, 46.74, 24.78)
out.mkdir(parents=True, exist_ok=True)
p = SuhailProvider("riyadh", max_workers=4, min_interval_s=0.02)
par = out / "parcels.parquet"
if par.exists():
    import geopandas as gpd
    gdf = gpd.read_parquet(par)
    print("reusing parcels", len(gdf), flush=True)
else:
    t = time.time()
    fb = p.fetch_features("parcel", bbox, checkpoint_path=out / "tiles.checkpoint.json")
    gdf = fb.gdf
    gdf.to_parquet(par)
    print(f"parcels {len(gdf)} in {time.time()-t:.0f}s stats={p.stats.__dict__} warnings={fb.warnings}", flush=True)
    if fb.warnings:
        sys.exit("tile crawl incomplete; re-run to resume")
ids = gdf["parcel_objectid"].astype(str).tolist()
CH = 4000
parts = out / "rules_parts"
parts.mkdir(exist_ok=True)
for i in range(0, len(ids), CH):
    f = parts / f"rules_{i // CH:04d}.csv"
    if f.exists():
        continue
    t = time.time()
    d = p.fetch_details("parcel", ids[i:i + CH])
    got = set(d["parcel_objectid"].astype(str)) if len(d) else set()
    miss = len(set(ids[i:i + CH]) - got)
    if p.stats.aborted:
        print("ABORTED:", p.stats.aborted, flush=True)
        sys.exit(1)
    d.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"chunk {i // CH + 1}/{(len(ids) + CH - 1) // CH}: {len(d)} rows, {miss} parcels missing (will be retried on re-run? no: re-run this chunk by deleting {f.name}) in {time.time()-t:.0f}s", flush=True)
rules = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(parts.glob("*.csv"))], ignore_index=True)
rules.to_csv(out / "rules.csv", index=False, encoding="utf-8-sig")
(out / "manifest.json").write_text(json.dumps({"provider": "suhail", "authorization_ref": p.authorization_ref, "bbox": bbox, "parcels": len(gdf),
                                               "rules_rows": len(rules), "api_status": rules["api_status"].value_counts().to_dict()}, indent=2), encoding="utf-8")
print("done", len(gdf), len(rules), flush=True)
