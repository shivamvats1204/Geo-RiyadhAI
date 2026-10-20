"""Authorized Urbi/2GIS parking acquisition (scout/drill), resumable output.
Requires GEOACQUIRER_ENABLE_URBI=1, GEOACQUIRER_URBI_AUTHORIZATION_REF, GEOACQUIRER_URBI_API_KEY (env only; never stored).
Usage: python scripts/run_urbi.py [out_dir] [west south east north]
"""
import json, sys, time
from pathlib import Path
from geoacquirer.providers.urbi import UrbiProvider

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_urbi")
bbox = tuple(float(x) for x in sys.argv[2:6]) if len(sys.argv) >= 6 else (46.62, 24.64, 46.74, 24.78)
out.mkdir(parents=True, exist_ok=True)
p = UrbiProvider(max_workers=4, min_interval_s=0.05)
t = time.time()
fb = p.fetch_features("parking", bbox)
g = fb.gdf
g.to_parquet(out / "parking.parquet")
g.drop(columns="geometry").assign(wkt=g.geometry.to_wkt()).to_csv(out / "parking.csv", index=False, encoding="utf-8-sig")
(out / "manifest.json").write_text(json.dumps({"provider": "urbi", "authorization_ref": p.authorization_ref, "bbox": bbox, "rows": len(g),
    "crawl": p.stats.__dict__, "warnings": fb.warnings}, indent=2, default=str))
print(f"done {len(g)} parking rows in {time.time()-t:.0f}s stats={p.stats.__dict__} warnings={fb.warnings}", flush=True)
