"""City-wide authorized Suhail acquisition: parcel polygons by tile strips (bounded memory), then zoning rules in resumable chunks.
Env: GEOACQUIRER_ENABLE_SUHAIL=1, GEOACQUIRER_SUHAIL_AUTHORIZATION_REF. Args: out_dir tile_workers rules_workers
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from geoacquirer.acquisition import AdaptiveCrawler, Checkpoint
from geoacquirer.acquisition.grid import tiles_for_bbox
from geoacquirer.providers.suhail import SuhailProvider, ZOOM, merge_parts

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
TW = int(sys.argv[2]) if len(sys.argv) > 2 else 12
RW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
BBOX = (46.40, 24.40, 47.10, 25.15)  # Riyadh bounds from suhail_ai.ipynb (W,S,E,N)
sp = SuhailProvider("riyadh", max_workers=TW, min_interval_s=0.0)

# ---- stage 1: parcels, tile strips
tiles = sorted(tiles_for_bbox(BBOX, ZOOM))
xs = sorted({x for x, _ in tiles})
strips = [xs[i:i + 12] for i in range(0, len(xs), 12)]
print(f"tiles={len(tiles)} strips={len(strips)}", flush=True)
for k, sx in enumerate(strips):
    f = out / f"parcels_part{k:02d}.parquet"
    if f.exists():
        continue
    sxs = set(sx)
    cells = [t for t in tiles if t[0] in sxs]
    t0 = time.time()
    c = AdaptiveCrawler(sp._fetch_tile, key=lambda it: (it["parcel_id"], it["geometry"].wkb_hex[:64]), cell_id=lambda c: f"{c[0]}/{c[1]}",
                        max_workers=TW, limiter=sp.limiter, checkpoint=Checkpoint(out / f"tiles_strip{k:02d}.json"), max_rate_limited=300)
    st = c.run(cells)
    if st.aborted or st.cells_failed:
        print("strip", k, "INCOMPLETE", st.aborted, st.cells_failed, flush=True)
        sys.exit(1)
    g = merge_parts(list(c.items.values()))
    g.to_parquet(f)
    print(f"strip {k + 1}/{len(strips)}: tiles={len(cells)} parcels={len(g)} {time.time() - t0:.0f}s rate_limited={st.rate_limited}", flush=True)

final = out / "parcels.parquet"
if not final.exists():
    parts = [gpd.read_parquet(f) for f in sorted(out.glob("parcels_part*.parquet"))]
    g = pd.concat(parts, ignore_index=True)
    dup = g["parcel_id"].duplicated(keep=False)
    merged = g[dup].groupby("parcel_id").agg(parcel_objectid=("parcel_objectid", "first"), n_tile_parts=("n_tile_parts", "sum"),
                                              geometry=("geometry", lambda s: unary_union(list(s)))).reset_index()
    g = pd.concat([g[~dup], gpd.GeoDataFrame(merged, geometry="geometry", crs=4326)], ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    g.to_parquet(final)
    print("parcels total", len(g), "rows merged across strips:", int(dup.sum()), flush=True)
else:
    g = gpd.read_parquet(final)
ids = g["parcel_objectid"].astype(str).tolist()
print(f"RULES STAGE: {len(ids)} parcels, workers={RW}", flush=True)

# ---- stage 2: zoning rules in resumable chunks
sp2 = SuhailProvider("riyadh", max_workers=RW, min_interval_s=0.0)
parts_dir = out / "rules_parts"
parts_dir.mkdir(exist_ok=True)
CH = 10000
for i in range(0, len(ids), CH):
    f = parts_dir / f"rules_{i // CH:05d}.csv"
    if f.exists():
        continue
    t0 = time.time()
    d = sp2.fetch_details("parcel", ids[i:i + CH])
    if sp2.stats.aborted:
        print("ABORTED:", sp2.stats.aborted, flush=True)
        sys.exit(1)
    d.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"rules chunk {i // CH + 1}/{(len(ids) + CH - 1) // CH}: rows={len(d)} failed_batches={sp2.stats.cells_failed} 429s={sp2.stats.rate_limited} {time.time() - t0:.0f}s", flush=True)

# ---- retry parcels with no response, then merge
rules = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(parts_dir.glob("*.csv"))], ignore_index=True)
miss = sorted(set(ids) - set(rules["parcel_objectid"]))
print("missing after pass 1:", len(miss), flush=True)
if miss:
    d = sp2.fetch_details("parcel", miss)
    rules = pd.concat([rules, d.astype(str)], ignore_index=True)
rules.to_csv(out / "rules.csv", index=False, encoding="utf-8-sig")
(out / "manifest.json").write_text(json.dumps({"provider": "suhail", "authorization_ref": sp.authorization_ref, "bbox": BBOX, "parcels": len(g),
                                               "rules_rows": len(rules), "api_status": rules["api_status"].value_counts().to_dict(),
                                               "still_missing": len(set(ids) - set(rules["parcel_objectid"]))}, indent=2))
print("DONE", len(g), len(rules), flush=True)
