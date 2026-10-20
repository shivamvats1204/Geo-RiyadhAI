"""Pre-generate the vector tiles of the partitioned city store (low memory; resumable per layer).

Phase A  stream every stored parquet file; for each feature compute the tiles (at the layer's generation zoom) its bounds touch and append it to
         that tile's bucket. A feature spanning several tiles is replicated into each of them, so no tile is missing a feature that crosses into it.
Phase B  per tile: read its bucket, encode to MVT (polygons, lines, points; sub-4-unit^2 polygons become centroid points), write
         tiles/<layer>/<z>/<x>/<y>.pbf + .json (input/polygon/point counts).
Usage: python scripts/pregen_tiles.py [layer ...]
"""
import json
import math
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riyadhai.city_api import LAYERS, ORIGIN, PART_COLS, TILE_Z, CityTiles, tile_lonlat_bbox  # noqa: E402
from riyadhai.paths import partition_root  # noqa: E402

PART = partition_root()
_ct = None


def tile_ranges(g: gpd.GeoDataFrame, z: int):
    b = g.geometry.bounds
    n = 2 ** z
    size = 2 * ORIGIN / n
    unit = size / 4096
    pad = 64 * unit
    def mx(lon): return np.asarray(lon) * ORIGIN / 180.0
    def my(lat): return np.log(np.tan((90.0 + np.asarray(lat)) * math.pi / 360.0)) * 6378137.0
    x0 = np.floor((mx(b.minx.to_numpy()) - pad + ORIGIN) / size).astype(int)
    x1 = np.floor((mx(b.maxx.to_numpy()) + pad + ORIGIN) / size).astype(int)
    y0 = np.floor((ORIGIN - (my(b.maxy.to_numpy()) + pad)) / size).astype(int)
    y1 = np.floor((ORIGIN - (my(b.miny.to_numpy()) - pad)) / size).astype(int)
    return np.clip(x0, 0, n - 1), np.clip(x1, 0, n - 1), np.clip(y0, 0, n - 1), np.clip(y1, 0, n - 1)


def phase_a(layer: str, z: int, log):
    idcol, _, props = LAYERS[layer]
    want = ["geometry", idcol, *PART_COLS, *[p for p in props if p != idcol]]
    bdir = PART / "buckets" / layer / str(z)
    if bdir.exists():
        shutil.rmtree(bdir)
    n_in = n_assign = 0
    files = sorted((PART / layer).glob("t*__*.parquet"))
    for k, f in enumerate(files):
        have = set(pq.read_schema(f).names)
        g = gpd.read_parquet(f, columns=[c for c in want if c in have])
        n_in += len(g)
        x0, x1, y0, y1 = tile_ranges(g, z)
        rows, tx, ty = [], [], []
        geoms = g.geometry.values
        span = (x1 - x0 + 1) * (y1 - y0 + 1)
        for i in range(len(g)):
            if span[i] <= 4:  # compact feature: its bbox tiles are the tiles it touches (within the clip margin)
                for xx in range(x0[i], x1[i] + 1):
                    for yy in range(y0[i], y1[i] + 1):
                        rows.append(i); tx.append(xx); ty.append(yy)
            else:  # long/large feature (streets, big parcels): keep only the tiles whose rectangle it really intersects
                cx, cy = np.meshgrid(np.arange(x0[i], x1[i] + 1), np.arange(y0[i], y1[i] + 1))
                cx, cy = cx.ravel(), cy.ravel()
                boxes = shapely.box(*np.array([tile_lonlat_bbox(z, int(a), int(b)) for a, b in zip(cx, cy)]).T)
                shapely.prepare(geoms[i])
                ok = shapely.intersects(geoms[i], boxes)
                for a, b in zip(cx[ok], cy[ok]):
                    rows.append(i); tx.append(int(a)); ty.append(int(b))
        df = pd.DataFrame({"r": rows, "x": tx, "y": ty})
        n_assign += len(df)
        for (xx, yy), grp in df.groupby(["x", "y"]):
            d = bdir / f"{xx}_{yy}"
            d.mkdir(parents=True, exist_ok=True)
            g.iloc[grp.r.to_numpy()].to_parquet(d / f"{k:04d}.parquet")
        if k % 100 == 0:
            log(f"  A {layer} {k}/{len(files)} files")
    log(f"A {layer}: {n_in} features -> {n_assign} tile memberships ({n_assign - n_in} replicas for features spanning tiles)")
    return n_in, n_assign


def _encode_tile(args):
    global _ct
    layer, z, d = args
    if _ct is None:
        _ct = CityTiles(PART, gpd.read_parquet(PART / "registry.parquet"))
    x, y = (int(v) for v in Path(d).name.split("_"))
    g = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(f) for f in sorted(Path(d).glob("*.parquet"))], ignore_index=True), geometry="geometry", crs=4326)
    idcol = LAYERS[layer][0]
    g = g.drop_duplicates(subset=[idcol])  # a feature may be listed once per bucket file only; guard against any repeat
    data, info = _ct.encode_frame(layer, z, x, y, g)
    out = PART / "tiles" / layer / str(z) / str(x)
    out.mkdir(parents=True, exist_ok=True)
    if data:
        (out / f"{y}.pbf").write_bytes(data)
    (out / f"{y}.json").write_text(json.dumps(info))
    return info


def phase_b(layer: str, z: int, workers: int, log):
    out = PART / "tiles" / layer / str(z)
    if out.exists():
        shutil.rmtree(out)
    dirs = sorted(p for p in (PART / "buckets" / layer / str(z)).iterdir() if p.is_dir())
    tot = {"input": 0, "polygons": 0, "points": 0, "lines": 0, "centroid_fallback": 0, "tiles": 0}
    t0 = time.time()
    with Pool(workers) as pool:
        for i, info in enumerate(pool.imap_unordered(_encode_tile, [(layer, z, str(d)) for d in dirs], chunksize=2)):
            for k in ("input", "polygons", "points", "lines", "centroid_fallback"):
                tot[k] += info[k]
            tot["tiles"] += 1
            if i % 100 == 0:
                log(f"  B {layer} {i}/{len(dirs)} tiles {round(time.time() - t0)}s")
    log(f"B {layer}: {tot}")
    return tot


def main(layers, workers=3):
    log = lambda *a: print(*a, flush=True)
    report = {}
    rp = PART / "tiles_report.json"
    if rp.exists():
        report = json.loads(rp.read_text())
    for layer in layers:
        for z in TILE_Z[layer]:
            t0 = time.time()
            n_in, n_assign = phase_a(layer, z, log)
            tot = phase_b(layer, z, workers, log)
            report[f"{layer}@{z}"] = {"zoom": z, "stored_features": n_in, "tile_memberships": n_assign, **tot, "seconds": round(time.time() - t0)}
            rp.write_text(json.dumps(report, indent=1))
            shutil.rmtree(PART / "buckets" / layer / str(z), ignore_errors=True)
    log("done", report)


if __name__ == "__main__":
    main(sys.argv[1:] or ["metro", "attraction", "urbi_parking", "street", "urbi_building", "parcel"])
