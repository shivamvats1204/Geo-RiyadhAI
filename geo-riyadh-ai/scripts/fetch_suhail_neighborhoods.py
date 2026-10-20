"""Neighbourhood polygons from Suhail's vector tiles (the `neighborhoods` layer; pieces clipped at tile borders are re-joined by neighborhood_id).

Output: <data root>/suhail_neighborhoods/{neighborhoods,neighborhood_centroids}.parquet
Needs recorded authorization for Suhail (GEOACQUIRER_ENABLE_SUHAIL=1 and GEOACQUIRER_SUHAIL_AUTHORIZATION_REF); uses 4 polite workers and an honest User-Agent.
The extent defaults to the area covered by the city store's parcels; override with --bbox west south east north.
"""
import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import requests
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoacquirer.providers.registry import require_authorized  # noqa: E402
from riyadhai.citystore import CityStore  # noqa: E402
from riyadhai.paths import city_store, raw_dir  # noqa: E402

Z, EXT = 13, 4096
UA = "geoacquirer/0.1 (+authorized-research)"


def to_lonlat(arr, x, y):
    n = 2 ** Z
    lon = (x + arr[:, 0] / EXT) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + arr[:, 1] / EXT) / n))))
    return np.column_stack([lon, lat])


def default_bbox() -> tuple:
    tiles = CityStore(city_store()).index("parcel")["tiles"]
    xs = [int(t[1:].split("_")[0]) * 0.05 for t in tiles]
    ys = [int(t[1:].split("_")[1]) * 0.05 for t in tiles]
    return min(xs) - 0.05, min(ys) - 0.05, max(xs) + 0.10, max(ys) + 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    ap.add_argument("--slug", default="riyadh", help="Suhail region slug")
    ap.add_argument("--out-dir", type=Path, default=raw_dir("suhail_neighborhoods"))
    ns = ap.parse_args()
    require_authorized("suhail")
    w, s, e, n = ns.bbox or default_bbox()
    tx = lambda lon: int((lon + 180) / 360 * 2 ** Z)  # noqa: E731
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** Z)  # noqa: E731
    tiles = [(x, y) for x in range(tx(w), tx(e) + 1) for y in range(ty(n), ty(s) + 1)]
    print("bbox", (w, s, e, n), "tiles", len(tiles), flush=True)
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8))
    sess.headers["User-Agent"] = UA
    polys, cents, stat = [], [], {"ok": 0, "empty": 0, "fail": 0}

    def one(t):
        x, y = t
        for k in range(5):
            try:
                r = sess.get(f"https://tiles.suhail.ai/maps/{ns.slug}/{Z}/{x}/{y}.vector.pbf", timeout=30)
                if r.status_code == 404 or (r.status_code == 200 and not r.content):
                    stat["empty"] += 1
                    return
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1 + k)
        else:
            stat["fail"] += 1
            print("FAILED tile", t, flush=True)
            return
        stat["ok"] += 1
        tile = mapbox_vector_tile.decode(r.content, default_options={"y_coord_down": True})
        for f in tile.get("neighborhoods", {}).get("features", []):
            g = shape(f["geometry"])
            if g.geom_type in ("Polygon", "MultiPolygon"):
                polys.append((f["properties"], shapely.transform(g, lambda a: to_lonlat(a, x, y))))
        for f in tile.get("neighborhoods-centroids", {}).get("features", []):
            cents.append((f["properties"], shapely.transform(shape(f["geometry"]), lambda a: to_lonlat(a, x, y))))
        time.sleep(0.05)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, tiles))
    print(stat, "polygon pieces", len(polys), "centroids", len(cents), flush=True)
    if stat["fail"]:
        sys.exit("some tiles failed: not writing an incomplete result; run again")
    by = {}
    for p, g in polys:
        by.setdefault(p.get("neighborhood_id"), []).append((p, g))
    recs = [{"neighborhood_id": nid, "name_ar": v[0][0].get("neighborh_aname"), "province_id": v[0][0].get("province_id"), "region_id": v[0][0].get("region_id"),
             "shape_area": v[0][0].get("shape_area"), "price_of_meter": v[0][0].get("price_of_meter"), "n_pieces": len(v), "geometry": unary_union([g for _, g in v])} for nid, v in by.items()]
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(ns.out_dir / "neighborhoods.parquet")
    cs = gpd.GeoDataFrame([{"neighborhood_id": p.get("neighborhood_id"), "name_ar": p.get("neighborh_aname"), "geometry": g} for p, g in cents], geometry="geometry", crs=4326).drop_duplicates("neighborhood_id")
    cs.to_parquet(ns.out_dir / "neighborhood_centroids.parquet")
    print("neighbourhoods:", len(gdf), "| centroid-only ids:", len(set(cs.neighborhood_id) - set(gdf.neighborhood_id)))


if __name__ == "__main__":
    main()
