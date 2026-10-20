"""Riyadh Metro lines and stations from Suhail's vector tiles (`metro_lines`, `metro_stations`), saved into the city store for the analysis agent.

Output: <city store>/metro_network/{lines,stations}.parquet. Zoom 11 is enough for line geometry and station points. Lines cut at tile borders are re-joined per line name.
Needs recorded authorization for Suhail (GEOACQUIRER_ENABLE_SUHAIL=1 and GEOACQUIRER_SUHAIL_AUTHORIZATION_REF).
"""
import argparse
import math
import sys
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
from riyadhai.paths import city_store  # noqa: E402

Z, EXT = 11, 4096


def to_lonlat(arr, x, y):
    n = 2 ** Z
    return np.column_stack([(x + arr[:, 0] / EXT) / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + arr[:, 1] / EXT) / n))))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=(46.2, 24.2, 47.5, 25.4), metavar=("W", "S", "E", "N"))
    ap.add_argument("--slug", default="riyadh")
    ns = ap.parse_args()
    require_authorized("suhail")
    w, s, e, n = ns.bbox
    tx = lambda lon: int((lon + 180) / 360 * 2 ** Z)  # noqa: E731
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** Z)  # noqa: E731
    lines, stations = {}, {}
    for x in range(tx(w), tx(e) + 1):
        for y in range(ty(n), ty(s) + 1):
            r = requests.get(f"https://tiles.suhail.ai/maps/{ns.slug}/{Z}/{x}/{y}.vector.pbf", headers={"User-Agent": "geoacquirer/0.1 (+authorized-research)"}, timeout=30)
            if r.status_code != 200:
                continue
            tile = mapbox_vector_tile.decode(r.content, default_options={"y_coord_down": True})
            for f in tile.get("metro_lines", {}).get("features", []):
                lines.setdefault(f["properties"].get("track_name"), []).append((f["properties"], shapely.transform(shape(f["geometry"]), lambda a: to_lonlat(a, x, y))))
            for f in tile.get("metro_stations", {}).get("features", []):
                p = f["properties"]
                stations[(p.get("station_code"), p.get("station_name"))] = shapely.transform(shape(f["geometry"]), lambda a: to_lonlat(a, x, y))
    out = city_store() / "metro_network"
    out.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame([{"line": k, "color": v[0][0].get("track_color"), "geometry": unary_union([g for _, g in v])} for k, v in lines.items()], geometry="geometry", crs=4326).to_parquet(out / "lines.parquet")
    gpd.GeoDataFrame([{"station_code": k[0], "name": k[1], "geometry": g} for k, g in stations.items()], geometry="geometry", crs=4326).to_parquet(out / "stations.parquet")
    print("lines:", {k: len(v) for k, v in lines.items()}, "| stations:", len(stations), "| saved to", out)


if __name__ == "__main__":
    main()
