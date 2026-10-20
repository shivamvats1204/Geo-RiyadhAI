"""Verify the plotted city data at random places against independent sources.

For each random spot (a 300 m box) it compares
  raw       Urbi building records straight from the downloaded JSONL (never touched by the conversion/partition code)
  store     the partitioned store the map reads (what the vector tiles are cut from)
  tiles     the vector tiles actually served (decoded): every stored feature must be present
  suhail    a FRESH download of the Suhail z15 tile for the spot vs the stored parcels
  overture  Overture building footprints (a different dataset): match rate of Urbi buildings against them
and writes results/city_spot_verification.json. Browser rendering is checked separately in the UI (queryRenderedFeatures).
"""
import json
import math
import random
import sys
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import requests
import shapely
from shapely.geometry import box

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from riyadhai.city_api import CityTiles  # noqa: E402
from riyadhai.paths import partition_root, raw_dir  # noqa: E402

PART = partition_root()
RAW = raw_dir("urbi_city") / "buildings_data.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HALF_M = 150.0
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 20261020)


def bbox_around(lon, lat, half=HALF_M):
    dlat = half / 110_800.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def pick_spots(reg: gpd.GeoDataFrame, tiles: CityTiles):
    """Half the spots are random buildings (so dense areas), half random points inside random district polygons (so sparse ones too)."""
    spots = []
    idx = json.loads((PART / "urbi_building" / "index.json").read_text())["tiles"]
    names = sorted(idx)
    for _ in range(N // 2):
        t = random.choice(names)
        g = gpd.read_parquet(next((PART / "urbi_building").glob(f"{t}__*.parquet")), columns=["geometry"])
        p = g.geometry.iloc[random.randrange(len(g))].representative_point()
        spots.append({"how": "random building", "lon": p.x, "lat": p.y})
    ud = reg[(reg.source == "urbi") & (reg.kind == "district") & reg.has_polygon]
    while len(spots) < N:
        r = ud.iloc[random.randrange(len(ud))]
        minx, miny, maxx, maxy = r.geometry.bounds
        for _ in range(200):
            p = shapely.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if r.geometry.contains(p):
                spots.append({"how": f"random point in district {r['name']}", "lon": p.x, "lat": p.y})
                break
    for s in spots:
        s["bbox"] = bbox_around(s["lon"], s["lat"])
    return spots


def raw_ids(spots):
    """One pass over the raw JSONL: ids whose centroid lies in each spot box."""
    found = [dict() for _ in spots]
    for line in open(RAW, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        pt = d.get("point") or {}
        lon, lat = pt.get("lon"), pt.get("lat")
        if lon is None:
            continue
        for s, f in zip(spots, found):
            w, so, e, n = s["bbox"]
            if w <= lon <= e and so <= lat <= n:
                f[str(d["id"]).split("_")[0]] = (lon, lat)
    return found


def decode_tiles(base: str, layer: str, bbox, z=14):
    """Fetch the served vector tiles covering bbox and return the set of feature ids they contain (clipped duplicates collapse)."""
    w, s, e, n = bbox
    nn = 2 ** z
    x0, x1 = int((w + 180) / 360 * nn), int((e + 180) / 360 * nn)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * nn)
    ids = set()
    for x in range(x0, x1 + 1):
        for y in range(ty(n), ty(s) + 1):
            r = requests.get(f"{base}/v1/city/tiles/{layer}/{z}/{x}/{y}.pbf", timeout=120)
            if r.status_code == 200 and r.content:
                for lname, ly in mapbox_vector_tile.decode(r.content).items():
                    ids |= {str(f["properties"]["i"]) for f in ly["features"]}
    return ids


def suhail_fresh(bbox):
    from geoacquirer.providers.suhail import decode_tile_parcels
    w, s, e, n = bbox
    z = 15
    nn = 2 ** z
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * nn)
    out = {}
    for x in range(int((w + 180) / 360 * nn), int((e + 180) / 360 * nn) + 1):
        for y in range(ty(n), ty(s) + 1):
            r = requests.get(f"https://tiles.suhail.ai/maps/riyadh/{z}/{x}/{y}.vector.pbf", headers={"User-Agent": "geoacquirer/0.1 (+authorized-research)"}, timeout=60)
            if r.status_code == 200:
                for p in decode_tile_parcels(r.content, z, x, y):
                    if p["geometry"].intersects(box(w, s, e, n)):
                        out[p["parcel_objectid"]] = p["geometry"]
    return out


def overture_compare(bbox, urbi_polys: gpd.GeoDataFrame):
    from geoacquirer.providers.overture import OvertureProvider
    fb = OvertureProvider().fetch_features("building", bbox, max_age_days=None)
    og = fb.gdf if hasattr(fb, "gdf") else fb[0].gdf
    if og.empty or urbi_polys.empty:
        return {"overture_buildings": int(len(og)), "urbi_buildings": int(len(urbi_polys)), "urbi_matched_by_overture": None, "overture_matched_by_urbi": None}
    crs = 32638
    u, o = urbi_polys.to_crs(crs), og.to_crs(crs)
    ut, ot = shapely.STRtree(u.geometry.values), shapely.STRtree(o.geometry.values)

    def matched(a, tree_b, geoms_b):
        n = 0
        for g in a.geometry.values:
            if g is None or g.is_empty:
                continue
            c = g.representative_point()
            hit = [j for j in tree_b.query(g, predicate="intersects") if geoms_b[j].intersects(g) and geoms_b[j].intersection(g).area >= 0.3 * min(g.area, geoms_b[j].area)]
            n += bool(hit) or any(geoms_b[j].contains(c) for j in tree_b.query(c))
        return n

    return {"overture_buildings": int(len(o)), "urbi_buildings": int(len(u)),
            "urbi_matched_by_overture": matched(u, ot, o.geometry.values), "overture_matched_by_urbi": matched(o, ut, u.geometry.values)}


def main(base="http://127.0.0.1:8080"):
    reg = gpd.read_parquet(PART / "registry.parquet")
    tiles = CityTiles(PART, reg)
    spots = pick_spots(reg, tiles)
    print("reading raw JSONL for", len(spots), "spotsâ€¦", flush=True)
    raws = raw_ids(spots)
    results = []
    for s, raw in zip(spots, raws):
        bb = s["bbox"]
        st = tiles._features("urbi_building", bb)
        store_ids = set(st["urbi_id"].astype(str)) if len(st) else set()
        tile_ids = decode_tiles(base, "urbi_building", bb)
        r = {"spot": {k: v for k, v in s.items() if k != "bbox"}, "bbox": [round(v, 6) for v in bb],
             "buildings": {"raw_records": len(raw), "store": len(store_ids), "in_served_tiles": len(tile_ids & store_ids),
                           "raw_not_in_store": len(set(raw) - store_ids),
                           "store_missing_from_tiles": len(store_ids - tile_ids), "tiles_not_in_store": len(tile_ids - store_ids)}}
        # raw records are located by centroid; the store keeps every feature that intersects the box, so store >= raw is expected, never the reverse
        r["buildings"]["store_minus_raw_edge_features"] = len(store_ids - set(raw))
        r["buildings"]["geometry"] = {"polygons": int(st.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).sum()), "points": int(st.geometry.geom_type.isin(["Point"]).sum())}
        sp = tiles._features("parcel", bb)
        fresh = suhail_fresh(bb)
        store_p = set(sp["parcel_objectid"].astype(str)) if len(sp) else set()
        r["parcels"] = {"store": len(store_p), "fresh_suhail_tile": len(fresh), "fresh_not_in_store": len(set(fresh) - store_p), "store_not_in_fresh": len(store_p - set(fresh))}
        try:
            r["overture"] = overture_compare(bb, st[st.geometry.geom_type.isin(["Polygon", "MultiPolygon"])])
        except Exception as e:  # noqa: BLE001
            r["overture"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        for layer in ("urbi_parking", "street", "attraction", "metro"):
            g = tiles._features(layer, bb)
            ids = set(g["urbi_id"].astype(str)) if len(g) else set()
            r[layer] = {"store": len(ids), "in_served_tiles": len(decode_tiles(base, layer, bb) & ids) if ids else 0}
        results.append(r)
        print(json.dumps(r, default=str)[:600], flush=True)
    out = HERE / "benchmarks" / "results" / "city_spot_verification.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()


