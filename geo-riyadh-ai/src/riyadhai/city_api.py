"""City-wide map API: district registry + on-demand Mapbox vector tiles from the partitioned city store.

Nothing is sampled: every feature of a layer that intersects a tile is encoded into it. A polygon that collapses to under 4 tile units^2 is emitted
as a centroid point in the companion `<layer>_c` source-layer instead of being dropped, so features are polygons where they have an area and points
where they do not. The tile endpoint reports the encoded polygon/point counts in response headers for verification.
"""
from __future__ import annotations

import json
import math
import threading
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

from .citystore import CityStore
from .paths import partition_root as default_partition_root

EXT = 4096
R = 6378137.0
ORIGIN = math.pi * R
# layer -> (id column, minzoom, props kept)
LAYERS = {
    "urbi_building": ("urbi_id", 14, ["purpose_class", "floors", "name"]),
    "parcel": ("parcel_objectid", 15, ["landuse"]),
    "urbi_parking": ("urbi_id", 12, ["subtype", "capacity", "name"]),
    "street": ("urbi_id", 13, ["name"]),
    "attraction": ("urbi_id", 12, ["subtype", "name"]),
    "metro": ("urbi_id", 11, ["name"]),
}
TILE_Z = {"urbi_building": [14], "parcel": [14], "urbi_parking": [12, 14], "street": [13, 14], "attraction": [12, 14], "metro": [12, 14]}  # generation zooms; MapLibre overzooms above the last
PART_COLS = ["urbi_district", "suhail_nbh", "overture_nbh"]
_geo_lock = threading.Lock()


def _safe_clip(geoms, x0, y0, x1, y1):
    """clip_by_rect with a per-feature fallback: a degenerate ring (e.g. a 3-point ring) is repaired with make_valid before clipping."""
    try:
        return shapely.clip_by_rect(geoms, x0, y0, x1, y1)
    except shapely.errors.GEOSException:
        out = np.empty(len(geoms), dtype=object)
        for i, g in enumerate(geoms):
            out[i] = _clip_one(g, x0, y0, x1, y1)
        return out


def _clip_one(g, x0, y0, x1, y1):
    """Clip one geometry; if it is degenerate, repair it (make_valid, then buffer(0)); as a last resort draw its centroid so it is never lost."""
    for fix in (lambda a: a, shapely.make_valid, lambda a: a.buffer(0)):
        try:
            return shapely.clip_by_rect(fix(g), x0, y0, x1, y1)
        except shapely.errors.GEOSException:
            continue
    try:
        c = g.centroid
    except shapely.errors.GEOSException:
        c = shapely.Point(*np.asarray(shapely.get_coordinates(g)).mean(axis=0)) if len(shapely.get_coordinates(g)) else shapely.Point()
    return c if (x0 <= c.x <= x1 and y0 <= c.y <= y1) else shapely.Point()


def _safe_simplify(geoms, tol, keep):
    """Simplify the kept features; a feature that cannot be simplified is kept as it is rather than lost."""
    out = np.array(geoms, dtype=object)
    try:
        out[keep] = shapely.simplify(out[keep], tol)
    except shapely.errors.GEOSException:
        for i in np.where(keep)[0]:
            try:
                out[i] = shapely.simplify(out[i], tol)
            except shapely.errors.GEOSException:
                pass
    bad = keep & ~shapely.is_valid(out)
    if bad.any():
        out[bad] = shapely.make_valid(out[bad])
    return out


def _merc(lon, lat):
    x = np.asarray(lon) * ORIGIN / 180.0
    y = np.log(np.tan((90.0 + np.asarray(lat)) * math.pi / 360.0)) * R
    return x, y


def tile_bounds(z, x, y):
    n = 2 ** z
    size = 2 * ORIGIN / n
    return -ORIGIN + x * size, ORIGIN - (y + 1) * size, -ORIGIN + (x + 1) * size, ORIGIN - y * size


def tile_lonlat_bbox(z, x, y):
    n = 2 ** z
    w = x / n * 360 - 180
    e = (x + 1) / n * 360 - 180
    nlat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    slat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return w, slat, e, nlat


class CityTiles:
    def __init__(self, root: Path, registry: gpd.GeoDataFrame, cache_dir: Path | None = None):
        self.store = CityStore(root)
        self.root = Path(root)
        self.reg = registry.reset_index(drop=True)
        self.idx = {u: i for i, u in enumerate(self.reg.uid)}
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._read = lru_cache(maxsize=6)(self._read_file)

    def _read_file(self, path: str, cols: tuple):
        g = gpd.read_parquet(path, columns=list(cols))
        g.sindex  # build the spatial index once; every later tile request into this stored tile reuses it
        return g

    def _features(self, layer: str, bbox):
        """All stored features of `layer` intersecting bbox (WGS84), reading only the columns a tile needs."""
        idcol, _, props = LAYERS[layer]
        want = ["geometry", idcol, *PART_COLS, *[p for p in props if p != idcol]]
        w, s, e, n = bbox
        d = self.root / layer
        ext = self.store.extent(layer)
        r = ext["small_reach_deg"]
        files = {p for x, y in self.store.tiles_for((w - r, s - r, e + r, n + r)) for p in d.glob(f"t{x}_{y}__*.parquet")}
        for name, boxes in ext["large"].items():
            if any(not (bx[2] < w or bx[0] > e or bx[3] < s or bx[1] > n) for bx in boxes):
                files.add(d / name)
        parts = []
        from shapely.geometry import box
        q = box(w, s, e, n)
        for p in sorted(files):
            have = set(pq.read_schema(p).names)
            g = self._read(str(p), tuple(c for c in want if c in have))
            hit = g.sindex.query(q, predicate="intersects")
            if len(hit):
                parts.append(g.iloc[np.sort(hit)])
        if not parts:
            return gpd.GeoDataFrame(geometry=[], crs=4326)
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)

    def encode(self, layer: str, z: int, x: int, y: int) -> tuple[bytes, dict]:
        w, s, e, n = tile_lonlat_bbox(z, x, y)
        pad = 0.04 * (e - w)
        return self.encode_frame(layer, z, x, y, self._features(layer, (w - pad, s - pad, e + pad, n + pad)))

    def encode_frame(self, layer: str, z: int, x: int, y: int, g: gpd.GeoDataFrame) -> tuple[bytes, dict]:
        """Encode already-selected features (every stored feature intersecting the tile) into one MVT tile."""
        idcol, minz, props = LAYERS[layer]
        info = {"layer": layer, "input": int(len(g)), "polygons": 0, "points": 0, "lines": 0, "centroid_fallback": 0}
        xmin, ymin, xmax, ymax = tile_bounds(z, x, y)
        unit = (xmax - xmin) / EXT
        if g.empty:
            return b"", info
        geoms = shapely.transform(g.geometry.values, lambda a: np.column_stack(_merc(a[:, 0], a[:, 1])))
        buf = 64 * unit
        clipped = _safe_clip(geoms, xmin - buf, ymin - buf, xmax + buf, ymax + buf)
        keep = ~shapely.is_empty(clipped)
        if z < 17:
            clipped = _safe_simplify(clipped, unit * 0.5, keep)
        main, cfall = [], []
        d_idx = g["urbi_district"].map(self.idx).fillna(-1).astype(int).to_numpy()
        n_idx = g["suhail_nbh"].map(self.idx).fillna(-1).astype(int).to_numpy()
        o_idx = g["overture_nbh"].map(self.idx).fillna(-1).astype(int).to_numpy()
        ids = g[idcol].astype(str).to_numpy()
        extra = {p: g[p].to_numpy() for p in props if p in g.columns and p != idcol}
        for i in np.where(keep)[0]:
            geom = clipped[i]
            gt = geom.geom_type
            pr = {"i": ids[i], "d": int(d_idx[i]), "n": int(n_idx[i]), "o": int(o_idx[i])}
            for k, arr in extra.items():
                v = arr[i]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    pr[k] = v.item() if hasattr(v, "item") else v
            if gt in ("Polygon", "MultiPolygon", "GeometryCollection"):
                if gt == "GeometryCollection":
                    polys = [q for q in geom.geoms if q.geom_type in ("Polygon", "MultiPolygon")]
                    if not polys:
                        continue
                    geom = shapely.union_all(polys)
                if geom.area < 4 * unit * unit:
                    c = geom.representative_point()
                    cfall.append({"geometry": c, "properties": pr})
                    info["centroid_fallback"] += 1
                else:
                    main.append({"geometry": geom, "properties": pr})
                    info["polygons"] += 1
            elif gt in ("LineString", "MultiLineString"):
                main.append({"geometry": geom, "properties": pr})
                info["lines"] += 1
            else:
                main.append({"geometry": geom, "properties": pr})
                info["points"] += 1
        layers = []
        if main:
            layers.append({"name": layer, "features": main})
        if cfall:
            layers.append({"name": layer + "_c", "features": cfall})
        if not layers:
            return b"", info
        with _geo_lock:
            data = mapbox_vector_tile.encode(layers, default_options={"quantize_bounds": (xmin, ymin, xmax, ymax), "extents": EXT, "y_coord_down": False})
        return data, info

    def tile(self, layer: str, z: int, x: int, y: int) -> tuple[bytes, dict]:
        if self.cache_dir:
            f = self.cache_dir / layer / str(z) / str(x) / f"{y}.pbf"
            m = f.with_suffix(".json")
            if f.exists() and m.exists():
                return f.read_bytes(), json.loads(m.read_text())
        data, info = self.encode(layer, z, x, y)
        if self.cache_dir:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
            f.with_suffix(".json").write_text(json.dumps(info))
        return data, info


def add_city_routes(app, partition_root: str | Path | None = None):
    root = Path(partition_root or default_partition_root())
    reg_path = root / "registry.parquet"
    if not reg_path.exists():
        return None
    reg = gpd.read_parquet(reg_path)
    tiles = CityTiles(root, reg, root / "tilecache")
    summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else {}

    @app.get("/v1/city/summary")
    def city_summary():
        return summary

    @app.get("/v1/city/registry.geojson")
    def registry_geojson(source: str = "urbi", kind: str | None = None):
        g = tiles.reg
        g = g[g.source == source]
        if kind:
            g = g[g.kind == kind]
        cols = [c for c in g.columns if c not in ("geometry",)]
        out = g.copy()
        out["idx"] = [tiles.idx[u] for u in out.uid]
        return Response(out[cols + ["idx", "geometry"]].to_json(drop_id=True), media_type="application/geo+json")

    @app.get("/v1/city/registry.json")
    def registry_table():
        g = tiles.reg.drop(columns="geometry")
        g = g.assign(idx=range(len(g)))
        return JSONResponse(json.loads(g.to_json(orient="records")))

    @app.get("/v1/city/count")
    def city_count(bbox: str, ids: bool = False, max_ids: int = 80000):
        """What the store holds inside a viewport (features intersecting it), per layer; optionally the ids, to compare with what the browser rendered."""
        try:
            w, s, e, n = (float(v) for v in bbox.split(","))
        except ValueError:
            raise HTTPException(400, "bbox must be west,south,east,north")
        if (e - w) * (n - s) > 0.05:
            raise HTTPException(400, "viewport too large for an exact count; zoom in")
        out: dict = {}
        for layer, (idcol, _, _) in LAYERS.items():
            g = tiles._features(layer, (w, s, e, n))
            rec = {"count": int(len(g)), "polygons": int(g.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).sum()) if len(g) else 0,
                   "points": int(g.geometry.geom_type.isin(["Point", "MultiPoint"]).sum()) if len(g) else 0}
            if ids and len(g) <= max_ids:
                rec["ids"] = g[idcol].astype(str).tolist() if len(g) else []
            out[layer] = rec
        return out

    @app.get("/v1/city/tiles/{layer}/{z}/{x}/{y}.pbf")
    def city_tile(layer: str, z: int, x: int, y: int):
        """Pre-generated tiles (scripts/pregen_tiles.py). A missing file means the tile holds no features of this layer."""
        if layer not in LAYERS:
            raise HTTPException(404, f"unknown layer {layer!r}")
        f = root / "tiles" / layer / str(z) / str(x) / f"{y}.pbf"
        if not f.exists():
            return Response(status_code=204)
        return Response(f.read_bytes(), media_type="application/vnd.mapbox-vector-tile", headers={"Cache-Control": "public, max-age=3600"})

    return tiles
