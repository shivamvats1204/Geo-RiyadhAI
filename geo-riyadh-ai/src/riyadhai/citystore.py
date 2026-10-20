"""City-scale tiled store for Urbi (buildings, parking, streets, attractions, metro) and Suhail (parcels + zoning rules).

Layers are partitioned into 0.05 degree tiles (about 5 km) of GeoParquet, with an index of tile row counts, so any bounding box can be loaded
without holding 1-2 million polygons in memory. `CityStore.query(layer, bbox)` returns only the features whose geometry intersects the bbox.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from .urbi_ingest import purpose_class

TILE = 0.05
LAYERS = ["urbi_building", "urbi_parking", "street", "attraction", "metro", "parcel"]
RULE_COLS = ["parcel_objectid", "rule_id", "zoningId", "landuse", "description", "maxBuildingCoefficient", "maxBuildingHeight", "maxParcelCoverage",
             "mainStreetsSetback", "sideRearSetback"]


def landuse_class(t) -> str:
    """Coarse English class from Suhail's Arabic land-use text (keyword mapping; not an official classification)."""
    if not isinstance(t, str) or not t.strip():
        return "unknown"
    com = "تجاري" in t
    res = any(k in t for k in ("سكني", "شقق", "فلل", "دبلكس"))
    if com and res:
        return "mixed"
    if com:
        return "commercial"
    if "صناع" in t:
        return "industrial"
    if res:
        return "residential"
    return "other"


def _tile_ids(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.floor(lon / TILE).astype(int), np.floor(lat / TILE).astype(int)


class CityStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ writing
    def write(self, layer: str, gdf: gpd.GeoDataFrame, batch_id: int) -> int:
        """Append a batch (WGS84) by tile; one file per (tile, batch) so nothing is ever rewritten."""
        if gdf.empty:
            return 0
        c = gdf.geometry.representative_point()
        tx, ty = _tile_ids(c.x.to_numpy(), c.y.to_numpy())
        d = self.root / layer
        d.mkdir(parents=True, exist_ok=True)
        self._update_extent(layer, gdf, c)
        n = 0
        for (x, y), idx in pd.DataFrame({"x": tx, "y": ty}).groupby(["x", "y"]).groups.items():
            gdf.iloc[list(idx)].to_parquet(d / f"t{x}_{y}__{batch_id:04d}.parquet")  # idx are positions (groupby on a fresh RangeIndex)
            n += len(idx)
        return n

    LARGE_DEG = 0.02  # features reaching further than ~2 km from their representative point are tracked individually

    def _reach_arrays(self, g: gpd.GeoDataFrame):
        c = g.geometry.representative_point()
        b = g.geometry.bounds
        reach = np.maximum.reduce([np.abs(b.minx - c.x), np.abs(b.maxx - c.x), np.abs(b.miny - c.y), np.abs(b.maxy - c.y)])
        return np.asarray(reach), b

    def _update_extent(self, layer: str, gdf: gpd.GeoDataFrame, c) -> None:
        pass  # extents are computed after the build by compute_extent (needs the final file names)

    def compute_extent(self, layer: str) -> dict:
        """Scan the stored tiles once: the largest reach of ordinary features, plus the file and bounds of every oversized feature.

        A feature lives in the tile of its representative point but may extend far beyond it. Ordinary features are found by padding the
        tile lookup with `small_reach_deg`; oversized ones (long street polygons, huge parcels) are looked up by their recorded bounds.
        """
        small, large = 0.0, {}
        for p in (self.root / layer).glob("t*__*.parquet"):
            g = gpd.read_parquet(p, columns=["geometry"])
            reach, b = self._reach_arrays(g)
            is_large = reach > self.LARGE_DEG
            if (~is_large).any():
                small = max(small, float(reach[~is_large].max()))
            if is_large.any():
                large[p.name] = b[is_large][["minx", "miny", "maxx", "maxy"]].round(6).values.tolist()
        meta = {"small_reach_deg": small, "large": large, "n_large": int(sum(len(v) for v in large.values()))}
        (self.root / layer / "extent.json").write_text(json.dumps(meta))
        return meta

    def extent(self, layer: str) -> dict:
        f = self.root / layer / "extent.json"
        m = json.loads(f.read_text()) if f.exists() else {}
        return m if "small_reach_deg" in m else self.compute_extent(layer)

    def reset(self, layer: str) -> None:
        d = self.root / layer
        if d.exists():
            shutil.rmtree(d)

    def build_index(self, layer: str) -> dict:
        d = self.root / layer
        idx: dict[str, int] = {}
        for p in d.glob("t*__*.parquet"):
            tile = p.name.split("__")[0]
            idx[tile] = idx.get(tile, 0) + pq.ParquetFile(p).metadata.num_rows
        meta = {"layer": layer, "tile_deg": TILE, "tiles": idx, "rows": int(sum(idx.values()))}
        (d / "index.json").write_text(json.dumps(meta))
        self.compute_extent(layer)
        return meta

    # ------------------------------------------------------------------ reading
    def index(self, layer: str) -> dict:
        return json.loads((self.root / layer / "index.json").read_text())

    def tiles_for(self, bbox: tuple) -> list[tuple[int, int]]:
        w, s, e, n = bbox
        return [(x, y) for x in range(math.floor(w / TILE), math.floor(e / TILE) + 1) for y in range(math.floor(s / TILE), math.floor(n / TILE) + 1)]

    def query(self, layer: str, bbox: tuple, intersects: bool = True) -> gpd.GeoDataFrame:
        w, s, e, n = bbox
        d = self.root / layer
        parts = []
        ext = self.extent(layer)
        r = ext["small_reach_deg"]
        files = {p for x, y in self.tiles_for((w - r, s - r, e + r, n + r)) for p in d.glob(f"t{x}_{y}__*.parquet")}
        for name, boxes in ext["large"].items():  # tiles holding an oversized feature whose bounds touch the query box
            if any(not (bx[2] < w or bx[0] > e or bx[3] < s or bx[1] > n) for bx in boxes):
                files.add(d / name)
        for p in sorted(files):
            parts.append(gpd.read_parquet(p))
        if not parts:
            return gpd.GeoDataFrame(geometry=[], crs=4326)
        g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)
        from shapely.geometry import box
        box_ = box(w, s, e, n)
        keep = g.geometry.intersects(box_) if intersects else g.geometry.representative_point().within(box_)
        return g[keep].reset_index(drop=True)

    def summary(self) -> dict:
        out = {}
        for l in LAYERS:
            f = self.root / l / "index.json"
            if f.exists():
                m = json.loads(f.read_text())
                out[l] = {"rows": m["rows"], "tiles": len(m["tiles"])}
        return out


# ------------------------------------------------------------------ builders
def build_from_urbi(city: CityStore, parquet_dir: Path) -> dict:
    """Urbi parts -> tiled layers (buildings keep purpose class + floors)."""
    mapping = {"buildings": "urbi_building", "parking": "urbi_parking", "streets": "street", "attractions": "attraction", "metro": "metro"}
    report = {}
    for src, layer in mapping.items():
        city.reset(layer)
        n = 0
        for i, p in enumerate(sorted(Path(parquet_dir).glob(f"{src}_part*.parquet"))):
            g = gpd.read_parquet(p)
            g = g.drop(columns=[c for c in ("extra_json",) if c in g and layer in ("urbi_building",)], errors="ignore")  # keep payload small for the largest layer
            if layer == "urbi_building":
                g["purpose_class"] = g["purpose"].map(purpose_class)
                g["floors"] = pd.to_numeric(g["floors_ground"], errors="coerce")
            if layer == "urbi_parking":
                g["capacity"] = pd.to_numeric(g["capacity"], errors="coerce")
            n += city.write(layer, g, i)
        report[layer] = {"written": n, **{k: v for k, v in city.build_index(layer).items() if k in ("rows",)}}
    return report


def build_from_suhail(city: CityStore, suhail_dir: Path, batch_rows: int = 100_000) -> dict:
    """Parcels (polygons) + first zoning rule per parcel -> tiled `parcel` layer."""
    suhail_dir = Path(suhail_dir)
    rules = pd.read_csv(suhail_dir / "rules.csv", dtype=str, usecols=lambda c: c in RULE_COLS + ["api_status"])
    ok = rules[rules["api_status"] == "SUCCESS"]
    n_rules = ok.groupby("parcel_objectid").size().rename("n_rules")
    first = ok.drop_duplicates("parcel_objectid").drop(columns=["api_status"]).set_index("parcel_objectid").join(n_rules)
    del rules, ok
    city.reset("parcel")
    pf = pq.ParquetFile(suhail_dir / "parcels.parquet")
    n = 0
    for i, batch in enumerate(pf.iter_batches(batch_size=batch_rows)):
        df = batch.to_pandas()
        geom = shapely.from_wkb(df["geometry"].to_numpy())
        df = df.drop(columns="geometry")
        df["parcel_objectid"] = df["parcel_objectid"].astype(str)
        df = df.join(first, on="parcel_objectid")
        df["landuse_ar"] = df["landuse"]
        df["landuse"] = df["landuse_ar"].map(landuse_class)
        df["has_rules"] = df["landuse_ar"].notna()
        g = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
        g["geometry"] = shapely.make_valid(g.geometry.values) if (~shapely.is_valid(g.geometry.values)).any() else g.geometry
        n += city.write("parcel", g, i)
    meta = city.build_index("parcel")
    return {"written": n, "rows": meta["rows"], "tiles": len(meta["tiles"])}
