"""Local dataset inventory and loader (GeoJSON / GeoParquet / CSV with lon,lat)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from .base import BBox, Capabilities, CoverageEstimate, FeatureBatch, Provenance, Provider, utcnow_iso

_EXTS = {".geojson", ".json", ".parquet", ".geoparquet", ".csv"}

# column signatures used to guess the entity type of an untyped file
_SIGNATURES = {
    "parcel": {"parcel_id", "parcel_objectid"},
    "poi": {"poi_id", "category", "name"},
    "road": {"road_id", "class"},
    "building": {"building_id"},
    "connector": {"connector_id"},
    "parking": {"capacity", "is_paid"},
}


@dataclass
class InventoryItem:
    path: str
    entity_type: Optional[str]
    rows: int
    columns: list[str]
    bbox: Optional[BBox]
    geometry_types: list[str]
    modified: str
    has_polygons: bool


def _read(path: Path) -> gpd.GeoDataFrame:
    s = path.suffix.lower()
    if s in (".parquet", ".geoparquet"):
        return gpd.read_parquet(path)
    if s in (".geojson", ".json"):
        return gpd.read_file(path)
    df = pd.read_csv(path, low_memory=False)
    lon = next((c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "x")), None)
    lat = next((c for c in df.columns if c.lower() in ("lat", "latitude", "y")), None)
    if lon is None or lat is None:
        raise ValueError(f"{path.name}: CSV needs lon/lat columns")
    df = df.dropna(subset=[lon, lat])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326)


_KEY_COLUMNS = [("building_id", "building"), ("road_id", "road"), ("connector_id", "connector"),
                ("parcel_id", "parcel"), ("parcel_objectid", "parcel"), ("poi_id", "poi")]


def guess_entity_type(cols: set[str]) -> Optional[str]:
    low = {c.lower() for c in cols}
    for key, et in _KEY_COLUMNS:  # an id column is decisive; shared columns like `name` are not
        if key in low:
            return et
    best, n = None, 0
    for et, sig in _SIGNATURES.items():
        k = len(sig & low)
        if k > n:
            best, n = et, k
    return best


def scan_inventory(root: Path) -> list[InventoryItem]:
    items = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in _EXTS or not p.is_file():
            continue
        try:
            g = _read(p)
        except Exception:
            continue
        gt = sorted(g.geometry.geom_type.dropna().unique().tolist())
        b = tuple(float(x) for x in g.to_crs(4326).total_bounds) if len(g) else None
        items.append(InventoryItem(str(p), guess_entity_type(set(g.columns)), len(g), [c for c in g.columns if c != "geometry"], b, gt,
                                   time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(p.stat().st_mtime)),
                                   any("Polygon" in t for t in gt)))
    return items


class LocalProvider(Provider):
    provider_id = "local_files"

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else None
        self.inventory = scan_inventory(self.root) if self.root and self.root.exists() else []

    def discover_capabilities(self) -> Capabilities:
        et = sorted({i.entity_type for i in self.inventory if i.entity_type})
        return Capabilities(self.provider_id, et, {}, {}, notes=f"{len(self.inventory)} local files")

    def estimate_coverage(self, entity_type: str, bbox: BBox) -> CoverageEstimate:
        n = 0
        for i in self.inventory:
            if i.entity_type == entity_type and i.bbox and _overlaps(i.bbox, bbox):
                n += i.rows
        return CoverageEstimate(self.provider_id, entity_type, bbox, n or None, 0, 1.0, "sum of rows in overlapping local files (file bbox, not row-level)")

    def fetch_features(self, entity_type: str, bbox: BBox, *, budget=None, **kw) -> FeatureBatch:
        frames = []
        for i in self.inventory:
            if i.entity_type == entity_type and i.bbox and _overlaps(i.bbox, bbox):
                g = self.normalize(entity_type, _read(Path(i.path)))
                frames.append(g.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]])
        gdf = pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame(geometry=[], crs=4326)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)
        return FeatureBatch(gdf, Provenance(self.provider_id, entity_type, "local", "user-supplied (user responsible)", "user data",
                                            utcnow_iso(), bbox, 0, True, ""), [] if len(gdf) else [f"no local {entity_type} data in bbox"])

    def normalize(self, entity_type: str, raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = raw.to_crs(4326) if raw.crs else raw.set_crs(4326)
        if entity_type == "poi" and "category_path" in g.columns:
            g["category_path"] = g["category_path"].apply(lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else v)
        return g

    def report_provenance(self) -> dict:
        return {"provider": self.provider_id, "files": [i.path for i in self.inventory]}


def _overlaps(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])
