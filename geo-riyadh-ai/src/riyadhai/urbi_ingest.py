"""Parse raw Urbi/2GIS JSONL (buildings, attractions, metro, streets, parking) into GeoParquet layers and load them into an area."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt as shp_wkt
from shapely.geometry import Point

from .provenance import data_date

_PT = re.compile(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")


def _geom(geo: dict):
    """Best geometry: hover polygon > selection polygon > centroid point."""
    for k in ("hover", "selection"):
        w = (geo or {}).get(k)
        if w and ("POLYGON" in w):
            try:
                return shp_wkt.loads(w)
            except Exception:  # noqa: BLE001 - malformed WKT from source: fall through to the point
                pass
    m = _PT.search((geo or {}).get("centroid", "") or (geo or {}).get("selection", "") or "")
    return Point(float(m.group(1)), float(m.group(2))) if m else None


def _district(adm):
    return next((x.get("name") for x in (adm or []) if x.get("type") == "district"), None)


def _rows(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_buildings(path: Path) -> gpd.GeoDataFrame:
    recs = []
    for d in _rows(path):
        g = _geom(d.get("geometry"))
        if g is None:
            continue
        fl = d.get("floors") or {}
        recs.append({"urbi_id": str(d["id"]).split("_")[0], "purpose": d.get("purpose_name"), "purpose_code": d.get("purpose_code"),
                     "floors": fl.get("ground_count"), "address": d.get("address_name"), "name": d.get("building_name") or d.get("name"),
                     "district": _district(d.get("adm_div")), "geometry": g})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)


def load_attractions(path: Path) -> gpd.GeoDataFrame:
    recs = []
    for d in _rows(path):
        g = _geom(d.get("geometry"))
        if g is None:
            continue
        recs.append({"urbi_id": str(d["id"]).split("_")[0], "name": d.get("name") or d.get("full_name"), "subtype": d.get("subtype"), "subtype_name": d.get("subtype_name"),
                     "rubrics": ", ".join(r.get("name", "") for r in d.get("rubrics", []) or []), "district": _district(d.get("adm_div")), "geometry": g})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)


def load_metro(path: Path) -> gpd.GeoDataFrame:
    recs = []
    for d in _rows(path):
        g = _geom(d.get("geometry"))
        if g is None:
            continue
        recs.append({"urbi_id": str(d["id"]).split("_")[0], "name": d.get("name") or d.get("full_name"), "kind": d.get("type"), "subtype": d.get("subtype"),
                     "route_type": d.get("route_type"), "district": _district(d.get("adm_div")), "geometry": g})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)


def load_streets(path: Path) -> gpd.GeoDataFrame:
    recs = []
    for d in _rows(path):
        g = _geom(d.get("geometry"))
        if g is None:
            continue
        recs.append({"urbi_id": str(d["id"]).split("_")[0], "name": d.get("name") or d.get("full_name"), "district": _district(d.get("adm_div")), "geometry": g})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)


# purposes -> coarse land-use class used for demand and for labels (keyword mapping on Urbi's English purpose names)
RESIDENTIAL = ("villa", "apartment", "residential", "house", "dormitor", "housing", "residence", "flat", "duplex", "townhouse")
COMMERCIAL = ("shop", "retail", "mall", "market", "commercial", "office", "business", "bank", "restaurant", "cafe", "hotel", "store", "showroom", "supermarket", "shopping")


def purpose_class(p: Optional[str]) -> str:
    if not isinstance(p, str) or not p.strip():
        return "unknown"
    t = p.lower()
    res = any(k in t for k in RESIDENTIAL)
    com = any(k in t for k in COMMERCIAL)
    if res and com:
        return "mixed"
    if com:
        return "commercial"
    if res:
        return "residential"
    return "other"


def ingest_all(store, ds, src_dir: Path) -> dict:
    """Load every Urbi type found in `src_dir` into the area (canonical GeoParquet + provenance)."""
    src = Path(src_dir)
    report = {}
    loaders = {"urbi_building": ("buildings_data.jsonl", load_buildings), "attraction": ("attractions_data.jsonl", load_attractions),
               "metro": ("metro_data.jsonl", load_metro), "street": ("streets_data.jsonl", load_streets)}
    area = ds.area_polygon()
    for layer, (fn, loader) in loaders.items():
        f = src / fn
        if not f.exists():
            report[layer] = "missing file"
            continue
        g = loader(f).to_crs(ds.crs)
        g = g[g.geometry.intersects(area.buffer(300))].reset_index(drop=True)
        if layer == "urbi_building":
            g["purpose_class"] = g["purpose"].map(purpose_class)
            g["floors"] = pd.to_numeric(g["floors"], errors="coerce")
        ds.layers[layer] = g
        ds.provenance[layer] = {"provider_id": "urbi", "dataset_version": f"live-api {data_date()}", "license": "Urbi/2GIS, user-stated permission (no document seen)",
                                "retrieved_at": data_date(), "from_cache": True}
        d = store.root / "canonical" / ds.name
        g.to_crs(4326).to_parquet(d / f"{layer}.parquet")
        report[layer] = int(len(g))
    import json as _j
    (store.root / "provenance" / f"{ds.name}.json").write_text(_j.dumps(ds.provenance, indent=2, default=str))
    return report
