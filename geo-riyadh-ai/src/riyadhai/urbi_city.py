"""Load the city-wide Urbi/2GIS JSONL dumps into GeoParquet (streamed in chunks) and verify them.

Every record keeps: id, names, type/subtype, purpose, floors, address, district, the centroid point (lon, lat), the best polygon geometry
(hover polygon > selection polygon > point), which geometry kind was used, the selection geometry kind, and every remaining scalar/small
field as `extra_json`, so nothing the source returned is silently dropped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt as shp_wkt
from shapely.geometry import Point

_PT = re.compile(r"POINT\s*\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)")
FILES = {"buildings": "buildings_data.jsonl", "parking": "parking_data.jsonl", "streets": "streets_data.jsonl",
         "attractions": "attractions_data.jsonl", "metro": "metro_data.jsonl"}
IDFILES = {"buildings": "buildings_ids.txt", "parking": "parking_ids.txt", "streets": "streets_ids.txt", "attractions": "attractions_ids.txt", "metro": "metro_ids.txt"}
CORE = {"id", "geometry", "adm_div", "name", "full_name", "address_name", "building_name", "purpose_name", "purpose_code", "floors", "type", "subtype", "subtype_name",
        "address", "capacity", "access", "is_paid", "schedule", "rubrics", "for_trucks", "is_incentive", "city_alias", "route_type", "point"}


def _point(s: Optional[str]):
    m = _PT.search(s or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def _district(adm) -> Optional[str]:
    return next((x.get("name") for x in (adm or []) if x.get("type") == "district"), None)


def _best_geom(geo: dict):
    """(geometry, kind) with kind in hover_polygon | selection_polygon | point | none."""
    geo = geo or {}
    for key, kind in (("hover", "hover_polygon"), ("selection", "selection_polygon")):
        w = geo.get(key)
        if w and "POLYGON" in w:
            try:
                return shp_wkt.loads(w), kind
            except Exception:  # noqa: BLE001 - malformed WKT is counted, then we fall back to the point
                pass
    pt = _point(geo.get("centroid")) or _point(geo.get("selection"))
    return (Point(*pt), "point") if pt else (None, "none")


def parse_record(d: dict, layer: str) -> dict:
    geo = d.get("geometry") or {}
    g, kind = _best_geom(geo)
    c = _point(geo.get("centroid"))
    sel = geo.get("selection") or ""
    fl = d.get("floors") or {}
    cap = (d.get("capacity") or {}).get("total") if isinstance(d.get("capacity"), dict) else d.get("capacity")
    extra = {k: v for k, v in d.items() if k not in CORE and v not in (None, "", [], {})}
    # the rest of the response can be large (links, external_content...): keep it, but compactly
    rec = {"urbi_id": str(d.get("id", "")).split("_")[0], "raw_id": str(d.get("id", "")), "layer": layer, "type": d.get("type"), "subtype": d.get("subtype"),
           "subtype_name": d.get("subtype_name"), "name": d.get("building_name") or d.get("name") or d.get("full_name"), "full_name": d.get("full_name"),
           "address": d.get("address_name"), "purpose": d.get("purpose_name"), "purpose_code": d.get("purpose_code"),
           "floors_ground": fl.get("ground_count"), "floors_json": json.dumps(fl, ensure_ascii=False) if fl else None,
           "capacity": cap, "access": d.get("access"), "is_paid": d.get("is_paid"), "for_trucks": d.get("for_trucks"), "is_incentive": d.get("is_incentive"),
           "schedule_json": json.dumps(d.get("schedule"), ensure_ascii=False) if d.get("schedule") else None,
           "rubrics": ", ".join(r.get("name", "") for r in (d.get("rubrics") or [])) or None, "route_type": d.get("route_type"),
           "district": _district(d.get("adm_div")), "city_alias": d.get("city_alias"),
           "centroid_lon": c[0] if c else None, "centroid_lat": c[1] if c else None,
           "geometry_kind": kind, "selection_kind": (sel.split("(")[0].strip() if sel else None),
           "extra_json": json.dumps(extra, ensure_ascii=False, default=str)[:20000] if extra else None, "geometry": g}
    return rec


def iter_jsonl(path: Path) -> Iterator[tuple[int, Optional[dict]]]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError:
                yield n, None


def convert_layer(src_dir: Path, layer: str, out_dir: Path, chunk: int = 100_000) -> dict:
    """Stream one JSONL into parquet parts. Returns exact counts (parsed, bad json lines, no-geometry rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{layer}_part*.parquet"):
        old.unlink()
    rows, part, stats = [], 0, {"lines": 0, "bad_json_lines": 0, "no_geometry": 0, "written": 0, "parts": 0}

    def flush():
        nonlocal rows, part
        if not rows:
            return
        g = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
        g.to_parquet(out_dir / f"{layer}_part{part:03d}.parquet")
        stats["written"] += len(g)
        part += 1
        rows = []

    for n, d in iter_jsonl(src_dir / FILES[layer]):
        stats["lines"] += 1
        if d is None:
            stats["bad_json_lines"] += 1
            continue
        r = parse_record(d, layer)
        if r["geometry"] is None:
            stats["no_geometry"] += 1
        rows.append(r)
        if len(rows) >= chunk:
            flush()
    flush()
    stats["parts"] = part
    return stats


def read_layer(out_dir: Path, layer: str) -> gpd.GeoDataFrame:
    parts = sorted(out_dir.glob(f"{layer}_part*.parquet"))
    return gpd.GeoDataFrame(pd.concat([gpd.read_parquet(p) for p in parts], ignore_index=True), geometry="geometry", crs=4326)
