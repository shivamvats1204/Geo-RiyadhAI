"""Verification of the converted Urbi layers: completeness, correctness and cross-run consistency. Every check returns plain numbers."""
from __future__ import annotations

import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from .urbi_city import FILES, IDFILES, iter_jsonl, parse_record, read_layer

NOTEBOOK_BOUND = (45.9930, 23.6740, 48.1804, 25.2848)  # W,S,E,N from urbi_master.ipynb


def completeness(src: Path, out: Path, layer: str) -> dict:
    ids_file = src / IDFILES[layer]
    ids = set(ids_file.read_text().split())
    g = read_layer(out, layer)
    got = set(g["urbi_id"])
    return {"ids_scanned": len(ids), "records": int(len(g)), "unique_record_ids": len(got), "duplicate_records": int(len(g) - len(got)),
            "ids_without_record": len(ids - got), "records_not_in_scan_list": len(got - ids), "missing_id_sample": sorted(ids - got)[:5]}


def quality(out: Path, layer: str) -> dict:
    g = read_layer(out, layer)
    geom = g.geometry
    kinds = g["geometry_kind"].value_counts().to_dict()
    valid = shapely.is_valid(geom.values)
    nonpoint = ~geom.geom_type.isin(["Point"])
    w, s, e, n = NOTEBOOK_BOUND
    lon, lat = g["centroid_lon"], g["centroid_lat"]
    inb = lon.between(w, e) & lat.between(s, n)
    res = {"rows": int(len(g)), "geometry_kind": kinds, "geometry_types": geom.geom_type.value_counts().to_dict(),
           "invalid_geometries": int((~valid).sum()), "empty_geometries": int(geom.is_empty.sum()),
           "centroid_present": int(lon.notna().sum()), "centroid_inside_notebook_bounds": int(inb.sum()), "centroid_outside_bounds": int((~inb & lon.notna()).sum()),
           "null_rates": {c: round(float(g[c].isna().mean()), 4) for c in ["name", "address", "district", "purpose", "floors_ground", "capacity", "access", "rubrics", "subtype"] if c in g}}
    # centroid consistency: for polygons, the reported centroid should lie inside or very near the polygon
    poly = g[nonpoint & lon.notna()]
    if len(poly):
        samp = poly.sample(min(20000, len(poly)), random_state=0)
        cpts = gpd.GeoSeries(gpd.points_from_xy(samp["centroid_lon"], samp["centroid_lat"]), crs=4326, index=samp.index)
        pm, cm = samp.geometry.to_crs(32638), cpts.to_crs(32638)
        d = shapely.distance(pm.values, cm.values)
        res["polygon_centroid_consistency"] = {"sampled": int(len(samp)), "centroid_within_1m_of_polygon": round(float((d <= 1).mean()), 4),
                                                "within_10m": round(float((d <= 10).mean()), 4), "max_distance_m": round(float(d.max()), 1)}
        a = pm.area
        res["polygon_area_m2"] = {"min": round(float(a.min()), 2), "median": round(float(a.median()), 1), "max": round(float(a.max()), 1), "zero_area": int((a <= 0).sum())}
    return res


def spot_check(src: Path, out: Path, layer: str, n: int = 2000, seed: int = 0) -> dict:
    """Re-parse random raw lines and compare to what the parquet holds (round-trip fidelity)."""
    random.seed(seed)
    lines = sum(1 for _ in open(src / FILES[layer], encoding="utf-8"))
    pick = set(random.sample(range(1, lines + 1), min(n, lines)))
    raw = {}
    for i, d in iter_jsonl(src / FILES[layer]):
        if i in pick and d is not None:
            raw[str(d["id"]).split("_")[0]] = parse_record(d, layer)
    g = read_layer(out, layer).drop_duplicates("urbi_id").set_index("urbi_id")
    bad = {"missing": 0, "geometry": 0, "fields": 0}
    for uid, r in raw.items():
        if uid not in g.index:
            bad["missing"] += 1
            continue
        row = g.loc[uid]
        if not shapely.equals_exact(r["geometry"], row.geometry, 1e-9):
            bad["geometry"] += 1
        for f in ("name", "address", "purpose", "district", "floors_ground", "capacity", "access", "centroid_lon", "centroid_lat"):
            a, b = r.get(f), row.get(f)
            if not ((pd.isna(a) and pd.isna(b)) or a == b or (isinstance(a, float) and isinstance(b, float) and abs(a - b) < 1e-9)):
                bad["fields"] += 1
                break
    return {"checked": len(raw), "mismatches": bad}


def cross_run(earlier_dir: Path, out: Path, layer: str) -> dict:
    """The earlier study-area run used the same notebook settings: every id it found inside its box should also be in the city run."""
    f = earlier_dir / IDFILES[layer]
    if not f.exists():
        return {"skipped": "no earlier run"}
    old = set(f.read_text().split())
    new = set(read_layer(out, layer)["urbi_id"])
    return {"earlier_ids": len(old), "also_in_city_run": len(old & new), "share": round(len(old & new) / max(1, len(old)), 5), "earlier_only_sample": sorted(old - new)[:5]}
