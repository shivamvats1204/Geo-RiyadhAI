"""Streamed quality check + repair of the converted Urbi parts (one part in memory at a time)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

BOUND = (45.9930, 23.6740, 48.1804, 25.2848)


def check_and_repair(out: Path, layer: str, repair: bool = True) -> dict:
    parts = sorted(out.glob(f"{layer}_part*.parquet"))
    agg = {"parts": len(parts), "rows": 0, "invalid": 0, "repaired": 0, "still_invalid_after_repair": 0, "empty": 0, "outside_bounds": 0, "zero_area": 0,
           "centroid_not_in_polygon": 0, "polygons": 0, "points": 0, "area_min": np.inf, "area_max": 0.0, "area_sum": 0.0, "null": {}}
    areas = []
    w, s, e, n = BOUND
    for p in parts:
        g = gpd.read_parquet(p)
        agg["rows"] += len(g)
        geom = g.geometry.values
        valid = shapely.is_valid(geom)
        agg["invalid"] += int((~valid).sum())
        agg["empty"] += int(shapely.is_empty(geom).sum())
        if repair and (~valid).any():
            fixed = shapely.make_valid(geom[~valid])
            arr = np.array(geom, dtype=object)
            arr[~valid] = fixed
            g["geometry"] = gpd.GeoSeries(arr, index=g.index, crs=4326)
            agg["repaired"] += int((~valid).sum())
            agg["still_invalid_after_repair"] += int((~shapely.is_valid(g.geometry.values)).sum())
            g["geometry_repaired"] = ~valid
            g.to_parquet(p)
        elif "geometry_repaired" not in g:
            pass
        lon, lat = g["centroid_lon"], g["centroid_lat"]
        agg["outside_bounds"] += int((~(lon.between(w, e) & lat.between(s, n))).sum())
        is_poly = np.isin(shapely.get_type_id(g.geometry.values), [3, 6])
        agg["polygons"] += int(is_poly.sum())
        agg["points"] += int((~is_poly).sum())
        if is_poly.any():
            pm = g.loc[is_poly].geometry.to_crs(32638)
            a = pm.area.to_numpy()
            agg["zero_area"] += int((a <= 0).sum())
            agg["area_min"] = min(agg["area_min"], float(a.min()))
            agg["area_max"] = max(agg["area_max"], float(a.max()))
            agg["area_sum"] += float(a.sum())
            areas.append(np.random.default_rng(0).choice(a, min(2000, len(a)), replace=False))
            cpts = gpd.GeoSeries(gpd.points_from_xy(g.loc[is_poly, "centroid_lon"], g.loc[is_poly, "centroid_lat"]), crs=4326, index=pm.index)
            samp = pm.sample(min(10000, len(pm)), random_state=0)
            d = shapely.distance(samp.values, cpts.to_crs(32638).loc[samp.index].values)
            agg["centroid_not_in_polygon"] += int((d > 1).sum())
        for c in ("name", "address", "district", "purpose", "floors_ground", "capacity", "access", "subtype"):
            if c in g:
                agg["null"][c] = agg["null"].get(c, 0) + int(g[c].isna().sum())
    agg["null_rate"] = {c: round(v / max(1, agg["rows"]), 4) for c, v in agg["null"].items()}
    del agg["null"]
    if areas:
        a = np.concatenate(areas)
        agg["area_m2_sample_percentiles"] = {str(q): round(float(np.percentile(a, q)), 1) for q in (1, 25, 50, 75, 99)}
    agg["area_min"] = None if agg["area_min"] == np.inf else round(agg["area_min"], 2)
    agg["area_max"] = round(agg["area_max"], 1)
    agg["area_sum_km2"] = round(agg["area_sum"] / 1e6, 2)
    del agg["area_sum"]
    return agg
