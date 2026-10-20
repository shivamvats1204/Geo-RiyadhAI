"""Data-quality verification of acquired layers (independent of the constraint verifier).

Checks: geometry validity, inclusion in the requested area, duplicate ids, attribute completeness, and - when two sources describe the same
entity type - cross-source agreement (spatial overlap / proximity), so a layer is never trusted on one provider's word alone.
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
from shapely.geometry import box



def check_layer(name: str, gdf: gpd.GeoDataFrame, bbox_wgs84: tuple, id_col: str, attrs: Optional[list[str]] = None, tolerance_deg: float = 0.0) -> dict:
    """`gdf` in WGS84. Returns counts and the rows that fail each check (ids only)."""
    out: dict = {"layer": name, "rows": int(len(gdf))}
    if gdf.empty:
        out["problems"] = ["layer is empty"]
        return out
    w, s, e, n = bbox_wgs84
    area = box(w - tolerance_deg, s - tolerance_deg, e + tolerance_deg, n + tolerance_deg)
    invalid = ~gdf.geometry.is_valid
    empty = gdf.geometry.is_empty | gdf.geometry.isna()
    inside = gdf.geometry.intersects(area)
    dup = gdf[id_col].duplicated(keep=False) if id_col in gdf else gdf.geometry.duplicated(keep=False)
    out.update({
        "invalid_geometries": int(invalid.sum()), "empty_geometries": int(empty.sum()),
        "outside_requested_area": int((~inside).sum()), "outside_ids": gdf.loc[~inside, id_col].astype(str).head(10).tolist() if id_col in gdf else [],
        "duplicate_ids": int(dup.sum()),
        "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
    })
    comp = {}
    for a in attrs or []:
        if a in gdf:
            col = gdf[a]
            comp[a] = round(float(1 - (col.isna() | (col.astype(str).str.strip() == "")).mean()), 4)
    out["attribute_completeness"] = comp
    problems = []
    if out["invalid_geometries"]:
        problems.append(f"{out['invalid_geometries']} invalid geometries")
    if out["outside_requested_area"]:
        problems.append(f"{out['outside_requested_area']} features outside the requested area")
    if out["duplicate_ids"]:
        problems.append(f"{out['duplicate_ids']} rows share an id")
    out["problems"] = problems
    return out


def cross_source(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame, crs, max_dist_m: float = 60.0, label_a: str = "A", label_b: str = "B") -> dict:
    """How much two independent sources agree on the same entity type (metric CRS conversion done here)."""
    if a.empty or b.empty:
        return {"compared": f"{label_a} vs {label_b}", "note": "one side is empty", "a": int(len(a)), "b": int(len(b))}
    am, bm = a.to_crs(crs).copy(), b.to_crs(crs).copy()
    for g in (am, bm):
        g["geometry"] = g.geometry.representative_point()
    am["poi_id"] = am["poi_id"].astype(str)
    bm["poi_id"] = bm["poi_id"].astype(str)
    sj = gpd.sjoin_nearest(am[["poi_id", "geometry"]], bm[["poi_id", "geometry"]], how="left", max_distance=max_dist_m, distance_col="d")
    sj = sj[~sj.index.duplicated()]
    matched = sj["d"].notna()
    return {"compared": f"{label_a} vs {label_b}", f"{label_a}_rows": int(len(am)), f"{label_b}_rows": int(len(bm)),
            f"{label_a}_with_{label_b}_within_{max_dist_m:g}m": int(matched.sum()),
            f"share_{label_a}_confirmed_by_{label_b}": round(float(matched.mean()), 4),
            "median_distance_m_of_matches": round(float(sj.loc[matched, "d"].median()), 1) if matched.any() else None}


def anchor_check(gdf: gpd.GeoDataFrame, anchor_lonlat: tuple, crs, radius_m: float) -> dict:
    """Do the results lie where the user said (e.g. within `radius_m` of the airport reference point)?"""
    from shapely.geometry import Point
    p = gpd.GeoSeries([Point(*anchor_lonlat)], crs=4326).to_crs(crs).iloc[0]
    d = gdf.to_crs(crs).geometry.distance(p)
    return {"anchor_lonlat": list(anchor_lonlat), "radius_m": radius_m, "within": int((d <= radius_m).sum()), "beyond": int((d > radius_m).sum()),
            "median_distance_m": round(float(d.median()), 1), "max_distance_m": round(float(d.max()), 1)}
