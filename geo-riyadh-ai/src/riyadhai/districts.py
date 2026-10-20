"""Unified registry of every district / neighbourhood-type area of Riyadh from three independent sources.

Sources
  urbi     2GIS/Urbi catalogue `adm_div` objects referenced by the downloaded records: district, living_area, settlement, district_area
  suhail   Suhail vector-tile `neighborhoods` layer (Arabic names, ids) + `neighborhoods-centroids` for ids with no polygon
  overture Overture `division_area` neighborhood / macrohood / microhood polygons

Each registry row has a polygon when the source supplied one, otherwise its centroid is the geometry (`has_polygon` False).
Rows are never merged across sources: they are different datasets. `crosswalk` records, for each Suhail/Overture polygon, the Urbi district with the
largest overlap and the share of its area it covers, so disagreement between sources is visible instead of hidden.
"""
from __future__ import annotations


import geopandas as gpd
import pandas as pd
import shapely
from shapely import wkt

URBI_KINDS = ("district", "living_area", "settlement", "district_area")


def _urbi_rows(raw: list[dict], ar: dict[str, dict]) -> list[dict]:
    rows = []
    for it in raw:
        uid = it["id"].split("_")[0]
        geo = it.get("geometry", {}) or {}
        poly = None
        for k in ("hover", "selection"):
            if geo.get(k) and geo[k].startswith(("POLYGON", "MULTIPOLYGON")):
                poly = wkt.loads(geo[k])
                break
        cen = wkt.loads(geo["centroid"]) if geo.get("centroid") else None
        g = poly if poly is not None else cen
        if g is None:
            continue
        a = ar.get(uid, {})
        rows.append({"uid": f"urbi:{uid}", "source": "urbi", "kind": it.get("subtype") or it.get("type"), "src_id": uid, "name": it.get("name"),
                     "name_ar": a.get("name"), "full_name": it.get("full_name"), "has_polygon": poly is not None, "geometry": g,
                     "centroid_lon": (cen or g.centroid).x, "centroid_lat": (cen or g.centroid).y})
    return rows


def build_registry(urbi_raw: list[dict], urbi_ar: dict, suhail_poly: gpd.GeoDataFrame, suhail_cent: gpd.GeoDataFrame, overture: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = _urbi_rows(urbi_raw, urbi_ar)
    for r in suhail_poly.itertuples():
        c = r.geometry.representative_point()
        rows.append({"uid": f"suhail:{r.neighborhood_id}", "source": "suhail", "kind": "neighborhood", "src_id": str(r.neighborhood_id), "name": None, "name_ar": r.name_ar,
                     "full_name": r.name_ar, "has_polygon": True, "geometry": r.geometry, "centroid_lon": c.x, "centroid_lat": c.y})
    have = set(suhail_poly.neighborhood_id)
    for r in suhail_cent.itertuples():
        if r.neighborhood_id in have:
            continue
        rows.append({"uid": f"suhail:{r.neighborhood_id}", "source": "suhail", "kind": "neighborhood", "src_id": str(r.neighborhood_id), "name": None, "name_ar": r.name_ar,
                     "full_name": r.name_ar, "has_polygon": False, "geometry": r.geometry, "centroid_lon": r.geometry.x, "centroid_lat": r.geometry.y})
    for r in overture.itertuples():
        if r.geometry is None or r.geometry.is_empty:
            continue
        c = r.geometry.representative_point()
        rows.append({"uid": f"overture:{r.district_id}", "source": "overture", "kind": r.subtype, "src_id": str(r.district_id), "name": r.name, "name_ar": None,
                     "full_name": r.name, "has_polygon": True, "geometry": r.geometry, "centroid_lon": c.x, "centroid_lat": c.y})
    reg = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    bad = ~shapely.is_valid(reg.geometry.values)
    if bad.any():
        reg.loc[bad, "geometry"] = shapely.make_valid(reg.geometry.values[bad])
    return reg


def crosswalk(reg: gpd.GeoDataFrame) -> pd.DataFrame:
    """For every non-Urbi polygon: the Urbi district polygon covering most of it (area share of the polygon, metric CRS)."""
    ud = reg[(reg.source == "urbi") & (reg.kind == "district") & reg.has_polygon]
    other = reg[(reg.source != "urbi") & reg.has_polygon]
    crs = 32638  # UTM 38N covers Riyadh
    udm, om = ud.to_crs(crs), other.to_crs(crs)
    tree = shapely.STRtree(udm.geometry.values)
    out = []
    for uid, g in zip(om.uid, om.geometry):
        best, best_inter = None, 0.0
        for j in tree.query(g, predicate="intersects"):
            inter = g.intersection(udm.geometry.values[j]).area
            if inter > best_inter:
                best_inter, best = inter, udm.uid.values[j]
        out.append({"uid": uid, "best_urbi": best, "overlap_share": round(best_inter / g.area, 4) if g.area else 0.0})
    return pd.DataFrame(out)
