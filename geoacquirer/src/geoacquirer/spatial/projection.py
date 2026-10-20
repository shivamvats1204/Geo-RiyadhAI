"""Projection helpers. All metric computations happen in a local UTM CRS, never in degrees."""
from __future__ import annotations

import math

import geopandas as gpd
from pyproj import CRS


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def utm_crs_for_bbox(bbox: tuple[float, float, float, float]) -> CRS:
    w, s, e, n = bbox
    return utm_crs_for((w + e) / 2, (s + n) / 2)


def to_metric(gdf: gpd.GeoDataFrame, crs: CRS) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(crs)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
