"""Spatial feature engineering for candidate locations (all distances in metres, UTM CRS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from geoacquirer.ontology import category_mask

from .store import AreaDataset

NON_RESIDENTIAL_SUBTYPES = {"commercial", "industrial", "education", "religious", "medical", "civic", "transportation", "entertainment",
                            "service", "agricultural", "outbuilding"}
ROAD_SCORE = {"motorway": 0.0, "trunk": 0.0, "primary": 1.0, "secondary": 0.85, "tertiary": 0.65, "residential": 0.4,
              "unclassified": 0.3, "living_street": 0.2, "service": 0.1}


@dataclass
class FeatureConfig:
    demand_radius_m: float = 800.0
    competitor_radius_m: float = 500.0
    retail_radius_m: float = 500.0
    parking_radius_m: float = 300.0
    highway_exclusion_m: float = 60.0
    street_frontage_m: float = 80.0
    grid_step_m: float = 150.0
    competitor_concepts: tuple = ("supermarket",)
    exclude_occupied: bool = True  # drop cells inside existing building footprints


def _within_counts(cands: gpd.GeoSeries, objs: gpd.GeoSeries, r: float) -> np.ndarray:
    out = np.zeros(len(cands), dtype=int)
    if len(objs) == 0 or len(cands) == 0:
        return out
    tree = objs.sindex
    idx = tree.query(cands.values, predicate="dwithin", distance=r)
    np.add.at(out, idx[0], 1)
    return out


def _weighted_within(cands: gpd.GeoSeries, objs: gpd.GeoSeries, w: np.ndarray, r: float) -> np.ndarray:
    out = np.zeros(len(cands))
    if len(objs) == 0 or len(cands) == 0:
        return out
    idx = objs.sindex.query(cands.values, predicate="dwithin", distance=r)
    np.add.at(out, idx[0], w[idx[1]])
    return out


def _nearest_dist(cands: gpd.GeoSeries, objs: gpd.GeoSeries, cap: float = 5000.0) -> np.ndarray:
    d = np.full(len(cands), cap)
    if len(objs) == 0 or len(cands) == 0:
        return d
    pairs, dist = objs.sindex.nearest(cands.values, return_all=False, max_distance=cap, return_distance=True)
    d[pairs[0]] = dist
    return d


def residential_like_mask(b: gpd.GeoDataFrame) -> pd.Series:
    """Buildings that are not positively labelled as something other than housing.

    Overture labels only a minority of buildings (mostly OSM-sourced). Unlabelled footprints (largely ML-detected) are
    treated as residential-like, which is an assumption; `demand` is a proxy, not a population estimate.
    """
    sub = b["subtype"].fillna("")
    cls = b["class"].fillna("")
    non = sub.isin(NON_RESIDENTIAL_SUBTYPES) | cls.isin({"commercial", "retail", "industrial", "school", "hospital", "mosque", "university", "warehouse", "office"})
    return ~non


def candidate_cells(ds: AreaDataset, cfg: FeatureConfig, region: Optional[object] = None) -> tuple[gpd.GeoDataFrame, dict]:
    """Square-grid candidate points that pass the hard eligibility filters. Returns (cells, filter_counts)."""
    poly = region if region is not None else ds.area_polygon()
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + cfg.grid_step_m / 2, maxx, cfg.grid_step_m)
    ys = np.arange(miny + cfg.grid_step_m / 2, maxy, cfg.grid_step_m)
    pts = gpd.GeoSeries([Point(x, y) for x in xs for y in ys], crs=ds.crs)
    pts = pts[pts.within(poly)].reset_index(drop=True)
    counts = {"grid_cells": len(pts)}
    road = ds.layers["road"]
    hw = road[road["class"].isin(["motorway", "trunk"])]
    keep = _nearest_dist(pts, hw.geometry, cap=1e6) > cfg.highway_exclusion_m
    counts["removed_near_highway"] = int((~keep).sum())
    pts = pts[keep].reset_index(drop=True)
    drivable = road[~road["class"].isin(["motorway", "trunk", "service"])]
    on_street = _nearest_dist(pts, drivable.geometry, cap=1e6) <= cfg.street_frontage_m
    counts["removed_no_street_within_%dm" % cfg.street_frontage_m] = int((~on_street).sum())
    pts = pts[on_street].reset_index(drop=True)
    if cfg.exclude_occupied and ds.has("building"):
        occ = ds.layers["building"]
        hit = occ.sindex.query(pts.values, predicate="within")
        occupied = np.zeros(len(pts), dtype=bool)
        occupied[hit[0]] = True
        counts["removed_inside_existing_building"] = int(occupied.sum())
        pts = pts[~occupied].reset_index(drop=True)
    counts["eligible_cells"] = len(pts)
    cells = gpd.GeoDataFrame({"cell_id": [f"c{i}" for i in range(len(pts))]}, geometry=pts.values, crs=ds.crs)
    return cells, counts


def network_demand(ds: AreaDataset, cands: gpd.GeoSeries, weights_gdf: gpd.GeoDataFrame, weight_col: str, cutoff_m: float) -> np.ndarray:
    """Sum of `weight_col` of demand objects reachable within `cutoff_m` along the road graph (undirected distance)."""
    net = ds.network()
    nodes, snap = net.snap(weights_gdf.geometry.centroid, max_snap_m=150.0)
    wmap: dict[str, float] = {}
    for n, w in zip(nodes, weights_gdf[weight_col].to_numpy()):
        if n is not None:
            wmap[n] = wmap.get(n, 0.0) + float(w)
    cn, cs = net.snap(cands, max_snap_m=250.0)
    out = np.zeros(len(cands))
    cache: dict[str, float] = {}
    for i, n in enumerate(cn):
        if n is None:
            out[i] = np.nan
            continue
        if n not in cache:
            reach = nx.single_source_dijkstra_path_length(net.graph, n, cutoff=cutoff_m, weight="weight")
            cache[n] = sum(wmap.get(k, 0.0) for k in reach)
        out[i] = cache[n]
    return out


def compute_features(ds: AreaDataset, cells: gpd.GeoDataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = cells.geometry
    feats = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()})
    junc = ds.layers["junction"]
    four = junc[junc["kind"] == "four_way"]
    feats["dist_to_four_way_m"] = _nearest_dist(g, four.geometry, 3000.0)
    feats["four_way_within_300m"] = _within_counts(g, four.geometry, 300.0)
    road = ds.layers["road"]
    near_roads = road[~road["class"].isin(["service"])]
    pr = near_roads.sindex.query(g.values, predicate="dwithin", distance=60.0)
    score = np.zeros(len(g))
    cls = near_roads["class"].to_numpy()
    for i, j in zip(*pr):
        score[i] = max(score[i], ROAD_SCORE.get(cls[j], 0.2))
    feats["road_class_score"] = score
    hw = road[road["class"].isin(["motorway", "trunk"])]
    feats["dist_to_highway_m"] = _nearest_dist(g, hw.geometry, 5000.0)
    poi = ds.layers["poi"]
    comp = poi[category_mask(poi["category_path"], list(cfg.competitor_concepts))]
    feats["competitors_500m"] = _within_counts(g, comp.geometry, cfg.competitor_radius_m)
    feats["nearest_competitor_m"] = _nearest_dist(g, comp.geometry, 5000.0)
    shops = poi[category_mask(poi["category_path"], ["shop"])]
    feats["retail_pois_500m"] = _within_counts(g, shops.geometry, cfg.retail_radius_m)
    park = ds.layers.get("parking")
    feats["parking_pois_300m"] = _within_counts(g, park.geometry, cfg.parking_radius_m) if park is not None and len(park) else 0
    feats["demand_source"] = "none"
    if ds.has("urbi_building"):
        # Urbi buildings carry an explicit purpose and floor count: demand = residential/mixed floor area (footprint x floors)
        ub = ds.layers["urbi_building"]
        res = ub[ub["purpose_class"].isin(["residential", "mixed"])]
        res = res.assign(area_m2=res.geometry.area * res["floors"].fillna(1).clip(lower=1, upper=40))
        cen = gpd.GeoDataFrame(geometry=res.geometry.centroid.values, crs=ds.crs)
        feats["resid_like_built_area_800m"] = _weighted_within(g, cen.geometry, res["area_m2"].to_numpy(), cfg.demand_radius_m)
        feats["network_demand_800m"] = network_demand(ds, g, res.assign(geometry=res.geometry.centroid), "area_m2", cfg.demand_radius_m)
        feats["demand_source"] = "urbi residential floor area (footprint x floors)"
    elif ds.has("building"):
        b = ds.layers["building"]
        res = b[residential_like_mask(b)]
        res = res.assign(area_m2=res.geometry.area)
        cen = gpd.GeoDataFrame(geometry=res.geometry.centroid.values, crs=ds.crs)
        feats["resid_like_built_area_800m"] = _weighted_within(g, cen.geometry, res["area_m2"].to_numpy(), cfg.demand_radius_m)
        feats["network_demand_800m"] = network_demand(ds, g, res.assign(geometry=res.geometry.centroid), "area_m2", cfg.demand_radius_m)
        feats["demand_source"] = "overture residential-like footprint area"
    else:
        feats["resid_like_built_area_800m"] = np.nan
        feats["network_demand_800m"] = np.nan
    metro = ds.layers.get("metro")
    feats["dist_to_metro_m"] = _nearest_dist(g, metro.geometry, 10000.0) if metro is not None and len(metro) else np.nan
    feats["landuse_zoning"] = "UNVERIFIED"
    if ds.has("parcel") and "landuse" in ds.layers["parcel"].columns:
        sj = gpd.sjoin(cells[["cell_id", "geometry"]], ds.layers["parcel"][["landuse", "geometry"]], how="left", predicate="within")
        sj = sj[~sj.cell_id.duplicated()]
        feats["landuse_zoning"] = sj["landuse"].fillna("no_parcel_match").to_numpy()
    return feats
