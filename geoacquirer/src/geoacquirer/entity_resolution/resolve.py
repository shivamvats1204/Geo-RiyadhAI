"""Cross-source entity resolution with explicit, inspectable evidence.

We do not emit numeric match probabilities: they would be uncalibrated. Each link carries a
categorical `match_type` plus the raw evidence (distance, containment, name similarity) so
a reviewer can judge it, and precision/recall are measured on labelled samples in benchmarks/.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_STOP = {"shop", "store", "market", "co", "company", "ltd", "est", "establishment", "مؤسسة", "شركة", "محل", "the", "al", "ال"}
_STOP = _STOP | {s.translate(_AR_MAP) for s in _STOP}  # stop words must be normalized like the text they are matched against


def normalize_name(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and s != s) or s is pd.NA or not str(s).strip():
        return ""  # NaN/None/NA must never normalize to the string 'nan' (two nameless POIs are not the same business)
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip diacritics/tashkeel
    s = s.translate(_AR_MAP)
    s = re.sub(r"[^\w\s؀-ۿ]", " ", s)
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks)


def name_similarity(a: Optional[str], b: Optional[str]) -> float:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    sa, sb = set(na.split()), set(nb.split())
    jacc = len(sa & sb) / len(sa | sb)
    seq = SequenceMatcher(None, na, nb).ratio()
    return round(max(jacc, seq), 3)


def resolve_poi_to_building(poi: gpd.GeoDataFrame, buildings: Optional[gpd.GeoDataFrame], max_dist_m: float = 20.0) -> pd.DataFrame:
    """Link each POI to a building footprint (metric CRS in, evidence out)."""
    n = len(poi)
    out = pd.DataFrame({"building_id": [None] * n, "building_match": ["none"] * n, "building_distance_m": [np.nan] * n,
                        "building_candidates": [0] * n}, index=poi.index)
    if buildings is None or buildings.empty or n == 0:
        out["building_match"] = "layer_unavailable" if buildings is None else "none"
        return out
    bid_col = "building_id" if "building_id" in buildings.columns else None
    b = buildings.reset_index(drop=True)
    p = poi.reset_index()
    sj = gpd.sjoin(p[["index", "geometry"]], b[["geometry"]].assign(_area=b.geometry.area), how="inner", predicate="within")
    for pidx, grp in sj.groupby("index"):
        best = grp.sort_values("_area").iloc[0]  # smallest containing footprint (innermost)
        bi = int(best["index_right"])
        pos = poi.index.get_loc(pidx)
        out.iat[pos, 0] = b.iloc[bi][bid_col] if bid_col else bi
        out.iat[pos, 1] = "contained"
        out.iat[pos, 2] = 0.0
        out.iat[pos, 3] = len(grp)
    todo = out.index[out["building_match"] == "none"]
    if len(todo):
        sub = poi.loc[todo]
        pairs, d = b.sindex.nearest(sub.geometry.values, return_all=False, max_distance=max_dist_m, return_distance=True)
        for si, bi, dd in zip(pairs[0], pairs[1], d):
            pos = poi.index.get_loc(todo[si])
            out.iat[pos, 0] = b.iloc[bi][bid_col] if bid_col else int(bi)
            out.iat[pos, 1] = f"nearest_within_{max_dist_m:g}m"
            out.iat[pos, 2] = round(float(dd), 2)
            out.iat[pos, 3] = 1
    return out


def resolve_building_to_parcel(buildings: gpd.GeoDataFrame, parcels: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    n = len(buildings)
    out = pd.DataFrame({"parcel_id": [None] * n, "parcel_match": ["none"] * n, "parcel_overlap_fraction": [np.nan] * n}, index=buildings.index)
    if parcels is None or parcels.empty or n == 0:
        out["parcel_match"] = "layer_unavailable"
        return out
    has_poly = parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any()
    if has_poly:
        b = buildings.reset_index()
        sj = gpd.sjoin(b[["index", "geometry"]], parcels[["parcel_id", "geometry"]], how="inner", predicate="intersects")
        for bidx, grp in sj.groupby("index"):
            geom = b.loc[b["index"] == bidx, "geometry"].iloc[0]
            best, frac = None, 0.0
            for _, r in grp.iterrows():
                pg = parcels.loc[parcels["parcel_id"] == r["parcel_id"], "geometry"].iloc[0]
                f = geom.intersection(pg).area / geom.area if geom.area else 0.0
                if f > frac:
                    best, frac = r["parcel_id"], f
            pos = buildings.index.get_loc(bidx)
            out.iat[pos, 0] = best
            out.iat[pos, 1] = "polygon_overlap" if frac >= 0.5 else "partial_overlap"
            out.iat[pos, 2] = round(float(frac), 3)
    else:  # centroid-only parcels (as in the Suhail tile pull): evidence is much weaker
        sj = gpd.sjoin(parcels[["parcel_id", "geometry"]], buildings[["geometry"]], how="inner", predicate="within")
        for pid, bi in zip(sj["parcel_id"], sj["index_right"]):
            pos = buildings.index.get_loc(bi)
            if out.iat[pos, 0] is None:
                out.iat[pos, 0] = pid
                out.iat[pos, 1] = "parcel_centroid_in_building"
    return out


def resolve_poi_to_roads(poi: gpd.GeoDataFrame, roads: Optional[gpd.GeoDataFrame], radius_m: float = 60.0, k: int = 3) -> pd.Series:
    """Nearest named/classified roads within radius, as a compact string per POI."""
    if roads is None or roads.empty or len(poi) == 0:
        return pd.Series([""] * len(poi), index=poi.index)
    buf = poi[["geometry"]].copy().reset_index()
    buf["geometry"] = buf.geometry.buffer(radius_m)
    sj = gpd.sjoin(buf, roads[["geometry", "road_name", "class"]], how="inner", predicate="intersects")
    pts = poi.geometry
    res = {}
    for pidx, grp in sj.groupby("index"):
        rows = []
        for _, r in grp.iterrows():
            rg = roads.geometry.loc[r["index_right"]]
            rows.append((pts.loc[pidx].distance(rg), r["road_name"] if isinstance(r["road_name"], str) else "(unnamed)", r["class"]))
        rows.sort()
        seen, parts = set(), []
        for d, nm, cl in rows:
            key = (nm, cl)
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"{nm} [{cl}] {d:.0f}m")
            if len(parts) >= k:
                break
        res[pidx] = "; ".join(parts)
    return pd.Series([res.get(i, "") for i in poi.index], index=poi.index)


def _specific_tokens(path) -> set:
    if path is None or isinstance(path, float):
        return set()
    p = list(path)
    return set(p[1:]) if len(p) > 1 else set()


def match_pois(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame, max_dist_m: float = 60.0, min_name_sim: float = 0.6,
               key_a: str = "poi_id", key_b: str = "poi_id") -> pd.DataFrame:
    """Cross-source POI matching: stable id > proximity+name > proximity+category (flagged weaker)."""
    rows = []
    if a.empty or b.empty:
        return pd.DataFrame(columns=["a_id", "b_id", "match_type", "distance_m", "name_similarity", "category_compatible"])
    bb = b.reset_index(drop=True)
    best: dict[int, tuple] = {}
    cand = bb.sindex.query(a.geometry.buffer(max_dist_m).values, predicate="intersects")
    for ai, bi in zip(cand[0], cand[1]):
        ra, rb = a.iloc[ai], bb.iloc[bi]
        d = ra.geometry.distance(rb.geometry)
        if d > max_dist_m:
            continue
        sim = name_similarity(ra.get("name"), rb.get("name"))
        ca, cb = _specific_tokens(ra.get("category_path")), _specific_tokens(rb.get("category_path"))
        cat_ok = bool(ca & cb) if ca and cb else None  # the shared top-level branch ('shopping') alone is not compatibility
        if ra.get(key_a) is not None and ra.get(key_a) == rb.get(key_b):
            mt = "same_id"
        elif sim >= min_name_sim:
            mt = "proximity_and_name"
        elif cat_ok and d <= 15:
            mt = "proximity_and_category_only"
        else:
            continue
        score = (0 if mt == "same_id" else 1 if mt == "proximity_and_name" else 2, -sim, d)
        if ai not in best or score < best[ai][0]:
            best[ai] = (score, int(bi), mt, d, sim, cat_ok)
    used_b: set = set()
    for ai in sorted(best, key=lambda k: best[k][0]):
        _, bi, mt, d, sim, cat_ok = best[ai]
        if bi in used_b:
            continue
        used_b.add(bi)
        rows.append({"a_id": a.iloc[ai][key_a], "b_id": bb.iloc[bi][key_b], "match_type": mt, "distance_m": round(float(d), 2),
                     "name_similarity": sim, "category_compatible": cat_ok})
    return pd.DataFrame(rows)
