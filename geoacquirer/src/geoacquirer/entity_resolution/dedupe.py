"""Cross-source POI de-duplication (keeps the better record, records what was merged)."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .resolve import match_pois


def dedupe_pois(poi: gpd.GeoDataFrame, source_col: str = "source_dataset", max_dist_m: float = 60.0) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Merge duplicate POIs that come from different sources (metric CRS in/out).

    Only pairs from *different* sources are considered duplicates. The record with higher reported confidence is
    kept; ties keep the first. Returns (deduped frame with `merged_ids`, merge log).
    """
    if source_col not in poi.columns or poi[source_col].nunique() < 2:
        return poi, pd.DataFrame(columns=["kept", "dropped", "match_type", "distance_m", "name_similarity"])
    sources = list(poi[source_col].dropna().unique())
    poi = poi.reset_index(drop=True).copy()
    poi["merged_ids"] = ""
    drop: set = set()
    log = []
    base = poi[poi[source_col] == sources[0]]
    for other in sources[1:]:
        o = poi[poi[source_col] == other]
        m = match_pois(base, o, max_dist_m=max_dist_m)
        for r in m.itertuples():
            a = poi.index[poi["poi_id"] == r.a_id][0]
            b = poi.index[poi["poi_id"] == r.b_id][0]
            ca, cb = poi.at[a, "confidence"] if "confidence" in poi else None, poi.at[b, "confidence"] if "confidence" in poi else None
            keep, gone = (a, b) if (pd.isna(cb) or (not pd.isna(ca) and ca >= cb)) else (b, a)
            poi.at[keep, "merged_ids"] = (poi.at[keep, "merged_ids"] + ";" if poi.at[keep, "merged_ids"] else "") + str(poi.at[gone, "poi_id"])
            drop.add(gone)
            log.append({"kept": poi.at[keep, "poi_id"], "dropped": poi.at[gone, "poi_id"], "match_type": r.match_type,
                        "distance_m": r.distance_m, "name_similarity": r.name_similarity})
    out = poi.drop(index=list(drop)).reset_index(drop=True)
    return out, pd.DataFrame(log)
