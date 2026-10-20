"""Snapshot comparison. Never infers history from a single snapshot."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

TRACKED = ("name", "category", "operating_status")


def classify_change(old: gpd.GeoDataFrame, new: gpd.GeoDataFrame, subject: gpd.GeoDataFrame, key: str = "poi_id") -> pd.DataFrame:
    """Per-subject change type against two snapshots: added / attribute_changed / unchanged / removed."""
    old_i = old.drop_duplicates(key).set_index(key)
    rows = []
    for k in subject[key]:
        if k not in old_i.index:
            rows.append(("added", True))
            continue
        o = old_i.loc[k]
        n = new.drop_duplicates(key).set_index(key).loc[k] if k in set(new[key]) else None
        if n is None:
            rows.append(("removed", True))
            continue
        diff = [c for c in TRACKED if c in old_i.columns and c in new.columns and str(o[c]) != str(n[c])]
        rows.append((f"attribute_changed:{','.join(diff)}" if diff else "unchanged", bool(diff)))
    return pd.DataFrame(rows, columns=["change_type", "changed"], index=subject.index)


def removed_entities(old: gpd.GeoDataFrame, new: gpd.GeoDataFrame, key: str = "poi_id") -> gpd.GeoDataFrame:
    return old[~old[key].isin(set(new[key]))]


def coverage_caveat(old: gpd.GeoDataFrame, new: gpd.GeoDataFrame) -> dict:
    """Large count swings usually mean source coverage changed, not the city."""
    a, b = len(old), len(new)
    ratio = (b / a) if a else float("inf")
    return {"old_count": a, "new_count": b, "ratio": ratio,
            "suspicious_coverage_shift": bool(a and (ratio > 1.5 or ratio < 0.67))}
