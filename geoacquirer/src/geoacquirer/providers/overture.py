"""Overture Maps adapter (public, no credentials) via DuckDB over the official S3 GeoParquet.

Overture's transportation theme carries explicit connectors, which is what makes
genuine-junction detection possible (see spatial/topology.py).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import duckdb
import geopandas as gpd
import pandas as pd
import requests
from shapely import wkb

from ..security import check_domain, safe_release, sql_float
from ..storage import LayerCache
from .base import (BBox, Capabilities, CoverageEstimate, FeatureBatch, Provenance, Provider,
                   utcnow_iso)

FALLBACK_RELEASE = "2026-09-23.1"
S3_BASE = "s3://overturemaps-us-west-2/release"
SCHEMA_VERSION = 3  # bump to invalidate caches when normalization changes

_THEME = {
    "poi": ("places", "place"),
    "road": ("transportation", "segment"),
    "connector": ("transportation", "connector"),
    "building": ("buildings", "building"),
    "district": ("divisions", "division_area"),
}
_LICENSE = {
    "poi": "CDLA Permissive 2.0 (verify per-record `sources.license`)",
    "road": "ODbL 1.0 (OpenStreetMap-derived; share-alike) (verify per-record `sources.license`)",
    "connector": "ODbL 1.0 (OpenStreetMap-derived; share-alike)",
    "building": "ODbL 1.0 / CDLA mix by source (verify per-record `sources.license`)",
    "district": "ODbL 1.0 (OSM-derived administrative areas; verify per-record `sources.license`)",
}


def latest_release(timeout: float = 20.0) -> str:
    env = os.environ.get("GEOACQUIRER_OVERTURE_RELEASE")
    if env:
        return safe_release(env)
    try:
        check_domain("https://stac.overturemaps.org")
        r = requests.get("https://stac.overturemaps.org/catalog.json", timeout=timeout,
                         headers={"User-Agent": "geoacquirer/0.1"})
        r.raise_for_status()
        return safe_release(r.json().get("latest", FALLBACK_RELEASE))
    except Exception:
        return FALLBACK_RELEASE


class OvertureProvider(Provider):
    provider_id = "overture"

    def __init__(self, release: Optional[str] = None, cache: Optional[LayerCache] = None):
        self.release = safe_release(release) if release else None
        self.cache = cache or LayerCache()
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self.requests_made = 0
        self.last_latency_s = 0.0

    # ---- capabilities -------------------------------------------------
    def discover_capabilities(self) -> Capabilities:
        return Capabilities(
            provider_id=self.provider_id,
            entity_types=["poi", "road", "connector", "building", "district"],
            attributes={
                "poi": ["name", "name_ar", "category", "category_path", "confidence", "operating_status", "address"],
                "road": ["class", "subclass", "road_name", "connectors", "is_link", "is_bridge", "is_tunnel"],
                "connector": ["id"],
                "building": ["height", "num_floors", "subtype", "class", "name"],
            },
            geometry={"poi": "Point", "road": "LineString", "connector": "Point", "building": "Polygon"},
            notes="No parcels. Building footprints are OSM/ML-derived and not cadastral.",
        )

    def estimate_coverage(self, entity_type: str, bbox: BBox) -> CoverageEstimate:
        return CoverageEstimate(
            self.provider_id, entity_type, bbox, expected_features=None, estimated_requests=1,
            estimated_seconds=45.0,
            basis="one DuckDB scan with bbox predicate pushdown; ~30-60 s per layer measured in this project",
        )

    # ---- connection ---------------------------------------------------
    def _connection(self):
        if self._con is None:
            check_domain("overturemaps-us-west-2.s3.us-west-2.amazonaws.com")
            con = duckdb.connect()
            con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
            con.execute("SET s3_region='us-west-2';")
            self._con = con
        return self._con

    def _release(self) -> str:
        if self.release is None:
            self.release = latest_release()
        return self.release

    # ---- fetch --------------------------------------------------------
    def fetch_features(self, entity_type: str, bbox: BBox, *, budget=None, use_cache: bool = True,
                       max_age_days: Optional[float] = 30, **kwargs) -> FeatureBatch:
        if entity_type not in _THEME:
            raise ValueError(f"unsupported entity type {entity_type!r}")
        w, s, e, n = (round(v, 6) for v in bbox)
        release = self._release()
        key = LayerCache.key(p="overture", r=release, t=entity_type, b=(w, s, e, n), v=SCHEMA_VERSION)
        if use_cache:
            hit = self.cache.get(key, max_age_days) or self.cache.find_covering(entity_type, release, (w, s, e, n), SCHEMA_VERSION)
            if hit:
                gdf, meta = hit
                return FeatureBatch(gdf, Provenance(
                    self.provider_id, entity_type, release, _LICENSE[entity_type],
                    "© Overture Maps Foundation; © OpenStreetMap contributors where applicable",
                    meta.get("retrieved_at", ""), (w, s, e, n), 0, True, key))
        if budget is not None:
            budget.charge_request(1)
        theme, typ = _THEME[entity_type]
        path = f"{S3_BASE}/{release}/theme={theme}/type={typ}/*"
        where = (f"bbox.xmin <= {sql_float(e)} AND bbox.xmax >= {sql_float(w)} "
                 f"AND bbox.ymin <= {sql_float(n)} AND bbox.ymax >= {sql_float(s)}")
        sql = _SQL[entity_type].format(path=path, where=where)
        t0 = time.monotonic()
        df = self._connection().execute(sql).df()
        self.last_latency_s = time.monotonic() - t0
        self.requests_made += 1
        gdf = self.normalize(entity_type, df)
        if budget is not None:
            budget.charge_features(len(gdf))
        retrieved = utcnow_iso()
        self.cache.put(key, gdf, {"retrieved_at": retrieved, "release": release, "entity": entity_type,
                                  "latency_s": self.last_latency_s, "bbox": [w, s, e, n], "schema": SCHEMA_VERSION})
        warns = []
        if gdf.empty:
            warns.append(f"overture returned 0 {entity_type} features for bbox {bbox}")
        return FeatureBatch(gdf, Provenance(
            self.provider_id, entity_type, release, _LICENSE[entity_type],
            "© Overture Maps Foundation; © OpenStreetMap contributors where applicable",
            retrieved, (w, s, e, n), 1, False, key,
            {"latency_s": round(self.last_latency_s, 2)}), warns)

    # ---- normalization ------------------------------------------------
    def normalize(self, entity_type: str, raw: pd.DataFrame) -> gpd.GeoDataFrame:
        df = raw.copy()
        geom = df.pop("wkb").apply(lambda b: wkb.loads(bytes(b)) if b is not None else None)
        if entity_type == "poi":
            df["category_path"] = df["category_path"].apply(_as_str_list)
        for col in ("connectors_json", "road_flags_json"):
            if col in df:
                df[col] = df[col].astype(object).where(df[col].notna(), None)
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)

    def report_provenance(self) -> dict:
        return {"provider": self.provider_id, "release": self._release(), "requests_made": self.requests_made,
                "license": _LICENSE}


def _as_str_list(x) -> list[str]:
    if x is None:
        return []
    try:
        return [str(v) for v in list(x)]
    except TypeError:
        return []


_SQL = {
    "poi": """
        SELECT id AS poi_id, names.primary AS name,
               taxonomy.primary AS category, taxonomy.hierarchy AS category_path,
               basic_category, confidence, operating_status,
               CAST(addresses[1].freeform AS VARCHAR) AS address,
               CAST(sources[1].dataset AS VARCHAR) AS source_dataset,
               ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{path}', hive_partitioning=1) WHERE {where}""",
    "road": """
        SELECT id AS road_id, class AS "class", subclass, subtype, names.primary AS road_name,
               CAST(to_json(connectors) AS VARCHAR) AS connectors_json,
               CAST(to_json(road_flags) AS VARCHAR) AS road_flags_json,
               ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{path}', hive_partitioning=1) WHERE subtype = 'road' AND {where}""",
    "connector": """
        SELECT id AS connector_id, ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{path}', hive_partitioning=1) WHERE {where}""",
    "district": """
        SELECT id AS district_id, names.primary AS name, subtype, "class", admin_level, ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{path}', hive_partitioning=1)
        WHERE subtype IN ('macrohood', 'neighborhood', 'microhood') AND {where}""",
    "building": """
        SELECT id AS building_id, names.primary AS name, subtype, "class", height, num_floors,
               CAST(sources[1].dataset AS VARCHAR) AS source_dataset, ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{path}', hive_partitioning=1) WHERE {where}""",
}


def segments_frame(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Adapt a normalized Overture road layer to the shape `spatial.topology` expects."""
    seg = roads.rename(columns={"road_id": "id"}).copy()
    seg["connectors"] = seg["connectors_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    seg["road_flags"] = seg["road_flags_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    return seg


def connectors_frame(conn: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return conn.rename(columns={"connector_id": "id"}).copy()
