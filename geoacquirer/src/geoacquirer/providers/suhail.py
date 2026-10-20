"""Suhail parcel adapter (REFERENCE PORT of suhail_ai.ipynb) - DISABLED unless authorization is recorded.

What changed relative to the notebook (see docs/notebook_audit.md):
* gating: refuses to run without GEOACQUIRER_ENABLE_SUHAIL=1 + GEOACQUIRER_SUHAIL_AUTHORIZATION_REF
* parcel POLYGONS are preserved (the notebook saved centroids only)
* tile y handling made explicit (`y_coord_down=True`) and covered by a round-trip test
* parts of a parcel clipped across tile borders are unioned instead of silently de-duplicated
* bounded concurrency + rate limiting + backoff; 401/403 abort, repeated 429 abort; no evasion
* honest User-Agent; bare `except: pass` replaced by counted, reported failures
"""
from __future__ import annotations

import math
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape
from shapely.ops import unary_union

from ..acquisition import (AdaptiveCrawler, Checkpoint, FatalSourceError, RateLimitedError, TransientError)
from ..acquisition.grid import tiles_for_bbox
from ..security import RateLimiter
from .base import BBox, Capabilities, CoverageEstimate, FeatureBatch, Provenance, Provider, utcnow_iso
from .registry import require_authorized

ZOOM = 15
EXTENT = 4096
TILE_URL = "https://tiles.suhail.ai/maps/{slug}/{z}/{x}/{y}.vector.pbf"
RULES_URL = "https://api2.suhail.ai/parcel/buildingRules?parcelObjectId={}"
UA = "geoacquirer/0.1 (+authorized-research)"

# (display name, slug, (north, west, south, east)) - copied from the reference notebook
REGION_CONFIG = {
    "riyadh": ("Riyadh", "riyadh", (25.15, 46.40, 24.40, 47.10)),
    "eastern": ("Eastern Region (Dammam/Khobar)", "eastern_region", (26.60, 49.90, 26.20, 50.25)),
    "madinah": ("Madinah", "al_madenieh", (24.58, 39.45, 24.35, 39.75)),
    "makkah": ("Makkah", "makkah_region", (21.55, 39.70, 21.25, 40.05)),
    "qassim": ("Al Qassim (Buraydah)", "al_qassim", (26.45, 43.85, 26.25, 44.05)),
    "asir": ("Asir (Abha)", "asir_region", (18.35, 42.35, 18.15, 42.60)),
    "tabuk": ("Tabuk", "tabuk", (28.48, 36.45, 28.32, 36.65)),
    "hail": ("Hail", "hail", (27.60, 41.60, 27.40, 41.80)),
    "najran": ("Najran", "najran", (17.58, 44.08, 17.45, 44.25)),
    "jazan": ("Jazan", "jazan", (16.95, 42.50, 16.82, 42.65)),
    "jawf": ("Al Jouf (Sakaka)", "jawf", (30.00, 40.10, 29.85, 40.25)),
    "northern_borders": ("Northern Borders (Arar)", "northern_borders", (30.99, 41.00, 30.90, 41.15)),
    "bahah": ("Al Bahah", "bahah", (20.06, 41.42, 19.98, 41.52)),
}

RULE_FIELDS = ["zoningId", "zoningColor", "landuse", "description", "maxBuildingCoefficient",
               "maxBuildingHeight", "maxParcelCoverage", "mainStreetsSetback", "sideRearSetback"]


def tile_px_to_lonlat(px: float, py: float, z: int, tile_x: int, tile_y: int, extent: int = EXTENT):
    """Tile-local pixel (origin top-left, y down) -> lon/lat (Web Mercator tile scheme)."""
    n = 2 ** z
    x_frac = tile_x + px / extent
    y_frac = tile_y + py / extent
    lon = x_frac / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_frac / n))))
    return lon, lat


def decode_tile_parcels(content: bytes, z: int, x: int, y: int) -> list[dict]:
    """Decode an MVT tile to parcel polygons in lon/lat. Returns dicts with parcel_id, parcel_objectid, geometry."""
    import mapbox_vector_tile  # optional dependency
    import numpy as np
    import shapely

    def to_lonlat(arr):
        n = 2 ** z
        lon = (x + arr[:, 0] / EXTENT) / n * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + arr[:, 1] / EXTENT) / n))))
        return np.column_stack([lon, lat])

    tile = mapbox_vector_tile.decode(content, default_options={"y_coord_down": True})
    out = []
    for layer in tile.values():
        for f in layer.get("features", []):
            props = f.get("properties", {})
            oid = props.get("parcel_objectid") or props.get("OBJECTID")
            pid = props.get("parcel_id") or props.get("id") or oid
            geom = f.get("geometry")
            if not (pid and oid and geom):
                continue
            g = shape(geom)
            if g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            g = shapely.transform(g, to_lonlat)
            out.append({"parcel_id": str(pid), "parcel_objectid": str(oid), "geometry": g})
    return out


def merge_parts(rows: list[dict]) -> gpd.GeoDataFrame:
    """Union the pieces of parcels that were clipped by tile borders."""
    by_id: dict[str, list] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        by_id.setdefault(r["parcel_id"], []).append(r["geometry"])
        meta.setdefault(r["parcel_id"], r)
    recs = []
    for pid, geoms in by_id.items():
        g = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        recs.append({"parcel_id": pid, "parcel_objectid": meta[pid]["parcel_objectid"], "n_tile_parts": len(geoms), "geometry": g})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)


def parse_rules_response(data: dict, base: dict) -> list[dict]:
    """buildingRules JSON -> one row per rule (same semantics as the notebook, explicit statuses)."""
    items = (data or {}).get("data", []) or []
    if not items:
        return [dict(base, api_status="NO_RULES")]
    rows = []
    for item in items:
        r = dict(base)
        r["rule_id"] = item.get("id")
        for k in RULE_FIELDS:
            if k in item:
                r[k] = item[k]
        r["api_status"] = "SUCCESS"
        rows.append(r)
    return rows


class SuhailProvider(Provider):
    provider_id = "suhail"

    def __init__(self, region: str = "riyadh", max_workers: int = 4, min_interval_s: float = 0.05, session: Optional[requests.Session] = None):
        self.authorization_ref = require_authorized("suhail")
        if region not in REGION_CONFIG:
            raise ValueError(f"unknown region {region!r}; choose from {sorted(REGION_CONFIG)}")
        self.region = region
        self.slug = REGION_CONFIG[region][1]
        self.max_workers = max_workers
        self.limiter = RateLimiter(min_interval_s)
        self.session = session or requests.Session()
        # the default pool keeps only 10 connections: with dozens of threads every request would redo the TLS handshake (CPU-bound)
        adapter = requests.adapters.HTTPAdapter(pool_connections=max(10, max_workers * 2), pool_maxsize=max(10, max_workers * 2))
        self.session.mount("https://", adapter)
        self.session.headers["User-Agent"] = UA
        self.stats = None

    def discover_capabilities(self) -> Capabilities:
        return Capabilities(self.provider_id, ["parcel"], {"parcel": ["parcel_id", "parcel_objectid", *RULE_FIELDS]}, {"parcel": "Polygon"},
                            crs="EPSG:4326", notes=f"authorization ref: {self.authorization_ref}")

    def estimate_coverage(self, entity_type: str, bbox: BBox) -> CoverageEstimate:
        tiles = len(tiles_for_bbox(bbox, ZOOM))
        return CoverageEstimate(self.provider_id, entity_type, bbox, None, tiles, tiles * 0.3 / max(1, self.max_workers),
                                f"{tiles} zoom-{ZOOM} tiles; feature count unknown until fetched")

    def _fetch_tile(self, cell) -> list[dict]:
        x, y = cell
        r = self.session.get(TILE_URL.format(slug=self.slug, z=ZOOM, x=x, y=y), timeout=15)
        if r.status_code in (401, 403):
            raise FatalSourceError(f"HTTP {r.status_code} from tile server")
        if r.status_code == 429:
            raise RateLimitedError("429")
        if r.status_code >= 500:
            raise TransientError(f"HTTP {r.status_code}")
        if r.status_code != 200 or not r.content:
            return []
        return decode_tile_parcels(r.content, ZOOM, x, y)

    def fetch_features(self, entity_type: str, bbox: BBox, *, budget=None, checkpoint_path=None, cancel=None, **kw) -> FeatureBatch:
        if entity_type != "parcel":
            raise ValueError("suhail adapter only provides parcels")
        parts: list[dict] = []
        crawler = AdaptiveCrawler(self._fetch_tile, key=lambda it: (it["parcel_id"], it["geometry"].wkb_hex[:64]), cell_id=lambda c: f"{c[0]}/{c[1]}",
                                  max_workers=self.max_workers, limiter=self.limiter, budget=budget,
                                  checkpoint=Checkpoint(checkpoint_path), cancel=cancel)
        self.stats = crawler.run(tiles_for_bbox(bbox, ZOOM))
        parts = list(crawler.items.values())
        gdf = self.normalize("parcel", parts)
        warns = []
        if self.stats.aborted:
            warns.append(f"crawl aborted: {self.stats.aborted}")
        if self.stats.cells_failed:
            warns.append(f"{self.stats.cells_failed} tiles failed after retries; coverage incomplete")
        return FeatureBatch(gdf, Provenance(self.provider_id, "parcel", "live-tiles", "UNESTABLISHED - see authorization ref " + self.authorization_ref,
                                            "Suhail", utcnow_iso(), bbox, self.stats.cells_done), warns)

    def normalize(self, entity_type: str, raw) -> gpd.GeoDataFrame:
        if isinstance(raw, list):
            return merge_parts(raw) if raw else gpd.GeoDataFrame({"parcel_id": [], "parcel_objectid": []}, geometry=[], crs=4326)
        return raw

    def fetch_details(self, entity_type: str, ids: list[str], *, budget=None) -> gpd.GeoDataFrame:
        def one(oid):
            r = self.session.get(RULES_URL.format(oid), timeout=25)
            if r.status_code in (401, 403):
                raise FatalSourceError(f"HTTP {r.status_code}")
            if r.status_code == 429:
                raise RateLimitedError("429")
            if r.status_code >= 500:
                raise TransientError(str(r.status_code))
            if r.status_code == 404:
                return [{"parcel_objectid": oid, "api_status": "NOT_FOUND"}]
            if r.status_code != 200:
                return [{"parcel_objectid": oid, "api_status": f"HTTP_{r.status_code}"}]
            try:
                return parse_rules_response(r.json(), {"parcel_objectid": oid})
            except ValueError:
                return [{"parcel_objectid": oid, "api_status": "JSON_ERROR"}]

        crawler = AdaptiveCrawler(one, key=lambda it: (it["parcel_objectid"], it.get("rule_id")), cell_id=str, max_workers=self.max_workers,
                                  limiter=self.limiter, budget=budget)
        self.stats = crawler.run(ids)
        return pd.DataFrame(list(crawler.items.values()))

    def report_provenance(self) -> dict:
        return {"provider": self.provider_id, "authorization_ref": self.authorization_ref, "region": self.region,
                "crawl": self.stats.__dict__ if self.stats else None}
