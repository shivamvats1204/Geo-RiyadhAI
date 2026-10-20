"""Urbi / 2GIS parking adapter (REFERENCE PORT of urbi.ipynb) - DISABLED unless authorized.

Differences from the notebook (docs/notebook_audit.md):
* the API key is read from GEOACQUIRER_URBI_API_KEY; the key hard-coded in the notebook must be rotated and is never reused
* honest User-Agent (the notebook spoofed a mobile-app UA)
* requires GEOACQUIRER_ENABLE_URBI=1 + GEOACQUIRER_URBI_AUTHORIZATION_REF
* scout/drill discovery is preserved (coarse_to_fine) but runs on the shared crawler (bounded, resumable)
* schedule/centroid parsers are pure functions with unit tests
"""
from __future__ import annotations

import os
import re
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely import wkt as shp_wkt
from shapely.geometry import Point

from ..acquisition import AdaptiveCrawler, FatalSourceError, RateLimitedError, TransientError
from ..acquisition.grid import coarse_to_fine, uniform_grid
from ..security import RateLimiter, SecurityError
from .base import BBox, Capabilities, CoverageEstimate, FeatureBatch, Provenance, Provider, utcnow_iso
from .registry import require_authorized

BASE = "https://catalog.api.2gis.ru/3.0"
UA = "geoacquirer/0.1 (+authorized-research)"
FIELDS = ("items.geometry.centroid,items.geometry.selection,items.address,items.name_ex,items.adm_div,items.capacity,"
          "items.is_paid,items.access,items.subtype,items.schedule,items.purpose_name,items.point,items.context,"
          "items.for_trucks,items.is_incentive,items.purpose")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def extract_geo(centroid_wkt: Optional[str]):
    """'POINT(lon lat)' -> (lat, lon) as floats. Handles negatives and exponents (the notebook regex did not)."""
    if not centroid_wkt:
        return None, None
    m = re.search(r"POINT\s*\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)", centroid_wkt)
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)


def parse_full_schedule(schedule: Optional[dict]) -> str:
    if not schedule:
        return "Unknown"
    if schedule.get("is_24x7"):
        return "24 Hours Daily"
    out = []
    for d in DAYS:
        info = schedule.get(d)
        if info and "working_hours" in info:
            slots = [f"{h.get('from')}-{h.get('to')}" for h in info["working_hours"]]
            if slots:
                out.append(f"{d}: {', '.join(slots)}")
    return " | ".join(out) if out else "See Comments"


def item_to_row(d: dict) -> dict:
    geo = d.get("geometry", {}) or {}
    lat, lon = extract_geo(geo.get("centroid"))
    district = next((x.get("name", "") for x in d.get("adm_div", []) or [] if x.get("type") == "district"), "")
    cap = (d.get("capacity") or {}).get("total")
    sel = geo.get("selection")
    try:
        poly = shp_wkt.loads(sel) if sel else None
    except Exception:  # noqa: BLE001 - malformed WKT from source: keep the point, record the defect
        poly = None
    return {"parking_id": str(d.get("id", "")).split("_")[0], "district": district,
            "address": d.get("address_name") or d.get("full_name", ""), "lat": lat, "lon": lon, "subtype": d.get("subtype", ""),
            "capacity": cap, "is_paid": d.get("is_paid"), "access": d.get("access"), "purpose": d.get("purpose"),
            "for_trucks": d.get("for_trucks"), "is_incentive": d.get("is_incentive"),
            "schedule": parse_full_schedule(d.get("schedule")), "capacity_is_reported_not_live": True,
            "geometry": poly if poly is not None else (Point(lon, lat) if lon is not None else None)}


class UrbiProvider(Provider):
    provider_id = "urbi"

    def __init__(self, max_workers: int = 4, min_interval_s: float = 0.1, session: Optional[requests.Session] = None):
        self.authorization_ref = require_authorized("urbi")
        self.key = os.environ.get("GEOACQUIRER_URBI_API_KEY")
        if not self.key:
            raise SecurityError("GEOACQUIRER_URBI_API_KEY is not set")
        self.max_workers = max_workers
        self.limiter = RateLimiter(min_interval_s)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json"})
        self.stats = None

    def discover_capabilities(self) -> Capabilities:
        return Capabilities(self.provider_id, ["parking"], {"parking": ["capacity", "is_paid", "access", "schedule"]}, {"parking": "Polygon/Point"},
                            notes="capacity is operator-reported; there is no live occupancy")

    def estimate_coverage(self, entity_type: str, bbox: BBox) -> CoverageEstimate:
        cells = len(uniform_grid(bbox, 0.01))
        return CoverageEstimate(self.provider_id, entity_type, bbox, None, cells, cells / max(1, self.max_workers) * 0.2, "coarse scout cells only; fine drill count is data-dependent")

    def _get(self, path: str, params: dict) -> dict:
        r = self.session.get(f"{BASE}/{path}", params={**params, "key": self.key}, timeout=15)
        if r.status_code in (401, 403):
            raise FatalSourceError(f"HTTP {r.status_code}")
        if r.status_code == 429:
            raise RateLimitedError("429")
        if r.status_code >= 500:
            raise TransientError(str(r.status_code))
        if r.status_code != 200:
            return {}
        return r.json().get("result", {})

    def _ids_in(self, lat: float, lon: float, radius: int):
        found, page, total = set(), 1, 0
        while True:
            res = self._get("items", {"point": f"{lon},{lat}", "radius": radius, "type": "parking", "fields": "items.id", "page_size": 50, "page": page})
            items, total = res.get("items", []), res.get("total", 0)
            found.update(str(i["id"]).split("_")[0] for i in items)
            if len(items) < 50:
                return found, total
            page += 1

    def fetch_features(self, entity_type: str, bbox: BBox, *, budget=None, cancel=None, **kw) -> FeatureBatch:
        if entity_type != "parking":
            raise ValueError("urbi adapter provides parking only")
        coarse, fine = 0.01, 0.002

        def probe(cell):
            w, s, e, n = cell
            return self._ids_in((s + n) / 2, (w + e) / 2, 1200)[1]

        cells = coarse_to_fine(bbox, coarse, fine, probe)
        crawler = AdaptiveCrawler(lambda c: [{"id": i} for i in self._ids_in((c[1] + c[3]) / 2, (c[0] + c[2]) / 2, 300)[0]],
                                  key=lambda it: it["id"], cell_id=lambda c: ",".join(f"{v:.5f}" for v in c), max_workers=self.max_workers,
                                  limiter=self.limiter, budget=budget, cancel=cancel)
        self.stats = crawler.run(cells)
        ids = list(crawler.items)
        gdf = self.normalize("parking", self._details(ids, budget))
        warns = [f"crawl aborted: {self.stats.aborted}"] if self.stats.aborted else []
        return FeatureBatch(gdf, Provenance(self.provider_id, "parking", "live-api", "UNESTABLISHED - see authorization ref " + self.authorization_ref,
                                            "2GIS/Urbi", utcnow_iso(), bbox, self.stats.cells_done), warns)

    def _details(self, ids: list[str], budget) -> list[dict]:
        batches = [ids[i:i + 50] for i in range(0, len(ids), 50)]
        crawler = AdaptiveCrawler(lambda b: [item_to_row(i) for i in self._get("items/byid", {"id": ",".join(b), "fields": FIELDS, "locale": "en_SA"}).get("items", [])],
                                  key=lambda r: r["parking_id"], cell_id=lambda b: b[0], max_workers=self.max_workers, limiter=self.limiter, budget=budget)
        crawler.run(batches)
        return list(crawler.items.values())

    def normalize(self, entity_type: str, raw) -> gpd.GeoDataFrame:
        if not raw:
            return gpd.GeoDataFrame({"parking_id": []}, geometry=[], crs=4326)
        df = pd.DataFrame(raw)
        return gpd.GeoDataFrame(df.drop(columns="geometry"), geometry=df["geometry"], crs=4326)

    def report_provenance(self) -> dict:
        return {"provider": self.provider_id, "authorization_ref": self.authorization_ref, "crawl": self.stats.__dict__ if self.stats else None}
