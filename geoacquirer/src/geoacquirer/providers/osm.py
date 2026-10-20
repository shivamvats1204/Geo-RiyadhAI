"""OpenStreetMap adapter (Overpass API) producing the same normalized schema as Overture.

Ways are split at nodes shared by two or more highways, so connectors come from genuine shared
nodes (a bridge crossing another road shares no node and therefore creates no junction).
Public Overpass servers are often busy: the adapter tries mirrors with bounded retries and reports failure explicitly.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Optional

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, Polygon

from ..security import check_domain
from ..storage import LayerCache
from .base import BBox, Capabilities, CoverageEstimate, FeatureBatch, Provenance, Provider, utcnow_iso

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]
UA = "geoacquirer/0.1 (research prototype)"
SCHEMA_VERSION = 1

_AMENITY_TOKENS = {
    "restaurant": ["food_and_drink", "restaurant"], "cafe": ["food_and_drink", "coffee_shop"], "fast_food": ["food_and_drink", "restaurant"],
    "pharmacy": ["health_care", "pharmacy"], "clinic": ["health_care", "clinic"], "hospital": ["health_care", "hospital"],
    "doctors": ["health_care", "doctor"], "bank": ["services_and_business", "bank"], "school": ["education", "school"],
    "place_of_worship": ["community_and_government", "place_of_worship"], "parking": ["travel_and_transportation", "parking"],
    "fuel": ["travel_and_transportation", "gas_station"],
}
_HW = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "unclassified", "living_street",
       "footway", "pedestrian", "steps", "path", "cycleway", "track"}


def _hw_class(tags: dict) -> tuple[str, Optional[str]]:
    h = tags.get("highway", "")
    if h.endswith("_link"):
        return h[:-5], "link"
    return (h if h in _HW else "unknown"), None


def build_query(bbox: BBox) -> str:
    w, s, e, n = bbox
    b = f"({s},{w},{n},{e})"
    return (f"[out:json][timeout:120];(way[\"highway\"]{b};node[\"shop\"]{b};node[\"amenity\"]{b};way[\"shop\"]{b};);out geom;")


def parse_overpass(js: dict) -> dict[str, gpd.GeoDataFrame]:
    ways = [el for el in js.get("elements", []) if el["type"] == "way" and "highway" in el.get("tags", {}) and "geometry" in el]
    usage = Counter(nid for w in ways for nid in w["nodes"])
    seg_rows, conn_pts = [], {}
    for w in ways:
        nodes, geom = w["nodes"], w["geometry"]
        cut = [0] + [i for i in range(1, len(nodes) - 1) if usage[nodes[i]] >= 2] + [len(nodes) - 1]
        cls, sub = _hw_class(w["tags"])
        flags = []
        if w["tags"].get("bridge") not in (None, "no"):
            flags.append({"values": ["is_bridge"], "between": None})
        if w["tags"].get("tunnel") not in (None, "no"):
            flags.append({"values": ["is_tunnel"], "between": None})
        if sub == "link":
            flags.append({"values": ["is_link"], "between": None})
        for k, (a, b) in enumerate(zip(cut, cut[1:])):
            pts = [(g["lon"], g["lat"]) for g in geom[a:b + 1]]
            if len(pts) < 2:
                continue
            for idx in (a, b):
                conn_pts[nodes[idx]] = Point(geom[idx]["lon"], geom[idx]["lat"])
            seg_rows.append({"road_id": f"w{w['id']}_{k}", "class": cls, "subclass": sub, "subtype": "road", "road_name": w["tags"].get("name"),
                             "connectors_json": json.dumps([{"connector_id": f"n{nodes[a]}", "at": 0.0}, {"connector_id": f"n{nodes[b]}", "at": 1.0}]),
                             "road_flags_json": json.dumps(flags) if flags else None, "geometry": LineString(pts)})
    pois = []
    for el in js.get("elements", []):
        t = el.get("tags", {})
        if "highway" in t and el["type"] == "way":
            continue
        if "shop" in t:
            path = ["shopping", t["shop"]]
        elif t.get("amenity") in _AMENITY_TOKENS:
            path = _AMENITY_TOKENS[t["amenity"]]
        else:
            continue
        if el["type"] == "node":
            geom = Point(el["lon"], el["lat"])
        elif "geometry" in el:
            geom = Polygon([(g["lon"], g["lat"]) for g in el["geometry"]]).centroid if len(el["geometry"]) > 3 else None
        else:
            geom = None
        if geom is None:
            continue
        pois.append({"poi_id": f"osm_{el['type'][0]}{el['id']}", "name": t.get("name"), "category": path[-1], "category_path": path,
                     "basic_category": None, "confidence": None, "operating_status": None, "address": t.get("addr:street"),
                     "source_dataset": "OpenStreetMap", "geometry": geom})
    return {
        "road": gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=4326) if seg_rows else gpd.GeoDataFrame(geometry=[], crs=4326),
        "connector": gpd.GeoDataFrame([{"connector_id": f"n{k}", "geometry": v} for k, v in conn_pts.items()], geometry="geometry", crs=4326) if conn_pts else gpd.GeoDataFrame(geometry=[], crs=4326),
        "poi": gpd.GeoDataFrame(pois, geometry="geometry", crs=4326) if pois else gpd.GeoDataFrame(geometry=[], crs=4326),
    }


class OsmOverpassProvider(Provider):
    provider_id = "osm_overpass"

    def __init__(self, cache: Optional[LayerCache] = None, mirrors=None, max_attempts: int = 3, session=None, sleep=time.sleep):
        self.cache = cache or LayerCache()
        self.mirrors = mirrors or MIRRORS
        self.max_attempts = max_attempts
        self.session = session or requests.Session()
        self.sleep = sleep
        self.requests_made = 0

    def discover_capabilities(self) -> Capabilities:
        return Capabilities(self.provider_id, ["poi", "road", "connector"], {"poi": ["name", "category_path"], "road": ["class", "road_name"]},
                            {"poi": "Point", "road": "LineString", "connector": "Point"}, notes="live OSM; servers may be busy")

    def estimate_coverage(self, entity_type: str, bbox: BBox) -> CoverageEstimate:
        return CoverageEstimate(self.provider_id, entity_type, bbox, None, 1, 30.0, "single Overpass query; size unknown a priori")

    def _query(self, q: str, budget=None) -> dict:
        last = None
        for attempt in range(self.max_attempts):
            for url in self.mirrors:
                check_domain(url)
                try:
                    if budget:
                        budget.charge_request()
                    self.requests_made += 1
                    r = self.session.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=150)
                    if r.status_code == 200:
                        return r.json()
                    last = f"{url}: HTTP {r.status_code}"
                except requests.RequestException as e:
                    last = f"{url}: {type(e).__name__}"
            self.sleep(min(30, 2 ** attempt * 2))
        raise ConnectionError(f"all Overpass mirrors failed ({last})")

    def fetch_features(self, entity_type: str, bbox: BBox, *, budget=None, use_cache=True, **kw) -> FeatureBatch:
        if entity_type not in ("poi", "road", "connector"):
            raise ValueError(entity_type)
        key = LayerCache.key(p="osm", b=tuple(round(v, 5) for v in bbox), v=SCHEMA_VERSION, t=entity_type)
        if use_cache:
            hit = self.cache.get(key, 7)
            if hit:
                return FeatureBatch(hit[0], Provenance(self.provider_id, entity_type, hit[1].get("retrieved_at", ""), "ODbL 1.0", "© OpenStreetMap contributors",
                                                       hit[1].get("retrieved_at", ""), bbox, 0, True, key))
        layers = parse_overpass(self._query(build_query(bbox), budget))
        now = utcnow_iso()
        for et, g in layers.items():
            self.cache.put(LayerCache.key(p="osm", b=tuple(round(v, 5) for v in bbox), v=SCHEMA_VERSION, t=et), g, {"retrieved_at": now})
        return FeatureBatch(layers[entity_type], Provenance(self.provider_id, entity_type, now, "ODbL 1.0", "© OpenStreetMap contributors", now, bbox, 1, False, key))

    def normalize(self, entity_type: str, raw):
        return parse_overpass(raw)[entity_type]

    def report_provenance(self) -> dict:
        return {"provider": self.provider_id, "requests_made": self.requests_made}
