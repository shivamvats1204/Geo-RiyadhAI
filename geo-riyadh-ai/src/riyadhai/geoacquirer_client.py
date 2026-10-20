"""Client for the separate GeoAcquirer project (SDK in-process, or HTTP API). No acquisition logic is duplicated here."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import geopandas as gpd

BBox = tuple[float, float, float, float]
LAYERS = ["poi", "road", "connector", "building", "district"]


@dataclass
class AcquisitionResult:
    layers: dict[str, gpd.GeoDataFrame]
    provenance: dict[str, dict]
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    from_cache: dict[str, bool] = field(default_factory=dict)


class GeoAcquirerUnavailable(RuntimeError):
    pass


class GeoAcquirerClient:
    """mode='sdk' uses the installed `geoacquirer` package; mode='http' talks to `geoacquirer serve`."""

    def __init__(self, mode: str = "sdk", base_url: Optional[str] = None, cache_dir=None,
                 approve: Optional[Callable[[str], bool]] = None):
        self.mode = mode
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.approve = approve  # called before any live (non-cached) acquisition; return False to refuse

    # ---------------------------------------------------------------- availability
    def available(self) -> tuple[bool, str]:
        if self.mode == "sdk":
            try:
                import geoacquirer  # noqa: F401
                return True, f"geoacquirer {geoacquirer.__version__} (SDK)"
            except ImportError as e:
                return False, f"geoacquirer not installed: {e}"
        try:
            import requests
            r = requests.get(f"{self.base_url}/v1/health", timeout=5)
            return r.ok, f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def sources(self) -> dict:
        from geoacquirer.providers.registry import is_authorized, load_registry
        return {k: {"enabled": is_authorized(k, v)[0], "licence": v["allowed_usage"]} for k, v in load_registry().items()}

    # ---------------------------------------------------------------- acquisition
    def cached_layers(self, bbox: BBox, layers=LAYERS) -> dict[str, bool]:
        """Which layers are already cached (so no live acquisition is needed)."""
        from geoacquirer.providers.overture import OvertureProvider, SCHEMA_VERSION
        from geoacquirer.storage import LayerCache
        prov = OvertureProvider(cache=LayerCache(self.cache_dir) if self.cache_dir else None)
        rel = prov._release()
        out = {}
        w, s, e, n = (round(v, 6) for v in bbox)
        for l in layers:
            key = LayerCache.key(p="overture", r=rel, t=l, b=(w, s, e, n), v=SCHEMA_VERSION)
            out[l] = prov.cache.get(key, 365) is not None or prov.cache.find_covering(l, rel, (w, s, e, n), SCHEMA_VERSION) is not None
        return out

    def fetch_layers(self, bbox: BBox, layers=LAYERS, allow_live: bool = True) -> AcquisitionResult:
        ok, why = self.available()
        if not ok:
            raise GeoAcquirerUnavailable(why)
        if self.mode != "sdk":
            raise NotImplementedError("HTTP layer fetching is exposed through run_query(); use sdk mode for raw layers")
        from geoacquirer.providers.overture import OvertureProvider
        from geoacquirer.storage import LayerCache
        t0 = time.monotonic()
        cached = self.cached_layers(bbox, layers)
        need_live = [l for l, c in cached.items() if not c]
        if need_live:
            if not allow_live:
                raise GeoAcquirerUnavailable(f"layers not cached and live acquisition disabled: {need_live}")
            if self.approve is not None and not self.approve(f"live acquisition of {need_live} for bbox {bbox} (several minutes)"):
                raise PermissionError("live acquisition not approved")
        from concurrent.futures import ThreadPoolExecutor

        def one(l):
            p = OvertureProvider(cache=LayerCache(self.cache_dir) if self.cache_dir else None)
            return l, p.fetch_features(l, bbox)

        out, prov, warns, fc = {}, {}, [], {}
        with ThreadPoolExecutor(4) as ex:
            for l, fb in ex.map(one, layers):
                out[l] = fb.gdf
                prov[l] = fb.provenance.to_dict()
                warns += fb.warnings
                fc[l] = fb.provenance.from_cache
        return AcquisitionResult(out, prov, warns, time.monotonic() - t0, fc)

    def run_query(self, text: str, bbox: BBox, work_dir, include_statuses=None):
        """Natural-language retrieval through GeoAcquirer (compile, plan, evaluate, verify, export)."""
        from geoacquirer.agent import GeoAcquirerAgent
        from geoacquirer.storage import LayerCache
        agent = GeoAcquirerAgent(work_dir, cache=LayerCache(self.cache_dir) if self.cache_dir else None,
                                 approve=(lambda plan: True) if (self.approve is None or self.approve("GeoAcquirer query")) else None)
        return agent.run(text, area_bbox=bbox, include_statuses=include_statuses)
