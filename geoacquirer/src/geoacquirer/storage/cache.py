"""On-disk cache of acquired layers (GeoParquet + JSON metadata) with freshness checks."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd


def default_cache_dir() -> Path:
    return Path(os.environ.get("GEOACQUIRER_CACHE", Path.home() / ".geoacquirer" / "cache"))


class LayerCache:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(**parts) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def _paths(self, key: str):
        return self.root / f"{key}.parquet", self.root / f"{key}.json"

    def get(self, key: str, max_age_days: Optional[float] = None):
        data, meta = self._paths(key)
        if not data.exists() or not meta.exists():
            return None
        m = json.loads(meta.read_text(encoding="utf-8"))
        if max_age_days is not None and (time.time() - m["saved_at"]) > max_age_days * 86400:
            return None
        return gpd.read_parquet(data), m

    def find_covering(self, entity: str, release: str, bbox, schema) -> Optional[tuple]:
        """A cached layer whose bbox fully contains `bbox` (same release/schema); returns (gdf clipped by bbox, meta)."""
        w, s, e, n = bbox
        best = None
        for mp in self.root.glob("*.json"):
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                continue
            b = m.get("bbox")
            if not b or m.get("entity") != entity or m.get("release") != release or m.get("schema") != schema:
                continue
            if b[0] <= w and b[1] <= s and b[2] >= e and b[3] >= n:
                area = (b[2] - b[0]) * (b[3] - b[1])
                if best is None or area < best[0]:
                    best = (area, mp.with_suffix(".parquet"), m)
        if not best:
            return None
        gdf = gpd.read_parquet(best[1])
        if (w, s, e, n) != tuple(best[2]["bbox"]):
            gdf = gdf[gdf.geometry.bounds.pipe(lambda b: (b.maxx >= w) & (b.minx <= e) & (b.maxy >= s) & (b.miny <= n))].reset_index(drop=True)
        return gdf, best[2]

    def put(self, key: str, gdf: gpd.GeoDataFrame, meta: dict) -> None:
        data, mp = self._paths(key)
        gdf.to_parquet(data)
        mp.write_text(json.dumps({**meta, "saved_at": time.time(), "rows": len(gdf)}, default=str), encoding="utf-8")
