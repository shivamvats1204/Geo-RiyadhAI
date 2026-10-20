"""Versioned result sets (GeoParquet + JSON metadata) with lineage, used for follow-up questions and exports."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd


@dataclass
class ResultSet:
    id: str
    gdf: gpd.GeoDataFrame            # WGS84
    meta: dict = field(default_factory=dict)


class ResultStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "analyses" / "results"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, ResultSet] = {}

    def put(self, gdf: gpd.GeoDataFrame, *, kind: str, producer: str, parent: Optional[str] = None, params: Optional[dict] = None,
            notes: Optional[list] = None) -> ResultSet:
        rid = uuid.uuid4().hex[:10]
        version = 1 + (self._mem[parent].meta.get("version", 0) if parent and parent in self._mem else 0)
        meta = {"id": rid, "kind": kind, "producer": producer, "parent": parent, "version": version, "params": params or {}, "notes": notes or [],
                "rows": int(len(gdf)), "created": time.time()}
        g = gdf.to_crs(4326) if gdf.crs else gdf
        rs = ResultSet(rid, g, meta)
        self._mem[rid] = rs
        try:
            g.to_parquet(self.dir / f"{rid}.parquet")
        except Exception:  # noqa: BLE001 - list/dict columns that parquet cannot hold: keep GeoJSON instead
            g.drop(columns=[c for c in g.columns if g[c].map(lambda v: isinstance(v, (list, dict))).any()]).to_parquet(self.dir / f"{rid}.parquet")
        (self.dir / f"{rid}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return rs

    def get(self, rid: str) -> ResultSet:
        if rid in self._mem:
            return self._mem[rid]
        p = self.dir / f"{rid}.parquet"
        if not p.exists():
            raise KeyError(f"unknown result set {rid!r}")
        rs = ResultSet(rid, gpd.read_parquet(p), json.loads((self.dir / f"{rid}.json").read_text()))
        self._mem[rid] = rs
        return rs

    def lineage(self, rid: str) -> list[dict]:
        out, cur = [], rid
        while cur:
            m = self.get(cur).meta
            out.append({k: m[k] for k in ("id", "kind", "producer", "version", "rows", "params")})
            cur = m.get("parent")
        return out
