"""Explainable spatial knowledge graph stored as relational edge tables (SQLite), per the project brief.

Edges carry the relation, the calculation parameters and provenance, so every traversal can be explained.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

import geopandas as gpd

RELATIONS = ["LOCATED_IN", "NEAR", "AT_JUNCTION", "HAS_FRONTAGE_ON", "BELONGS_TO", "OVERLAPS", "CONNECTED_TO"]


class SpatialKG:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS node(id TEXT PRIMARY KEY, type TEXT, name TEXT, attrs TEXT);
            CREATE TABLE IF NOT EXISTS edge(src TEXT, rel TEXT, dst TEXT, params TEXT, provenance TEXT);
            CREATE INDEX IF NOT EXISTS ix_edge_src ON edge(src, rel);
            CREATE INDEX IF NOT EXISTS ix_edge_dst ON edge(dst, rel);
        """)

    def add_node(self, id: str, type: str, name: Optional[str] = None, **attrs) -> None:
        self.db.execute("INSERT OR REPLACE INTO node VALUES (?,?,?,?)", (id, type, name, json.dumps(attrs, default=str)))

    def add_edge(self, src: str, rel: str, dst: str, params: Optional[dict] = None, provenance: Optional[dict] = None) -> None:
        self.db.execute("INSERT INTO edge VALUES (?,?,?,?,?)", (src, rel, dst, json.dumps(params or {}), json.dumps(provenance or {})))

    def neighbors(self, node: str, rel: Optional[str] = None) -> list[dict]:
        q = "SELECT src, rel, dst, params, provenance FROM edge WHERE (src=? OR dst=?)" + (" AND rel=?" if rel else "") + " ORDER BY rowid"
        args = (node, node) + ((rel,) if rel else ())
        return [dict(zip(("src", "rel", "dst", "params", "provenance"), r[:3] + (json.loads(r[3]), json.loads(r[4])))) for r in self.db.execute(q, args)]

    def explain(self, node: str) -> list[str]:
        out = []
        for e in self.neighbors(node):
            p = ", ".join(f"{k}={v}" for k, v in e["params"].items())
            out.append(f"{e['src']} -[{e['rel']}{(' ' + p) if p else ''}]-> {e['dst']}")
        return out

    def find(self, rel: str, node_type: Optional[str] = None) -> list[str]:
        q = "SELECT DISTINCT e.src FROM edge e JOIN node n ON n.id=e.src WHERE e.rel=?" + (" AND n.type=?" if node_type else "")
        return [r[0] for r in self.db.execute(q, (rel,) + ((node_type,) if node_type else ()))]


def build_from_results(flat: gpd.GeoDataFrame, provenance: Optional[dict] = None) -> SpatialKG:
    """Edges from a flattened result frame: shop LOCATED_IN building, shop AT_JUNCTION junction, shop HAS_FRONTAGE_ON road."""
    kg = SpatialKG()
    prov = provenance or {}
    for r in flat.itertuples():
        kg.add_node(str(r.id), "poi", r.name, status=r.status, category=r.category)
        if r.building_id:
            kg.add_node(str(r.building_id), "building")
            kg.add_edge(str(r.id), "LOCATED_IN", str(r.building_id), {"match": r.building_match, "distance_m": r.building_distance_m}, prov)
        if r.nearest_junction_id:
            kg.add_node(str(r.nearest_junction_id), "junction")
            kg.add_edge(str(r.id), "AT_JUNCTION", str(r.nearest_junction_id), {"distance_m": r.nearest_junction_distance_m}, prov)
        for part in str(r.nearby_roads or "").split("; "):
            if part:
                kg.add_node(part.split(" [")[0], "road")
                kg.add_edge(str(r.id), "HAS_FRONTAGE_ON", part.split(" [")[0], {"detail": part}, prov)
    kg.db.commit()
    return kg
