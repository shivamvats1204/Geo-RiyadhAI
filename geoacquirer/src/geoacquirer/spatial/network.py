"""Road-network graph built from explicit connectors (no snapping of crossing lines).

Limitations (also surfaced in evidence): edges are treated as undirected because Overture's
direction/oneway information is incomplete, and travel *distance* (not time) is returned.
"""
from __future__ import annotations

from typing import Iterable, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from .topology import prepare_segments, JunctionConfig, _line_parts


class RoadNetwork:
    def __init__(self, segments: gpd.GeoDataFrame, connectors: gpd.GeoDataFrame, cfg: Optional[JunctionConfig] = None):
        seg = prepare_segments(segments, cfg or JunctionConfig(exclude_minor_service=False))
        self.crs = segments.crs
        self.segments = seg
        self.connectors = connectors.set_index("id")
        self.graph = nx.Graph()
        self._seg_nodes: dict[str, list[str]] = {}
        for row in seg.itertuples():
            line = _line_parts(row.geometry)
            conns = sorted(row.connectors or [], key=lambda c: c["at"])
            ids = [c["connector_id"] for c in conns]
            self._seg_nodes[row.id] = ids
            for a, b in zip(conns, conns[1:]):
                w = (b["at"] - a["at"]) * line.length
                u, v = a["connector_id"], b["connector_id"]
                if u == v or w <= 0:
                    continue
                if self.graph.has_edge(u, v):
                    if w < self.graph[u][v]["weight"]:
                        self.graph[u][v]["weight"] = w
                else:
                    self.graph.add_edge(u, v, weight=w, segment_id=row.id)
        for n in self.graph.nodes:
            if n in self.connectors.index:
                g = self.connectors.loc[n, "geometry"]
                self.graph.nodes[n]["x"], self.graph.nodes[n]["y"] = g.x, g.y
        self._node_ids = [n for n in self.graph.nodes if "x" in self.graph.nodes[n]]
        self._node_points = gpd.GeoSeries([Point(self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in self._node_ids], crs=self.crs)
        self._tree = self._node_points.sindex if len(self._node_ids) else None
        self._comp: Optional[dict] = None

    # -- snapping ---------------------------------------------------------
    def snap(self, geoms: gpd.GeoSeries, max_snap_m: float = 150.0):
        """Nearest graph node per geometry -> (node ids or None, snap distances)."""
        n = len(geoms)
        ids = [None] * n
        dist = np.full(n, np.inf)
        if self._tree is None or n == 0:
            return ids, dist
        pairs, d = self._tree.nearest(geoms.values, return_all=False, max_distance=max_snap_m, return_distance=True)
        for i, j, dd in zip(pairs[0], pairs[1], d):
            ids[i] = self._node_ids[j]
            dist[i] = dd
        return ids, dist

    def snap_edges(self, geoms: gpd.GeoSeries, max_snap_m: float = 150.0) -> list:
        """Snap each geometry to its nearest road *segment* (not just the nearest node).

        Returns per geometry None or {segment_id, snap_m, along_m, candidates:[(node, offset_m), ...]} where the
        two candidates are the connectors bracketing the snap point, each with perpendicular + along-segment offset.
        """
        out: list = [None] * len(geoms)
        seg = self.segments
        if len(seg) == 0 or len(geoms) == 0:
            return out
        pairs, dist = seg.sindex.nearest(geoms.values, return_all=False, max_distance=max_snap_m, return_distance=True)
        for gi, si, d in zip(pairs[0], pairs[1], dist):
            row = seg.iloc[si]
            line = _line_parts(row.geometry)
            p = geoms.iloc[gi]
            p = p if p.geom_type == "Point" else p.representative_point()
            along = float(line.project(p))
            conns = sorted(row.connectors or [], key=lambda c: c["at"])
            if len(conns) < 2:
                continue
            f = along / line.length if line.length else 0.0
            lo = conns[0]
            hi = conns[-1]
            for a, b in zip(conns, conns[1:]):
                if a["at"] <= f <= b["at"]:
                    lo, hi = a, b
                    break
            cands = [(lo["connector_id"], float(d) + max(0.0, along - lo["at"] * line.length)),
                     (hi["connector_id"], float(d) + max(0.0, hi["at"] * line.length - along))]
            out[gi] = {"segment_id": row.id, "snap_m": float(d), "along_m": along, "candidates": [c for c in cands if c[0] in self.graph]}
        return out

    # -- components ------------------------------------------------------
    def component_of(self, node: str) -> Optional[int]:
        if self._comp is None:
            self._comp = {}
            for k, comp in enumerate(sorted(nx.connected_components(self.graph), key=len, reverse=True)):
                for n in comp:
                    self._comp[n] = k
        return self._comp.get(node)

    def component_sizes(self) -> list[int]:
        return sorted((len(c) for c in nx.connected_components(self.graph)), reverse=True)

    # -- shortest paths --------------------------------------------------
    def distances_from(self, sources: Iterable[str], cutoff: float) -> dict[str, float]:
        srcs = [s for s in sources if s in self.graph]
        if not srcs:
            return {}
        return nx.multi_source_dijkstra_path_length(self.graph, srcs, cutoff=cutoff, weight="weight")

    def nodes_of_segments(self, segment_ids: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for s in segment_ids:
            out.update(self._seg_nodes.get(s, []))
        return out
