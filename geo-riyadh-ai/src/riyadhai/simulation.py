"""What-if scenarios. Every result is labelled `simulated: True`; nothing here touches the canonical data."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point

from .store import AreaDataset

OPS = {"add_facility", "remove_facility", "remove_connector", "remove_segment", "set_capacity"}


@dataclass
class Scenario:
    name: str
    operations: list[dict]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    simulated: bool = True
    created: float = field(default_factory=time.time)

    def validate(self) -> None:
        for op in self.operations:
            if op.get("type") not in OPS:
                raise ValueError(f"unsupported operation {op.get('type')!r}; supported: {sorted(OPS)}")
            if op["type"] == "add_facility" and not (("lon" in op and "lat" in op) or ("x" in op and "y" in op)):
                raise ValueError("add_facility needs lon/lat or x/y")
            if op["type"] in ("remove_connector", "remove_segment") and not (op.get("connector_id") or op.get("segment_id")):
                raise ValueError("remove_* needs connector_id or segment_id")

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "operations": self.operations, "simulated": True, "created": self.created}


def _metrics(ds: AreaDataset, G: nx.Graph, demand_nodes: dict, facilities_xy: list, radius_m: float, cap_m: float, snap_nodes) -> dict:
    SRC = "__fac__"
    for n, off in snap_nodes:
        if n is not None and n in G:
            if G.has_edge(SRC, n):
                G[SRC][n]["weight"] = min(G[SRC][n]["weight"], off)
            else:
                G.add_edge(SRC, n, weight=off)
    try:
        dist = nx.single_source_dijkstra_path_length(G, SRC, cutoff=cap_m, weight="weight") if SRC in G else {}
    finally:
        if SRC in G:
            G.remove_node(SRC)
    tot = sum(w for w, _ in demand_nodes.values())
    cost = 0.0
    cov = 0.0
    unreached = 0.0
    for n, (w, off) in demand_nodes.items():
        d = dist.get(n)
        if d is None:
            cost += w * cap_m
            unreached += w
        else:
            c = min(d + off, cap_m)
            cost += w * c
            if c <= radius_m:
                cov += w
    return {"facilities": len(facilities_xy), "mean_distance_m": round(cost / tot, 1), "coverage_within_radius": round(cov / tot, 4),
            "demand_weight": round(tot, 1), "share_unreached_within_cap": round(unreached / tot, 4),
            "components": nx.number_connected_components(G)}


def run_scenario(ds: AreaDataset, scenario: Scenario, demand: gpd.GeoDataFrame, weights: Optional[np.ndarray] = None,
                 radius_m: float = 600.0, cap_m: float = 2000.0, facilities: Optional[gpd.GeoDataFrame] = None) -> dict:
    """Compare baseline vs scenario accessibility (nearest facility by road distance)."""
    scenario.validate()
    net = ds.network()
    to_m = Transformer.from_crs(4326, ds.crs, always_xy=True)
    G0 = net.graph
    G1 = G0.copy()
    base_fac = facilities if facilities is not None else ds.layers.get("parking")
    fac_pts = [(c.x, c.y, "existing") for c in base_fac.geometry.centroid] if base_fac is not None else []  # polygons -> centroids
    existing_pts = list(fac_pts)  # the unmodified baseline world
    removed_edges = []
    applied = []
    for op in scenario.operations:
        t = op["type"]
        if t == "add_facility":
            x, y = to_m.transform(op["lon"], op["lat"]) if "lon" in op else (op["x"], op["y"])
            fac_pts.append((x, y, "added"))
            applied.append(f"added facility at ({x:.0f}, {y:.0f})")
        elif t == "remove_facility":
            i = op.get("index")
            if i is None or not (0 <= i < len(fac_pts)):
                raise ValueError("remove_facility needs a valid index")
            fac_pts.pop(i)
            applied.append(f"removed facility #{i}")
        elif t == "remove_connector":
            cid = op["connector_id"]
            if cid not in G1:
                raise ValueError(f"connector {cid} is not in the road graph")
            removed_edges += [(cid, nb) for nb in list(G1.neighbors(cid))]
            G1.remove_node(cid)
            applied.append(f"closed connector {cid} (all its road connections)")
        elif t == "remove_segment":
            sid = op["segment_id"]
            hit = [(u, v) for u, v, d in G1.edges(data=True) if d.get("segment_id") == sid]
            if not hit:
                raise ValueError(f"segment {sid} has no edges in the graph")
            G1.remove_edges_from(hit)
            removed_edges += hit
            applied.append(f"closed road segment {sid} ({len(hit)} edge(s))")
        elif t == "set_capacity":
            applied.append("set_capacity noted (capacity is not part of the distance model; no effect on these metrics)")
    # demand aggregated per node
    dn, ds_off = net.snap(demand.geometry.centroid, 200.0)
    w = np.ones(len(demand)) if weights is None else np.asarray(weights, float)
    dem: dict[str, list] = {}
    for n, off, wi in zip(dn, ds_off, w):
        if n is not None:
            a = dem.setdefault(n, [0.0, 0.0]); a[0] += wi; a[1] += wi * off
    dem = {n: (a[0], a[1] / a[0]) for n, a in dem.items() if a[0] > 0}
    if not dem:  # no demand reaches the road graph (empty/sparse area, or every point > 200 m from a road): accessibility is undefined
        raise ValueError("no demand points could be snapped to the road network, so accessibility before/after is undefined for this area")

    def snap_fac(pts):
        gs = gpd.GeoSeries([Point(x, y) for x, y, _ in pts], crs=ds.crs)
        nodes, off = net.snap(gs, 200.0)
        return list(zip(nodes, off))

    base_pts = existing_pts
    # baseline = the unmodified world (existing facilities only, original graph)
    before = _metrics(ds, G0.copy(), dem, base_pts, radius_m, cap_m, snap_fac(base_pts) if base_pts else [])
    scen_pts = [p for p in fac_pts]
    after = _metrics(ds, G1, dem, scen_pts, radius_m, cap_m, [s for s in snap_fac(scen_pts) if s[0] in G1] if scen_pts else [])
    delta = {k: round(after[k] - before[k], 4) for k in ("mean_distance_m", "coverage_within_radius", "share_unreached_within_cap", "components")}
    return {"scenario": scenario.to_dict(), "simulated": True, "applied": applied, "before": before, "after": after, "delta": delta,
            "assumptions": ["road-network distance (undirected); closing a connector removes all its connections", "demand weights are caller-supplied proxies"]}


def compare_scenarios(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        rows.append({"scenario": r["scenario"]["name"], "id": r["scenario"]["id"], "simulated": True, **{f"after_{k}": v for k, v in r["after"].items() if k in ("mean_distance_m", "coverage_within_radius", "facilities")},
                     **{f"delta_{k}": v for k, v in r["delta"].items() if k in ("mean_distance_m", "coverage_within_radius")}})
    return sorted(rows, key=lambda x: x["delta_mean_distance_m"])


def save_scenario(root: Path, result: dict) -> Path:
    d = Path(root) / "analyses" / "scenarios"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{result['scenario']['id']}.json"
    p.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return p
