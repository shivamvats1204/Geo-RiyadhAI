"""Topology-aware junction detection.

Core ideas (see docs/topology.md):

* A visual crossing is NOT a junction. Two roads are joined only if they share a connector
  (Overture's explicit topology). Bridges/tunnels therefore never create junctions;
  `find_grade_separated_crossings` reports the geometric crossings that were *excluded*.
* Divided carriageways produce several nearby 3-way nodes for what a human calls one
  crossroad. We merge connectors linked by short segments into one *cluster* and count
  *approaches* (arms merged by bearing and road name), not raw node degree.
* Ramps (`subclass == "link"`) never count as ordinary approaches. A cluster that mixes
  ramps with motorway/trunk arms is classified `highway_interchange`, not a crossroad.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.strtree import STRtree

HIGHWAY_CLASSES = frozenset({"motorway", "trunk"})
NON_DRIVABLE = frozenset({"footway", "steps", "path", "pedestrian", "cycleway", "bridleway", "track"})
MINOR_SERVICE = frozenset({"driveway", "parking_aisle", "alley", "drive_through"})


@dataclass
class JunctionConfig:
    merge_radius_m: float = 25.0  # segments up to this long link connectors into one cluster
    bearing_tol_deg: float = 25.0  # arms within this angle (and same name) are one approach
    arm_probe_m: float = 15.0  # distance along the segment used to measure the bearing
    exclude_minor_service: bool = True
    drivable_only: bool = True
    min_arm_angle_deg: float = 25.0  # approaches closer than this to another are merged regardless of name
    extra_excluded_classes: frozenset = field(default_factory=frozenset)


def _has_flag(flags, name: str) -> bool:
    if flags is None or (isinstance(flags, float) and np.isnan(flags)):
        return False
    try:
        for item in flags:
            vals = item.get("values") if isinstance(item, dict) else None
            if vals and name in vals:
                return True
    except TypeError:
        return False
    return False


def prepare_segments(segments: gpd.GeoDataFrame, cfg: Optional[JunctionConfig] = None) -> gpd.GeoDataFrame:
    """Keep drivable road segments and derive normalized flags. Input must be in a metric CRS."""
    cfg = cfg or JunctionConfig()
    seg = segments.copy()
    if "subtype" in seg:
        seg = seg[seg["subtype"].fillna("road") == "road"]
    seg["class"] = seg["class"].fillna("unknown")
    if cfg.drivable_only:
        seg = seg[~seg["class"].isin(NON_DRIVABLE | cfg.extra_excluded_classes)]
    if "subclass" not in seg:
        seg["subclass"] = None
    if cfg.exclude_minor_service:
        seg = seg[~((seg["class"] == "service") & seg["subclass"].isin(MINOR_SERVICE))]
    seg["is_link"] = seg["subclass"].fillna("").eq("link") | (
        seg["road_flags"].apply(lambda f: _has_flag(f, "is_link")) if "road_flags" in seg else False
    )
    seg["is_bridge"] = seg["road_flags"].apply(lambda f: _has_flag(f, "is_bridge")) if "road_flags" in seg else False
    seg["is_tunnel"] = seg["road_flags"].apply(lambda f: _has_flag(f, "is_tunnel")) if "road_flags" in seg else False
    if "names" in seg:
        seg["road_name"] = seg["names"].apply(
            lambda n: (n.get("primary") if isinstance(n, dict) else None)
        )
    elif "road_name" not in seg:
        seg["road_name"] = None
    seg["road_name"] = seg["road_name"].astype(object).where(seg["road_name"].notna(), None)
    seg = seg[seg.geometry.notna() & ~seg.geometry.is_empty & (seg.geometry.length > 0)]
    return seg.reset_index(drop=True)


def _line_parts(geom) -> LineString:
    if geom.geom_type == "LineString":
        return geom
    # MultiLineString: use the longest part (documented simplification)
    return max(geom.geoms, key=lambda g: g.length)


def _bearing(p0: Point, p1: Point) -> float:
    return (math.degrees(math.atan2(p1.x - p0.x, p1.y - p0.y)) + 360.0) % 360.0


def _angdiff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _arms_for_segment(row, probe_m: float):
    """Yield (connector_id, bearing_deg) for every arm a segment contributes."""
    line = _line_parts(row.geometry)
    length = line.length
    conns = row.connectors
    if conns is None or (isinstance(conns, float) and np.isnan(conns)):
        return
    for c in conns:
        cid = c["connector_id"]
        at = float(c["at"])
        pos = at * length
        eps = 1e-6
        if at <= eps:
            p0 = line.interpolate(0)
            p1 = line.interpolate(min(length, probe_m))
            yield cid, _bearing(p0, p1)
        elif at >= 1 - eps:
            p0 = line.interpolate(length)
            p1 = line.interpolate(max(0.0, length - probe_m))
            yield cid, _bearing(p0, p1)
        else:
            p0 = line.interpolate(pos)
            yield cid, _bearing(p0, line.interpolate(min(length, pos + probe_m)))
            yield cid, _bearing(p0, line.interpolate(max(0.0, pos - probe_m)))


def build_junctions(
    segments: gpd.GeoDataFrame,
    connectors: gpd.GeoDataFrame,
    cfg: Optional[JunctionConfig] = None,
) -> gpd.GeoDataFrame:
    """Return one row per junction cluster with its normalized kind.

    `segments` and `connectors` must share a metric CRS; segments need `connectors` (list of
    {connector_id, at}), `class`, `id`; connectors need `id` and point geometry.
    """
    cfg = cfg or JunctionConfig()
    seg = prepare_segments(segments, cfg)
    crs = segments.crs
    conn_xy = {r.id: (r.geometry.x, r.geometry.y) for r in connectors.itertuples() if r.geometry is not None}

    arms_at: dict[str, list[dict]] = defaultdict(list)
    seg_conns: dict[str, list[str]] = {}
    seg_len: dict[str, float] = {}
    for row in seg.itertuples():
        seg_len[row.id] = _line_parts(row.geometry).length
        seg_conns[row.id] = [c["connector_id"] for c in (row.connectors or [])]
        for cid, brg in _arms_for_segment(row, cfg.arm_probe_m):
            arms_at[cid].append(
                {
                    "segment_id": row.id,
                    "bearing": brg,
                    "cls": None,  # filled below (itertuples cannot expose the `class` column)
                    "name": row.road_name,
                    "is_link": bool(row.is_link),
                }
            )
    # itertuples renames `class`; fetch it explicitly
    cls_by_seg = dict(zip(seg["id"], seg["class"]))
    for cid, arms in arms_at.items():
        for a in arms:
            a["cls"] = cls_by_seg[a["segment_id"]]

    # union-find over connectors joined by a short segment (divided-carriageway merge)
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    multi = {cid for cid, arms in arms_at.items() if len(arms) >= 3}
    for sid, cids in seg_conns.items():
        if seg_len[sid] <= cfg.merge_radius_m:
            ends = [c for c in cids if c in multi]
            for i in range(len(ends) - 1):
                union(ends[i], ends[i + 1])
    for cid in multi:
        find(cid)

    clusters: dict[str, list[str]] = defaultdict(list)
    for cid in multi:
        clusters[find(cid)].append(cid)

    rows = []
    for root, members in clusters.items():
        mset = set(members)
        arms = []
        for cid in members:
            for a in arms_at[cid]:
                # an arm whose segment is short and ends inside the cluster is internal
                other_ends = [c for c in seg_conns[a["segment_id"]] if c != cid]
                internal = (
                    seg_len[a["segment_id"]] <= cfg.merge_radius_m
                    and other_ends
                    and all(o in mset for o in other_ends)
                )
                if not internal:
                    arms.append(a)
        link_arms = [a for a in arms if a["is_link"]]
        main_arms = [a for a in arms if not a["is_link"]]
        approaches = _merge_arms(main_arms, cfg)
        n_app = len(approaches)
        classes = [a["cls"] for a in arms]
        involves_hw = any(c in HIGHWAY_CLASSES for c in classes)
        has_motorway = any(c == "motorway" for c in classes)
        if has_motorway or (link_arms and involves_hw):
            kind = "highway_interchange"
        elif n_app >= 5:
            kind = "complex"
        elif n_app == 4:
            kind = "four_way"
        elif n_app == 3:
            kind = "t_junction"
        else:
            kind = "other"
        xs = [conn_xy[c] for c in members if c in conn_xy]
        if not xs:
            continue
        mp = MultiPoint([Point(x) for x in xs])
        names = sorted({a["name"] for a in arms if a["name"]})
        rows.append(
            {
                "junction_id": f"jn_{min(members)[-12:]}",
                "geometry": mp.centroid,
                "kind": kind,
                "n_approaches": n_app,
                "n_connectors": len(members),
                "has_ramp": bool(link_arms),
                "involves_highway": involves_hw,
                "max_class": _max_class(classes),
                "road_names": "; ".join(names),
                "approach_bearings": ",".join(f"{b:.0f}" for b in sorted(x["bearing"] for x in approaches)),
                "connector_ids": ",".join(sorted(members)),
            }
        )
    cols = ["junction_id", "kind", "n_approaches", "n_connectors", "has_ramp", "involves_highway", "max_class", "road_names", "approach_bearings", "connector_ids", "geometry"]
    if not rows:  # a network with no junction at all (a single road, a sparse edge area) must yield an empty, correctly-typed frame
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


_CLASS_RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified", "living_street", "service"]


def _max_class(classes: Iterable[str]) -> str:
    best = None
    for c in classes:
        r = _CLASS_RANK.index(c) if c in _CLASS_RANK else len(_CLASS_RANK)
        if best is None or r < best[0]:
            best = (r, c)
    return best[1] if best else "unknown"


def _merge_arms(arms: list[dict], cfg: JunctionConfig) -> list[dict]:
    """Merge parallel arms (dual carriageways) into approaches."""
    merged: list[dict] = []
    for a in sorted(arms, key=lambda x: x["bearing"]):
        for m in merged:
            d = _angdiff(a["bearing"], m["bearing"])
            same_name = (a["name"] is None) or (m["name"] is None) or (a["name"] == m["name"])
            if d <= cfg.min_arm_angle_deg or (d <= cfg.bearing_tol_deg and same_name):
                break
        else:
            merged.append(dict(a))
    return merged


def find_grade_separated_crossings(
    segments: gpd.GeoDataFrame, min_class_rank_gap: int = 0
) -> gpd.GeoDataFrame:
    """Geometric crossings between road segments that share *no* connector.

    These are bridges, tunnels, or data gaps; they are never junctions. Returned so the
    caller can report what was deliberately excluded.
    """
    seg = prepare_segments(segments, JunctionConfig(exclude_minor_service=True))
    geoms = list(seg.geometry)
    tree = STRtree(geoms)
    conn_sets = [set(c["connector_id"] for c in (cs or [])) for cs in seg["connectors"]]
    ids = seg["id"].tolist()
    bridge = (seg["is_bridge"] | seg["is_tunnel"]).tolist()
    rows = []
    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="crosses"):
            if j <= i:
                continue
            if conn_sets[i] & conn_sets[j]:
                continue
            inter = g.intersection(geoms[j])
            if inter.is_empty:
                continue
            pt = inter if inter.geom_type == "Point" else inter.centroid
            rows.append(
                {
                    "segment_a": ids[i],
                    "segment_b": ids[j],
                    "geometry": pt,
                    "flagged_bridge_or_tunnel": bool(bridge[i] or bridge[j]),
                }
            )
    cols = ["segment_a", "segment_b", "flagged_bridge_or_tunnel", "geometry"]
    return gpd.GeoDataFrame(rows, columns=cols, geometry="geometry", crs=segments.crs) if rows else gpd.GeoDataFrame({c: [] for c in cols}, geometry="geometry", crs=segments.crs)
