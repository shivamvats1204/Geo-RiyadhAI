"""Deterministic GeoQuery executor.

Every leaf predicate returns one of five codes per subject row. Boolean logic (AND/OR/NOT)
is evaluated over those codes, so "uncertain" and "unverifiable" never silently become
true or false. No LLM output is executed: this module only interprets validated IR.
"""
from __future__ import annotations

import json
from enum import IntEnum
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ..ontology import CONCEPTS, category_mask, resolve
from ..query_ir import And, EntityType, GeoQuery, Not, ObjectSpec, Op, Or, Predicate
from .context import DataContext


class Code(IntEnum):
    FAIL = 0
    FAIL_NEAR = 1
    UNKNOWN = 2
    PASS_NEAR = 3
    PASS = 4


class Status:
    VERIFIED = "VERIFIED_MATCH"
    PROBABLE = "PROBABLE_MATCH"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"


_AND_RANK = np.array([0, 2, 1, 3, 4])  # self-inverse: FAIL < UNKNOWN < FAIL_NEAR < PASS_NEAR < PASS
_STATUS = {
    Code.PASS: Status.VERIFIED, Code.PASS_NEAR: Status.PROBABLE, Code.FAIL: Status.REJECTED,
    Code.FAIL_NEAR: Status.INSUFFICIENT, Code.UNKNOWN: Status.INSUFFICIENT,
}
_CODE_NAME = {c.value: c.name for c in Code}


def and_combine(arrs: list[np.ndarray]) -> np.ndarray:
    ranks = np.min(np.stack([_AND_RANK[a] for a in arrs]), axis=0)
    return _AND_RANK[ranks]


def or_combine(arrs: list[np.ndarray]) -> np.ndarray:
    return np.max(np.stack(arrs), axis=0)


def not_codes(a: np.ndarray) -> np.ndarray:
    return 4 - a


def near_codes(d: np.ndarray, threshold: float, u: np.ndarray) -> np.ndarray:
    """Codes for 'distance <= threshold' given positional uncertainty u (metres)."""
    codes = np.full(len(d), Code.FAIL, dtype=np.int8)
    codes[(d > threshold) & (d <= threshold + u)] = Code.FAIL_NEAR
    codes[(d <= threshold) & (d > threshold - u)] = Code.PASS_NEAR
    codes[d <= threshold - u] = Code.PASS
    return codes


class Evaluator:
    def __init__(self, ctx: DataContext, query: GeoQuery):
        self.ctx = ctx
        self.q = query
        self.leaf_n = 0
        self.leaf_info: list[dict] = []
        self.subject = self._subject_frame()
        n = len(self.subject)
        self.n = n
        is_pt = (self.subject.geometry.geom_type == "Point").to_numpy() if n else np.zeros(0, bool)
        self.u = np.where(is_pt, query.position_uncertainty_m, 0.0)
        self.evidence: list[list[dict]] = [[] for _ in range(n)]
        self.extra: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ subject
    def _subject_frame(self) -> gpd.GeoDataFrame:
        t = self.q.primary()
        layer_name = {EntityType.POI: "poi", EntityType.PARKING: "parking", EntityType.BUILDING: "building",
                      EntityType.PARCEL: "parcel"}.get(t.entity_type)
        base = self._layer_for(layer_name, t.entity_type, t.categories)
        if base is None:
            raise LookupError(f"layer for target {t.entity_type.value} not available")
        if t.entity_type == EntityType.PARCEL and t.categories and "landuse" in base.columns:
            base = base[base["landuse"].astype(str).str.lower().isin([c.lower() for c in t.categories])]
        elif t.categories or t.exclude_categories:
            ids = self._concept_ids(t.categories)
            ex = self._concept_ids(t.exclude_categories)
            if "category_path" in base and ids:
                base = base[category_mask(base["category_path"], ids, ex)]
        if self.ctx.area is not None and len(base):
            base = base[base.geometry.intersects(self.ctx.area)]
        return base.reset_index(drop=True).copy()

    def _concept_ids(self, terms: list[str]) -> list[str]:
        out: list[str] = []
        for t in terms:
            if t in CONCEPTS:
                out.append(t)
            else:
                out.extend(c.id for c in resolve(t)[:1])
        return out

    def _layer_for(self, layer_name, etype: EntityType, cats: list[str]) -> Optional[gpd.GeoDataFrame]:
        if etype == EntityType.PARKING:
            if self.ctx.has("parking"):
                return self.ctx.layers["parking"]
            poi = self.ctx.layers.get("poi")
            if poi is None:
                return None
            return poi[category_mask(poi["category_path"], ["parking"])]
        return self.ctx.layers.get(layer_name) if layer_name else None

    # ----------------------------------------------------------- objects
    def _object_gdf(self, spec: ObjectSpec, pred: Predicate) -> Optional[gpd.GeoDataFrame]:
        et = spec.entity_type
        if et == EntityType.ROAD:
            g = self.ctx.layers.get("road")
            if g is not None and spec.road_classes:
                g = g[g["class"].isin(spec.road_classes)]
            return g
        if et == EntityType.INTERSECTION:
            return self.ctx.layers.get("junction")
        if et == EntityType.POI:
            g = self.ctx.layers.get("poi")
            if g is not None and spec.categories:
                g = g[category_mask(g["category_path"], self._concept_ids(spec.categories))]
            return g
        if et == EntityType.PARKING:
            return self._layer_for("parking", et, [])
        if et == EntityType.BUILDING:
            return self.ctx.layers.get("building")
        if et == EntityType.PARCEL:
            g = self.ctx.layers.get("parcel")
            if g is not None and spec.categories and "landuse" in g.columns:
                g = g[g["landuse"].astype(str).str.lower().isin([c.lower() for c in spec.categories])]
            return g
        return None

    # --------------------------------------------------------- recursion
    def evaluate(self, expr) -> np.ndarray:
        if expr is None:
            return np.full(self.n, Code.PASS, dtype=np.int8)
        if isinstance(expr, And):
            return and_combine([self.evaluate(c) for c in expr.children])
        if isinstance(expr, Or):
            return or_combine([self.evaluate(c) for c in expr.children])
        if isinstance(expr, Not):
            return not_codes(self.evaluate(expr.child))
        return self._leaf(expr)

    def run(self) -> gpd.GeoDataFrame:
        codes = self.evaluate(self.q.constraint)
        out = self.subject.copy()
        out["status"] = [_STATUS[Code(int(c))] for c in codes]
        out["result_code"] = [_CODE_NAME[int(c)] for c in codes]
        out["evidence_json"] = [json.dumps(ev, default=_json_default) for ev in self.evidence]
        for k, v in self.extra.items():
            out[k] = v
        return out

    # -------------------------------------------------------------- leaf
    def _leaf(self, p: Predicate) -> np.ndarray:
        self.leaf_n += 1
        lid = f"c{self.leaf_n}"
        label = _label(p)
        self.leaf_info.append({"id": lid, "label": label, "op": p.op.value, "exactness": p.exactness.value})
        if self.n == 0:
            return np.zeros(0, dtype=np.int8)
        try:
            handler = _HANDLERS[p.op]
            codes, ev = handler(self, p)
        except _Unavailable as e:
            codes = np.full(self.n, Code.UNKNOWN, dtype=np.int8)
            ev = [{"reason": str(e)}] * self.n
        for i in range(self.n):
            rec = {"id": lid, "predicate": label, "result": _CODE_NAME[int(codes[i])], "exactness": p.exactness.value}
            rec.update(ev[i] if ev[i] else {})
            self.evidence[i].append(rec)
        return codes


class _Unavailable(Exception):
    pass


def _label(p: Predicate) -> str:
    obj = ""
    if p.object is not None:
        bits = p.object.road_classes or p.object.categories
        obj = f"{p.object.entity_type.value}{'[' + ','.join(bits) + ']' if bits else ''}"
    dist = f"<={p.distance_m:g}m" if p.distance_m else ""
    jk = f" {p.junction_kind}" if p.junction_kind else ""
    return f"{p.op.value}({obj}{jk}{dist})".replace("()", "")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if np.isinf(o) or np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def _need(ev: Evaluator, gdf: Optional[gpd.GeoDataFrame], what: str) -> gpd.GeoDataFrame:
    if gdf is None:
        raise _Unavailable(f"{what} layer was not acquired; constraint cannot be verified")
    return gdf


def nearest(ev: Evaluator, obj: gpd.GeoDataFrame, max_distance: float):
    """(distance array (inf where none within max), object positional index array (-1 where none))."""
    d = np.full(ev.n, np.inf)
    idx = np.full(ev.n, -1, dtype=int)
    if obj is None or obj.empty:
        return d, idx
    pairs, dist = obj.sindex.nearest(ev.subject.geometry.values, return_all=False,
                                     max_distance=max_distance, return_distance=True)
    d[pairs[0]] = dist
    idx[pairs[0]] = pairs[1]
    return d, idx


def _ident(obj: gpd.GeoDataFrame, i: int) -> dict:
    if i < 0:
        return {}
    row = obj.iloc[i]
    out = {}
    for k in ("junction_id", "road_id", "id", "poi_id", "building_id", "parcel_id"):
        if k in obj.columns and pd.notna(row.get(k)):
            out["object_id"] = str(row[k])
            break
    for k in ("kind", "class", "road_name", "name", "road_names"):
        if k in obj.columns and pd.notna(row.get(k)):
            out[f"object_{k}"] = row[k]
    return out


def _distance_leaf(ev: Evaluator, p: Predicate, obj: gpd.GeoDataFrame, negate: bool, what: str):
    D = float(p.distance_m)
    d, idx = nearest(ev, obj, D + float(ev.u.max(initial=0.0)) + 1e-6)
    codes = near_codes(d, D, ev.u)
    if negate:
        codes = not_codes(codes)
    recs = []
    for i in range(ev.n):
        r = {"threshold_m": D, "uncertainty_m": float(ev.u[i]), "object_layer": what}
        r["nearest_distance_m"] = None if not np.isfinite(d[i]) else round(float(d[i]), 2)
        if not np.isfinite(d[i]):
            r["note"] = f"no {what} feature within {D + ev.u[i]:g} m"
        r.update(_ident(obj, idx[i]))
        recs.append(r)
    return codes, recs


def h_at_intersection(ev: Evaluator, p: Predicate):
    j = _need(ev, ev.ctx.layers.get("junction"), "road junction")
    kinds = {"four_way": {"four_way"}, "t_junction": {"t_junction"}, "any": {"four_way", "t_junction", "complex"}}[p.junction_kind]
    obj = j[j["kind"].isin(kinds)]
    codes, recs = _distance_leaf(ev, p, obj, False, f"junction[{p.junction_kind}]")
    return codes, recs


def h_near(ev: Evaluator, p: Predicate, negate=False):
    obj = _need(ev, ev._object_gdf(p.object, p), p.object.entity_type.value)
    return _distance_leaf(ev, p, obj, negate, p.object.entity_type.value + (str(p.object.road_classes) if p.object.road_classes else ""))


def h_farther(ev: Evaluator, p: Predicate):
    return h_near(ev, p, negate=True)


def h_not_within(ev: Evaluator, p: Predicate):
    if p.distance_m:
        return h_near(ev, p, negate=True)
    codes, recs = h_within(ev, p)
    return not_codes(codes), recs


def _poly_join(ev: Evaluator, obj: gpd.GeoDataFrame, predicate: str):
    sj = gpd.sjoin(ev.subject[["geometry"]], obj[["geometry"]], how="inner", predicate=predicate)
    return sj


def h_within(ev: Evaluator, p: Predicate):
    obj = _need(ev, ev._object_gdf(p.object, p), p.object.entity_type.value)
    sj = _poly_join(ev, obj, "within")
    hit = np.zeros(ev.n, dtype=bool)
    first: dict[int, int] = {}
    for sidx, oidx in zip(sj.index, sj["index_right"]):
        hit[sidx] = True
        first.setdefault(int(sidx), int(oidx))
    codes = np.where(hit, Code.PASS, Code.FAIL).astype(np.int8)
    recs = [{**_ident(obj, obj.index.get_loc(first[i])) , "contained": bool(hit[i])} if i in first else {"contained": False} for i in range(ev.n)]
    return codes, recs


def _topo(name):
    def h(ev: Evaluator, p: Predicate):
        obj = _need(ev, ev._object_gdf(p.object, p), p.object.entity_type.value)
        sj = _poly_join(ev, obj, name)
        hit = np.zeros(ev.n, dtype=bool)
        hit[sj.index.unique()] = True
        return np.where(hit, Code.PASS, Code.FAIL).astype(np.int8), [{"predicate_geometry": name}] * ev.n
    return h


def h_inside_building(ev: Evaluator, p: Predicate):
    b = _need(ev, ev.ctx.layers.get("building"), "building")
    sj = gpd.sjoin(ev.subject[["geometry"]], b[["geometry"]].assign(_bid=b.get("building_id", pd.Series(b.index.astype(str), index=b.index))),
                   how="inner", predicate="within")
    first = {}
    for sidx, bid in zip(sj.index, sj["_bid"]):
        first.setdefault(int(sidx), bid)
    d, idx = nearest(ev, b, float(ev.u.max(initial=0.0)) + 1e-6)
    codes = np.full(ev.n, Code.FAIL, dtype=np.int8)
    recs = []
    for i in range(ev.n):
        if i in first:
            codes[i] = Code.PASS
            recs.append({"building_id": str(first[i]), "contained": True})
        elif np.isfinite(d[i]) and d[i] <= ev.u[i] and ev.u[i] > 0:
            codes[i] = Code.PASS_NEAR
            recs.append({"building_id": str(b.iloc[idx[i]].get("building_id", idx[i])), "contained": False,
                         "distance_to_footprint_m": round(float(d[i]), 2),
                         "note": "point lies outside footprint but within positional uncertainty"})
        else:
            recs.append({"contained": False})
    return codes, recs


def h_inside_parcel(ev: Evaluator, p: Predicate):
    parcels = _need(ev, ev.ctx.layers.get("parcel"), "parcel polygon")
    if not (parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any()):
        raise _Unavailable("parcel layer has no polygon geometry (centroids only)")
    sj = gpd.sjoin(ev.subject[["geometry"]], parcels[["geometry", "parcel_id"]], how="inner", predicate="within")
    first = {}
    for sidx, pid in zip(sj.index, sj["parcel_id"]):
        first.setdefault(int(sidx), pid)
    codes = np.array([Code.PASS if i in first else Code.FAIL for i in range(ev.n)], dtype=np.int8)
    return codes, [{"parcel_id": str(first[i]), "contained": True} if i in first else {"contained": False} for i in range(ev.n)]


def h_frontage(ev: Evaluator, p: Predicate):
    roads = _need(ev, ev._object_gdf(p.object, p), "road")
    tol = float(p.distance_m or 10.0)
    buf = ev.subject[["geometry"]].copy()
    buf["geometry"] = ev.subject.geometry.buffer(tol)
    sj = gpd.sjoin(buf, roads[["geometry", "road_name", "id"]], how="inner", predicate="intersects")
    counts = np.zeros(ev.n, dtype=int)
    names: dict[int, set] = {}
    for sidx, name, rid in zip(sj.index, sj["road_name"], sj["id"]):
        key = name if isinstance(name, str) and name else f"segment:{rid}"
        names.setdefault(int(sidx), set()).add(key)
    for i, s in names.items():
        counts[i] = len(s)
    codes = np.where(counts >= p.min_count, Code.PASS, Code.FAIL).astype(np.int8)
    recs = [{"frontage_tolerance_m": tol, "roads": sorted(names.get(i, [])), "count": int(counts[i]),
             "required": p.min_count,
             "note": "unnamed segments counted individually; may over-count"} for i in range(ev.n)]
    return codes, recs


def _subject_nodes(ev: Evaluator):
    net = ev.ctx.network()
    return net, *net.snap(ev.subject.geometry)


def h_connected(ev: Evaluator, p: Predicate):
    roads = _need(ev, ev._object_gdf(p.object, p), "road")
    net, nodes, sd = _subject_nodes(ev)
    comps = {net.component_of(n) for n in net.nodes_of_segments(roads["id"])}
    codes = np.full(ev.n, Code.UNKNOWN, dtype=np.int8)
    recs = []
    for i, n in enumerate(nodes):
        if n is None:
            recs.append({"note": "no road node within 150 m"})
            continue
        ok = net.component_of(n) in comps
        codes[i] = Code.PASS if ok else Code.FAIL
        recs.append({"component": net.component_of(n), "snap_m": round(float(sd[i]), 1)})
    return codes, recs


def h_distance_by_road(ev: Evaluator, p: Predicate):
    obj = _need(ev, ev._object_gdf(p.object, p), p.object.entity_type.value)
    net = ev.ctx.network()
    subj = net.snap_edges(ev.subject.geometry)
    objs = net.snap_edges(obj.geometry)
    g = net.graph
    SRC = "__src__"
    by_seg: dict[str, list] = {}
    for o in objs:
        if o is None:
            continue
        by_seg.setdefault(o["segment_id"], []).append(o)
        for node, off in o["candidates"]:
            if g.has_edge(SRC, node):
                g[SRC][node]["weight"] = min(g[SRC][node]["weight"], off)
            else:
                g.add_edge(SRC, node, weight=off)
    try:
        import networkx as nx
        dist = nx.single_source_dijkstra_path_length(g, SRC, cutoff=float(p.distance_m) + 400.0, weight="weight") if SRC in g else {}
    finally:
        if SRC in g:
            g.remove_node(SRC)
    d = np.full(ev.n, np.inf)
    unknown = np.zeros(ev.n, dtype=bool)
    for i, s in enumerate(subj):
        if s is None:
            unknown[i] = True
            continue
        best = min((dist.get(node, np.inf) + off for node, off in s["candidates"]), default=np.inf)
        for o in by_seg.get(s["segment_id"], []):  # same segment: direct path along the segment
            best = min(best, abs(s["along_m"] - o["along_m"]) + s["snap_m"] + o["snap_m"])
        d[i] = best
    codes = near_codes(d, float(p.distance_m), ev.u)
    codes[unknown] = Code.UNKNOWN
    recs = [{"road_distance_m": None if not np.isfinite(d[i]) else round(float(d[i]), 1), "threshold_m": p.distance_m,
             "note": "undirected graph distance incl. snapping offsets; not travel time"
             + ("; subject could not be snapped to a road within 150 m" if unknown[i] else "")} for i in range(ev.n)]
    return codes, recs


def h_changed(ev: Evaluator, p: Predicate):
    a, b = p.snapshots
    if a not in ev.ctx.snapshots or b not in ev.ctx.snapshots:
        raise _Unavailable("two dated snapshots are required; single-snapshot data cannot show change")
    from ..temporal import classify_change
    status = classify_change(ev.ctx.snapshots[a], ev.ctx.snapshots[b], ev.subject)
    codes = np.where(status["changed"].to_numpy(), Code.PASS, Code.FAIL).astype(np.int8)
    return codes, [{"change": r, "caveat": "difference may reflect source coverage, not physical change"} for r in status["change_type"]]


_HANDLERS = {
    Op.AT_INTERSECTION: h_at_intersection, Op.NEAR: h_near, Op.FARTHER_THAN: h_farther,
    Op.NOT_WITHIN: h_not_within, Op.WITHIN: h_within, Op.INTERSECTS: _topo("intersects"),
    Op.TOUCHES: _topo("touches"), Op.INSIDE_BUILDING: h_inside_building, Op.INSIDE_PARCEL: h_inside_parcel,
    Op.HAS_FRONTAGE_ON: h_frontage, Op.CONNECTED_TO: h_connected, Op.DISTANCE_BY_ROAD: h_distance_by_road,
    Op.CHANGED_BETWEEN: h_changed,
}
