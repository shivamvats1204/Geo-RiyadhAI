"""Typed tool registry. Every tool validates its arguments, enforces geographic/size budgets, returns structured output with a
calculation record and provenance, and never fabricates data. Language models (if any) may only *select* tools and arguments."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from shapely.geometry import box
from shapely.ops import unary_union

from geoacquirer.ontology import CONCEPTS, category_mask, resolve

from . import features as F
from . import ml as ML
from . import optimization as OPT
from . import simulation as SIM
from . import suitability as SUIT
from .geoacquirer_client import GeoAcquirerClient
from .rag import KnowledgeBase
from .results import ResultStore
from .store import AreaDataset, UrbanStore

MAX_ROWS_DEFAULT = 50_000


class NeedsApproval(Exception):
    def __init__(self, tool: str, why: str):
        super().__init__(f"{tool} needs approval: {why}")
        self.tool, self.why = tool, why


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    store: UrbanStore
    area: AreaDataset
    results: ResultStore
    kb: KnowledgeBase
    client: GeoAcquirerClient
    out_dir: Path
    approved: set = field(default_factory=set)       # tool names (or "tool:key") the user approved for this job
    max_rows: int = MAX_ROWS_DEFAULT
    cache: dict = field(default_factory=dict)

    def provenance_of(self, *layers) -> dict:
        return {l: self.area.provenance.get(l, {"note": "derived"}) for l in layers}


@dataclass
class ToolResult:
    tool: str
    summary: dict
    calc_id: str = field(default_factory=lambda: "calc-" + uuid.uuid4().hex[:8])
    result_id: Optional[str] = None
    table: Optional[pd.DataFrame] = None
    provenance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    facts: list = field(default_factory=list)   # {"type": observed|model|inferred, "text", "calc"}
    seconds: float = 0.0

    def fact(self, kind: str, text: str):
        self.facts.append({"type": kind, "text": text, "calc": self.calc_id})
        return self


# --------------------------------------------------------------------------- arg models
class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegionSpec(_A):
    kind: Literal["all", "north", "south", "east", "west", "central", "district", "bbox"] = "all"
    district: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None


class NearFilter(_A):
    layer: Literal["poi", "road", "parking", "junction", "building", "metro", "attraction"]
    categories: list[str] = Field(default_factory=list)
    road_classes: list[str] = Field(default_factory=list)
    junction_kind: Optional[Literal["four_way", "t_junction", "any"]] = None
    distance_m: float = Field(gt=0, le=20000)
    negate: bool = False


class FilterArgs(_A):
    layer: Literal["poi", "building", "parking", "junction", "district", "metro", "attraction", "urbi_building", "parcel"] = "poi"
    categories: list[str] = Field(default_factory=list)
    region: RegionSpec = Field(default_factory=RegionSpec)
    near: list[NearFilter] = Field(default_factory=list)
    base_result_id: Optional[str] = None
    limit: int = Field(default=20000, ge=1, le=200000)


def resolve_region(ds: AreaDataset, spec: RegionSpec):
    poly = ds.area_polygon()
    if spec.kind == "all":
        return poly, ds.scope_label() if ds.scope else "entire prepared area"
    minx, miny, maxx, maxy = poly.bounds
    if spec.kind == "bbox":
        if spec.bbox is None:
            raise ToolError("region kind 'bbox' needs bbox")
        g = gpd.GeoSeries([box(*spec.bbox)], crs=4326).to_crs(ds.crs).iloc[0].intersection(poly)
        if g.is_empty:
            raise ToolError("that box does not intersect the selected area")
        return g, f"bbox {tuple(round(v, 4) for v in spec.bbox)}"
    if spec.kind == "district":
        d = ds.layers.get("district")
        if d is None or not spec.district:
            raise ToolError("district region needs a district name and a district layer")
        names = d["name"].fillna("")
        wanted = [w.strip() for w in spec.district.split("|") if w.strip()]
        hit = d[names.apply(lambda n: any(w.lower() in n.lower() for w in wanted))]
        if hit.empty:
            sample = ", ".join(names[names != ""].head(8))
            raise ToolError(f"no district matching {spec.district!r}; examples: {sample}")
        return unary_union(hit.geometry.values).intersection(poly), f"district(s) {', '.join(hit['name'].head(3))}"
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2
    w, h = maxx - minx, maxy - miny
    boxes = {"north": box(minx, my, maxx, maxy), "south": box(minx, miny, maxx, my), "east": box(mx, miny, maxx, maxy),
             "west": box(minx, miny, mx, maxy), "central": box(minx + w / 3, miny + h / 3, maxx - w / 3, maxy - h / 3)}
    return boxes[spec.kind].intersection(poly), f"the {spec.kind} part of {ds.scope_label()} (a geometric split of its bounding box, not an administrative boundary)"


def _layer(ctx: ToolContext, name: str) -> gpd.GeoDataFrame:
    try:
        return ctx.area.get(name)
    except KeyError as e:
        raise ToolError(str(e))


def _cap(ctx: ToolContext, gdf: gpd.GeoDataFrame, limit: int):
    lim = min(limit, ctx.max_rows)
    if len(gdf) > lim:
        return gdf.iloc[:lim], f"result truncated to {lim} of {len(gdf)} rows (row budget)"
    return gdf, None


def _ids(gdf: gpd.GeoDataFrame) -> str:
    for c in ("poi_id", "building_id", "district_id", "junction_id", "road_id", "cell_id", "parking_id"):
        if c in gdf.columns:
            return c
    return ""


def _object_layer(ctx: ToolContext, nf: NearFilter) -> gpd.GeoDataFrame:
    g = _layer(ctx, nf.layer if nf.layer != "junction" else "junction")
    if nf.layer == "poi" and nf.categories:
        ids = [c.id for t in nf.categories for c in resolve(t)[:1]] or [c for c in nf.categories if c in CONCEPTS]
        if not ids:
            raise ToolError(f"unknown category terms {nf.categories}")
        g = g[category_mask(g["category_path"], ids)]
    if nf.layer == "road" and nf.road_classes:
        g = g[g["class"].isin(nf.road_classes)]
    if nf.layer == "junction" and nf.junction_kind:
        kinds = {"four_way": ["four_way"], "t_junction": ["t_junction"], "any": ["four_way", "t_junction", "complex"]}[nf.junction_kind]
        g = g[g["kind"].isin(kinds)]
    return g


def _within(g: gpd.GeoDataFrame, objs: gpd.GeoDataFrame, d: float) -> np.ndarray:
    hit = np.zeros(len(g), dtype=bool)
    if len(objs):
        idx = objs.sindex.query(g.geometry.values, predicate="dwithin", distance=d)
        hit[idx[0]] = True
    return hit


# --------------------------------------------------------------------------- tools
TOOLS: dict[str, tuple[type, Callable, bool, str]] = {}


def tool(name: str, model: type, expensive: bool = False, doc: str = ""):
    def deco(fn):
        TOOLS[name] = (model, fn, expensive, doc or (fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco


class SearchArgs(_A):
    query: str = ""


@tool("search_local_data", SearchArgs, doc="List areas, layers and row counts available locally (no network).")
def search_local_data(ctx: ToolContext, a: SearchArgs) -> ToolResult:
    cov = ctx.area.coverage()
    rows = [{"layer": k, **v} for k, v in cov["layers"].items()]
    t = ToolResult("search_local_data", {"areas": ctx.store.areas(), "active_area": ctx.area.name, "layers": cov["layers"]}, table=pd.DataFrame(rows))
    return t.fact("observed", f"Local store holds area '{ctx.area.name}' with " + ", ".join(f"{k}: {v['rows']}" for k, v in cov["layers"].items() if k in ("poi", "building", "road_raw", "district", "parking")))


class RequestArgs(_A):
    bbox: tuple[float, float, float, float]
    layers: list[Literal["poi", "road", "connector", "building", "district"]] = Field(default_factory=lambda: ["poi", "road", "connector", "building", "district"])
    area_name: str = "custom_area"


@tool("request_geoacquirer_data", RequestArgs, expensive=True, doc="Acquire missing layers through GeoAcquirer (cache first; live acquisition needs approval).")
def request_geoacquirer_data(ctx: ToolContext, a: RequestArgs) -> ToolResult:
    w, s, e, n = a.bbox
    if not (e - w) * (n - s) <= 0.1:  # ~ 1000 km2 at this latitude: bounded request
        raise ToolError("requested bbox is larger than the 0.1 deg2 acquisition budget")
    cached = ctx.client.cached_layers(a.bbox, a.layers)
    live = [l for l, c in cached.items() if not c]
    if live and "request_geoacquirer_data" not in ctx.approved:
        raise NeedsApproval("request_geoacquirer_data", f"layers {live} are not cached; live acquisition takes minutes and calls public data servers")
    ds = ctx.store.build_area(a.area_name, a.bbox, ctx.client, allow_live=bool(live))
    ctx.area = ds
    r = ToolResult("request_geoacquirer_data", {"area": a.area_name, "layers_from_cache": [l for l, c in cached.items() if c], "layers_live": live,
                                                  "rows": {k: len(v) for k, v in ds.layers.items() if k in ("poi", "road_raw", "building", "district")}},
                   provenance=ds.provenance)
    return r.fact("observed", f"Acquired area {a.area_name} through GeoAcquirer ({len(live)} live layer(s), {len(cached) - len(live)} cached).")


class SchemaArgs(_A):
    layer: str


@tool("inspect_layer_schema", SchemaArgs, doc="Columns, dtypes, null rates and a sample row of a layer.")
def inspect_layer_schema(ctx: ToolContext, a: SchemaArgs) -> ToolResult:
    g = _layer(ctx, a.layer)
    cols = [{"column": c, "dtype": str(g[c].dtype), "null_rate": round(float(g[c].isna().mean()), 4)} for c in g.columns]
    return ToolResult("inspect_layer_schema", {"layer": a.layer, "rows": len(g), "geometry": sorted(g.geometry.geom_type.unique().tolist())}, table=pd.DataFrame(cols),
                      provenance=ctx.provenance_of(a.layer)).fact("observed", f"Layer {a.layer} has {len(g)} rows and {len(g.columns)} columns.")


@tool("get_geographic_coverage", SearchArgs, doc="Coverage, topology statistics, provenance and known data gaps for the active area.")
def get_geographic_coverage(ctx: ToolContext, a: SearchArgs) -> ToolResult:
    cov = ctx.area.coverage()
    gaps = []
    if not ctx.area.has("parcel"):
        gaps.append("no parcel polygons / zoning: zoning-dependent questions are UNVERIFIED")
    if len(ctx.area.layers.get("parking", [])) < 50:
        gaps.append(f"only {len(ctx.area.layers.get('parking', []))} mapped parking facilities: parking supply is largely unmapped")
    gaps.append("no population, traffic volume, land price, revenue or occupancy data")
    return ToolResult("get_geographic_coverage", {**cov, "gaps": gaps}, provenance=ctx.area.provenance).fact(
        "observed", "Data gaps: " + "; ".join(gaps))


@tool("filter_features", FilterArgs, doc="Filter a layer by category, region and proximity rules (deterministic spatial predicates).")
def filter_features(ctx: ToolContext, a: FilterArgs) -> ToolResult:
    region, rdesc = resolve_region(ctx.area, a.region)
    base = ctx.results.get(a.base_result_id).gdf.to_crs(ctx.area.crs) if a.base_result_id else _layer(ctx, a.layer)
    g = base
    if a.categories and "category_path" in g.columns:
        ids = [c.id for t in a.categories for c in resolve(t)[:1]] or [c for c in a.categories if c in CONCEPTS]
        if not ids:
            raise ToolError(f"unknown category terms {a.categories}")
        g = g[category_mask(g["category_path"], ids)]
    g = g[g.geometry.intersects(region)]
    steps = [{"step": "category+region", "rows": int(len(g))}]
    for nf in a.near:
        objs = _object_layer(ctx, nf)
        hit = _within(g, objs, nf.distance_m)
        g = g[~hit] if nf.negate else g[hit]
        steps.append({"step": f"{'NOT ' if nf.negate else ''}within {nf.distance_m:g} m of {nf.layer}{nf.categories or nf.road_classes or ''}", "rows": int(len(g)), "object_rows": int(len(objs))})
    g, trunc = _cap(ctx, g, a.limit)
    rs = ctx.results.put(g, kind="features", producer="filter_features", parent=a.base_result_id, params=a.model_dump(), notes=[trunc] if trunc else [])
    t = ToolResult("filter_features", {"rows": len(g), "region": rdesc, "steps": steps}, result_id=rs.id, provenance=ctx.provenance_of(a.layer),
                   warnings=[trunc] if trunc else [], table=pd.DataFrame(steps))
    return t.fact("observed", f"{len(g)} {a.layer} features remain after: " + "; ".join(f"{s['step']} -> {s['rows']}" for s in steps) + f" (region: {rdesc}).")


class JunctionArgs(_A):
    kind: Literal["four_way", "t_junction", "any", "highway_interchange"] = "four_way"
    region: RegionSpec = Field(default_factory=RegionSpec)


@tool("find_intersections", JunctionArgs, doc="Genuine road junctions from connector topology (overpasses and ramps excluded).")
def find_intersections(ctx: ToolContext, a: JunctionArgs) -> ToolResult:
    j = _layer(ctx, "junction")
    region, rdesc = resolve_region(ctx.area, a.region)
    kinds = {"any": ["four_way", "t_junction", "complex"]}.get(a.kind, [a.kind])
    g = j[j["kind"].isin(kinds) & j.geometry.within(region)]
    rs = ctx.results.put(g, kind="junctions", producer="find_intersections", params=a.model_dump())
    excl = len(ctx.area.layers.get("excluded_crossings", []))
    return ToolResult("find_intersections", {"rows": len(g), "kind": a.kind, "excluded_grade_separated_crossings": excl, "region": rdesc}, result_id=rs.id,
                      provenance=ctx.provenance_of("road", "connector")).fact(
        "observed", f"{len(g)} {a.kind} junction clusters in {rdesc}; {excl} geometric crossings without a shared connector were excluded as grade-separated.")


class ProxArgs(_A):
    result_id: str
    to: NearFilter
    name: str = Field(default="nearest", pattern=r"^[a-z][a-z0-9_]{0,24}$")


@tool("measure_proximity", ProxArgs, doc="Add the distance (m) from each feature in a result set to the nearest object of another layer.")
def measure_proximity(ctx: ToolContext, a: ProxArgs) -> ToolResult:
    base = ctx.results.get(a.result_id).gdf.to_crs(ctx.area.crs)
    objs = _object_layer(ctx, a.to)
    d = np.full(len(base), np.nan)
    ident = np.array([None] * len(base), dtype=object)
    if len(objs):
        pairs, dist = objs.sindex.nearest(base.geometry.values, return_all=False, return_distance=True)
        d[pairs[0]] = dist
        idc = _ids(objs)
        if idc:
            ident[pairs[0]] = objs[idc].to_numpy()[pairs[1]]
    out = base.copy()
    out[f"dist_{a.name}_m"] = d.round(1)
    out[f"{a.name}_id"] = ident
    rs = ctx.results.put(out, kind="features", producer="measure_proximity", parent=a.result_id, params=a.model_dump())
    med = float(np.nanmedian(d)) if np.isfinite(d).any() else None
    return ToolResult("measure_proximity", {"rows": len(out), "median_distance_m": None if med is None else round(med, 1), "object_rows": int(len(objs))},
                      result_id=rs.id, provenance=ctx.provenance_of(a.to.layer)).fact(
        "observed", f"Median distance from the {len(out)} features to the nearest {a.to.layer} object ({a.name}) is {med:.0f} m." if med is not None else "No objects found.")


class NetAccessArgs(_A):
    origin_result_id: Optional[str] = None
    origin_layer: Literal["poi", "building", "parking"] = "poi"
    origin_categories: list[str] = Field(default_factory=list)
    destination: NearFilter
    cutoff_m: float = Field(default=800, gt=0, le=5000)
    region: RegionSpec = Field(default_factory=RegionSpec)
    limit: int = Field(default=3000, ge=1, le=20000)


@tool("calculate_network_accessibility", NetAccessArgs, doc="For each origin, count destinations and find the nearest one by road-network distance.")
def calculate_network_accessibility(ctx: ToolContext, a: NetAccessArgs) -> ToolResult:
    import networkx as nx
    net = ctx.area.network()
    region, rdesc = resolve_region(ctx.area, a.region)
    if a.origin_result_id:
        o = ctx.results.get(a.origin_result_id).gdf.to_crs(ctx.area.crs)
    else:
        o = _layer(ctx, a.origin_layer)
        if a.origin_categories and "category_path" in o:
            o = o[category_mask(o["category_path"], [c.id for t in a.origin_categories for c in resolve(t)[:1]])]
    o = o[o.geometry.intersects(region)].head(min(a.limit, ctx.max_rows)).reset_index(drop=True)
    dest = _object_layer(ctx, a.destination)
    dn, dsn = net.snap(dest.geometry.centroid, 200.0)
    dnodes: dict[str, int] = {}
    for n in dn:
        if n is not None:
            dnodes[n] = dnodes.get(n, 0) + 1
    on, osn = net.snap(o.geometry.centroid, 200.0)
    cnt = np.full(len(o), np.nan)
    nearest = np.full(len(o), np.nan)
    cache = {}
    for i, n in enumerate(on):
        if n is None:
            continue
        if n not in cache:
            cache[n] = nx.single_source_dijkstra_path_length(net.graph, n, cutoff=a.cutoff_m, weight="weight")
        reach = cache[n]
        cnt[i] = sum(c for k, c in dnodes.items() if k in reach)
        ds_ = [reach[k] for k in dnodes if k in reach]
        if ds_:
            nearest[i] = min(ds_) + osn[i]
    out = o.copy()
    out["destinations_reachable"] = cnt
    out["nearest_network_distance_m"] = nearest.round(1)
    rs = ctx.results.put(out, kind="features", producer="calculate_network_accessibility", parent=a.origin_result_id, params=a.model_dump())
    ok = np.isfinite(cnt)
    summary = {"origins": len(o), "origins_snapped": int(ok.sum()), "destinations": int(len(dest)), "cutoff_m": a.cutoff_m,
               "share_with_any_destination": round(float((cnt[ok] > 0).mean()), 4) if ok.any() else None,
               "median_nearest_network_m": float(np.nanmedian(nearest)) if np.isfinite(nearest).any() else None,
               "note": "undirected road-network distance; not travel time"}
    return ToolResult("calculate_network_accessibility", summary, result_id=rs.id, provenance=ctx.provenance_of("road", a.destination.layer)).fact(
        "observed", f"{summary['share_with_any_destination']:.1%} of {summary['origins_snapped']} snapped origins have at least one matching destination within {a.cutoff_m:g} m by road."
        if summary["share_with_any_destination"] is not None else "No origin could be snapped to the road network.")


class AggArgs(_A):
    layer: Literal["poi", "building", "parking", "metro", "attraction", "urbi_building", "parcel"] = "poi"
    categories: list[str] = Field(default_factory=list)
    by: Literal["district", "grid"] = "district"
    grid_m: float = Field(default=1000, ge=200, le=5000)
    district_level: Literal["macrohood", "neighborhood", "all"] = "macrohood"
    capacity_sum: bool = False
    top: int = Field(default=30, ge=1, le=500)
    focus_names: list[str] = Field(default_factory=list)  # restrict the returned table to units whose name contains one of these


@tool("aggregate_spatial_statistics", AggArgs, doc="Count (and density per km2) of features per district or grid cell.")
def aggregate_spatial_statistics(ctx: ToolContext, a: AggArgs) -> ToolResult:
    g = _layer(ctx, a.layer)
    if a.categories and "category_path" in g:
        g = g[category_mask(g["category_path"], [c.id for t in a.categories for c in resolve(t)[:1]])]
    warnings = []
    if a.by == "district":
        d = _layer(ctx, "district")
        if not getattr(ctx.area, "_urbi_districts", False):  # the Urbi districts have one level; the subtype filter is for Overture's macro/neighbourhood/micro levels
            d = d if a.district_level == "all" else d[d["subtype"] == a.district_level]
        if ctx.area.scope:  # count only inside the selected area, over the districts it touches
            poly = ctx.area.area_polygon()
            d = d[d.geometry.intersects(poly)]
            g = g[g.geometry.intersects(poly)]
        d = d.reset_index(drop=True)
        d["area_km2"] = d.geometry.area / 1e6
        sj = gpd.sjoin(g[[c for c in ["geometry", "capacity"] if c in g.columns]], d[["district_id", "geometry"]], how="inner", predicate="within")
        cnt = sj.groupby("district_id").size().rename("count")
        agg = d.merge(cnt, left_on="district_id", right_index=True, how="left")
        agg["count"] = agg["count"].fillna(0).astype(int)
        if a.capacity_sum and "capacity" in sj:
            cap = sj.groupby("district_id")["capacity"].sum(min_count=1).rename("capacity_reported_sum")
            agg = agg.merge(cap, left_on="district_id", right_index=True, how="left")
        outside = len(g) - int(len(sj.index.unique()))
        if outside:
            warnings.append(f"{outside} features fall outside every {a.district_level} polygon (districts do not tile the whole area)")
    else:
        minx, miny, maxx, maxy = ctx.area.area_polygon().bounds
        cells = [box(x, y, x + a.grid_m, y + a.grid_m) for x in np.arange(minx, maxx, a.grid_m) for y in np.arange(miny, maxy, a.grid_m)]
        d = gpd.GeoDataFrame({"district_id": [f"g{i}" for i in range(len(cells))], "name": [f"grid {i}" for i in range(len(cells))]}, geometry=cells, crs=ctx.area.crs)
        d["area_km2"] = d.geometry.area / 1e6
        sj = gpd.sjoin(g[["geometry"]], d[["district_id", "geometry"]], how="inner", predicate="within")
        cnt = sj.groupby("district_id").size().rename("count")
        agg = d.merge(cnt, left_on="district_id", right_index=True, how="left")
        agg["count"] = agg["count"].fillna(0).astype(int)
    agg["density_per_km2"] = (agg["count"] / agg["area_km2"]).round(2)
    agg = agg.sort_values("count", ascending=False)
    rs = ctx.results.put(agg, kind="aggregate", producer="aggregate_spatial_statistics", params=a.model_dump())
    shown = agg.head(a.top)
    if a.focus_names:
        pat = "|".join(__import__("re").escape(n) for n in a.focus_names)
        shown = agg[agg["name"].fillna("").str.contains(pat, case=False)]
        if shown.empty:
            raise ToolError(f"no {a.by} unit matches {a.focus_names}")
    tbl = pd.DataFrame(shown.drop(columns="geometry"))
    top = shown.iloc[0] if len(shown) else None
    focus_text = ""
    if a.focus_names:
        focus_text = "; ".join(f"{r['name']}: {int(r['count'])} ({r['density_per_km2']:.1f} per km2, area {r['area_km2']:.2f} km2)" for r in shown.to_dict("records"))
    t = ToolResult("aggregate_spatial_statistics", {"units": len(agg), "total_counted": int(agg["count"].sum()), "by": a.by}, result_id=rs.id, table=tbl,
                   provenance=ctx.provenance_of(a.layer, "district" if a.by == "district" else a.layer), warnings=warnings)
    if focus_text:
        t.fact("observed", f"Compared units ({a.layer}): {focus_text}.")
    elif top is not None:
        t.fact("observed", f"{int(agg['count'].sum())} {a.layer} features counted over {len(agg)} {a.by} units; highest: {top.get('name')} with {int(top['count'])} ({top['density_per_km2']:.1f} per km2).")
    return t


class LandUseArgs(_A):
    action: Literal["metrics", "train", "predict"] = "metrics"
    max_rows: int = Field(default=30000, ge=2000, le=60000)
    predict_limit: int = Field(default=20000, ge=100, le=100000)
    label_source: Literal["overture", "urbi"] = "overture"   # 'urbi' needs the Urbi building layer (purpose labels, ~10x more labelled buildings)


@tool("classify_land_use", LandUseArgs, expensive=True, doc="Building land-use (residential vs commercial) model: report spatial-CV metrics, train, or predict unlabelled buildings.")
def classify_land_use(ctx: ToolContext, a: LandUseArgs) -> ToolResult:
    mdir = ctx.store.root / "models"
    urbi = a.label_source == "urbi"
    if urbi and not ctx.area.has("urbi_building"):
        raise ToolError("label_source='urbi' needs the Urbi building layer (urbi_ingest.ingest_all); not loaded for this area")
    mp = mdir / ("building_landuse_urbi_metrics.json" if urbi else "building_landuse_metrics.json")
    if a.action == "metrics" and mp.exists():
        res = json.loads(mp.read_text())
    else:
        if "classify_land_use" not in ctx.approved:
            raise NeedsApproval("classify_land_use", "training runs LightGBM with spatial cross-validation (about 30-110 s)")
        res = ML.run_experiment_urbi(ctx.area, max_rows=max(a.max_rows, 2000), out_dir=mdir) if urbi else ML.run_experiment(ctx.area, max_rows=a.max_rows, out_dir=mdir)
    s = res["models_spatial_cv"]
    summary = {"dataset": res["dataset"], "lightgbm_spatial_cv": {k: s["lightgbm_all_features"][k] for k in ("pr_auc", "roc_auc", "balanced_accuracy", "f1_commercial")},
               "baselines": {k: {kk: vv for kk, vv in v.items() if kk in ("pr_auc", "balanced_accuracy")} for k, v in s.items() if k.startswith("baseline")},
               "random_split_contrast": s.get("lightgbm_random_split_for_contrast_only"), "ablations": res["ablations_spatial_cv"], "limitations": res["limitations"]}
    t = ToolResult("classify_land_use", summary, provenance=ctx.provenance_of("building", "poi", "road"))
    if a.action == "predict":
        pred = ML.predict_unlabelled(ctx.area, mdir / "building_landuse_lgbm.txt", a.predict_limit)
        rs = ctx.results.put(pred, kind="model_output", producer="classify_land_use", params=a.model_dump(),
                             notes=["MODEL OUTPUT on unlabelled buildings (mostly ML-detected footprints); labelled-building CV does not validate this population"])
        t.result_id = rs.id
        t.warnings.append("Predictions are for buildings without a source label, a different population than the labelled training set; scores are uncalibrated.")
    m = summary["lightgbm_spatial_cv"]
    t.fact("model", f"Building land-use model (LightGBM, 1 km spatial block CV): PR-AUC {m['pr_auc']}, balanced accuracy {m['balanced_accuracy']} versus a majority-class baseline PR-AUC "
                    f"{summary['baselines']['baseline_majority_class']['pr_auc']}; labels are OSM-derived and noisy.")
    return t


class SuitArgs(_A):
    facility: Literal["supermarket", "retail", "clinic", "restaurant"] = "supermarket"
    k: int = Field(default=5, ge=1, le=20)
    region: RegionSpec = Field(default_factory=RegionSpec)
    weights: dict[str, float] = Field(default_factory=dict)
    min_highway_distance_m: float = Field(default=60, ge=0, le=2000)
    min_competitor_distance_m: float = Field(default=100, ge=0, le=2000)
    max_junction_distance_m: Optional[float] = Field(default=None, gt=0, le=3000)
    allowed_landuse: list[str] = Field(default_factory=list)
    min_separation_m: float = Field(default=800, ge=100, le=5000)
    grid_step_m: float = Field(default=150, ge=75, le=500)

    @field_validator("weights")
    @classmethod
    def _w(cls, v):
        bad = [k for k in v if k not in SUIT.CRITERIA]
        if bad or any(x < 0 for x in v.values()):
            raise ValueError(f"weights must be non-negative and keys in {list(SUIT.CRITERIA)}; got {bad or v}")
        return v


_COMP = {"supermarket": ("supermarket",), "retail": ("shop",), "clinic": ("clinic",), "restaurant": ("restaurant",)}


def _features_for(ctx: ToolContext, cfg: F.FeatureConfig):
    key = "features:" + hashlib.md5(json.dumps(cfg.__dict__, default=str, sort_keys=True).encode()).hexdigest()[:10]
    if key not in ctx.cache:
        cells, counts = F.candidate_cells(ctx.area, cfg)
        feats = F.compute_features(ctx.area, cells, cfg)
        ctx.cache[key] = (cells, feats, counts)
        d = ctx.store.root / "derived" / ctx.area.name
        d.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(d / f"{key.split(':')[1]}.parquet")
    return ctx.cache[key]


@tool("rank_site_suitability", SuitArgs, doc="Rank candidate locations with hard filters, weighted criteria, sensitivity analysis and explanations.")
def rank_site_suitability(ctx: ToolContext, a: SuitArgs) -> ToolResult:
    cfg = F.FeatureConfig(competitor_concepts=_COMP[a.facility], highway_exclusion_m=a.min_highway_distance_m, grid_step_m=a.grid_step_m)
    cells, feats, cell_counts = _features_for(ctx, cfg)
    region, rdesc = resolve_region(ctx.area, a.region)
    inreg = cells.geometry.within(region).to_numpy()
    sub_f, sub_c = feats[inreg].reset_index(drop=True), cells[inreg].reset_index(drop=True)
    xy = pd.DataFrame({"x": sub_c.geometry.x.to_numpy(), "y": sub_c.geometry.y.to_numpy()})
    weights = {**SUIT.DEFAULT_WEIGHTS, **a.weights}
    if sum(weights.values()) <= 0:
        raise ToolError("weights must sum to a positive value (all criteria are zero)")
    flt = SUIT.HardFilters(a.min_highway_distance_m, a.min_competitor_distance_m, a.max_junction_distance_m, a.allowed_landuse or None)
    res = SUIT.rank_sites(sub_f, xy, k=a.k, weights=weights, filters=flt, min_separation_m=a.min_separation_m)
    if len(res.top) == 0:
        raise ToolError("no candidate survives the hard filters: " + json.dumps(res.filter_report))
    top = sub_c.loc[res.top.index].copy()
    for c in ["score", "top_k_frequency", "explanation"] + [c for c in res.top.columns if c.startswith(("contrib_", "norm_"))] + [v[0] for v in SUIT.CRITERIA.values()] + ["landuse_zoning", "dist_to_highway_m", "nearest_competitor_m", "four_way_within_300m", "dist_to_metro_m", "demand_source"]:
        top[c] = res.top[c].to_numpy()
    top["rank"] = range(1, len(top) + 1)
    top["verification"] = "heuristic ranking; land availability and zoning UNVERIFIED"
    if ctx.area.has("parcel"):  # parcel layer present: say where zoning comes from instead of the generic 'UNVERIFIED' note
        res.notes = [n if "UNVERIFIED" not in n else
                     "Candidates are grid cells. Zoning comes from the parcel layer's land-use attribute (e.g. Suhail), mapped to coarse classes by keyword; it is data, not a verified "
                     "legal determination. Land availability and ownership are not checked." for n in res.notes]
    rs = ctx.results.put(top, kind="suitability_top", producer="rank_site_suitability", params=a.model_dump(), notes=res.notes)
    allr = sub_c.loc[res.ranked.index].copy(); allr["score"] = res.ranked["score"].to_numpy()
    all_rs = ctx.results.put(allr, kind="suitability_all", producer="rank_site_suitability", parent=rs.id, params={"note": "all eligible candidates with scores"})
    tbl = pd.DataFrame({"rank": top["rank"], "cell": top["cell_id"], "score": top["score"].round(3), "top5_frequency": top["top_k_frequency"].round(2),
                        "reachable_built_area_800m": top["network_demand_800m"].round(0), "dist_four_way_m": top["dist_to_four_way_m"].round(0),
                        "competitors_500m": top["competitors_500m"], "retail_500m": top["retail_pois_500m"], "parking_300m": top["parking_pois_300m"],
                        "dist_highway_m": top["dist_to_highway_m"].round(0)})
    summary = {"rows": len(top), "region": rdesc, "weights": {k: round(v / sum(weights.values()), 3) for k, v in weights.items()}, "filters": res.filter_report,
               "candidate_generation": cell_counts, "sensitivity": res.sensitivity, "all_ranked_result_id": all_rs.id, "notes": res.notes}
    t = ToolResult("rank_site_suitability", summary, result_id=rs.id, table=tbl, warnings=res.notes, provenance=ctx.provenance_of("poi", "road", "building", "junction"))
    mc = res.sensitivity["monte_carlo"]
    if len(top) < a.k:
        msg = (f"Only {len(top)} of the {a.k} requested locations could be returned: the {a.min_separation_m:g} m minimum separation between recommendations "
               f"and the size of the region ({rdesc}) leave no further eligible cell.")
        t.warnings.append(msg)
        t.fact("observed", msg)
    t.fact("inferred", f"Top {len(top)} heuristic {a.facility} locations in {rdesc}: " + ", ".join(f"{r.cell_id} (score {r.score:.2f})" for r in top.itertuples()) + ".")
    ll = top.to_crs(4326)
    for r, g in zip(top.itertuples(), ll.geometry):
        drivers = sorted(SUIT.CRITERIA, key=lambda k: -getattr(r, "contrib_" + k))[:2]
        t.fact("inferred", f"#{r.rank} {r.cell_id} at {g.y:.5f}N, {g.x:.5f}E (score {r.score:.3f}): reachable residential-like built area {r.network_demand_800m:,.0f} m2 within 800 m by road, "
                           f"{r.dist_to_four_way_m:.0f} m to the nearest four-way junction, {int(r.competitors_500m)} competitor(s) within 500 m, {int(r.retail_pois_500m)} other retail POIs within 500 m, "
                           f"{r.dist_to_highway_m:.0f} m to the nearest motorway/trunk road, {int(r.parking_pois_300m)} mapped parking facility(ies) within 300 m, "
                           + (f"{r.dist_to_metro_m:.0f} m to the nearest metro station/entrance; " if r.dist_to_metro_m == r.dist_to_metro_m else "")
                           + f"zoning {r.landuse_zoning}; "
                           f"strongest score drivers: {', '.join(drivers)}.")
    t.fact("model", f"Rank stability over {mc['n']} random weight vectors: mean Spearman {mc['mean_spearman']}, min {mc['min_spearman']}; a one-at-a-time weight change keeps at least "
                    f"{res.sensitivity['min_top_k_overlap_oat']:.0%} of the top-{len(top)}.")
    fr = res.filter_report
    removed_comp = fr.get("removed_competitor_within_%dm" % a.min_competitor_distance_m, 0)
    t.fact("observed", f"Candidate screening: {cell_counts['grid_cells']} grid cells, {cell_counts['eligible_cells']} left after highway ({a.min_highway_distance_m:g} m), "
                       f"street-frontage and existing-building filters; {removed_comp} removed for a competitor within {a.min_competitor_distance_m:g} m; {fr['eligible']} scored. "
                       + (fr.get("landuse_filter") or "Zoning filter applied."))
    return t


def existing_parking(ctx: "ToolContext", public_only: bool, min_capacity: float):
    """Existing facilities used as the baseline: optionally public-access only and above a reported capacity."""
    g = ctx.area.layers.get("parking")
    if g is None or len(g) == 0:
        return g, "no existing parking layer"
    n0 = len(g)
    notes = []
    if public_only and "access" in g.columns:
        keep = g["access"].isna() | g["access"].astype(str).eq("public")  # unknown access (e.g. Overture places) is kept
        g = g[keep]
        notes.append(f"public/unknown-access only ({n0} -> {len(g)})")
    if min_capacity and "capacity" in g.columns:
        cap = pd.to_numeric(g["capacity"], errors="coerce")
        g = g[cap.isna() | (cap >= min_capacity)]
        notes.append(f"reported capacity >= {min_capacity:g} or unknown ({len(g)} left)")
    return g, "; ".join(notes) or "all mapped parking"


class OptArgs(_A):
    facility: Literal["parking"] = "parking"
    p: int = Field(default=3, ge=1, le=20)
    objective: Literal["p_median", "max_coverage"] = "p_median"
    radius_m: float = Field(default=500, ge=100, le=3000)
    region: RegionSpec = Field(default_factory=RegionSpec)
    demand_categories: list[str] = Field(default_factory=lambda: ["shop"])
    time_limit_s: float = Field(default=45, ge=1, le=300)
    grid_step_m: float = Field(default=200, ge=100, le=600)
    exclude_near_existing_m: float = Field(default=0, ge=0, le=1000)
    max_candidates: int = Field(default=400, ge=20, le=2000)  # candidate pruning budget (documented heuristic)
    existing_public_only: bool = True       # ignore residents-only / handicapped-only / taxi-only lots as public supply
    existing_min_capacity: float = Field(default=0, ge=0, le=2000)  # ignore lots with a smaller reported capacity


@tool("optimize_facility_locations", OptArgs, doc="p-median / max-coverage placement of new facilities on the road network, with greedy and random baselines.")
def optimize_facility_locations(ctx: ToolContext, a: OptArgs) -> ToolResult:
    cfg = F.FeatureConfig(grid_step_m=a.grid_step_m, exclude_occupied=False)
    region, rdesc = resolve_region(ctx.area, a.region)
    cells, _ = F.candidate_cells(ctx.area, cfg, region)
    poi = _layer(ctx, "poi")
    demand = poi[category_mask(poi["category_path"], [c.id for t in a.demand_categories for c in resolve(t)[:1]])]
    demand = demand[demand.geometry.within(region.buffer(300))]
    existing, existing_note = existing_parking(ctx, a.existing_public_only, a.existing_min_capacity)
    if a.exclude_near_existing_m and existing is not None and len(existing):
        cells = cells[~_within(cells, existing, a.exclude_near_existing_m)].reset_index(drop=True)
    if len(cells) == 0 or len(demand) == 0:
        raise ToolError("no candidate cells or no demand points in the region")
    n_cells_total = len(cells)
    if len(cells) > 2 * a.max_candidates:  # cheap Euclidean prefilter so the network step stays bounded
        near = F._within_counts(cells.geometry, demand.geometry, 1.5 * a.radius_m)
        cells = cells.iloc[np.argsort(-near, kind="stable")[: 2 * a.max_candidates]].reset_index(drop=True)
    prob = OPT.build_problem(ctx.area, demand, None, cells, existing, radius_m=a.radius_m)
    prob.meta["candidate_cells_before_prefilter"] = n_cells_total
    res = OPT.optimize(prob, a.p, a.objective, a.time_limit_s, max_candidates=a.max_candidates)
    chosen = res["cpsat"]["selected"]
    sel = cells[cells["cell_id"].isin(chosen)].copy()
    sel["objective"] = a.objective
    sel["verification"] = "mathematical optimum on candidate grid; buildable land NOT verified"
    rs = ctx.results.put(sel, kind="optimization", producer="optimize_facility_locations", params=a.model_dump(), notes=res["assumptions"])
    tbl = pd.DataFrame([{"strategy": "existing facilities only", **res["before"]},
                        {"strategy": "greedy", **{k: res["greedy"][k] for k in ("mean_distance_m", "coverage_within_radius")}},
                        {"strategy": "greedy + swap local search", **{k: res["greedy_swap"][k] for k in ("mean_distance_m", "coverage_within_radius")}},
                        {"strategy": f"CP-SAT ({res['cpsat']['status']})", **{k: res["cpsat"][k] for k in ("mean_distance_m", "coverage_within_radius")}},
                        {"strategy": "random (mean of %d)" % res["random_baseline"]["n"], "mean_distance_m": res["random_baseline"]["mean_distance_m_mean"],
                         "coverage_within_radius": res["random_baseline"]["coverage_mean"]}])
    to_ll = sel.to_crs(4326)
    sites = [{"cell_id": r.cell_id, "lon": round(r.geometry.x, 6), "lat": round(r.geometry.y, 6)} for r in to_ll.itertuples()]
    proposed = [{"name": f"add {x['cell_id']}", "operations": [{"type": "add_facility", "lon": x["lon"], "lat": x["lat"]}]} for x in sites]
    if len(sites) > 1:
        proposed.append({"name": "add all %d selected" % len(sites), "operations": [{"type": "add_facility", "lon": x["lon"], "lat": x["lat"]} for x in sites]})
    summary = {k: v for k, v in res.items() if k != "selected_idx"} | {"selected_sites": sites, "proposed_scenarios": proposed, "region": rdesc, "demand_proxy": f"{len(demand)} {a.demand_categories} POIs (proxy for parking demand; no occupancy data)"}
    t = ToolResult("optimize_facility_locations", summary, result_id=rs.id, table=tbl, warnings=res["assumptions"], provenance=ctx.provenance_of("poi", "road", "parking"))
    b, c = res["before"], res["cpsat"]
    t.fact("model", f"{a.objective} with p={a.p} on {res['problem']['candidates']} candidate cells and {res['problem']['demand_nodes']} demand nodes: mean road distance {b['mean_distance_m']} m -> {c['mean_distance_m']} m; "
                    f"coverage within {a.radius_m:g} m {b['coverage_within_radius']:.1%} -> {c['coverage_within_radius']:.1%} (solver {c['status']}, {c['seconds']} s, optimality gap {c['optimality_gap_pct']}%); "
                    f"greedy {res['greedy']['mean_distance_m']} m, greedy+swap {res['greedy_swap']['mean_distance_m']} m, random baseline {res['random_baseline']['mean_distance_m_mean']} m.")
    t.fact("observed", f"{res['problem']['existing_facilities']} existing parking facilities were used as the baseline ({existing_note}). Reported capacity is operator-supplied, not occupancy; "
                       f"demand is a shop-location proxy.")
    return t


class ScenArgs(_A):
    existing_public_only: bool = True
    existing_min_capacity: float = Field(default=0, ge=0, le=2000)
    scenarios: list[dict] = Field(min_length=1, max_length=8)
    demand_categories: list[str] = Field(default_factory=lambda: ["shop"])
    radius_m: float = Field(default=500, ge=100, le=3000)
    region: RegionSpec = Field(default_factory=RegionSpec)


@tool("compare_scenarios", ScenArgs, doc="Simulate what-if scenarios (add facility, close road) and compare accessibility before/after.")
def compare_scenarios(ctx: ToolContext, a: ScenArgs) -> ToolResult:
    region, rdesc = resolve_region(ctx.area, a.region)
    poi = _layer(ctx, "poi")
    demand = poi[category_mask(poi["category_path"], [c.id for t in a.demand_categories for c in resolve(t)[:1]])]
    demand = demand[demand.geometry.within(region)]
    results = []
    for sc in a.scenarios:
        s = SIM.Scenario(name=sc.get("name", "scenario"), operations=sc.get("operations", []))
        try:
            r = SIM.run_scenario(ctx.area, s, demand, radius_m=a.radius_m, facilities=existing_parking(ctx, a.existing_public_only, a.existing_min_capacity)[0])
        except ValueError as e:
            raise ToolError(f"scenario '{s.name}': {e}")
        SIM.save_scenario(ctx.store.root, r)
        results.append(r)
    table = pd.DataFrame(SIM.compare_scenarios(results))
    t = ToolResult("compare_scenarios", {"scenarios": [r["scenario"]["name"] for r in results], "simulated": True, "results": results}, table=table,
                   provenance=ctx.provenance_of("poi", "road", "parking"), warnings=["ALL VALUES ARE SIMULATED; nothing in the canonical data changed"])
    for r in results:
        t.fact("model", f"[SIMULATED] {r['scenario']['name']}: mean road distance {r['before']['mean_distance_m']} -> {r['after']['mean_distance_m']} m; coverage {r['before']['coverage_within_radius']:.1%} -> {r['after']['coverage_within_radius']:.1%}.")
    return t


class DocArgs(_A):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=3, ge=1, le=8)


@tool("retrieve_reference_documents", DocArgs, doc="Retrieve methodology / data-definition passages with citation ids (no numeric geographic answers).")
def retrieve_reference_documents(ctx: ToolContext, a: DocArgs) -> ToolResult:
    hits = ctx.kb.search(a.query, a.k)
    t = ToolResult("retrieve_reference_documents", {"hits": hits}, table=pd.DataFrame([{k: h[k] for k in ("cite", "score")} | {"text": h["text"][:240]} for h in hits]))
    for h in hits:
        t.fact("reference", f"{h['cite']} {h['text'][:300]}")
    if not hits:
        t.warnings.append("no reference passage matched the query")
    return t


class MapArgs(_A):
    result_id: str
    max_features: int = Field(default=5000, ge=1, le=50000)


@tool("render_map_layer", MapArgs, doc="Write a result set as GeoJSON for the map UI.")
def render_map_layer(ctx: ToolContext, a: MapArgs) -> ToolResult:
    rs = ctx.results.get(a.result_id)
    g = rs.gdf.head(a.max_features)
    for c in g.columns:
        if g[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
            g = g.drop(columns=c)
    p = ctx.out_dir / f"layer_{a.result_id}.geojson"
    p.write_text(g.to_json(), encoding="utf-8")
    return ToolResult("render_map_layer", {"path": str(p), "features": len(g), "bounds": [round(x, 5) for x in g.total_bounds] if len(g) else None}, result_id=a.result_id)


class ExportArgs(_A):
    result_id: str
    format: Literal["csv", "geojson", "geoparquet"] = "geojson"


@tool("export_dataset", ExportArgs, doc="Export a result set (csv, geojson, geoparquet) with its lineage.")
def export_dataset(ctx: ToolContext, a: ExportArgs) -> ToolResult:
    rs = ctx.results.get(a.result_id)
    g = rs.gdf.copy()
    for c in g.columns:
        if c != "geometry" and g[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
            g[c] = g[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict, np.ndarray)) else v)
    ext = {"csv": "csv", "geojson": "geojson", "geoparquet": "parquet"}[a.format]
    p = ctx.out_dir / f"export_{a.result_id}.{ext}"
    if a.format == "csv":
        t = g.copy()
        cen = g.to_crs(ctx.area.crs).geometry.centroid.to_crs(4326)  # centroid in a metric CRS, reported in WGS84
        t["lon"], t["lat"] = cen.x.to_numpy(), cen.y.to_numpy()
        t.drop(columns="geometry").to_csv(p, index=False, encoding="utf-8-sig")
    elif a.format == "geojson":
        g.to_file(p, driver="GeoJSON")
    else:
        g.to_parquet(p)
    (ctx.out_dir / f"export_{a.result_id}.lineage.json").write_text(json.dumps({"lineage": ctx.results.lineage(a.result_id), "provenance": ctx.area.provenance}, indent=2, default=str))
    return ToolResult("export_dataset", {"path": str(p), "rows": len(g), "lineage": ctx.results.lineage(a.result_id)})


class GeoQueryArgs(_A):
    text: str = Field(min_length=3, max_length=1000)
    include_statuses: list[str] = Field(default_factory=lambda: ["VERIFIED_MATCH", "PROBABLE_MATCH"])


@tool("run_geoquery", GeoQueryArgs, doc="Evaluate a natural-language spatial constraint query with GeoAcquirer's deterministic evaluator on the local area.")
def run_geoquery(ctx: ToolContext, a: GeoQueryArgs) -> ToolResult:
    from geoacquirer.agent import RuleCompiler
    from geoacquirer.engine import Evaluator
    try:
        q = RuleCompiler().compile(a.text, ctx.area.bbox)
    except ValueError as e:
        errs = getattr(e, "errors", None)
        if callable(errs):  # pydantic ValidationError: report the field and the reason, not the internals
            first = e.errors()[0]
            detail = f"{'.'.join(str(x) for x in first.get('loc', ()))}: {first.get('msg')}"
        else:
            detail = str(e)
        raise ToolError(f"could not interpret the question as a supported spatial query ({detail})")
    try:
        res = Evaluator(ctx.area.geoacquirer_context(), q).run()
    except LookupError as e:
        raise ToolError(f"{e}. This data layer is not available for the active area (no authorized source integrated); "
                        "supply it with UrbanStore.ingest_local() or enable an authorized provider in GeoAcquirer.")
    counts = res["status"].value_counts().to_dict()
    keep = res[res["status"].isin(a.include_statuses)]
    rs = ctx.results.put(keep, kind="features", producer="run_geoquery", params={"text": a.text, "ir": json.loads(q.model_dump_json())})
    t = ToolResult("run_geoquery", {"status_counts": counts, "rows": len(keep), "defaults_applied": q.defaults_applied, "ambiguities": q.ambiguities,
                                    "unparsed_phrases": q.unparsed_phrases}, result_id=rs.id, provenance=ctx.provenance_of("poi", "road"))
    if q.unparsed_phrases:
        t.warnings.append("NOT APPLIED (unrecognised): " + "; ".join(q.unparsed_phrases))
    t.fact("observed", f"GeoAcquirer evaluator: {counts}. Defaults: {q.defaults_applied}.")
    return t


@tool("generate_evidence_report", SearchArgs, doc="Compile the facts and tables of this job into an HTML evidence report (see reporting.py).")
def generate_evidence_report(ctx: ToolContext, a: SearchArgs) -> ToolResult:  # filled in by the executor (needs the whole job)
    raise ToolError("generate_evidence_report is executed by the job executor with the full trace")


def call_tool(ctx: ToolContext, name: str, args: dict) -> ToolResult:
    if name not in TOOLS:
        raise ToolError(f"unknown tool {name!r}")
    model, fn, expensive, _ = TOOLS[name]
    try:
        parsed = model.model_validate(args)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"invalid arguments for {name}: {str(e)[:400]}")
    if expensive and name not in ctx.approved and name in ("classify_land_use", "request_geoacquirer_data"):
        pass  # the tool itself decides whether the concrete call is expensive (cache hits are free)
    t0 = time.perf_counter()
    try:
        r = fn(ctx, parsed)
    except KeyError as e:  # unknown result-set ids etc.: a user error, not a crash
        if "unknown result set" in str(e):
            raise ToolError(str(e).strip("'\""))
        raise
    r.seconds = round(time.perf_counter() - t0, 3)
    return r
