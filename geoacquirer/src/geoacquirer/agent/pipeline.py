"""GeoAcquirer agent: explicit state machine from natural language to verified, exported results.

States: COMPILE -> INVENTORY -> PLAN -> ACQUIRE -> BUILD_CONTEXT -> EVALUATE -> ENRICH -> VERIFY -> EXPORT.
Every stage is checkpointed in `<run_dir>/state.json`; acquisition results are additionally cached
on disk, so an interrupted run resumes without refetching. The LLM (if any) is only used in COMPILE.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from ..engine import DataContext, Evaluator
from ..entity_resolution import dedupe_pois, resolve_poi_to_building, resolve_poi_to_roads
from ..exports import export_all, flatten_results
from ..observability import AuditLog
from ..ontology import category_mask
from ..providers.base import FeatureBatch
from ..providers.local import LocalProvider
from ..providers.overture import OvertureProvider, connectors_frame, segments_frame
from ..providers.registry import score_sources
from ..query_ir import EntityType, GeoQuery, Op, iter_predicates
from ..security import BudgetExceeded, BudgetTracker
from ..spatial import to_metric, utm_crs_for_bbox
from ..storage import LayerCache
from ..verification import check_layer, verify_results
from .compiler import RuleCompiler

STAGES = ["COMPILE", "INVENTORY", "PLAN", "ACQUIRE", "BUILD_CONTEXT", "EVALUATE", "ENRICH", "VERIFY", "EXPORT"]
CONTEXT_MARGIN_DEG = 0.01  # ~1.1 km so edge POIs still see roads, junctions and competitors


def normalize_parking(gdf: gpd.GeoDataFrame, source: str) -> gpd.GeoDataFrame:
    """Common parking schema: poi_id, name, category, category_path, capacity, access, is_paid, source_dataset (+ originals)."""
    g = gdf.copy()
    if g.empty:
        for c in ("poi_id", "name", "category", "category_path", "capacity", "access", "is_paid", "source_dataset"):
            g[c] = None
        return g
    if source == "urbi":
        g["poi_id"] = g["parking_id"].astype(str)
        addr = g["address"].where(g["address"].astype(str).str.len() > 0)
        g["name"] = [a if isinstance(a, str) else f"Parking {i}" for a, i in zip(addr, g["poi_id"])]
        g["category"] = g["subtype"].fillna("parking") if "subtype" in g else "parking"
        g["source_dataset"] = "urbi"
    else:  # overture: keep POIs whose category path contains a parking token
        keep = category_mask(g["category_path"], ["parking"])
        g = g[keep].copy()
        for c in ("capacity", "access", "is_paid"):
            if c not in g:
                g[c] = None
        g["source_dataset"] = "overture:" + g.get("source_dataset", "places").astype(str)
    g["category_path"] = [["travel_and_transportation", "parking"]] * len(g) if source == "urbi" else g["category_path"]
    return g.reset_index(drop=True)


class Cancelled(RuntimeError):
    pass


@dataclass
class LayerNeed:
    layer: str  # poi | road | connector | building | parcel
    required: bool
    reason: str


@dataclass
class PlanStep:
    layer: str
    provider: Optional[str]
    required: bool
    reason: str
    est_seconds: float
    scores: list = field(default_factory=list)
    note: str = ""


@dataclass
class AcquisitionPlan:
    steps: list[PlanStep]
    est_total_seconds: float
    gaps: list[str]

    def to_dict(self):
        return {"steps": [s.__dict__ for s in self.steps], "est_total_seconds": self.est_total_seconds, "gaps": self.gaps}


@dataclass
class RunResult:
    run_dir: Path
    query: GeoQuery
    counts: dict
    files: dict
    manifest: dict
    flat: gpd.GeoDataFrame
    summary: list[str]
    plan: AcquisitionPlan


def needs_for(q: GeoQuery) -> list[LayerNeed]:
    needs: dict[str, LayerNeed] = {}

    def need(layer, req, why):
        if layer not in needs or (req and not needs[layer].required):
            needs[layer] = LayerNeed(layer, req, why)

    t = q.primary()
    if t.entity_type == EntityType.POI:
        need("poi", True, f"target entities ({t.entity_type.value})")
    if t.entity_type == EntityType.PARKING:
        need("parking", True, "target entities (parking facilities)")
    if t.entity_type == EntityType.BUILDING:
        need("building", True, "target entities")
    if t.entity_type == EntityType.PARCEL:
        need("parcel", True, "target entities")
    for p in iter_predicates(q.constraint):
        if p.op in (Op.AT_INTERSECTION, Op.CONNECTED_TO, Op.DISTANCE_BY_ROAD):
            need("road", True, f"{p.op.value} needs the road network")
            need("connector", True, f"{p.op.value} needs explicit connectors")
        if p.op == Op.HAS_FRONTAGE_ON:
            need("road", True, "frontage needs road geometry")
        if p.op == Op.INSIDE_BUILDING:
            need("building", True, "INSIDE_BUILDING")
        if p.op == Op.INSIDE_PARCEL:
            need("parcel", True, "INSIDE_PARCEL")
        if p.object is not None:
            et = p.object.entity_type
            if et in (EntityType.ROAD, EntityType.INTERSECTION):
                need("road", True, "road object")
                need("connector", True, "road object topology")
            elif et == EntityType.PARKING:
                need("parking", True, "object parking facilities")
            elif et == EntityType.POI:
                need("poi", True, "object POIs")
            elif et == EntityType.BUILDING:
                need("building", True, "object buildings")
            elif et == EntityType.PARCEL:
                need("parcel", True, "object parcels")
    attrs = set(q.output.requested_attributes)
    if "building" in attrs:
        need("building", False, "requested output attribute: associated building")
    if "parcel" in attrs:
        need("parcel", False, "requested output attribute: associated parcel")
    if "nearby_roads" in attrs:
        need("road", False, "requested output attribute: nearby roads")
        need("connector", False, "requested output attribute: nearby roads")
    return list(needs.values())


def geocode_bbox(name: str, timeout: float = 20.0):
    """Resolve a place name to a bbox via Nominatim (1 request; honest UA; allowlisted)."""
    import requests
    from ..security import check_domain
    check_domain("nominatim.openstreetmap.org")
    r = requests.get("https://nominatim.openstreetmap.org/search", params={"q": name, "format": "json", "limit": 1},
                     headers={"User-Agent": "geoacquirer/0.1 (research prototype)"}, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise LookupError(f"could not geocode {name!r}")
    s, n, w, e = (float(x) for x in js[0]["boundingbox"])
    return (w, s, e, n)


class GeoAcquirerAgent:
    def __init__(self, work_dir: Path = Path("runs"), *, local_dir: Optional[Path] = None, compiler=None,
                 cache: Optional[LayerCache] = None, approve_over_seconds: float = 3600.0,
                 approve: Optional[Callable[[AcquisitionPlan], bool]] = None, geocoder=geocode_bbox):
        self.work_dir = Path(work_dir)
        self.compiler = compiler or RuleCompiler()
        self.cache = cache or LayerCache()
        self.local = LocalProvider(local_dir) if local_dir else None
        self.approve_over_seconds = approve_over_seconds
        self.approve = approve
        self.geocoder = geocoder
        self.log: list[str] = []
        self.observed: dict[str, dict] = {}

    # ----------------------------------------------------------------- helpers
    def _log(self, msg: str):
        line = f"{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')} {msg}"
        self.log.append(line)
        if getattr(self, "audit", None):
            self.audit.event("log", message=msg)

    def _state(self, run_dir: Path, stage: str, **info):
        p = run_dir / "state.json"
        st = json.loads(p.read_text()) if p.exists() else {"completed": []}
        if stage not in st["completed"]:
            st["completed"].append(stage)
        st.update(info)
        p.write_text(json.dumps(st, indent=2, default=str))

    @staticmethod
    def _check(cancel: Optional[threading.Event]):
        if cancel is not None and cancel.is_set():
            raise Cancelled("run cancelled")

    # -------------------------------------------------------------------- plan
    def plan(self, q: GeoQuery) -> AcquisitionPlan:
        local_types = set()
        if self.local:
            for i in self.local.inventory:
                if i.entity_type:
                    local_types.add(i.entity_type)
        steps, gaps = [], []
        for need in needs_for(q):
            et = need.layer
            local_hit = self.local is not None and self.local.estimate_coverage(et, q.area.bbox).expected_features
            ranked = score_sources(et, {"local_files"} if local_hit else set(), observed=self.observed)
            elig = [s for s in ranked if s.eligible]
            if local_hit:
                pick = next((s for s in elig if s.provider_id == "local_files"), None)
            elif et == "parking":
                # a specialised, authorized parking source (rich attributes: capacity, access, schedule) beats POIs filtered by category
                pick = next((s for s in elig if s.provider_id == "urbi"), None) or next((s for s in elig if s.provider_id not in ("local_files",)), None)
            else:
                pick = next((s for s in elig if s.provider_id != "local_files"), None)
            if pick is None:
                dis = [s for s in ranked if not s.eligible]
                gaps.append(f"{et}: no authorized provider ({'; '.join(f'{s.provider_id}: {s.reason}' for s in dis) or 'none registered'})")
                steps.append(PlanStep(et, None, need.required, need.reason, 0.0, [s.__dict__ for s in ranked], "UNAVAILABLE"))
            else:
                reg_cost = 1.0 if pick.provider_id == "local_files" else 60.0 if pick.provider_id == "urbi" else 120.0
                steps.append(PlanStep(et, pick.provider_id, need.required, need.reason, reg_cost, [s.__dict__ for s in ranked[:3]]))
        total = sum(s.est_seconds for s in steps if s.provider != "local_files") / 2.0  # layers run in parallel
        return AcquisitionPlan(steps, total, gaps)

    # ---------------------------------------------------------------- acquire
    def _fetch_layer(self, step: PlanStep, bbox, budget: BudgetTracker) -> Optional[FeatureBatch]:
        if step.provider is None:
            return None
        if step.provider == "local_files":
            return self.local.fetch_features(step.layer, bbox, budget=budget)
        if step.provider == "urbi":
            from ..providers.urbi import UrbiProvider
            t0 = time.monotonic()
            key = LayerCache.key(p="urbi", t="parking", b=tuple(round(v, 6) for v in bbox), v=1)
            hit = self.cache.get(key, 30)
            if hit:  # cached copy of an earlier authorized acquisition
                from ..providers.base import Provenance
                gdf, meta = hit
                return FeatureBatch(gdf, Provenance("urbi", "parking", "live-api (cached)", meta.get("license", ""), "2GIS/Urbi", meta.get("retrieved_at", ""), bbox, 0, True, key))
            try:
                fb = UrbiProvider(max_workers=4).fetch_features("parking", bbox, budget=budget)
                fb.gdf = normalize_parking(fb.gdf, "urbi")
                self.cache.put(key, fb.gdf, {"retrieved_at": fb.provenance.retrieved_at, "license": fb.provenance.license})
                self.observed.setdefault("urbi", {"ok": 0, "fail": 0})["ok"] += 1
                return fb
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                self.observed.setdefault("urbi", {"ok": 0, "fail": 0})["fail"] += 1
                self._log(f"urbi parking failed after {time.monotonic()-t0:.0f}s: {type(e).__name__}: {str(e)[:200]}")
                return None
        if step.provider == "overture" and step.layer == "parking":
            prov = OvertureProvider(cache=self.cache)
            fb = prov.fetch_features("poi", bbox, budget=budget)
            fb.gdf = normalize_parking(fb.gdf, "overture")
            fb.provenance.entity_type = "parking"
            return fb
        if step.provider == "overture":
            prov = OvertureProvider(cache=self.cache)  # one DuckDB connection per thread
            t0 = time.monotonic()
            try:
                fb = prov.fetch_features(step.layer, bbox, budget=budget)
                self.observed.setdefault("overture", {"ok": 0, "fail": 0})["ok"] += 1
                return fb
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                self.observed.setdefault("overture", {"ok": 0, "fail": 0})["fail"] += 1
                self._log(f"overture {step.layer} failed after {time.monotonic()-t0:.0f}s: {type(e).__name__}: {str(e)[:200]}")
                return None
        raise NotImplementedError(step.provider)

    # --------------------------------------------------------------------- run
    def run(self, text: str, *, area_bbox=None, run_id: Optional[str] = None, cancel: Optional[threading.Event] = None,
            query: Optional[GeoQuery] = None, include_statuses: Optional[list[str]] = None, max_requests: Optional[int] = None) -> RunResult:
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log = []
        self.audit = AuditLog(run_dir / "audit.jsonl")
        self.audit.event("query_received", text=text)
        t_run = time.monotonic()

        # COMPILE
        self._check(cancel)
        q = query or self.compiler.compile(text, area_bbox)
        if max_requests:  # explicit, user-chosen request budget (Urbi scouting needs thousands of cell requests)
            q = q.model_copy(update={"budget": q.budget.model_copy(update={"max_requests": int(max_requests)})})
        if q.area.bbox is None:
            self._log(f"geocoding area {q.area.name!r}")
            q = q.model_copy(update={"area": q.area.model_copy(update={"bbox": self.geocoder(q.area.name)})})
        (run_dir / "geoquery.json").write_text(q.model_dump_json(indent=2))
        self._state(run_dir, "COMPILE")
        self._log(f"compiled query; defaults={q.defaults_applied}")

        # INVENTORY + PLAN
        plan = self.plan(q)
        self._state(run_dir, "PLAN", plan=plan.to_dict())
        for g in plan.gaps:
            self._log(f"GAP {g}")
        if plan.est_total_seconds > self.approve_over_seconds and not (self.approve and self.approve(plan)):
            raise PermissionError(f"estimated acquisition {plan.est_total_seconds:.0f}s exceeds approval threshold; approval required")
        budget = BudgetTracker(q.budget.max_requests, q.budget.max_seconds, q.budget.max_features)

        # ACQUIRE (layers in parallel)
        self._check(cancel)
        w, s, e, n = q.area.bbox
        ctx_bbox = (round(w - CONTEXT_MARGIN_DEG, 6), round(s - CONTEXT_MARGIN_DEG, 6),
                    round(e + CONTEXT_MARGIN_DEG, 6), round(n + CONTEXT_MARGIN_DEG, 6))
        batches: dict[str, FeatureBatch] = {}
        t_acq = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {st.layer: ex.submit(self._fetch_layer, st, ctx_bbox, budget) for st in plan.steps if st.provider}
            for layer, f in futs.items():
                fb = f.result()
                if fb is not None:
                    batches[layer] = fb
                    self._log(f"acquired {layer}: {len(fb.gdf)} features from {fb.provenance.provider_id} "
                              f"({'cache' if fb.provenance.from_cache else 'live'}) release={fb.provenance.dataset_version}")
        self._state(run_dir, "ACQUIRE", acquire_seconds=round(time.monotonic() - t_acq, 1), budget=budget.snapshot())

        # BUILD_CONTEXT
        self._check(cancel)
        crs = utm_crs_for_bbox(ctx_bbox)
        ctx = DataContext(crs=crs)
        ctx.area = to_metric(gpd.GeoDataFrame(geometry=[box(*q.area.bbox)], crs=4326), crs).geometry.iloc[0]
        retrieved = ""
        for layer, fb in batches.items():
            ctx.provenance[layer] = fb.provenance.to_dict()
            retrieved = retrieved or fb.provenance.retrieved_at
            for wmsg in fb.warnings:
                self._log("warning: " + wmsg)
        if "poi" in batches:
            poi_m = to_metric(batches["poi"].gdf, crs)
            if "source_dataset" in poi_m.columns and poi_m["source_dataset"].nunique() > 1 and "poi_id" in poi_m.columns:
                before = len(poi_m)
                poi_m, merge_log = dedupe_pois(poi_m)
                ctx.notes.append(f"cross-source POI de-duplication merged {before - len(poi_m)} duplicate records")
                self._log(ctx.notes[-1])
                merge_log.to_csv(run_dir / "poi_merge_log.csv", index=False)
            ctx.layers["poi"] = poi_m
        if "parking" in batches:
            ctx.layers["parking"] = to_metric(batches["parking"].gdf, crs)
        if "building" in batches:
            ctx.layers["building"] = to_metric(batches["building"].gdf, crs)
        if "parcel" in batches:
            ctx.layers["parcel"] = to_metric(batches["parcel"].gdf, crs)
        if "road" in batches and "connector" in batches:
            seg = segments_frame(to_metric(batches["road"].gdf, crs))
            con = connectors_frame(to_metric(batches["connector"].gdf, crs))
            ctx.add_roads(seg, con)
            self._log(ctx.notes[-1])
        self._state(run_dir, "BUILD_CONTEXT")

        # EVALUATE
        self._check(cancel)
        ev = Evaluator(ctx, q)
        res = ev.run()
        counts = res["status"].value_counts().to_dict()
        self._log(f"evaluated {len(res)} candidates -> {counts}")
        self._state(run_dir, "EVALUATE", counts=counts)

        # ENRICH (only rows that can be exported, to stay cheap)
        self._check(cancel)
        keep_mask = res["status"].isin(q.output.include_statuses if include_statuses is None else include_statuses)
        enrich_idx = res.index[keep_mask]
        res["building_id_resolved"] = None
        res["building_match"] = "not_evaluated"
        res["building_distance_m"] = np.nan
        res["parcel_id_resolved"] = None
        res["parcel_match"] = "not_evaluated"
        res["nearby_roads"] = ""
        if len(enrich_idx):
            sub = res.loc[enrich_idx]
            b = resolve_poi_to_building(sub, ctx.layers.get("building"))
            res.loc[enrich_idx, "building_id_resolved"] = b["building_id"].values
            res.loc[enrich_idx, "building_match"] = b["building_match"].values
            res.loc[enrich_idx, "building_distance_m"] = b["building_distance_m"].values
            if "parcel" in ctx.layers:
                pcl = ctx.layers["parcel"]
                if pcl.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                    sj = gpd.sjoin(sub[["geometry"]], pcl[["parcel_id", "geometry"]], how="left", predicate="within")
                    sj = sj[~sj.index.duplicated()]
                    res.loc[enrich_idx, "parcel_id_resolved"] = sj["parcel_id"].values
                    res.loc[enrich_idx, "parcel_match"] = np.where(sj["parcel_id"].notna(), "contained_in_parcel_polygon", "none")
                else:
                    res.loc[enrich_idx, "parcel_match"] = "parcel_layer_has_centroids_only"
            else:
                res.loc[enrich_idx, "parcel_match"] = "unavailable_no_authorized_parcel_source"
            res.loc[enrich_idx, "nearby_roads"] = resolve_poi_to_roads(sub, ctx.layers.get("road")).values
        self._state(run_dir, "ENRICH")

        # VERIFY
        self._check(cancel)
        data_checks = []
        id_cols = {"poi": "poi_id", "parking": "poi_id", "building": "building_id", "road": "road_id"}
        attrs = {"parking": ["name", "capacity", "access", "is_paid"], "poi": ["name", "category"]}
        for layer, fb in batches.items():
            if layer in id_cols and layer in ("poi", "parking", "building"):
                data_checks.append(check_layer(layer, fb.gdf, q.area.bbox, id_cols[layer], attrs.get(layer), tolerance_deg=CONTEXT_MARGIN_DEG if layer != "parking" else 0.0))
        for c in data_checks:
            self._log(f"data check {c['layer']}: rows={c['rows']} problems={c.get('problems')}")
        vrep = verify_results(res, ctx, q)
        self._log(f"independent verification: ok={vrep.ok} leaves={vrep.checked_leaves} max_err={vrep.max_abs_distance_error_m:.4f} m")
        self._state(run_dir, "VERIFY", verification=vrep.to_dict())

        # EXPORT
        self._check(cancel)
        release = next((fb.provenance.dataset_version for fb in batches.values()), "")
        run_info = {"poi_source": batches["poi"].provenance.provider_id if "poi" in batches else None,
                    "release": release, "retrieved_at": retrieved}
        flat = flatten_results(res, q, run_info)
        map_layers = {}
        if ctx.has("road"):
            r = ctx.layers["road"]
            map_layers["highways"] = r[r["class"].isin(["motorway", "trunk"])].to_crs(4326)
        if ctx.has("junction"):
            j = ctx.layers["junction"]
            map_layers["junctions"] = j[j.kind == "four_way"].to_crs(4326)
        coverage = self._coverage_notes(q, ctx, plan, batches)
        manifest = {
            "run_id": run_id, "provenance": ctx.provenance, "coverage_notes": coverage, "plan": plan.to_dict(),
            "budget": budget.snapshot(), "log": list(self.log), "verification": vrep.to_dict(), "data_checks": data_checks,
            "topology": {"notes": ctx.notes, "junction_config": ctx.junction_cfg.__dict__ | {"extra_excluded_classes": []}},
            "wall_seconds": round(time.monotonic() - t_run, 1),
            "reproduce": f'geoacquirer run "{q.raw_text}" --bbox {" ".join(str(x) for x in q.area.bbox)} --out <dir>   # same Overture release: {release}',
        }
        files = export_all(flat, q, run_dir, q.output.formats, manifest, map_layers, include_statuses)
        self._state(run_dir, "EXPORT", files={k: str(v) for k, v in files.items()})
        summary = self._summary(q, flat, plan, coverage, vrep)
        return RunResult(run_dir, q, {k: int(v) for k, v in counts.items()}, files, manifest, flat, summary, plan)

    # ------------------------------------------------------------ explanations
    def _coverage_notes(self, q, ctx, plan, batches) -> list[str]:
        notes = []
        for g in plan.gaps:
            notes.append("MISSING DATA: " + g)
        for layer in ("poi", "road", "connector", "building"):
            if layer in batches and batches[layer].gdf.empty:
                notes.append(f"{layer} layer returned no features for the requested area")
        if "poi" in batches:
            notes.append("POI completeness is unknown: Overture places aggregates several sources and under-represents small shops; "
                         "a missing shop in results does not mean it does not exist.")
            notes.append("POI coordinates are points, not entrances. Distances are point-based with the stated positional uncertainty; "
                         "near-threshold cases are PROBABLE_MATCH or INSUFFICIENT_EVIDENCE rather than VERIFIED.")
        if "parcel" not in batches:
            notes.append("No authorized parcel source: parcel association is reported as unavailable (never inferred).")
        if "building" in batches:
            notes.append("Building footprints are OSM/ML-derived, not cadastral; POI-to-building links are evidence-labelled, not guaranteed.")
        if ctx.has("junction"):
            ex = len(ctx.layers.get("excluded_crossings", []))
            notes.append(f"{ex} geometric road crossings without a shared connector were excluded as grade-separated/disconnected (not junctions).")
            notes.append("Divided carriageways within %.0f m are merged into one junction; ramps never count as approaches." % ctx.junction_cfg.merge_radius_m)
        return notes

    def _summary(self, q, flat, plan, coverage, vrep) -> list[str]:
        c = flat["status"].value_counts().to_dict()
        out = []
        if q.unparsed_phrases:
            out.append("NOT APPLIED (could not be interpreted as a supported constraint): " + "; ".join(q.unparsed_phrases)
                       + ". Results are NOT filtered by these phrases.")
        out += [f"{c.get('VERIFIED_MATCH', 0)} verified, {c.get('PROBABLE_MATCH', 0)} probable, "
               f"{c.get('INSUFFICIENT_EVIDENCE', 0)} insufficient-evidence, {c.get('REJECTED', 0)} rejected candidates."]
        for step in plan.steps:
            if step.provider is None:
                out.append(f"UNSATISFIED: {step.layer} ({'required' if step.required else 'optional'}) had no authorized source.")
        out.append("Independent re-verification of distance predicates: " + ("all agreed." if vrep.ok else f"{len(vrep.disagreements)} DISAGREEMENTS - see manifest."))
        out.extend(coverage[:4])
        return out
