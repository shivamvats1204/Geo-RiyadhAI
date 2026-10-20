"""Independent re-verification of distance predicates.

The evaluator uses spatial indexes; this verifier deliberately does NOT. It recomputes every
distance by brute force (bounding-box prefilter + exact shapely distance) and checks that the
recorded evidence and the resulting status agree. Disagreement is a defect, not a rounding issue.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely

from ..engine.context import DataContext
from ..engine.evaluator import Evaluator, near_codes
from ..query_ir import GeoQuery, Op, Predicate, iter_predicates


@dataclass
class VerificationReport:
    checked_rows: int = 0
    checked_leaves: int = 0
    disagreements: list[dict] = field(default_factory=list)
    not_independently_checkable: list[str] = field(default_factory=list)
    max_abs_distance_error_m: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.disagreements

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checked_rows": self.checked_rows, "checked_leaves": self.checked_leaves,
                "n_disagreements": len(self.disagreements), "disagreements": self.disagreements[:20],
                "not_independently_checkable": self.not_independently_checkable,
                "max_abs_distance_error_m": round(self.max_abs_distance_error_m, 4)}


def _brute_min_distance(geom, others: np.ndarray, bounds: np.ndarray, cutoff: float) -> float:
    gx0, gy0, gx1, gy1 = geom.bounds
    keep = ((bounds[:, 0] - cutoff <= gx1) & (bounds[:, 2] + cutoff >= gx0) &
            (bounds[:, 1] - cutoff <= gy1) & (bounds[:, 3] + cutoff >= gy0))
    if not keep.any():
        return np.inf
    return float(np.min(shapely.distance(geom, others[keep])))


def _objects_for(ev: Evaluator, p: Predicate):
    if p.op == Op.AT_INTERSECTION:
        j = ev.ctx.layers["junction"]
        kinds = {"four_way": {"four_way"}, "t_junction": {"t_junction"}, "any": {"four_way", "t_junction", "complex"}}[p.junction_kind]
        return j[j["kind"].isin(kinds)]
    return ev._object_gdf(p.object, p)


def verify_results(results: gpd.GeoDataFrame, ctx: DataContext, query: GeoQuery, sample_rejected: int = 300,
                   seed: int = 7) -> VerificationReport:
    rep = VerificationReport()
    ev = Evaluator(ctx, query)  # only used for object-layer resolution helpers
    leaves = [p for p in iter_predicates(query.constraint)]
    checkable = [p for p in leaves if p.op in (Op.AT_INTERSECTION, Op.NEAR, Op.FARTHER_THAN) or (p.op == Op.NOT_WITHIN and p.distance_m)]
    for p in leaves:
        if p not in checkable:
            rep.not_independently_checkable.append(f"{p.op.value} (no independent brute-force path)")
    if not checkable:
        return rep
    rng = np.random.default_rng(seed)
    rejected_idx = results.index[results["status"] == "REJECTED"].to_numpy()
    pick_rej = rng.choice(rejected_idx, min(sample_rejected, len(rejected_idx)), replace=False) if len(rejected_idx) else []
    rows = list(results.index[results["status"] != "REJECTED"]) + list(pick_rej)
    # leaf index within evidence list follows evaluation order == iter_predicates order
    cache = {}
    for li, p in enumerate(leaves):
        if p not in checkable:
            continue
        obj = _objects_for(ev, p)
        if obj is None:
            continue
        cache[li] = (np.asarray(obj.geometry.values), obj.geometry.bounds.to_numpy())
    for ridx in rows:
        row = results.loc[ridx]
        evid = json.loads(row["evidence_json"])
        rep.checked_rows += 1
        for li, p in enumerate(leaves):
            if li not in cache:
                continue
            others, bnds = cache[li]
            u = query.position_uncertainty_m if row.geometry.geom_type == "Point" else 0.0
            D = float(p.distance_m)
            d = _brute_min_distance(row.geometry, others, bnds, D + u + 1e-6)
            code = int(near_codes(np.array([d]), D, np.array([u]))[0])
            if p.op in (Op.FARTHER_THAN, Op.NOT_WITHIN):
                code = 4 - code
            rec = next((e for e in evid if e["id"] == f"c{li + 1}"), None)
            rep.checked_leaves += 1
            if rec is None:
                rep.disagreements.append({"row": str(ridx), "leaf": li + 1, "problem": "evidence missing"})
                continue
            from ..engine.evaluator import Code
            if Code[rec["result"]].value != code:
                rep.disagreements.append({"row": str(ridx), "leaf": li + 1, "recorded": rec["result"],
                                          "recomputed": Code(code).name, "distance_m": None if not np.isfinite(d) else round(d, 2)})
            rd = rec.get("nearest_distance_m")
            if rd is not None and np.isfinite(d):
                rep.max_abs_distance_error_m = max(rep.max_abs_distance_error_m, abs(rd - d))
    return rep
