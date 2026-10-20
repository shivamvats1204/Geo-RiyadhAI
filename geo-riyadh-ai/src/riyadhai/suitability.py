"""Configurable multi-criteria site suitability (heuristic ranking, NOT a validated business-success model).

Pipeline: hard eligibility filters -> robust min-max normalization -> weighted sum -> ranking with spatial separation ->
sensitivity analysis (one-at-a-time and Monte-Carlo weights) -> per-criterion contribution explanations computed from the
numbers (no language model involved).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# criterion -> (feature column, direction, human label, unit)
CRITERIA = {
    "demand": ("network_demand_800m", +1, "reachable residential-like built area within 800 m by road", "m2"),
    "junction_access": ("dist_to_four_way_m", -1, "distance to nearest four-way junction", "m"),
    "road_visibility": ("road_class_score", +1, "class of the street frontage (primary highest)", "score 0-1"),
    "competition": ("competitors_500m", -1, "competing stores within 500 m", "count"),
    "agglomeration": ("retail_pois_500m", +1, "other retail POIs within 500 m", "count"),
    "parking": ("parking_pois_300m", +1, "mapped parking facilities within 300 m", "count"),
    "transit": ("dist_to_metro_m", -1, "distance to the nearest metro station/entrance (Urbi)", "m"),
}
DEFAULT_WEIGHTS = {"demand": 0.30, "junction_access": 0.20, "road_visibility": 0.10, "competition": 0.25, "agglomeration": 0.10, "parking": 0.05, "transit": 0.0}


@dataclass
class HardFilters:
    min_highway_distance_m: float = 60.0
    min_competitor_distance_m: float = 100.0
    max_junction_distance_m: Optional[float] = None
    allowed_landuse: Optional[list] = None  # applied only if parcel landuse is known; else reported UNVERIFIED


@dataclass
class SuitabilityResult:
    ranked: pd.DataFrame
    top: pd.DataFrame
    weights: dict
    filter_report: dict
    sensitivity: dict
    notes: list[str] = field(default_factory=list)


def _norm(col: pd.Series, direction: int) -> pd.Series:
    if col.isna().all():
        return pd.Series(np.zeros(len(col)), index=col.index)  # criterion has no data: contributes nothing
    lo, hi = np.nanpercentile(col, 5), np.nanpercentile(col, 95)
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(col)), index=col.index)
    z = ((col.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)
    return z if direction > 0 else 1.0 - z


def normalize(features: pd.DataFrame, criteria=CRITERIA) -> pd.DataFrame:
    return pd.DataFrame({k: _norm(features[c], d) for k, (c, d, _, _) in criteria.items()}, index=features.index)


def apply_filters(features: pd.DataFrame, flt: HardFilters) -> tuple[pd.Series, dict]:
    ok = pd.Series(True, index=features.index)
    rep = {"candidates": int(len(features))}
    step = features["dist_to_highway_m"] > flt.min_highway_distance_m
    rep["removed_highway_%dm" % flt.min_highway_distance_m] = int((ok & ~step).sum()); ok &= step
    step = features["nearest_competitor_m"] >= flt.min_competitor_distance_m
    rep["removed_competitor_within_%dm" % flt.min_competitor_distance_m] = int((ok & ~step).sum()); ok &= step
    if flt.max_junction_distance_m is not None:
        step = features["dist_to_four_way_m"] <= flt.max_junction_distance_m
        rep["removed_beyond_junction_%dm" % flt.max_junction_distance_m] = int((ok & ~step).sum()); ok &= step
    step = features["network_demand_800m"].notna()
    rep["removed_not_snappable_to_road_network"] = int((ok & ~step).sum()); ok &= step
    if flt.allowed_landuse:
        known = features["landuse_zoning"] != "UNVERIFIED"
        if known.any():
            step = ~known | features["landuse_zoning"].isin(flt.allowed_landuse)
            rep["removed_landuse_not_allowed"] = int((ok & ~step).sum()); ok &= step
        else:
            rep["landuse_filter"] = "NOT APPLIED: no parcel/zoning layer, so zoning is UNVERIFIED for every candidate"
    rep["eligible"] = int(ok.sum())
    return ok, rep


def score(features: pd.DataFrame, weights: dict, criteria=CRITERIA) -> pd.DataFrame:
    w = pd.Series({k: weights.get(k, 0.0) for k in criteria}, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / w.sum()
    n = normalize(features, criteria)
    contrib = n * w
    out = features.copy()
    for k in criteria:
        out[f"norm_{k}"] = n[k]
        out[f"contrib_{k}"] = contrib[k]
    out["score"] = contrib.sum(axis=1)
    return out


def pick_top(scored: pd.DataFrame, cells_xy: pd.DataFrame, k: int, min_separation_m: float) -> pd.DataFrame:
    """Greedy non-maximum suppression so the k recommendations are not neighbouring cells of one hotspot."""
    order = scored.sort_values("score", ascending=False)
    chosen = []
    xy = cells_xy.loc[order.index, ["x", "y"]].to_numpy()
    for pos, idx in enumerate(order.index):
        if all(np.hypot(xy[pos, 0] - cells_xy.at[c, "x"], xy[pos, 1] - cells_xy.at[c, "y"]) >= min_separation_m for c in chosen):
            chosen.append(idx)
        if len(chosen) == k:
            break
    return scored.loc[chosen]


def explain(row: pd.Series, weights: dict, population: pd.DataFrame, criteria=CRITERIA) -> list[str]:
    """Sentences built from computed values; each states the value, the area median and the weighted contribution."""
    lines = []
    tot = sum(weights.get(k, 0.0) for k in criteria) or 1.0
    ranked = sorted(criteria, key=lambda k: -row[f"contrib_{k}"])
    for k in ranked:
        col, d, label, unit = criteria[k]
        med = float(population[col].median())
        v = row[col]
        lines.append(f"{k}: {label} = {v:.1f} {unit} (eligible-cell median {med:.1f}); contributes {row['contrib_' + k]:.3f} of score {row['score']:.3f} (weight {weights.get(k, 0) / tot:.2f})")
    return lines


def sensitivity(features: pd.DataFrame, base_weights: dict, k: int, n_mc: int = 400, seed: int = 0, criteria=CRITERIA) -> dict:
    """Rank stability under weight perturbation. All numbers are measured on the supplied candidates.

    Normalization does not depend on the weights, so it is computed once; each perturbed score is a matrix-vector product and the
    Spearman correlation is the Pearson correlation of ranks.
    """
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    keys = list(criteria)
    N = normalize(features, criteria)[keys].to_numpy()

    def scores(w: dict) -> np.ndarray:
        v = np.array([max(w.get(k, 0.0), 0.0) for k in keys], dtype=float)
        if v.sum() <= 0:
            raise ValueError("weights must sum to a positive value")
        return N @ (v / v.sum())

    base = scores(base_weights)
    base_rank = rankdata(base)
    base_top = set(np.argsort(-base, kind="stable")[:k])

    def rho(s: np.ndarray) -> float:
        r = rankdata(s)
        if np.ptp(r) == 0 or np.ptp(base_rank) == 0:
            return 1.0 if np.array_equal(r, base_rank) else 0.0  # constant scores: correlation undefined, avoid NaN
        return float(np.corrcoef(base_rank, r)[0, 1])

    oat = {}
    for key in keys:
        for f in (0.5, 1.5):
            w = dict(base_weights)
            w[key] = w.get(key, 0.0) * f
            s = scores(w)
            oat[f"{key} x{f}"] = {"spearman_vs_base": round(rho(s), 4), "top_k_overlap": len(base_top & set(np.argsort(-s, kind="stable")[:k])) / k}
    alpha = np.array([max(base_weights.get(c, 0.0), 1e-3) for c in keys]) * 20.0  # Dirichlet centred on the base weights
    freq = np.zeros(len(features))
    rhos = []
    for _ in range(n_mc):
        w = dict(zip(keys, rng.dirichlet(alpha)))
        s = scores(w)
        freq[np.argsort(-s, kind="stable")[:k]] += 1
        rhos.append(rho(s))
    return {"one_at_a_time": oat, "monte_carlo": {"n": n_mc, "dirichlet_concentration": 20.0, "mean_spearman": round(float(np.mean(rhos)), 4),
                                                  "min_spearman": round(float(np.min(rhos)), 4)},
            "top_k_frequency": pd.Series(freq / n_mc, index=features.index), "min_top_k_overlap_oat": min(v["top_k_overlap"] for v in oat.values())}


def rank_sites(features: pd.DataFrame, cells_xy: pd.DataFrame, k: int = 5, weights: Optional[dict] = None,
               filters: Optional[HardFilters] = None, min_separation_m: float = 800.0, n_mc: int = 400) -> SuitabilityResult:
    weights = dict(weights or DEFAULT_WEIGHTS)
    filters = filters or HardFilters()
    ok, rep = apply_filters(features, filters)
    elig = features[ok].copy()
    notes = ["Scores are heuristic suitability rankings from the chosen criteria and weights; they are not validated predictions of sales or business success.",
             "Candidates are grid cells, not parcels: land availability, ownership and zoning are UNVERIFIED unless a parcel layer was supplied."]
    if len(elig) == 0:
        return SuitabilityResult(elig, elig, weights, rep, {}, notes + ["no eligible candidates after hard filters"])
    sc = score(elig, weights)
    sc["rank_all"] = sc["score"].rank(ascending=False, method="first").astype(int)
    top = pick_top(sc, cells_xy, min(k, len(sc)), min_separation_m)
    sens = sensitivity(elig, weights, min(k, len(elig)), n_mc=n_mc)
    sc["top_k_frequency"] = sens["top_k_frequency"]
    top = top.assign(top_k_frequency=sc.loc[top.index, "top_k_frequency"])
    top["explanation"] = [" | ".join(explain(r, weights, elig)) for _, r in top.iterrows()]
    sens = {kk: v for kk, v in sens.items() if kk != "top_k_frequency"}
    return SuitabilityResult(sc.sort_values("score", ascending=False), top.sort_values("score", ascending=False), weights, rep, sens, notes)
