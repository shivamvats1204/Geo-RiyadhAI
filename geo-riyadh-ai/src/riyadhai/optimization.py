"""Facility-location optimization on the road network: greedy, truncated p-median, maximum coverage (OR-Tools CP-SAT).

Model (truncated, improvement form)
-----------------------------------
Demand nodes d carry weights w_d. Each has a current cost b_d = min(network distance to its nearest *existing* facility, CAP).
Opening candidate j serves d at cost c_dj (network distance incl. snapping offsets) only if c_dj < b_d. We choose p candidates
to minimize sum_d w_d * min(b_d, min_{j open} c_dj) (p-median) or to maximize newly covered weight within radius R (max coverage).
Distances are road-network distances (undirected, metres), not travel times.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import geopandas as gpd

from .store import AreaDataset

CAP_M = 2000.0


@dataclass
class FacilityProblem:
    demand_nodes: list            # node ids
    demand_w: np.ndarray          # weights per demand node
    demand_snap: np.ndarray       # snapping offset to node (m)
    base_cost: np.ndarray         # current cost per demand node (existing facilities, capped)
    cand_ids: list
    cand_nodes: list
    cand_snap: np.ndarray
    pairs: list                   # (demand_idx, cand_idx, cost_m)
    radius_m: float
    n_existing: int
    meta: dict = field(default_factory=dict)

    def total_weight(self) -> float:
        return float(self.demand_w.sum())

    def objective(self, chosen: list[int]) -> float:
        """Weighted mean cost (m) after opening `chosen` candidates, evaluated from the pair table."""
        best = self.base_cost.copy()
        cs = set(chosen)
        for d, j, c in self.pairs:
            if j in cs and c < best[d]:
                best[d] = c
        return float((best * self.demand_w).sum() / self.demand_w.sum())

    def coverage(self, chosen: list[int]) -> float:
        cov = self.base_cost <= self.radius_m
        cs = set(chosen)
        for d, j, c in self.pairs:
            if j in cs and c <= self.radius_m:
                cov[d] = True
        return float(self.demand_w[cov].sum() / self.demand_w.sum())


def build_problem(ds: AreaDataset, demand: gpd.GeoDataFrame, weight: Optional[np.ndarray], candidates: gpd.GeoDataFrame,
                  existing: Optional[gpd.GeoDataFrame], radius_m: float = 600.0, cap_m: float = CAP_M,
                  max_snap_m: float = 200.0) -> FacilityProblem:
    net = ds.network()
    G = net.graph
    if len(demand) == 0:
        raise ValueError("no demand points: cannot build a facility-location problem")
    dn, dsn = net.snap(demand.geometry.centroid, max_snap_m)
    w = np.ones(len(demand)) if weight is None else np.asarray(weight, float)
    agg: dict[str, list] = {}
    for n, s, wi in zip(dn, dsn, w):
        if n is None:
            continue
        a = agg.setdefault(n, [0.0, 0.0])
        a[0] += wi
        a[1] += wi * s
    nodes = list(agg)
    dw = np.array([agg[n][0] for n in nodes])
    dsnap = np.array([agg[n][1] / agg[n][0] for n in nodes])
    didx = {n: i for i, n in enumerate(nodes)}
    base = np.full(len(nodes), cap_m)
    n_exist = 0
    if existing is not None and len(existing):
        en, es = net.snap(existing.geometry, max_snap_m)
        for n, s in zip(en, es):
            if n is None:
                continue
            n_exist += 1
            reach = nx.single_source_dijkstra_path_length(G, n, cutoff=cap_m, weight="weight")
            for k, d in reach.items():
                if k in didx:
                    i = didx[k]
                    base[i] = min(base[i], d + s + dsnap[i])
    cn, cs = net.snap(candidates.geometry, max_snap_m)
    keep = [i for i, n in enumerate(cn) if n is not None]
    pairs, cache = [], {}
    cand_ids, cand_nodes, cand_snap = [], [], []
    for ci in keep:
        n = cn[ci]
        if n not in cache:
            cache[n] = nx.single_source_dijkstra_path_length(G, n, cutoff=cap_m, weight="weight")
        reach = cache[n]
        j = len(cand_ids)
        cand_ids.append(candidates.iloc[ci].get("cell_id", ci))
        cand_nodes.append(n)
        cand_snap.append(cs[ci])
        for k, d in reach.items():
            i = didx.get(k)
            if i is None:
                continue
            c = d + cs[ci] + dsnap[i]
            if c < base[i] and c <= cap_m:
                pairs.append((i, j, float(c)))
    return FacilityProblem(nodes, dw, dsnap, base, cand_ids, cand_nodes, np.array(cand_snap), pairs, radius_m, n_exist,
                           {"demand_points": int(len(demand)), "demand_nodes": len(nodes), "candidates": len(cand_ids), "pairs": len(pairs),
                            "cap_m": cap_m, "unsnapped_demand": int(sum(n is None for n in dn)), "unsnapped_candidates": int(len(cn) - len(keep))})


def dense_cost(prob: FacilityProblem) -> np.ndarray:
    C = np.full((len(prob.demand_w), len(prob.cand_ids)), np.inf)
    for d, j, c in prob.pairs:
        C[d, j] = c
    return C


def swap_local_search(prob: FacilityProblem, start: list[int], objective: str = "p_median", max_sweeps: int = 50) -> list[int]:
    """Teitz-Bart style interchange: repeatedly replace one chosen candidate by the best alternative while the objective improves."""
    C = dense_cost(prob)
    w = prob.demand_w
    base = prob.base_cost
    chosen = list(start)

    def value(best_cost: np.ndarray) -> float:
        if objective == "p_median":
            return float((w * best_cost).sum())  # lower is better
        return -float(w[best_cost <= prob.radius_m].sum())  # coverage: negate so lower is better

    for _ in range(max_sweeps):
        improved = False
        for pos in range(len(chosen)):
            others = [j for k, j in enumerate(chosen) if k != pos]
            bo = base.copy()
            if others:
                bo = np.minimum(bo, C[:, others].min(axis=1))
            cur = value(np.minimum(bo, C[:, chosen[pos]]))
            cand = np.minimum(bo[:, None], C)                     # D x J: cost if candidate j replaces the removed facility
            if objective == "p_median":
                vals = (w[:, None] * cand).sum(axis=0)
            else:
                vals = -(w[:, None] * (cand <= prob.radius_m)).sum(axis=0)
            vals[others] = np.inf                                 # cannot duplicate a kept facility
            j = int(np.argmin(vals))
            if vals[j] < cur - 1e-9:
                chosen[pos] = j
                improved = True
        if not improved:
            break
    return chosen


# ------------------------------------------------------------------ solvers
def greedy(prob: FacilityProblem, p: int, objective: str = "p_median") -> list[int]:
    """Classic greedy: repeatedly open the candidate with the best marginal gain."""
    best = prob.base_cost.copy()
    covered = best <= prob.radius_m
    by_cand: dict[int, list] = {}
    for d, j, c in prob.pairs:
        by_cand.setdefault(j, []).append((d, c))
    chosen: list[int] = []
    for _ in range(p):
        top, top_gain = None, 0.0
        for j, lst in by_cand.items():
            if j in chosen:
                continue
            if objective == "p_median":
                g = sum(prob.demand_w[d] * (best[d] - c) for d, c in lst if c < best[d])
            else:
                g = sum(prob.demand_w[d] for d, c in lst if c <= prob.radius_m and not covered[d])
            if g > top_gain:
                top, top_gain = j, g
        if top is None:
            break
        chosen.append(top)
        for d, c in by_cand[top]:
            if c < best[d]:
                best[d] = c
            if c <= prob.radius_m:
                covered[d] = True
    return chosen


def solve_cpsat(prob: FacilityProblem, p: int, objective: str = "p_median", time_limit_s: float = 60.0, workers: int = 8,
                forbidden: Optional[set] = None, hint: Optional[list[int]] = None) -> tuple[list[int], dict]:
    """CP-SAT with a tight "level" formulation of the truncated p-median.

    For demand d, candidates are sorted by cost c_1 <= ... <= c_m (all below the current cost b_d). Binary z_dk means "d is served at
    cost <= c_k" and requires an open candidate among those with cost <= c_k; the objective adds the cost step c_{k+1}-c_k for each
    level reached (c_{m+1} = b_d). This has a much stronger LP bound than the assignment (x_dj <= y_j) formulation.
    """
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    J = range(len(prob.cand_ids))
    y = {j: m.NewBoolVar(f"y{j}") for j in J if not (forbidden and prob.cand_ids[j] in forbidden)}
    m.Add(sum(y.values()) <= p)
    W = 100  # weight resolution
    obj_terms = []
    z_defs: list[tuple] = []   # (z var, candidate indices that make it true) for complete hints
    by_d: dict[int, list] = {}
    if objective == "p_median":
        for d, j, c in prob.pairs:
            if j in y:
                by_d.setdefault(d, []).append((c, j))
        for d, lst in by_d.items():
            lst.sort()
            levels: list[tuple[float, list[int]]] = []
            for c, j in lst:
                if levels and abs(levels[-1][0] - c) < 1e-6:
                    levels[-1][1].append(j)
                else:
                    levels.append((c, [j]))
            open_so_far: list[int] = []
            for k, (c, js) in enumerate(levels):
                open_so_far = open_so_far + js
                nxt = levels[k + 1][0] if k + 1 < len(levels) else prob.base_cost[d]
                z = m.NewBoolVar(f"z{d}_{k}")  # "some candidate with cost <= c_k is open"; nested sums make z monotone automatically
                m.Add(z <= sum(y[j] for j in open_so_far))
                z_defs.append((z, list(open_so_far)))
                obj_terms.append(int(round(prob.demand_w[d] * W * (nxt - c))) * z)  # telescoping: sum of steps from the first reached level = b_d - c_first
    else:
        for d, j, c in prob.pairs:
            if j in y and c <= prob.radius_m and prob.base_cost[d] > prob.radius_m:
                by_d.setdefault(d, []).append(j)
        for d, js in by_d.items():
            z = m.NewBoolVar(f"z{d}")
            m.Add(z <= sum(y[j] for j in set(js)))
            z_defs.append((z, list(set(js))))
            obj_terms.append(int(round(prob.demand_w[d] * W)) * z)
    if hint:  # a COMPLETE hint (y and every z) so the solver starts from the greedy solution instead of discarding it
        hs = set(hint)
        for j, v in y.items():
            m.AddHint(v, 1 if j in hs else 0)
        for z, js in z_defs:
            m.AddHint(z, 1 if any(j in hs for j in js) else 0)
    if obj_terms:
        m.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    t = time.perf_counter()
    status = solver.Solve(m)
    dt = time.perf_counter() - t
    chosen = [j for j, v in y.items() if solver.Value(v)] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else []
    bound = solver.BestObjectiveBound() / W if obj_terms else 0.0
    gain = solver.ObjectiveValue() / W if obj_terms and status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0.0
    info = {"status": solver.StatusName(status), "seconds": round(dt, 2), "best_bound_gain": round(bound, 1), "objective_gain": round(gain, 1),
            "optimality_gap_pct": round(100.0 * (bound - gain) / bound, 2) if bound > 0 else 0.0}
    return chosen, info


def prune_candidates(prob: FacilityProblem, max_candidates: int, objective: str = "p_median") -> FacilityProblem:
    """Keep the `max_candidates` candidates with the best single-facility gain (documented heuristic; the optimum is then exact only
    over the reduced candidate set)."""
    if len(prob.cand_ids) <= max_candidates:
        return prob
    gain = np.zeros(len(prob.cand_ids))
    for d, j, c in prob.pairs:
        if objective == "p_median":
            gain[j] += prob.demand_w[d] * (prob.base_cost[d] - c)
        elif c <= prob.radius_m and prob.base_cost[d] > prob.radius_m:
            gain[j] += prob.demand_w[d]
    keep = np.argsort(-gain)[:max_candidates]
    remap = {int(old): new for new, old in enumerate(keep)}
    pairs = [(d, remap[j], c) for d, j, c in prob.pairs if j in remap]
    meta = dict(prob.meta, candidates=len(keep), pairs=len(pairs), pruned_from=len(prob.cand_ids))
    return FacilityProblem(prob.demand_nodes, prob.demand_w, prob.demand_snap, prob.base_cost, [prob.cand_ids[i] for i in keep],
                           [prob.cand_nodes[i] for i in keep], prob.cand_snap[keep], pairs, prob.radius_m, prob.n_existing, meta)


def optimize(prob: FacilityProblem, p: int, objective: str = "p_median", time_limit_s: float = 60.0, n_random: int = 200, seed: int = 0,
             forbidden: Optional[set] = None, max_candidates: int = 400) -> dict:
    """Run greedy, CP-SAT (warm-started from greedy) and a random baseline; report objective values and improvements."""
    rng = np.random.default_rng(seed)
    prob = prune_candidates(prob, max_candidates, objective)
    t = time.perf_counter()
    g0 = greedy(prob, p, objective)
    t_greedy = time.perf_counter() - t
    t = time.perf_counter()
    g = swap_local_search(prob, g0, objective) if g0 else g0
    t_swap = time.perf_counter() - t
    sol, info = solve_cpsat(prob, p, objective, time_limit_s, forbidden=forbidden, hint=g)
    key = (lambda c: prob.objective(c)) if objective == "p_median" else (lambda c: -prob.coverage(c))
    info["improved_on_greedy_swap"] = bool(sol) and key(sol) < key(g) - 1e-9
    if not sol or key(sol) > key(g) + 1e-9:  # never report a solution worse than the best heuristic solution we already hold
        info["note"] = "CP-SAT returned no better solution than greedy+swap within the time limit; the greedy+swap solution is reported"
        sol = list(g)
    J = list(range(len(prob.cand_ids)))
    rnd_obj, rnd_cov = [], []
    for _ in range(n_random):
        pick = list(rng.choice(J, size=min(p, len(J)), replace=False))
        rnd_obj.append(prob.objective(pick)); rnd_cov.append(prob.coverage(pick))
    res = {
        "objective": objective, "p": p, "radius_m": prob.radius_m, "problem": prob.meta | {"existing_facilities": prob.n_existing},
        "before": {"mean_distance_m": round(float((prob.base_cost * prob.demand_w).sum() / prob.demand_w.sum()), 1),
                   "coverage_within_radius": round(float(prob.demand_w[prob.base_cost <= prob.radius_m].sum() / prob.demand_w.sum()), 4)},
        "greedy": {"selected": [prob.cand_ids[j] for j in g0], "mean_distance_m": round(prob.objective(g0), 1),
                   "coverage_within_radius": round(prob.coverage(g0), 4), "seconds": round(t_greedy, 2)},
        "greedy_swap": {"selected": [prob.cand_ids[j] for j in g], "mean_distance_m": round(prob.objective(g), 1),
                        "coverage_within_radius": round(prob.coverage(g), 4), "seconds": round(t_swap, 2)},
        "cpsat": {"selected": [prob.cand_ids[j] for j in sol], "mean_distance_m": round(prob.objective(sol), 1),
                  "coverage_within_radius": round(prob.coverage(sol), 4), **info},
        "random_baseline": {"n": n_random, "mean_distance_m_mean": round(float(np.mean(rnd_obj)), 1), "mean_distance_m_p5": round(float(np.percentile(rnd_obj, 5)), 1),
                            "coverage_mean": round(float(np.mean(rnd_cov)), 4), "coverage_p95": round(float(np.percentile(rnd_cov, 95)), 4)},
        "selected_idx": sol,
        "assumptions": ["road-network distance, undirected, no travel-time model", "demand weights are a proxy supplied by the caller",
                        "candidate cells are NOT verified buildable land: zoning, ownership and parcel size are unchecked",
                        f"demand beyond {prob.meta['cap_m']:.0f} m of any facility is capped at that distance"],
    }
    return res


def evaluate_sites(prob: FacilityProblem, chosen_ids: list) -> dict:
    """Score a user-proposed set of candidate ids (e.g. 'compare three proposed sites')."""
    idx = [prob.cand_ids.index(c) for c in chosen_ids if c in prob.cand_ids]
    return {"sites": chosen_ids, "mean_distance_m": round(prob.objective(idx), 1), "coverage_within_radius": round(prob.coverage(idx), 4)}
