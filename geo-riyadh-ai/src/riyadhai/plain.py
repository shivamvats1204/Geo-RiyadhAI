"""Plain-language summary of an analysis job, for people who are not GIS or data specialists.

Every sentence is built from numbers the tools computed (tables / summaries); nothing is invented here. The technical facts (with calculation ids)
stay available behind "technical details" in the UI. Output:
  headline        one sentence answer
  points          short bullet list (what to look at)
  cards           a few big numbers {label, value, hint}
  confidence      {level: high|medium|low, text}  how much to trust it, in words
  limits          what the data cannot tell you
  how_to_read     how to read the table / map
  next_questions  follow-ups the user can click
"""
from __future__ import annotations

from typing import Optional



def _name_of(row) -> str:
    n = row.get("name")
    return str(n) if n else str(row.get("name_ar") or "")


def district_of(ds, lon: float, lat: float) -> Optional[str]:
    d = ds.layers.get("district")
    if d is None or d.empty:
        return None
    from shapely.geometry import Point
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, ds.crs, always_xy=True).transform(lon, lat)
    p = Point(x, y)
    hit = d[d.geometry.contains(p)]
    if hit.empty:
        return None
    hit = hit.assign(_a=hit.geometry.area).sort_values("_a").iloc[0]
    return _name_of(hit)


def _pct(x) -> str:
    return f"{round(float(x) * 100):d}%"


def _m(x) -> str:
    x = float(x)
    return f"{x / 1000:.1f} km" if x >= 1000 else f"{round(x):d} m"


def _site_rows(ctx, job, sid, tbl):
    """Top table rows with lon/lat and district name."""
    res = job.results[sid]
    out = []
    try:
        g = ctx.results.get(res.result_id).gdf.to_crs(4326)
    except Exception:  # noqa: BLE001
        g = None
    for i, r in enumerate(tbl):
        lon = lat = None
        if g is not None and i < len(g):
            c = g.geometry.iloc[i].centroid
            lon, lat = c.x, c.y
        out.append({**r, "lon": lon, "lat": lat, "district": district_of(ctx.area, lon, lat) if lon is not None else None})
    return out


def _scored(job) -> Optional[int]:
    import re
    for f in job.get("answer", []):
        m = re.search(r"(\d[\d,]*) scored", f["text"]) if "Candidate screening" in f["text"] else None
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def _facts(job, kind):
    return [f for f in job.get("answer", []) if f["type"] == kind]


def summarize(job: dict, jobobj, ctx) -> dict:
    scope = ctx.area.scope_label()
    steps = job.get("plan", {}).get("steps", [])
    tools = [s["tool"] for s in steps]
    tables = job.get("tables", {})
    by_tool = {job["results"][sid]["tool"]: sid for sid in job.get("results", {})}
    out = {"headline": "", "points": [], "cards": [], "confidence": None, "limits": [], "how_to_read": "", "next_questions": [], "scope": scope}

    if job.get("status") == "unsupported":
        out["headline"] = "I can't answer that one yet."
        out["points"] = ["Try asking about places (shops, clinics, parking), counts by district, the best spot for a new facility, or what a planning term means."]
        out["next_questions"] = ["Find the best places for a new supermarket", "Count shops in each district", "Where should new parking be located?"]
        return out

    gaps = [f["text"] for f in _facts(job, "observed") if f["text"].lower().startswith("data gaps")]
    if gaps:
        out["limits"].append("There is no population, traffic, land-price or sales data in this system, so results are estimates from what is built and mapped, not from real customers.")

    if "rank_site_suitability" in by_tool:
        sid = by_tool["rank_site_suitability"]
        tbl = tables.get(sid) or []
        facility = next((s.get("args", {}).get("facility") for s in steps if s["tool"] == "rank_site_suitability"), "shop")
        if not tbl:
            out["headline"] = f"No good place for a new {facility} was found in {scope}."
            out["points"] = [f["text"] for f in _facts(job, "observed") if "Candidate screening" in f["text"]][:1] or ["Every possible spot was ruled out by the rules (too close to a highway, a competitor, or an existing building)."]
            return out
        rows = _site_rows(ctx, jobobj, sid, tbl[:5])
        best = rows[0]
        where = f" in {best['district']}" if best.get("district") else ""
        out["headline"] = f"Best place for a new {facility}: Site 1{where} - score {round(best['score'] * 100)} out of 100."
        for r in rows:
            reasons = []
            if r.get("competitors_500m") == 0:
                reasons.append(f"no other {facility} within 500 m")
            elif r.get("competitors_500m") is not None:
                reasons.append(f"{int(r['competitors_500m'])} other {facility}(s) within 500 m")
            if r.get("dist_four_way_m") is not None:
                reasons.append("on a major crossroads" if r["dist_four_way_m"] <= 30 else f"{_m(r['dist_four_way_m'])} from a major crossroads")
            if r.get("retail_500m"):
                reasons.append(f"{int(r['retail_500m'])} shops nearby, so people already come here")
            if r.get("parking_300m"):
                reasons.append(f"{int(r['parking_300m'])} parking places within 300 m")
            if r.get("dist_highway_m") is not None:
                reasons.append(f"{_m(r['dist_highway_m'])} from the nearest highway")
            dn = f" ({r['district']})" if r.get("district") else ""
            out["points"].append(f"Site {int(r['rank'])}{dn}: score {round(r['score'] * 100)}/100 - " + "; ".join(reasons[:4]))
        out["cards"] = [{"label": "Best score", "value": f"{round(best['score'] * 100)}/100", "hint": "higher is better; it is a ranking, not a prediction of sales"},
                        {"label": "Sites compared", "value": str(_scored(job) or len(tbl)), "hint": "locations that passed all the rules"}]
        freq = best.get("top5_frequency")
        if freq is not None:
            lvl = "high" if freq >= 0.8 else "medium" if freq >= 0.5 else "low"
            out["confidence"] = {"level": lvl, "text": f"Site 1 stayed in the top 5 in {_pct(freq)} of 400 'what if people cared about different things' tests. " +
                                 {"high": "The recommendation is robust.", "medium": "It is a good option but not the only one; compare with Sites 2-3.", "low": "Treat the top 5 as a shortlist, not a single answer."}[lvl]}
        out["how_to_read"] = "The red dots on the map are the recommended sites. Click a row to zoom to it. 'Score' blends demand (built-up area people can reach), access, visibility, competition and parking - you can change what matters on the left."
        out["next_questions"] = ["Exclude anything within 300 metres of a hospital", "Where should new parking be located?", "Count shops in each district"]
        return out

    if "aggregate_spatial_statistics" in by_tool:
        sid = by_tool["aggregate_spatial_statistics"]
        tbl = [r for r in (tables.get(sid) or [])]
        if tbl:
            tbl.sort(key=lambda r: -(r.get("count") or 0))
            total = sum(int(r.get("count") or 0) for r in tbl)
            top = tbl[0]
            out["headline"] = f"{_name_of(top)} has the most: {int(top['count']):,} out of {total:,} counted in {scope}."
            out["points"] = [f"{i + 1}. {_name_of(r)}: {int(r['count']):,}" + (f" ({r['density_per_km2']:.0f} per km2)" if r.get("density_per_km2") else "") for i, r in enumerate(tbl[:8])]
            quiet = [r for r in tbl if not r.get("count")]
            if quiet:
                out["points"].append(f"{len(quiet)} area(s) have none.")
            out["cards"] = [{"label": "Total counted", "value": f"{total:,}", "hint": "in " + scope}, {"label": "Areas compared", "value": str(len(tbl)), "hint": "districts"}]
            out["confidence"] = {"level": "high", "text": "These are straight counts of what is in the map data. Places missing from the map are not counted."}
            out["how_to_read"] = "The table is sorted by count. 'Density' is per square kilometre, so small busy districts can rank above big ones."
            out["next_questions"] = ["Find the best places for a new supermarket", "Show clinics near hospitals"]
            return out

    if "optimize_facility_locations" in by_tool:
        sid = by_tool["optimize_facility_locations"]
        tbl = tables.get(sid) or []
        summ = job["results"][sid].get("summary", {})
        p = summ.get("p")
        base = next((r for r in tbl if "existing" in str(r.get("strategy", ""))), tbl[0] if tbl else None)
        cand = [r for r in tbl if r is not base and r.get("mean_distance_m") is not None]
        best = min(cand, key=lambda r: r["mean_distance_m"]) if cand else None
        if base and best:
            d0, d1 = float(base["mean_distance_m"]), float(best["mean_distance_m"])
            gain = (d0 - d1) / d0 if d0 else 0
            c0, c1 = float(base.get("coverage_within_radius", 0)), float(best.get("coverage_within_radius", 0))
            if gain >= 0.03:
                out["headline"] = f"{p or 'New'} new facilities would cut the average trip to the nearest one from {_m(d0)} to {_m(d1)} ({_pct(gain)} shorter)."
            else:
                out["headline"] = f"Access is already good: adding {p or 'new'} facilities would only change the average trip from {_m(d0)} to {_m(d1)}."
            out["points"] = [f"Share of people within 500 m of one: {_pct(c0)} now -> {_pct(c1)} after.",
                             "The best locations are shown as red dots on the map (click a row to zoom).",
                             f"Compared against: {', '.join(str(r['strategy']) for r in tbl if r is not base)}."]
            out["cards"] = [{"label": "Average trip now", "value": _m(d0), "hint": "to the nearest existing facility"}, {"label": "After", "value": _m(d1), "hint": "with the new ones"}]
            out["confidence"] = {"level": "medium", "text": "Distances follow the road network. Real demand, parking capacity and land availability are not known, so check each site on the ground."}
            out["how_to_read"] = "Each row is a strategy; lower 'mean distance' is better. The recommended one is compared with simple alternatives (greedy, random) so you can see it is not luck."
            out["next_questions"] = ["Find the best places for a new supermarket", "Count parking in each district"]
            return out

    if "run_geoquery" in by_tool:
        sid = by_tool["run_geoquery"]
        sc = job["results"][sid].get("summary", {}).get("status_counts", {})
        v, pm, ins, rej = (int(sc.get(k, 0)) for k in ("VERIFIED_MATCH", "PROBABLE_MATCH", "INSUFFICIENT_EVIDENCE", "REJECTED"))
        out["headline"] = f"{v:,} places clearly match, {pm:,} probably match." if (v or pm) else "No place clearly matches your description."
        out["points"] = [f"Clearly match: {v:,} (every condition checked with enough accuracy).",
                         f"Probably match: {pm:,} (they pass, but map positions are only accurate to about 10 m, so a condition could be borderline).",
                         f"Could not be checked: {ins:,} (the map data is missing something needed).", f"Do not match: {rej:,}."]
        amb = job["results"][sid].get("summary", {}).get("ambiguities") or []
        out["limits"] += [str(a) for a in amb[:2]]
        out["cards"] = [{"label": "Clear matches", "value": f"{v:,}", "hint": "green list"}, {"label": "Probable", "value": f"{pm:,}", "hint": "borderline"}]
        out["confidence"] = {"level": "high" if v else "medium", "text": "Each place is checked against the rules one by one, and each result says which rule it passed or failed."}
        out["how_to_read"] = "The table lists the places; 'status' tells you how sure we are. Dots on the map are the matches."
        out["next_questions"] = ["Now exclude anything within 300 metres of a hospital", "Count shops in each district"]
        return out

    if "retrieve_reference_documents" in by_tool and len(tools) <= 2:
        tbl = tables.get(by_tool["retrieve_reference_documents"]) or []
        if tbl:
            out["headline"] = "Here is what that means:"
            out["points"] = [str(r["text"])[:420] for r in tbl[:2]]
            out["confidence"] = {"level": "high", "text": "These come from the project's reference notes (shown with their source)."}
            out["next_questions"] = ["What does maxBuildingHeight mean?", "How is site suitability scored?"]
            return out

    if "compare_scenarios" in by_tool:
        sid = by_tool["compare_scenarios"]
        tbl = tables.get(sid) or []
        if tbl:
            r = tbl[-1]
            out["headline"] = f"In this what-if, the average trip changes by {_m(abs(r.get('delta_mean_distance_m', 0)))} ({'shorter' if r.get('delta_mean_distance_m', 0) < 0 else 'longer'})."
            out["points"] = [f"Coverage within 500 m changes by {r.get('delta_coverage_within_radius', 0) * 100:+.1f} points."]
            out["limits"].append("This is a simulation: nothing in the real data changed.")
            return out

    # generic fallback: say the top facts in order
    facts = [f["text"] for f in job.get("answer", []) if f["type"] in ("inferred", "model", "observed") and not f["text"].lower().startswith(("data gaps", "evidence report"))]
    out["headline"] = facts[0][:220] if facts else "Done - see the details below."
    out["points"] = [t[:260] for t in facts[1:5]]
    out["confidence"] = {"level": "medium", "text": "See the technical details for exactly how each number was computed."}
    return out
