"""Question -> typed execution plan (task classes, tool calls, dependencies). Deterministic and inspectable.

An LLM could replace `plan()`, but its output would have to validate as an `ExecutionPlan` over the same tool registry, so it could
only choose tools and arguments, never produce numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from geoacquirer.ontology import find_concepts_in_text

TASK_CLASSES = ["RETRIEVAL", "SPATIAL_FILTERING", "SPATIAL_JOIN", "NETWORK_ANALYSIS", "STATISTICAL_ANALYSIS", "PREDICTION", "OPTIMIZATION",
                "SCENARIO_SIMULATION", "TEMPORAL_ANALYSIS", "REPORT_GENERATION"]
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUM = r"(\d+(?:\.\d+)?)\s*(km|kilomet(?:er|re)s?|m|met(?:er|re)s?)\b"


@dataclass
class Step:
    id: str
    tool: str
    args: dict
    why: str
    depends_on: list[str] = field(default_factory=list)
    expensive: bool = False


@dataclass
class ExecutionPlan:
    question: str
    task_classes: list[str]
    steps: list[Step]
    notes: list[str] = field(default_factory=list)
    supported: bool = True

    def to_dict(self) -> dict:
        return {"question": self.question, "task_classes": self.task_classes, "supported": self.supported, "notes": self.notes,
                "steps": [s.__dict__ for s in self.steps]}


def _count(text: str, default: int) -> int:
    m = re.search(r"\b(\d+)\b", text)
    if m and int(m.group(1)) <= 20:
        return int(m.group(1))
    for w, n in _WORDS.items():
        if re.search(rf"\b{w}\b", text):
            return n
    return default


def _dist(text: str, default: Optional[float] = None) -> Optional[float]:
    m = re.search(_NUM, text)
    if not m:
        return default
    v = float(m.group(1))
    return v * 1000 if m.group(2).lower().startswith("k") else v


def parse_region(low: str, district_names: list[str]) -> dict:
    found, rest = [], low
    for n in sorted(set(district_names), key=len, reverse=True):
        if n and len(n) > 2 and n.lower() in rest:
            found.append(n)
            rest = rest.replace(n.lower(), " ")
    if found:
        return {"kind": "district", "district": "|".join(found)}
    m = re.search(r"\b(?:district|neighbou?rhood|حي)\s+(?:of\s+)?([a-z؀-ۿ][\w؀-ۿ \-]{2,30}?)(?:[,.]|\s+(?:and|with|that|which|avoid)\b|$)", low)
    if m:
        return {"kind": "district", "district": m.group(1).strip()}
    for word, kind in (("northern", "north"), ("north", "north"), ("southern", "south"), ("south", "south"), ("eastern", "east"), ("east", "east"),
                       ("western", "west"), ("west", "west"), ("central", "central"), ("centre", "central"), ("center", "central")):
        if re.search(rf"\b{word}\b", low):
            return {"kind": kind}
    return {"kind": "all"}


def plan(question: str, *, district_names: Optional[list[str]] = None, last_result_id: Optional[str] = None,
         last_result_layer: str = "poi") -> ExecutionPlan:
    q = question.strip()
    low = q.lower()
    names = district_names or []
    region = parse_region(low, names)
    steps: list[Step] = []

    def S(tool, args, why, deps=None, expensive=False):
        sid = f"s{len(steps) + 1}"
        steps.append(Step(sid, tool, args, why, deps or [], expensive))
        return sid

    # ---- follow-up on the previous result set
    m = re.match(r"^(?:now|then|and|also|ok|okay)?[,\s]*(exclude|remove|drop|without|keep only|only keep|keep)\b(.*)$", low)
    if m and last_result_id and re.search(_NUM, low):
        d = _dist(low)
        conc = [c for c in find_concepts_in_text(low)]
        negate = m.group(1) in ("exclude", "remove", "drop", "without")
        if conc and d:
            near = {"layer": "parking" if conc[0].entity_type == "parking" else "poi", "categories": [] if conc[0].entity_type == "parking" else [conc[0].id],
                    "distance_m": d, "negate": negate}
            S("filter_features", {"layer": last_result_layer, "base_result_id": last_result_id, "near": [near]},
              f"follow-up: {'exclude' if negate else 'keep only'} features within {d:g} m of {conc[0].label}")
            return ExecutionPlan(q, ["SPATIAL_FILTERING"], steps, ["applied to the previous result set; lineage is kept"])

    # ---- reference / definition questions
    if re.search(r"\b(what (does|is|are)|define|meaning of|explain|how (is|are|does)|tell me about|describe)\b", low) and re.search(
            r"(zoning|setback|coefficient|coverage|building height|landuse|status|verified|probable|suitability|score|junction|crossroad|method|licen|parking capacity|data source|overture)", low) \
            and not re.search(r"\b(find|rank|where|count)\b", low):
        S("retrieve_reference_documents", {"query": q}, "definition/methodology question answered from the reference corpus")
        return ExecutionPlan(q, ["RETRIEVAL"], steps, ["numeric answers are never taken from documents"])

    # ---- forecasts: no temporal training data exists, so never answer a prediction question with a static list
    if re.search(r"(predict|forecast|will open|will be|next year|in the future|projection|trend)", low):
        return ExecutionPlan(q, [], [], ["Forecasting is not supported: only one data snapshot exists, so there is no history to train or validate a growth or change forecast. "
                                         "Supported analyses: constrained feature search, per-district counts, site suitability, parking placement optimisation, what-if scenarios, "
                                         "definitions of data fields and methods."], supported=False)

    # ---- 'underserved / shortage / deficient' parking: demand and occupancy are unknown, so no shortage can be asserted
    if re.search(r"(underserved|under-served|shortage|deficien|lack of|not enough|insufficient|crowded|congest|full at)", low) and re.search(r"parking", low):
        return ExecutionPlan(q, [], [], ["A parking shortage cannot be established: the data has reported capacities but no occupancy, demand or turnover. What can be computed: where shops are "
                                         "far from any mapped public parking (e.g. 'Where should three new parking facilities be located?'), and what a new facility would change in "
                                         "accessibility. Mapped parking proximity is not a shortage."], supported=False)

    # ---- report
    if re.search(r"\b(analy[sz]e|assess|report|evaluate)\b", low) and re.search(r"(opportunit|commercial development)", low):
        s1 = S("get_geographic_coverage", {}, "document data coverage and gaps first")
        s2 = S("aggregate_spatial_statistics", {"layer": "poi", "categories": ["shop"], "by": "district", "top": 12}, "retail density per district", [s1])
        s3 = S("rank_site_suitability", {"facility": "retail", "k": 5, "region": region}, "heuristic site ranking for the selected area", [s1])
        s4 = S("classify_land_use", {"action": "metrics"}, "report validation metrics of the building land-use model", [s1], expensive=True)
        s5 = S("retrieve_reference_documents", {"query": "suitability scoring method limitations"}, "methodology text for the report", [s3])
        S("generate_evidence_report", {}, "compile an evidence report with provenance, uncertainty and limitations", [s2, s3, s4, s5])
        return ExecutionPlan(q, ["STATISTICAL_ANALYSIS", "PREDICTION", "REPORT_GENERATION"], steps)

    # ---- facility location optimization
    if re.search(r"\b(where|which|find|suggest)\b.*\b(should|could|best|optimal|additional|more)\b.*\bparking\b", low) or \
            re.search(r"\b(locate|place|site)\b.*\b(new|additional)\b.*\bparking\b", low) or re.search(r"(additional|new|more|extra)\s+(public\s+)?parking", low) or re.search(r"(places?|spots?|sites?|locations?).{0,25}(add|build|put|place|open).{0,25}parking", low) or re.search(r"(add|build|put|place|open).{0,30}parking", low) and re.search(r"(where|which|best|room|how many)", low):
        p = _count(low, 3)
        obj = "max_coverage" if re.search(r"(coverage|cover)", low) else "p_median"
        s1 = S("get_geographic_coverage", {}, "check parking and road data coverage")
        s2 = S("optimize_facility_locations", {"facility": "parking", "p": p, "objective": obj, "region": region, "radius_m": _dist(low, 500) if re.search(_NUM, low) else 500},
               f"{obj} placement of {p} new parking facilities on the road network", [s1])
        s3 = S("compare_scenarios", {"scenarios": "$s2.proposed_scenarios"}, "simulate each proposed site and all of them together", [s2])
        S("generate_evidence_report", {}, "report with baselines and assumptions", [s2, s3])
        return ExecutionPlan(q, ["OPTIMIZATION", "SCENARIO_SIMULATION", "NETWORK_ANALYSIS"], steps,
                             ["demand is a proxy (shops); there is no parking occupancy data", "candidate cells are not verified buildable land"])

    # ---- site suitability
    if (re.search(r"\b(suitable|best|ideal|good|recommend|rank|candidate|top)\b", low) and re.search(r"\b(locations?|sites?|places?|spots?|areas?)\b", low) or re.search(r"(open|build|start|set up).{0,40}(where|which).{0,30}(put|place|locate|site|open)", low) or re.search(r"where (would|should|could) (you|i|we) (put|place|open|locate|build)", low)) \
            and re.search(r"(supermarket|grocery|store|retail|shop|clinic|restaurant|facility)", low):
        concepts = [c.id for c in find_concepts_in_text(low)]
        fac = "supermarket" if any(c in ("supermarket",) for c in concepts) or "grocery" in low else "clinic" if "clinic" in concepts else "restaurant" if "restaurant" in concepts else "retail"
        k = _count(low, 5)
        weights: dict = {}
        notes = []
        if re.search(r"(crossroad|intersection|junction)", low):
            weights["junction_access"] = 0.35
            notes.append("junction_access weight raised to 0.35 because the question asks to prioritise crossroad accessibility")
        if re.search(r"parking", low):
            weights["parking"] = 0.12
            notes.append("parking weight raised to 0.12 (mapped parking is sparse, so this criterion has little discriminating power)")
        if re.search(r"(metro|transit|train|station|public transport)", low):
            weights["transit"] = 0.12
            notes.append("transit weight raised to 0.12 because the question mentions metro/transit access")
        if re.search(r"competitor", low):
            weights["competition"] = 0.30
            notes.append("competition weight raised to 0.30 because the question asks to compare nearby competitors")
        allowed = ["commercial", "mixed"] if re.search(r"zoning|zoned|commercial", low) else []
        if allowed:
            notes.append("zoning restriction requested: applied only if a parcel layer exists; otherwise reported UNVERIFIED")
        s1 = S("get_geographic_coverage", {}, "document coverage and gaps before ranking")
        s2 = S("rank_site_suitability", {"facility": fac, "k": k, "region": region, "weights": weights, "allowed_landuse": allowed,
                                         "min_highway_distance_m": 60}, f"rank {fac} sites", [s1])
        s3 = S("retrieve_reference_documents", {"query": "suitability scoring demand proxy limitations"}, "methodology and limitations text", [s2])
        S("generate_evidence_report", {}, "evidence-backed decision report", [s2, s3])
        return ExecutionPlan(q, ["SPATIAL_FILTERING", "NETWORK_ANALYSIS", "STATISTICAL_ANALYSIS", "REPORT_GENERATION"], steps, notes)

    # ---- aggregation per district
    if re.search(r"\b(count|number of|how many|compare)\b", low) and (re.search(r"(district|neighbou?rhood|each|per)", low) or region["kind"] == "district"):
        conc = find_concepts_in_text(low)
        layer = "parking" if conc and conc[0].entity_type == "parking" else "poi"
        cats = [] if layer == "parking" else [conc[0].id] if conc else ["shop"]
        focus = []
        mm = re.search(r"(?:between|of)\s+(.+?)\s+and\s+(.+?)(?:[?.]|$)", q)
        if "compare" in low and mm:
            focus = [mm.group(1).strip(), mm.group(2).strip()]
        S("aggregate_spatial_statistics", {"layer": layer, "categories": cats, "by": "district", "top": 40, "focus_names": focus},
          "count features per district polygon and normalise by area")
        return ExecutionPlan(q, ["STATISTICAL_ANALYSIS", "SPATIAL_JOIN"], steps, ["districts are OSM-derived polygons and may not tile the whole area"])

    # ---- quality / popularity rankings need rating data we do not have: say so instead of returning an unranked list
    if re.search(r"\b(best|worst|top[- ]rated|highest[- ]rated|most popular|good|better)\b", low) and find_concepts_in_text(low):
        return ExecutionPlan(q, [], [], ["This asks for a quality ranking ('best/worst/popular'), but the data has no ratings, reviews, outcomes or quality measures, so the platform cannot rank "
                                         "by quality. Supported alternatives: count or density per district (e.g. 'Count schools in each district'), or accessibility (e.g. 'schools "
                                         "within 400 metres of mosques'). Supported analyses: constrained feature search, per-district counts, site suitability, parking placement "
                                         "optimisation, what-if scenarios, definitions of data fields and methods."], supported=False)

    # ---- generic constrained retrieval via GeoAcquirer's compiler/evaluator
    show_parking = bool(re.search(r"\b(and\s+)?show\s+(nearby\s+)?parking\b|\bwith\s+nearby\s+parking\b", low))
    text = re.sub(r"(,?\s*and\s+)?show\s+(nearby\s+)?parking|with\s+nearby\s+parking", "", q, flags=re.I).strip(" ,.")
    if (find_concepts_in_text(text.lower()) or re.search(r"\b(parcels?|plots?)\b", low)) and (re.search(r"\b(find|list|show|which|where|get|locate|give)\b", low) or re.search(r"(at|near|within|away from|not on|on|close to|next to|sit on)\b", low)):
        s1 = S("run_geoquery", {"text": text}, "compile to GeoQuery IR and evaluate deterministic constraints")
        classes = ["SPATIAL_FILTERING", "RETRIEVAL"]
        if show_parking:
            s2 = S("measure_proximity", {"result_id": f"${s1}.result_id", "to": {"layer": "parking", "distance_m": 1000}, "name": "parking"},
                   "attach distance to the nearest mapped parking facility", [s1])
            classes.append("SPATIAL_JOIN")
        return ExecutionPlan(q, classes, steps, ["POI positions are points (no entrances); status bands reflect a 10 m positional uncertainty"])

    return ExecutionPlan(q, [], [], ["This question does not match a supported analysis. Supported: constrained feature search, per-district counts, site suitability, "
                                    "parking placement optimisation, what-if scenarios, definitions of data fields and methods."], supported=False)
