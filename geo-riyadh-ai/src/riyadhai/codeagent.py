"""Plan -> act -> observe analysis agent: the question becomes runnable Python that is executed on the data, then plotted and explained.

Loop (ReAct / CodeAct style, see docs/agentic_analysis.md):
  PLAN     interpretation, steps and assumptions (the model, or a vetted recipe, proposes them)
  ACT      write Python (pandas / geopandas / shapely / numpy / matplotlib) that works on the in-memory layers and sets `result`
  OBSERVE  the sandbox runs it; the model sees only: errors, what the code printed, the result's shape / columns / head / statistics, automatic checks.
           The data itself is never put in the prompt (the RLM idea: the context lives in the REPL; the model inspects it with code).
  REPAIR   on an error or a failed check the model gets the observation and writes a fixed version (bounded number of turns)
  VERIFY   automatic sanity checks; the explanation's numbers are audited against what the code actually produced
  EXPLAIN  plain-language answer built from the observed result (model-written when an LLM is configured, deterministic otherwise)
With no LLM configured only vetted recipes (recipes.py) can run; anything else is declined with a clear message rather than guessed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from . import recipes
from . import sandbox
from .acquire import AcquisitionTools, NeedsApproval
from .llmbridge import LLMBridge
from .paths import city_store

LLM = Callable[[str, list], str]   # (system, messages[{role, content}]) -> assistant text
MARGIN_M = 1500.0

SYSTEM = """You are a careful geospatial data analyst for the city of Riyadh. You answer the user's question by writing Python that runs on in-memory data.

ENVIRONMENT (already imported / defined, do not import them again): layers (dict name -> GeoDataFrame in a metric CRS, units = metres), crs, scope (shapely polygon of the selected area, or None),
pd, np, gpd, plt, shapely, nx (networkx), and helpers nearest_dist(a, b), count_within(a, b, distance_m, weight=None), district_of(points), in_scope(gdf), to_wgs84(gdf).
Layers contain the selected area plus a 1.5 km margin so distances near the edge are right; use in_scope(...) for the things the question is ABOUT.
RULES: never invent data or numbers; use only the layers and columns listed; no files, network or system access; put the answer in `result` (DataFrame/GeoDataFrame/dict/number); print the few key numbers you will
quote; draw one clear matplotlib figure when a chart or map helps; append short assumption strings to the list `notes`; keep the code short and vectorised; do not use .iterrows() on large layers.
Layers can describe the SAME real-world things from different sources (layers['building'] = Overture footprints, layers['urbi_building'] = Urbi buildings with a use class): never add counts from two sources together; pick the better-suited one (urbi_building for residential/commercial use) and say which in `notes`. Names listed under PLACES NAMED IN THE QUESTION are in `entities`; entities["Kingdom Centre"] (case-insensitive) has .x, .y (metric metres), .geometry (a shapely Point), ['kind'], ['district']; build a GeoDataFrame only with geometry=... .
NEVER substitute a different layer for the one the user asked for without saying so: a junction is not a connector, a POI is not a building, a street polygon is not a road segment. If the requested data is not listed under DATA, either obtain it (acquire, when available) or state plainly in the headline that it is not available and exactly what you used instead.
Be explicit about units. If something the user asks for is not in the data, say so in `notes` and answer with the closest thing that is.

Reply with ONE JSON object and nothing else:
 {"thought": "...", "action": "code", "plan": {"interpretation": "...", "steps": ["..."], "assumptions": ["..."]}, "code": "python source"}      (first turn: include plan)
 {"thought": "...", "action": "code", "code": "python source"}                                                                                           (fix or extend after an observation)
 {"thought": "...", "action": "final", "final": {"headline": "one plain sentence with the answer", "points": ["short plain bullets"], "caveats": ["what this cannot tell you"], "next_questions": ["..."]}}
 {"thought": "...", "action": "ask_user", "question": "..."}                                       (only if the request is truly ambiguous and a wrong guess would mislead)
Write for a non-expert reader in the final answer: short, concrete, no jargon, every number must come from the observation."""


LLM_QUERY_DOC = """
MODEL CALLS FROM CODE (recursive step): llm_query(prompt) -> str and llm_map(items, instruction, batch=25) -> list are available. Use them ONLY for what code cannot do: reading fuzzy
Arabic/English names, classifying free text, summarising text. Never for counting, distances, joins or ranking. When the question asks you to JUDGE or classify names/text by meaning (cuisine, type of place, language, sentiment), use llm_map on the candidate rows, not keyword lists: keywords misfire (for example 'mandi' is an Arabian dish, not Indian). Pass short texts in batches (calls are budgeted; prompts over 14,000 characters
are refused). Their answers are model judgements: say so in `notes`. Handle exceptions (budget exhausted) by falling back to a simpler rule."""

ACQUIRE_DOC = """
MISSING DATA: if a layer you need is not listed, you may ask for open data with {"action": "acquire", "tool": "list_sources" | "estimate" | "fetch", "args": {"layers": ["poi"|"road"|"connector"|"building"|"district"]}}.
Always `estimate` first. Fetched layers appear as layers['acq_<name>'] on the next code run and come with a verification report: read it before trusting them. Live downloads need the user's approval
(the analysis pauses). Do not ask for data you already have. An empty fetch is "unknown", not "zero"."""


# ----------------------------------------------------------------------------- LLM adapters

def _post_with_retry(url: str, headers: dict, payload: dict, timeout: float = 120.0, tries: int = 5):
    """POST with backoff on rate limits (429) and server errors; other HTTP errors raise a readable message that includes the provider's explanation (never the key)."""
    delay = 3.0
    for attempt in range(tries):
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            wait = float(r.headers.get("retry-after", delay)) if str(r.headers.get("retry-after", "")).replace(".", "", 1).isdigit() else delay
            time.sleep(min(wait, 30.0))
            delay *= 2
            continue
        if not r.ok:
            try:
                msg = r.json().get("error", {}).get("message") or r.text[:300]
            except ValueError:
                msg = r.text[:300]
            raise RuntimeError(f"language model request failed (HTTP {r.status_code}): {str(msg)[:300]}")
        return r
    raise RuntimeError("language model request failed after retries")

def env_llm() -> Optional[LLM]:
    """Anthropic Messages API (ANTHROPIC_API_KEY, optional ANTHROPIC_BASE_URL and RIYADHAI_LLM_MODEL) or any OpenAI-compatible endpoint (RIYADHAI_LLM_URL / GEOACQUIRER_LLM_URL + _MODEL [+ _KEY]); None if neither is configured."""
    model = os.environ.get("RIYADHAI_LLM_MODEL") or os.environ.get("GEOACQUIRER_LLM_MODEL")
    if os.environ.get("ANTHROPIC_API_KEY"):
        key, m = os.environ["ANTHROPIC_API_KEY"], model or "claude-sonnet-5-5"

        def call(system: str, messages: list) -> str:
            r = _post_with_retry(os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                                 {"model": m, "max_tokens": 4000, "system": system, "messages": messages})
            return "".join(b.get("text", "") for b in r.json()["content"])
        return call
    url = os.environ.get("RIYADHAI_LLM_URL") or os.environ.get("GEOACQUIRER_LLM_URL")
    if url and model:
        key = os.environ.get("RIYADHAI_LLM_KEY") or os.environ.get("GEOACQUIRER_LLM_KEY", "")

        def call2(system: str, messages: list) -> str:
            base = int(os.environ.get("RIYADHAI_LLM_MAX_TOKENS", "3000"))
            est_prompt = (len(system) + sum(len(m["content"]) for m in messages)) / 3.2     # rough token estimate; many providers count the requested output against a per-minute cap
            tokens = int(max(800, min(base, 7600 - est_prompt)))
            for attempt in range(2):
                payload = {"model": model, "messages": [{"role": "system", "content": system}] + messages, "temperature": 0, "max_tokens": tokens}
                if "gpt-oss" in model.lower():
                    payload["reasoning_effort"] = "low"
                r = _post_with_retry(url.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {key}"} if key else {}, payload)
                ch = r.json()["choices"][0]
                msg = ch["message"]
                content = msg.get("content") or ""
                if content.strip():
                    return content
                if ch.get("finish_reason") == "length" and attempt == 0:   # reasoning used up the output budget: ask again with more room
                    tokens = int(max(tokens, min(tokens * 2, 7600 - est_prompt)))
                    continue
                break
            return msg.get("reasoning") or msg.get("reasoning_content") or ""
        return call2
    return None


def extract_json(text: str) -> Optional[dict]:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    else:
        a, b = t.find("{"), t.rfind("}")
        t = t[a:b + 1] if a >= 0 and b > a else t
    try:
        return json.loads(t)
    except ValueError:
        return None


class RecipeLLM:
    """Deterministic stand-in for the model: answers the loop's turns from a vetted recipe. Used when no LLM is configured (or as a fast path)."""

    def __init__(self, params: dict):
        self.params, self.turn = params, 0

    def __call__(self, system: str, messages: list) -> str:
        self.turn += 1
        if self.turn == 1:
            plan, code = recipes.render(self.params)
            return json.dumps({"thought": "vetted recipe", "action": "code", "plan": plan, "code": code})
        last = messages[-1]["content"]
        if "RUN FAILED" in last:
            return json.dumps({"thought": "the recipe failed", "action": "final", "final": {"headline": "The analysis could not be completed with the available data.", "points": [], "caveats": [last[-400:]], "next_questions": []}})
        m = re.search(r"SUMMARY (\{.*\})", last)
        s = json.loads(m.group(1)) if m else {}
        return json.dumps({"thought": "explain from the computed summary", "action": "final", "final": explain_homes(s)})


def explain_homes(s: dict) -> dict:
    n_ok = s.get("with_enough_parking", 0)
    if not s or not n_ok:
        mx = s.get("max_capacity_within_radius")
        return {"headline": f"No residential building in this area has {s.get('min_vehicles', 10)} or more vehicles of known parking capacity within {s.get('radius_m', 1000) / 1000:g} km.",
                "points": [f"{s.get('residential_buildings', 0):,} residential buildings were checked; {s.get('with_any_parking_lot', 0):,} have at least one mapped parking lot nearby.",
                           f"The most reported parking capacity within {s.get('radius_m', 1000) / 1000:g} km of any home here is {mx:g} vehicles." if mx is not None else "No parking capacity is reported nearby."],
                "caveats": ["Many parking lots have no reported capacity and are not counted, so the real number may be higher.", "Try a smaller number of vehicles or a larger radius."],
                "next_questions": ["Show homes within 2 km of a parking lot", "Which districts have the most parking capacity?"]}
    return {"headline": f"Best home found: about {s['best_minutes']:.0f} minutes to the airport by metro, {s['best_walk_m']:.0f} m from {s['best_station']} station, with at least {s['min_vehicles']} parking places within {s['radius_m'] / 1000:g} km.",
            "points": [f"{n_ok:,} of {s['residential_buildings']:,} residential buildings meet the parking rule; the {s['shown']} fastest are listed and mapped.",
                       f"The best one is {s['best_walk_m']:.0f} m from the station and {s['best_ride_km']:.1f} km along the metro to the airport.",
                       f"For comparison, the typical home here needs about {s['median_minutes_all_homes']:.0f} minutes."]
                      + (["Every residential building here already meets the parking rule, so the ranking depends only on the metro connection; the top homes differ by just a few minutes."] if n_ok == s["residential_buildings"] else []),
            "caveats": ["Times are estimates: walking is straight-line x 1.3 at 5 km/h, 5 minutes waiting and 35 km/h on the metro - not a timetable.",
                        f"Parking counts only lots with a reported capacity; about {s['unknown_capacity_share'] * 100:.0f}% of mapped lots have none, so true parking may be better than shown.",
                        "Metro connection means the distance along the metro lines to the airport terminals; transfers between lines are not penalised."],
            "next_questions": ["Show the same for homes with at least 20 parking places", "Which districts have the fastest metro access to the airport?"]}



def explain_generic(df, stdout: str = "", jsonval=None) -> dict:
    """Plain description of a result built only from the result itself (used when the model did not write an explanation)."""
    caveat = ["The language model did not write an explanation for this result, so this summary was generated directly from the computed table. Check the code and the table."]
    if df is None and isinstance(jsonval, dict):
        scal = [f"{str(k).replace('_', ' ')}: {v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{str(k).replace('_', ' ')}: {v}" for k, v in jsonval.items() if not isinstance(v, (list, dict))]
        lists = [f"{str(k).replace('_', ' ')}: " + "; ".join(str(x) for x in v[:10]) for k, v in jsonval.items() if isinstance(v, list)]
        return {"headline": "; ".join(scal)[:240] or "The analysis finished.", "points": lists, "caveats": caveat, "next_questions": []}
    if df is None and isinstance(jsonval, (int, float, str)):
        return {"headline": f"Result: {jsonval}", "points": [], "caveats": caveat, "next_questions": []}
    if df is None:
        return {"headline": (stdout.strip().splitlines() or ["The analysis finished; see the code and any chart."])[-1][:200], "points": [], "caveats": caveat, "next_questions": []}
    show = pd.DataFrame(df.drop(columns="geometry", errors="ignore"))
    if len(show) == 0:
        return {"headline": "The analysis found nothing that matches.", "points": ["An empty result means no feature met every condition in the data, which is not the same as the thing not existing."], "caveats": caveat, "next_questions": []}

    def fmt(v):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
    if len(show) == 1:
        row = show.iloc[0]
        head = "; ".join(f"{str(c).replace('_', ' ')}: {fmt(row[c])}" for c in show.columns[:5])
        return {"headline": head, "points": [], "caveats": caveat, "next_questions": []}
    num = show.select_dtypes("number")
    pts = [f"{len(show):,} rows in the result."]
    if len(num.columns):
        c = num.columns[0]
        pts.append(f"{str(c).replace('_', ' ')} ranges from {fmt(float(num[c].min()))} to {fmt(float(num[c].max()))} (median {fmt(float(num[c].median()))}).")
    return {"headline": f"The analysis returned {len(show):,} rows.", "points": pts, "caveats": caveat, "next_questions": []}


LAYER_WORDS = {"connector": r"\bconnectors?\b", "building": r"\bbuildings?\b|\bfootprints?\b", "parcel": r"\bparcels?\b", "poi": r"\bpois?\b|points of interest", "parking": r"\bparking\b",
               "district": r"\bdistricts?\b", "street": r"\bstreets?\b", "attraction": r"\battractions?\b|\blandmarks?\b", "metro": r"\bmetro\b", "junction": r"\bjunctions?\b"}


def missing_layers(question: str, loaded) -> list[str]:
    """Layers the question names that are NOT loaded (so the model cannot quietly substitute a different one)."""
    import re as _re
    low = question.lower()
    keys = list(loaded)
    return [name for name, pat in LAYER_WORDS.items() if _re.search(pat, low) and not any(name in k for k in keys)]

# ----------------------------------------------------------------------------- data export & prompts
def _clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()
    for c in list(g.columns):
        if c != "geometry" and g[c].map(lambda v: isinstance(v, (dict, set))).any():
            g = g.drop(columns=c)
    return g


def export_layers(ds, dest: Path, metro_dir: Optional[Path] = None) -> dict:
    """Write the layers (clipped to the area + margin) as parquet for the worker; return {layer: meta} used to describe them to the model."""
    from shapely import STRtree
    dest.mkdir(parents=True, exist_ok=True)
    poly = ds.area_polygon()
    clip = poly.buffer(MARGIN_M)
    info = {}
    for name in ("urbi_building", "building", "poi", "parking", "district", "parcel", "street", "attraction", "metro", "junction", "road"):
        g = ds.layers.get(name)
        if g is None or len(g) == 0:
            continue
        tree = STRtree(g.geometry.values)
        g = g.iloc[np.sort(tree.query(clip, predicate="intersects"))]
        if len(g) == 0:
            continue
        g = _clean(g)
        g.to_parquet(dest / f"{name}.parquet")
        info[name] = g
    if metro_dir is not None and (metro_dir / "stations.parquet").exists():
        for name in ("stations", "lines"):
            g = gpd.read_parquet(metro_dir / f"{name}.parquet").to_crs(ds.crs)
            g = g.rename(columns={"station_name": "name"}) if "station_name" in g else g
            g.to_parquet(dest / f"metro_{name}.parquet")
            info[f"metro_{name}"] = g
    return info


def describe(info: dict) -> str:
    lines = []
    for name, g in info.items():
        cols = []
        for c in g.columns:
            if c == "geometry":
                continue
            s = g[c].dropna()
            ex = ""
            if len(s) and not isinstance(s.iloc[0], (list, np.ndarray)):
                vals = s.astype(str).str.slice(0, 16).drop_duplicates().head(2).tolist()
                ex = " e.g. " + ", ".join(vals)
            cols.append(f"{c} ({str(g[c].dtype)}){ex}")
        gt = "/".join(sorted(set(g.geometry.geom_type.dropna()))) if len(g) else "-"
        lines.append(f"- layers['{name}']: {len(g):,} rows, {gt}; columns: " + "; ".join(cols[:14]))
    return "\n".join(lines)


def result_summary(res: sandbox.RunResult) -> tuple[str, Optional[pd.DataFrame]]:
    """What the model gets to see of the result (never the whole data)."""
    if res.result_kind in ("geodataframe", "dataframe"):
        df = gpd.read_parquet(res.result_path) if res.result_kind == "geodataframe" else pd.read_parquet(res.result_path)
        show = pd.DataFrame(df.drop(columns="geometry", errors="ignore"))
        txt = [f"result: {res.result_kind}, {len(df):,} rows x {len(show.columns)} columns: {', '.join(map(str, show.columns))}"]
        if len(show):
            txt.append("head:\n" + show.head(8).round(3).to_string(max_colwidth=28))
            num = show.select_dtypes("number")
            if len(num.columns):
                txt.append("numeric summary:\n" + num.describe().loc[["min", "50%", "max"]].round(3).to_string())
        return "\n".join(txt), df
    if res.result_kind == "json":
        return "result: " + Path(res.result_path).read_text(encoding="utf-8")[:1500], None
    return "result: none", None


def sanity_checks(res: sandbox.RunResult, df: Optional[pd.DataFrame]) -> list[str]:
    out = []
    if df is not None:
        if len(df) == 0:
            out.append("the result is empty: check the filters (or say clearly that nothing matches)")
        else:
            num = pd.DataFrame(df.drop(columns="geometry", errors="ignore")).select_dtypes("number")
            for c in num.columns:
                if num[c].isna().all():
                    out.append(f"column '{c}' is all NaN")
                if num[c].abs().max() > 1e7 and re.search(r"(dist|_m$|metre|meter)", str(c), re.I):
                    out.append(f"column '{c}' has values above 10,000 km: check the units / CRS")
            if isinstance(df, gpd.GeoDataFrame) and df.geometry.isna().any():
                out.append("some result geometries are missing")
    return out


def numbers_in(text: str) -> set[float]:
    out = set()
    for m in re.finditer(r"(?<![\w.])-?\d[\d,]*\.?\d*", text):
        try:
            out.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def audit_numbers(explanation: str, corpus: str) -> list[float]:
    """Numbers quoted in the explanation that cannot be found (as is, rounded, or as a percentage) in what the code produced."""
    have = numbers_in(corpus)
    variants = set()
    for v in have:
        variants |= {v, round(v), round(v, 1), round(v, 2), round(v * 100), round(v * 100, 1), round(v / 1000, 1), round(v / 1000, 2), round(v * 1000)}
    bad = []
    for n in numbers_in(explanation):
        if n in variants or abs(n) <= 3 or (1900 <= n <= 2100):
            continue
        if any(abs(n - h) <= max(0.51, abs(h) * 0.006) for h in have):
            continue
        bad.append(n)
    return sorted(bad)


# ----------------------------------------------------------------------------- the agent
@dataclass
class AnalysisJob:
    id: str
    question: str
    status: str = "running"        # running | done | failed | needs_input | needs_approval | unsupported
    steps: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    code: str = ""
    final: dict = field(default_factory=dict)
    table: list = field(default_factory=list)
    result_id: Optional[str] = None
    figure: Optional[str] = None
    notes: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    unverified_numbers: list = field(default_factory=list)
    mode: str = ""
    error: str = ""
    question_back: str = ""
    seconds: float = 0.0
    approved: bool = False
    pending: Optional[dict] = None
    acquisition: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    llm_calls: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def log(self, kind: str, **kw):
        self.steps.append({"kind": kind, "t": round(time.time() % 100000, 1), **kw})


class CodeAnalyst:
    def __init__(self, ctx, llm: Optional[LLM] = None, max_turns: int = 6, timeout_s: float = 120.0, metro_dir: Optional[Path] = None, fig_dir: Optional[Path] = None, rag=None,
                 llm_calls: int = 30):
        self.ctx, self.llm, self.max_turns, self.timeout_s, self.rag, self.llm_calls = ctx, llm, max_turns, timeout_s, rag, llm_calls
        self.metro_dir = metro_dir or city_store() / "metro_network"
        self.fig_dir = fig_dir or Path(ctx.out_dir) / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def available(self, question: str) -> tuple[bool, str]:
        if recipes.match(question):
            return True, "vetted recipe"
        if self.llm is not None:
            return True, "LLM"
        return False, "custom analysis needs a language model: set ANTHROPIC_API_KEY (or RIYADHAI_LLM_URL + RIYADHAI_LLM_MODEL); only the built-in recipes run without one"

    @staticmethod
    def _compact(msgs: list, job) -> list:
        """Keep the prompt small (provider token limits, and long histories confuse models): the first message, a one-line summary of earlier attempts, and the last exchange."""
        if len(msgs) <= 3:
            return msgs
        earlier = [("ok: " if st.get("ok") else "failed: ") + str(st.get("text", ""))[:140] for st in job.steps if st["kind"] == "observe"][:-1][-4:]
        first = dict(msgs[0])
        if earlier:
            first["content"] += "\n\nEARLIER ATTEMPTS (already tried, do not repeat): " + " | ".join(earlier)
        return [first] + msgs[-2:]

    def _bbox_wgs84(self, ds, margin_m=500.0):
        return tuple(float(v) for v in gpd.GeoSeries([ds.area_polygon().buffer(margin_m)], crs=ds.crs).to_crs(4326).total_bounds)

    def run(self, job: AnalysisJob) -> AnalysisJob:
        t0 = time.time()
        q = job.question
        recipe = recipes.match(q)
        llm = RecipeLLM(recipe) if recipe else self.llm
        job.mode = "vetted recipe" if recipe else "LLM-written code"
        job.status, job.pending = "running", None
        if llm is None:
            job.status, job.error = "unsupported", self.available(q)[1]
            return job
        bridge = LLMBridge(self.llm, max_calls=self.llm_calls) if (self.llm is not None and not recipe) else None
        client = getattr(self.ctx, "client", None)
        acq = AcquisitionTools(client, approved=job.approved) if (client is not None and not recipe and self.llm is not None) else None
        system = SYSTEM + (LLM_QUERY_DOC if bridge else "") + (ACQUIRE_DOC if acq else "")
        work = Path(tempfile.mkdtemp(prefix="riyadhai_layers_"))
        try:
            ds = self.ctx.area
            info = export_layers(ds, work, self.metro_dir)
            crs_s = ds.crs.to_string() if hasattr(ds.crs, "to_string") else str(ds.crs)
            meta = {"crs": crs_s, "scope_wkt": ds.area_polygon().wkt if ds.scope else None, "entities": {}}
            scope_name = ds.scope_label()
            job.log("data", text=f"loaded {len(info)} layers for {scope_name}: " + ", ".join(f"{k} ({len(v):,})" for k, v in info.items()))
            retrieved = ""
            if self.rag is not None:
                from pyproj import Transformer
                tr = Transformer.from_crs(4326, ds.crs, always_xy=True)
                self.rag.add_schema_stats(info)
                rt = self.rag.retrieve(q, to_metric=lambda lon, lat: tr.transform(lon, lat))
                retrieved = rt.prompt_text(budget=3600)
                job.entities = [{k: e[k] for k in ("name", "kind", "district", "score")} for e in rt.entities]
                meta["entities"] = {e["name"].lower(): {"x": e["x"], "y": e["y"], "kind": e["kind"], "district": e.get("district")} for e in rt.entities}
                job.log("retrieve", text=f"{len(rt.catalogue)} data notes, {len(rt.entities)} named places, {len(rt.examples)} similar past analyses, {len(rt.docs)} reference notes")
            gaps = missing_layers(q, info)
            gap_txt = ("\nNOT LOADED (the question names " + ", ".join(f"'{g}'" for g in gaps) + " but no such layer is loaded). Do NOT substitute another layer"
                       + ("; ask for it with the acquire action" if acq and all(g in ("poi", "building", "district", "connector") for g in gaps) else "; say it is unavailable") + ".\n") if gaps else ""
            if gaps:
                job.log("gap", text="layers named in the question but not loaded: " + ", ".join(gaps))
            user0 = f"QUESTION: {q}\nSELECTED AREA: {scope_name}\nDATA:\n{describe(info)}\n" + gap_txt + (f"\n{retrieved}\n" if retrieved else "") + "\nStart with the plan and the first code."
            msgs = [{"role": "user", "content": user0}]
            ran_ok, last_corpus, last_res_df, last_res = False, "", None, None
            for turn in range(self.max_turns):
                raw = llm(system, self._compact(msgs, job))
                obj = extract_json(raw)
                if obj is None:
                    nudge = "Your reply was empty. Reply with exactly one JSON object as specified, and keep any reasoning short." if not raw.strip() else "That was not valid JSON. Reply with exactly one JSON object as specified."
                    msgs += [{"role": "assistant", "content": raw[:2000] or "(empty)"}, {"role": "user", "content": nudge}]
                    job.log("format", text="the model reply was not valid JSON; asking again. Started with: " + raw[:160].replace("\n", " "))
                    continue
                msgs.append({"role": "assistant", "content": json.dumps(obj)[:6000]})
                job.log("thought", text=str(obj.get("thought", ""))[:300])
                act = obj.get("action")
                if act == "ask_user":
                    job.status, job.question_back = "needs_input", str(obj.get("question", ""))
                    return job
                if act == "acquire":
                    if acq is None:
                        msgs.append({"role": "user", "content": "Data acquisition is not available. Work with the layers you have."})
                        continue
                    try:
                        obs_d, newdata = acq.call(str(obj.get("tool", "")), obj.get("args", {}) or {}, self._bbox_wgs84(ds))
                        for lname, g in newdata.items():
                            gm = g.to_crs(ds.crs)
                            gm.to_parquet(work / f"acq_{lname}.parquet")
                            info[f"acq_{lname}"] = gm
                        job.acquisition.append({"tool": obj.get("tool"), "args": obj.get("args"), "result": json.loads(json.dumps(obs_d, default=str))})
                        job.log("acquire", text=f"{obj.get('tool')}: " + json.dumps(obs_d, default=str)[:240])
                        msgs.append({"role": "user", "content": "ACQUISITION RESULT:\n" + json.dumps(obs_d, default=str)[:2500]})
                    except NeedsApproval as e:
                        job.status, job.pending = "needs_approval", {"why": e.why, **e.action}
                        job.log("approval", text=e.why)
                        return job
                    except Exception as e:  # noqa: BLE001
                        job.log("acquire", text=f"refused: {e}")
                        msgs.append({"role": "user", "content": f"ACQUISITION REFUSED: {type(e).__name__}: {e}"})
                    continue
                if act == "final":
                    if not ran_ok:
                        msgs.append({"role": "user", "content": "You must run code that produces a result before giving the final answer."})
                        continue
                    job.final = obj.get("final", {})
                    break
                if act != "code" or not obj.get("code"):
                    msgs.append({"role": "user", "content": "Unknown action. Use action 'code', 'acquire', 'final' or 'ask_user'."})
                    continue
                if obj.get("plan"):
                    job.plan = obj["plan"]
                    job.log("plan", **{k: obj["plan"].get(k) for k in ("interpretation", "steps", "assumptions")})
                cand_code = obj["code"]
                job.log("act", text=f"running code (turn {turn + 1})")
                res = sandbox.run_code(cand_code, work, meta, self.timeout_s, bridge=bridge)
                if not res.ok:
                    obs = f"RUN FAILED:\n{res.error}\nstdout so far: {res.stdout[-600:]}"
                    job.log("observe", ok=False, text=res.error.splitlines()[-1][:300] if res.error else "failed")
                    msgs.append({"role": "user", "content": obs + "\nFix the code and try again."})
                    sandbox.cleanup(res)
                    continue
                rs_txt, df = result_summary(res)
                checks = sanity_checks(res, df)
                bsum = f"\nmodel calls used: {bridge.summary()['calls']} of {bridge.max_calls}" if bridge and bridge.calls else ""
                obs = f"RUN OK ({res.seconds:.1f}s)\nstdout:\n{res.stdout[-1500:] or '(nothing printed)'}\n{rs_txt}\nfigures: {len(res.figure_paths)}\nnotes: {res.notes}\nchecks: {checks or 'passed'}{bsum}"
                job.log("observe", ok=True, text=rs_txt.splitlines()[0], checks=checks)
                msgs.append({"role": "user", "content": obs + "\nIf the result answers the question, reply with action 'final'. Otherwise fix or extend the code."})
                if last_res is not None:
                    sandbox.cleanup(last_res)
                ran_ok, last_corpus, last_res_df, last_res = True, obs, df, res
                job.code = cand_code   # the code shown to the user is always the version that produced the result
                job.checks, job.notes = checks, res.notes
            if not ran_ok:
                job.status, job.error = "failed", "no working analysis was produced within the turn limit"
                return job
            if bridge is not None:
                job.llm_calls = bridge.summary()
            if not job.final:  # the model never gave a usable explanation: describe the result deterministically from the numbers themselves
                jv = None
                if last_res is not None and last_res.result_kind == "json":
                    try:
                        jv = json.loads(Path(last_res.result_path).read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        jv = None
                job.final = explain_generic(last_res_df, last_res.stdout if last_res else "", jv)
            self._finish(job, last_res, last_res_df, last_corpus)
            return job
        except Exception as e:  # noqa: BLE001
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            return job
        finally:
            shutil.rmtree(work, ignore_errors=True)
            job.seconds = round(time.time() - t0, 1)

    def _finish(self, job: AnalysisJob, res: sandbox.RunResult, df: Optional[pd.DataFrame], corpus: str):
        if df is not None and len(df):
            show = pd.DataFrame(df.drop(columns="geometry", errors="ignore"))
            job.table = json.loads(show.head(50).round(3).to_json(orient="records"))
            if isinstance(df, gpd.GeoDataFrame) and len(df):
                rs = self.ctx.results.put(df, kind="code_analysis", producer="codeagent", params={"question": job.question})
                job.result_id = rs.id
        if res.figure_paths:
            name = f"fig_{job.id}.png"
            shutil.copy(res.figure_paths[0], self.fig_dir / name)
            job.figure = name
        text = " ".join([str(job.final.get("headline", ""))] + [str(x) for x in job.final.get("points", [])])
        job.unverified_numbers = audit_numbers(text, corpus + " " + json.dumps(job.table)[:4000] + " " + job.question)   # numbers the user typed are not claims
        if job.llm_calls.get("calls"):
            job.final.setdefault("caveats", []).append(f"Some labels in this analysis were assigned by a language model ({job.llm_calls['calls']} calls): treat them as judgements, not measurements.")
        if job.acquisition:
            job.final.setdefault("caveats", []).append("Extra open data was downloaded for this analysis (see the acquisition log and its verification report).")
        job.status = "done"
        if self.rag is not None and not job.unverified_numbers and not job.checks and job.code:
            self.rag.remember(job.question, job.plan, job.code)   # only analyses that ran, passed the checks and had traceable numbers become examples
        sandbox.cleanup(res)
        job.log("final", text=str(job.final.get("headline", ""))[:200])
