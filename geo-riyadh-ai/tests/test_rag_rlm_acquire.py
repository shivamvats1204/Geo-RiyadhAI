"""Geo-RAG, the budgeted model-call bridge (RLM step) and guarded data acquisition. The model is a scripted stand-in (no API key in CI)."""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from riyadhai import acquire as AQ
from riyadhai import codeagent as CA
from riyadhai import sandbox
from riyadhai.georag import GeoRAG
from riyadhai.llmbridge import BudgetExceeded, LLMBridge


# ------------------------------------------------------------------ geo-RAG
def _entities():
    return pd.DataFrame([
        {"name": "Kingdom Centre", "name_ar": "برج المملكة", "kind": "attraction", "lon": 46.674, "lat": 24.711, "district": "Al Olaya"},
        {"name": "Al Murabba", "name_ar": "المربع", "kind": "urbi district", "lon": 46.71, "lat": 24.65, "district": None},
        {"name": "Murabba park", "name_ar": None, "kind": "attraction", "lon": 46.72, "lat": 24.66, "district": "Al Murabba"},
        {"name": None, "name_ar": "محطة وزارة المالية", "kind": "metro station", "lon": 46.71, "lat": 24.65, "district": None},
        {"name": "Park", "name_ar": None, "kind": "attraction", "lon": 46.7, "lat": 24.7, "district": None},
        {"name": "King Fahd road", "name_ar": None, "kind": "street", "lon": 46.69, "lat": 24.7, "district": "Al Olaya"},
    ])


def test_entity_resolution_english_arabic_and_generic_names():
    r = GeoRAG()
    r.set_entities(_entities())
    top = lambda q: [(e["name"], e["kind"]) for e in r.find_entities(q)]
    assert top("homes near Kingdom Centre with parking")[0] == ("Kingdom Centre", "attraction")
    assert top("what about Al Murabba")[0] == ("Al Murabba", "urbi district")             # the district beats 'Murabba park'
    assert ("محطة وزارة المالية", "metro station") in top("houses near the Ministry of Finance")  # English reaches an Arabic-only name
    assert ("Park", "attraction") not in top("close to Riyadh Park mall")                      # a bare generic name says nothing about WHICH park
    assert top("count shops in each district") == []                                          # no place named: nothing invented


def test_catalogue_retrieval_surfaces_the_pitfall_notes_for_the_question():
    r = GeoRAG()
    keys = [c["key"] for c in r.retrieve("parking with at least 10 vehicles capacity").catalogue]
    assert "layers['parking'].capacity" in keys
    assert "layers['urbi_building'].purpose_class" in [c["key"] for c in r.retrieve("residential houses").catalogue]


def test_examples_are_remembered_and_retrieved_and_survive_a_restart(tmp_path):
    f = tmp_path / "ex.jsonl"
    r = GeoRAG(example_path=f)
    r.remember("homes near metro with parking", {"steps": []}, "result = 1")
    assert r.similar_examples("residential homes close to the metro and parking")[0]["code"] == "result = 1"
    assert GeoRAG(example_path=f).similar_examples("homes near metro with parking")           # persisted
    assert r.similar_examples("count pharmacies per district") == []


# ------------------------------------------------------------------ bridge
def test_bridge_enforces_budget_size_cache_and_logs():
    calls = []
    b = LLMBridge(lambda system, msgs: calls.append(msgs[0]["content"]) or "ok", max_calls=2, max_chars_per_call=50)
    assert b("one") == "ok" and b("one") == "ok" and len(calls) == 1                         # second identical prompt served from cache
    assert b("two") == "ok"
    with pytest.raises(BudgetExceeded, match="budget"):
        b("three")
    with pytest.raises(BudgetExceeded, match="characters"):
        LLMBridge(lambda s, m: "x", max_chars_per_call=10)("x" * 11)
    s = b.summary()
    assert s["calls"] == 2 and s["cached"] == 1 and s["log"][0]["prompt"] == "one"


def _layers(tmp):
    d = Path(tempfile.mkdtemp(dir=tmp))
    gpd.GeoDataFrame({"name": ["Pharmacy Nahdi", "Bakery", "Al Dawaa pharmacy"]}, geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs=32638).to_parquet(d / "poi.parquet")
    return d


def test_sandbox_services_model_calls_through_the_bridge_and_surfaces_budget_errors(tmp_path):
    d = _layers(tmp_path)
    meta = {"crs": "EPSG:32638", "scope_wkt": None}
    bridge = LLMBridge(lambda system, msgs: json.dumps(["pharmacy" if "harm" in x or "awa" in x else "other" for x in json.loads(msgs[0]["content"].split("INPUTS:")[1])]), max_calls=1)
    code = "p = layers['poi']\np['kind'] = llm_map(list(p['name']), 'Classify each name as pharmacy or other')\nresult = p[['name', 'kind']]"
    r = sandbox.run_code(code, d, meta, bridge=bridge)
    assert r.ok and pd.read_parquet(r.result_path)["kind"].tolist() == ["pharmacy", "other", "pharmacy"]
    assert bridge.summary()["calls"] == 1
    again = sandbox.run_code("try:\n    llm_query('a different prompt')\nexcept Exception as e:\n    result = str(e)", d, meta, bridge=bridge)   # budget of 1 is already used
    assert again.ok and "budget" in open(again.result_path, encoding="utf-8").read()
    assert not sandbox.run_code("result = llm_query('hi')", d, meta).ok                       # without a configured model the helper does not exist


# ------------------------------------------------------------------ acquisition guardrails
@dataclass
class FakeResult:
    layers: dict
    from_cache: dict
    provenance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    seconds: float = 0.1


class FakeClient:
    def __init__(self, cached=False):
        self.cached, self.fetched = cached, []

    def cached_layers(self, bbox, layers):
        return {l: self.cached for l in layers}

    def fetch_layers(self, bbox, layers, allow_live=True):
        self.fetched.append(tuple(layers))
        w, s, e, n = bbox
        g = {l: gpd.GeoDataFrame({"name": ["x", None]}, geometry=[Point((w + e) / 2, (s + n) / 2), Point(w + 1e-4, s + 1e-4)], crs=4326) for l in layers}
        return FakeResult(g, {l: self.cached for l in layers})


BBOX = (46.70, 24.65, 46.72, 24.67)


def test_acquisition_guardrails_estimate_approval_budget_and_verification():
    live = AQ.AcquisitionTools(FakeClient(cached=False), approved=False)
    est = live.estimate(["poi"], BBOX)
    assert est["needs_live_download"] and est["approval_needed"] and est["est_total_seconds"] > 0
    with pytest.raises(AQ.NeedsApproval):
        live.fetch(["poi"], BBOX)                                                           # live download never runs without approval
    assert live.client.fetched == []
    ok = AQ.AcquisitionTools(FakeClient(cached=False), approved=True)
    data, rep = ok.fetch(["poi"], BBOX)
    assert len(data["poi"]) == 2 and rep["verification"]["poi"]["rows"] == 2 and "missing_name" in rep["verification"]["poi"]
    cached = AQ.AcquisitionTools(FakeClient(cached=True), approved=False)
    assert cached.fetch(["poi"], BBOX)[0]                                                    # cached data needs no approval
    for _ in range(AQ.MAX_FETCHES - 1):
        cached.fetch(["poi"], BBOX)
    with pytest.raises(AQ.AcquisitionError, match="budget"):
        cached.fetch(["poi"], BBOX)
    with pytest.raises(AQ.AcquisitionError, match="layers"):
        ok.estimate(["secrets"], BBOX)
    with pytest.raises(AQ.AcquisitionError, match="too large"):
        ok.estimate(["poi"], (46.0, 24.0, 47.0, 25.0))
    assert not AQ.AcquisitionTools(FakeClient()).verify("poi", gpd.GeoDataFrame(geometry=[], crs=4326), BBOX)["ok"]   # empty = unknown, flagged
    src = AQ.AcquisitionTools(FakeClient()).list_sources()
    assert src["overture"]["usable_by_this_tool"] and not src["suhail"]["usable_by_this_tool"] and not src["urbi"]["usable_by_this_tool"]


# ------------------------------------------------------------------ agent end to end with all three
class Scripted:
    def __init__(self, replies):
        self.replies, self.agent_calls, self.sub_calls = list(replies), 0, 0

    def __call__(self, system, messages):
        if system.startswith("You are a precise helper"):                                   # a sub-call from llm_map inside the sandbox
            self.sub_calls += 1
            items = json.loads(messages[0]["content"].split("INPUTS:")[1])
            return json.dumps(["pharmacy" if "harm" in i or "awa" in i else "other" for i in items])
        self.agent_calls += 1
        return self.replies.pop(0)


def test_agent_uses_entities_model_calls_and_acquisition_with_approval(world):
    _, ds, _, ctx = world
    ctx.client = FakeClient(cached=False)
    rag = GeoRAG()
    cx, cy = ds.area_polygon().representative_point().coords[0]
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(ds.crs, 4326, always_xy=True).transform(cx, cy)
    rag.set_entities(pd.DataFrame([{"name": "Test Tower", "name_ar": None, "kind": "attraction", "lon": lon, "lat": lat, "district": None}]))
    est = json.dumps({"thought": "check cost", "action": "acquire", "tool": "estimate", "args": {"layers": ["building"]}})
    fetch = json.dumps({"thought": "need buildings", "action": "acquire", "tool": "fetch", "args": {"layers": ["building"]}})
    code = json.dumps({"thought": "classify names with the model, use the resolved place", "action": "code", "plan": {"interpretation": "classify poi names", "steps": ["map"], "assumptions": []},
                       "code": "p = layers['poi'].head(6).copy()\np['kind'] = llm_map(list(p['name'].fillna('')), 'Is this a pharmacy? answer pharmacy or other')\nt = entities['test tower']\nprint('acq', len(layers['acq_building']), 'tower', round(t['x']) is not None)\nresult = p[['name', 'kind']]"})
    final = json.dumps({"thought": "done", "action": "final", "final": {"headline": "Classified the sample.", "points": [], "caveats": [], "next_questions": []}})
    llm = Scripted([est, fetch, est, fetch, code, final])
    analyst = CA.CodeAnalyst(ctx, llm, rag=rag, metro_dir=Path("none"))
    job = analyst.run(CA.AnalysisJob(id="abc1234569", question="classify the pharmacies near Test Tower"))
    assert job.status == "needs_approval" and "building" in job.pending["why"] and ctx.client.fetched == []   # paused BEFORE any live download
    assert [e["name"] for e in job.entities] == ["Test Tower"]
    job.approved = True
    job = analyst.run(job)
    assert job.status == "done", job.error
    assert ctx.client.fetched == [("building",)] and job.llm_calls["calls"] >= 1 and llm.sub_calls >= 1
    assert any("language model" in c for c in job.final["caveats"]) and any("downloaded" in c for c in job.final["caveats"])
    assert job.acquisition and job.acquisition[-1]["result"]["verification"]["building"]["rows"] == 2
    assert rag.similar_examples("classify the pharmacies near the tower")                      # verified analysis became an example
