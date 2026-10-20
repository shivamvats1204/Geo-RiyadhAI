"""Code-writing analysis agent: sandbox safety, plan/act/observe/repair loop, number audit, recipe matching.
The model is replaced by a scripted stand-in (no API key in CI): these tests check the machinery around the model, not the model's quality."""
import json
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from riyadhai import codeagent as CA
from riyadhai import recipes, sandbox


def _layers(tmp):
    d = Path(tempfile.mkdtemp(dir=tmp))
    gpd.GeoDataFrame({"name": ["a", "b", "c"], "cap": [5, 12, 30]}, geometry=[Point(0, 0), Point(100, 0), Point(5000, 0)], crs=32638).to_parquet(d / "parking.parquet")
    return d


@pytest.mark.parametrize("bad", ["import os", "import requests", "x = open('f')", "x = ().__class__", "import pandas as pd\npd.read_csv('http://x')", "exec('1')", "getattr(1, 'real')",
                                 "import pandas as pd\npd.DataFrame().to_csv('x.csv')"])
def test_static_check_blocks_files_network_and_escapes(bad):
    assert sandbox.validate_code(bad)


def test_static_check_allows_normal_analysis_code():
    assert sandbox.validate_code("import numpy as np\nimport pandas as pd\nresult = pd.DataFrame({'a': np.arange(3)}).groupby('a').size()") == []


def test_sandbox_runs_code_returns_result_figure_and_reports_errors_and_timeouts(tmp_path):
    d = _layers(tmp_path)
    meta = {"crs": "EPSG:32638", "scope_wkt": None}
    ok = sandbox.run_code("p = layers['parking']\np['near'] = count_within(p, p, 200, weight='cap')\nplt.bar(p['name'], p['near'])\nprint('rows', len(p))\nresult = p[['name', 'near']]", d, meta)
    assert ok.ok and ok.result_kind == "dataframe" and len(ok.figure_paths) == 1 and "rows 3" in ok.stdout
    import pandas as pd
    assert pd.read_parquet(ok.result_path).set_index("name")["near"].to_dict() == {"a": 17.0, "b": 17.0, "c": 30.0}   # a and b are within 200 m of each other: 5 + 12
    bad = sandbox.run_code("result = 1 / 0", d, meta)
    assert not bad.ok and "ZeroDivisionError" in bad.error
    slow = sandbox.run_code("while True:\n    pass", d, meta, timeout_s=3)
    assert not slow.ok and "stopped" in slow.error
    assert not sandbox.run_code("import os", d, meta).ok  # rejected before any process starts


def test_number_audit_flags_numbers_the_code_never_produced():
    corpus = "SUMMARY {\"best_minutes\": 102.4, \"n\": 1321, \"share\": 0.043}"
    assert CA.audit_numbers("About 102 minutes; 1,321 homes; 4% of lots", corpus) == []
    assert CA.audit_numbers("About 250 minutes", corpus) == [250.0]


def test_recipe_matching_parses_the_question_and_ignores_unrelated_ones():
    m = recipes.match("find me those residential houses that has best connection to airports via metro and has a parking of at least 10 vehicles within 1 km")
    assert m and m["min_vehicles"] == 10 and m["radius_m"] == 1000.0 and m["airport"]
    assert recipes.match("homes near a metro station with parking for 25 cars within 500 m")["radius_m"] == 500.0
    assert recipes.match("Count shops in each district") is None
    plan, code = recipes.render(m)
    assert "MIN_VEHICLES = 10" in code and sandbox.validate_code(code) == [] and plan["assumptions"]


class Scripted:
    """Stand-in for the LLM: replays a fixed conversation (first a broken script, then a fixed one, then the final answer)."""
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, system, messages):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0)


def test_agent_repairs_from_the_error_message_then_answers_with_audited_numbers(world):
    _, ds, _, ctx = world
    broken = json.dumps({"thought": "count parking", "action": "code", "plan": {"interpretation": "count parking rows", "steps": ["count"], "assumptions": []},
                         "code": "result = layers['no_such_layer'].shape[0]"})
    fixed = json.dumps({"thought": "use the poi layer", "action": "code", "code": "n = len(layers['poi'])\nprint('n_poi', n)\nresult = {'n_poi': n}"})
    n = len(ds.layers["poi"])
    final = json.dumps({"thought": "done", "action": "final", "final": {"headline": f"There are {n} places mapped here.", "points": ["invented 987654 figure"], "caveats": [], "next_questions": []}})
    llm = Scripted([broken, fixed, final])
    analyst = CA.CodeAnalyst(ctx, llm, metro_dir=Path("does_not_exist"))
    job = analyst.run(CA.AnalysisJob(id="abc1234567", question="how many places are mapped"))
    assert job.status == "done" and job.mode == "LLM-written code"
    assert "RUN FAILED" in llm.seen[1] and "KeyError" in llm.seen[1]                      # the model was shown the error and fixed it
    assert job.unverified_numbers == [987654.0]                                        # an invented number in the explanation is flagged
    kinds = [s["kind"] for s in job.steps]
    assert kinds.index("plan") < kinds.index("act") < kinds.index("observe") and kinds[-1] == "final"


def test_agent_without_llm_declines_non_recipe_questions_and_never_guesses(world):
    _, _, _, ctx = world
    analyst = CA.CodeAnalyst(ctx, None)
    job = analyst.run(CA.AnalysisJob(id="abc1234568", question="what is the correlation between building height and parking"))
    assert job.status == "unsupported" and "language model" in job.error


def test_dotenv_loader_parses_quotes_comments_and_never_overrides_real_variables(tmp_path, monkeypatch):
    from riyadhai import envfile
    f = tmp_path / ".env"
    f.write_text("# comment\nANTHROPIC_API_KEY=\"sk-test\"\nexport RIYADHAI_LLM_MODEL=my-model  # trailing\nRIYADHAI_LLM_URL='http://localhost:11434/v1'\nBAD LINE\nKEPT=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RIYADHAI_LLM_MODEL", raising=False)
    monkeypatch.delenv("RIYADHAI_LLM_URL", raising=False)
    monkeypatch.setenv("KEPT", "from-env")
    monkeypatch.delenv("RIYADHAI_NO_DOTENV")
    assert envfile.load_dotenv([f]) == [str(f)]
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test" and os.environ["RIYADHAI_LLM_MODEL"] == "my-model" and os.environ["RIYADHAI_LLM_URL"] == "http://localhost:11434/v1"
    assert os.environ["KEPT"] == "from-env"                      # a real environment variable wins over the file
    assert envfile.load_dotenv([tmp_path / "missing.env"]) == []  # a missing file is fine
    monkeypatch.delenv("ANTHROPIC_API_KEY"); monkeypatch.delenv("RIYADHAI_LLM_MODEL"); monkeypatch.delenv("RIYADHAI_LLM_URL")


def test_generic_explanation_is_built_only_from_the_result_numbers():
    import pandas as pd
    one = CA.explain_generic(pd.DataFrame({"total_residential_buildings": [1200], "within_500m": [87]}))
    assert "total residential buildings: 1,200" in one["headline"] and "within 500m: 87" in one["headline"] and one["caveats"]
    many = CA.explain_generic(pd.DataFrame({"est_minutes": [102.0, 110.0, 120.0]}))
    assert "3 rows" in many["headline"] and "102" in many["points"][1] and "120" in many["points"][1]
    assert "nothing" in CA.explain_generic(pd.DataFrame({"a": []}))["headline"]


def test_gap_detector_flags_layers_the_question_names_but_that_are_not_loaded():
    loaded = ["urbi_building", "poi", "junction", "metro_stations"]
    assert CA.missing_layers("How many road connectors are mapped? Use the connector layer.", loaded) == ["connector"]
    assert CA.missing_layers("homes near a metro station with parking", loaded) == ["parking"]
    assert CA.missing_layers("How many buildings and junctions are here?", loaded) == []
    d = CA.explain_generic(None, "", {"indian_restaurant_count": 7, "example_names": ["Makani", "Shezan"]})
    assert "indian restaurant count: 7" in d["headline"] and "Makani; Shezan" in d["points"][0]
