import pytest
from riyadhai import planner

CASES = [
    ("I want to open a supermarket in the north, where would you put it?", "rank_site_suitability"),
    ("top 3 spots for a new grocery in central riyadh", "rank_site_suitability"),
    ("Is there room for more parking near shops?", "optimize_facility_locations"),
    ("give me 4 places to add public parking", "optimize_facility_locations"),
    ("how many shops are there per neighbourhood", "aggregate_spatial_statistics"),
    ("number of clinics in each district", "aggregate_spatial_statistics"),
    ("show me cafes that sit on a four way junction", "run_geoquery"),
    ("list all pharmacies close to hospitals", "run_geoquery"),
    ("banks within 100 m of a mosque, away from motorways", "run_geoquery"),
    ("what is a T-junction in your data", "retrieve_reference_documents"),
    ("tell me about zoning setbacks", "retrieve_reference_documents"),
    ("shops at crossroads", "run_geoquery"),
    ("which areas are underserved by parking?", None),
    ("how crowded is parking in Olaya at 6pm?", None),
    ("estimate the population of Malqa", None),
    ("draw a route from Olaya to the airport", None),
    ("Which neighbourhood has the best schools?", None),
]


@pytest.mark.parametrize("q,expected", CASES)
def test_paraphrase_routing(q, expected):
    p = planner.plan(q, district_names=["Olaya", "Malqa"])
    if expected is None:
        assert not p.supported, [s.tool for s in p.steps]
    else:
        assert expected in [s.tool for s in p.steps], [s.tool for s in p.steps]


def test_forecast_questions_are_refused():
    for q in ("predict where new shops will open next year", "forecast parking demand", "what will the population be in 2030"):
        assert not planner.plan(q).supported
