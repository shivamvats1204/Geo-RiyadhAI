import json

import pytest
from pydantic import ValidationError

from geoacquirer.agent import LLMCompiler, RuleCompiler
from geoacquirer.ontology import category_mask, find_concepts_in_text, resolve
from geoacquirer.query_ir import (And, EntityType, GeoQuery, Not, ObjectSpec, Op, Predicate, iter_predicates)
import pandas as pd

FLAGSHIP = ("Find shops in Riyadh that are located at crossroads but are not on highways. Return their names, categories, "
            "coordinates, associated buildings/parcels, nearby roads, and geographic evidence. Show the results on a map "
            "and export them to CSV and GeoJSON.")


def test_flagship_compiles_to_expected_ir_with_visible_defaults():
    q = RuleCompiler().compile(FLAGSHIP)
    preds = list(iter_predicates(q.constraint))
    assert [p.op for p in preds] == [Op.AT_INTERSECTION, Op.NEAR]
    assert preds[0].junction_kind == "four_way" and preds[0].distance_m == 30
    assert isinstance(q.constraint, And) and isinstance(q.constraint.children[1], Not)
    hw = preds[1]
    assert hw.object.road_classes == ["motorway", "trunk"] and hw.distance_m == 60
    assert q.defaults_applied["crossroad_distance_m"] == 30 and q.defaults_applied["highway_exclusion_m"] == 60
    assert set(q.output.formats) >= {"csv", "geojson", "map"}
    assert q.primary().categories == ["shop"] and q.area.name == "Riyadh"


def test_explicit_distances_override_defaults():
    q = RuleCompiler().compile("Find shops within 25 metres of four-way intersections, exclude those within 80 m of motorways, in Riyadh")
    preds = list(iter_predicates(q.constraint))
    assert preds[0].distance_m == 25 and preds[1].distance_m == 80
    assert "crossroad_distance_m" not in q.defaults_applied


def test_parking_target_and_object_concept():
    q = RuleCompiler().compile("Find parking facilities within 150 metres of shops in Riyadh")
    assert q.primary().entity_type == EntityType.PARKING
    p = q.constraint
    assert p.op == Op.NEAR and p.distance_m == 150 and p.object.categories == ["shop"]


def test_road_distance_clinics_hospitals():
    q = RuleCompiler().compile("Find clinics within 2 km road distance of hospitals in Riyadh")
    p = q.constraint
    assert p.op == Op.DISTANCE_BY_ROAD and p.distance_m == 2000 and p.object.categories == ["hospital"]
    assert p.exactness.value == "approximate"


def test_parcels_touching_two_roads_and_zone_exclusion():
    q = RuleCompiler().compile("Find commercial parcels touching two roads in Riyadh")
    assert q.primary().entity_type == EntityType.PARCEL and q.primary().categories == ["commercial"]
    assert q.constraint.op == Op.HAS_FRONTAGE_ON and q.constraint.min_count == 2
    q2 = RuleCompiler().compile("Find warehouses outside residential zones in Riyadh")
    assert q2.constraint.op == Op.NOT_WITHIN and q2.constraint.object.entity_type == EntityType.PARCEL


def test_negated_proximity():
    q = RuleCompiler().compile("Find shops in Riyadh that are not within 300 metres of a hospital")
    assert isinstance(q.constraint, Not) and q.constraint.child.distance_m == 300


def test_unknown_query_raises_cleanly():
    with pytest.raises(ValueError):
        RuleCompiler().compile("tell me a joke")


def test_ir_validation_is_strict():
    with pytest.raises(ValidationError):
        Predicate(op=Op.NEAR, distance_m=50)  # no object
    with pytest.raises(ValidationError):
        Predicate(op=Op.NEAR, object=ObjectSpec(entity_type=EntityType.ROAD))  # no distance
    with pytest.raises(ValidationError):
        Predicate(op=Op.NEAR, distance_m=-5, object=ObjectSpec(entity_type=EntityType.ROAD))
    with pytest.raises(ValidationError):
        GeoQuery.model_validate({"raw_text": "x", "targets": [], "area": {"name": "a"}})
    with pytest.raises(ValidationError):  # extra fields (e.g. smuggled SQL) are rejected
        GeoQuery.model_validate({"raw_text": "x", "targets": [{"entity_type": "poi"}], "area": {"name": "a"}, "sql": "DROP TABLE x"})
    with pytest.raises(ValidationError):
        GeoQuery.model_validate({"raw_text": "x", "targets": [{"entity_type": "poi"}], "area": {"name": "a", "bbox": [47, 24, 46, 25]}})


def test_ir_roundtrip_json():
    q = RuleCompiler().compile(FLAGSHIP)
    again = GeoQuery.model_validate_json(q.model_dump_json())
    assert again == q


def test_llm_compiler_validates_and_falls_back():
    good = RuleCompiler().compile(FLAGSHIP).model_dump_json()
    c = LLMCompiler(lambda prompt: "```json\n" + good + "\n```")
    assert not c.used_fallback and c.compile(FLAGSHIP).primary().categories == ["shop"]
    bad = LLMCompiler(lambda prompt: json.dumps({"raw_text": "x", "sql": "DROP TABLE users"}))
    q = bad.compile(FLAGSHIP)
    assert bad.used_fallback and q.primary().categories == ["shop"]
    boom = LLMCompiler(lambda prompt: (_ for _ in ()).throw(RuntimeError("network down")))
    boom.compile(FLAGSHIP)
    assert boom.used_fallback and "network down" in boom.last_error


def test_ontology_multilingual_and_hierarchy():
    assert resolve("متاجر")[0].id == "shop"
    assert resolve("pharmacies")[0].id == "pharmacy"
    assert [c.id for c in find_concepts_in_text("restaurants near the mosque")][:2] == ["restaurant", "mosque"]
    paths = pd.Series([["shopping", "food_and_beverage_store", "grocery_store"], ["food_and_drink", "restaurant"],
                       ["health_care", "pharmacy"], ["health_care", "hospital"], None])
    assert category_mask(paths, ["shop"]).tolist() == [True, False, False, False, False]
    assert category_mask(paths, ["healthcare"]).tolist() == [False, False, False, True, False]  # pharmacy excluded by default
    assert category_mask(paths, ["clinic"]).tolist() == [False] * 5  # a hospital is not a clinic
    assert category_mask(pd.Series([["health_care", "clinic"]]), ["clinic"]).tolist() == [True]
    assert category_mask(paths, ["supermarket"]).tolist() == [True, False, False, False, False]
