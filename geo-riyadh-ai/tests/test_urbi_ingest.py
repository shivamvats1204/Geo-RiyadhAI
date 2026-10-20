import json
from shapely.geometry import Polygon

from riyadhai.urbi_ingest import load_attractions, load_buildings, load_metro, purpose_class


def _write(tmp_path, name, recs):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return p


def test_purpose_mapping():
    assert purpose_class("Villa") == "residential" and purpose_class("Residential building") == "residential"
    assert purpose_class("Commercial building") == "commercial" and purpose_class("Hotel") == "commercial"
    assert purpose_class("Mosque") == "other" and purpose_class("Utility structure") == "other"
    assert purpose_class(None) == "unknown" and purpose_class("") == "unknown"


def test_building_parser_prefers_polygon_and_reads_floors(tmp_path):
    rec = {"id": "123_456", "purpose_name": "Villa", "purpose_code": 520, "floors": {"ground_count": 2}, "address_name": "3 Street",
           "adm_div": [{"type": "district", "name": "Al Olaya"}],
           "geometry": {"centroid": "POINT(46.68 24.67)", "hover": "POLYGON((46.68 24.67,46.681 24.67,46.681 24.671,46.68 24.671,46.68 24.67))"}}
    bad = {"id": "9", "geometry": {}}  # no geometry -> skipped, never invented
    g = load_buildings(_write(tmp_path, "b.jsonl", [rec, bad]))
    assert len(g) == 1 and g.urbi_id[0] == "123" and g.floors[0] == 2 and g.district[0] == "Al Olaya"
    assert isinstance(g.geometry[0], Polygon)


def test_point_fallback_and_metro_attraction_parsers(tmp_path):
    a = load_attractions(_write(tmp_path, "a.jsonl", [{"id": "1", "name": "Park", "subtype": "park", "rubrics": [{"name": "Parks"}], "geometry": {"centroid": "POINT(46.7 24.7)"}}]))
    assert a.geometry[0].geom_type == "Point" and a.rubrics[0] == "Parks"
    m = load_metro(_write(tmp_path, "m.jsonl", [{"id": "2", "name": "Station", "type": "station_entrance", "geometry": {"centroid": "POINT(46.7 24.7)"}}]))
    assert m.kind[0] == "station_entrance"
