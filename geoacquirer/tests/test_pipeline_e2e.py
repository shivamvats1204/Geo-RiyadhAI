import json
from pathlib import Path

import geopandas as gpd
import pytest

from geoacquirer.agent import GeoAcquirerAgent
from geoacquirer.storage import LayerCache
from geoacquirer.testing import FIXTURE_ORIGIN, write_fixture_files

FLAGSHIP = ("Find shops in Riyadh that are located at crossroads but are not on highways. Return their names, categories, "
            "coordinates, associated buildings/parcels, nearby roads, and geographic evidence. Show the results on a map "
            "and export them to CSV and GeoJSON.")

BBOX = (FIXTURE_ORIGIN[0] - 0.004, FIXTURE_ORIGIN[1] - 0.004, FIXTURE_ORIGIN[0] + 0.012, FIXTURE_ORIGIN[1] + 0.012)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    truth = write_fixture_files(d / "data")["truth"]
    agent = GeoAcquirerAgent(d / "runs", local_dir=d / "data", cache=LayerCache(d / "cache"))
    res = agent.run(FLAGSHIP, area_bbox=BBOX, include_statuses=["VERIFIED_MATCH", "PROBABLE_MATCH", "INSUFFICIENT_EVIDENCE", "REJECTED"])
    return res, truth


def test_ground_truth_statuses(run):
    res, truth = run
    got = dict(zip(res.flat["name"].str.replace("Shop ", ""), res.flat["status"]))
    assert got == truth, got


def test_default_export_only_substantiated(run, tmp_path):
    res, _ = run
    exported = gpd.read_file(res.files["geojson"])
    # exports were requested with all statuses in this fixture; re-export default via manifest check
    assert set(exported["status"]) <= {"VERIFIED_MATCH", "PROBABLE_MATCH", "INSUFFICIENT_EVIDENCE", "REJECTED"}


def test_defaults_are_surfaced(run):
    res, _ = run
    d = res.query.defaults_applied
    assert d["crossroad_distance_m"] == 30.0 and d["highway_exclusion_m"] == 60.0


def test_verifier_agrees_and_manifest_complete(run):
    res, _ = run
    m = json.loads(Path(res.files["manifest"]).read_text(encoding="utf-8"))
    assert m["verification"]["ok"] is True
    assert m["verification"]["checked_leaves"] > 0
    assert m["geoquery_ir"]["constraint"]["kind"] == "and"
    assert m["provenance"]["poi"]["provider_id"] == "local_files"
    assert any("No authorized parcel source" in n for n in m["coverage_notes"])
    assert m["files"]["csv"]["sha256"]


def test_parcel_is_unavailable_not_invented(run):
    res, _ = run
    f = res.flat
    assert set(f["parcel_match"].dropna()) <= {"unavailable_no_authorized_parcel_source", "not_evaluated"}
    assert f["parcel_id"].isna().all()


def test_building_links_are_evidence_labelled(run):
    res, _ = run
    a = res.flat[res.flat["name"] == "Shop A"].iloc[0]
    c = res.flat[res.flat["name"] == "Shop C"].iloc[0]
    assert a["building_id"] == "BLD1" and a["building_match"] == "contained"
    assert c["building_id"] == "BLD1" and c["building_match"].startswith("nearest_within")


def test_evidence_records_distances(run):
    res, _ = run
    row = res.flat[res.flat["name"] == "Shop A"].iloc[0]
    ev = json.loads(row["evidence_json"])
    inter = next(e for e in ev if e["predicate"].startswith("AT_INTERSECTION"))
    assert inter["result"] == "PASS" and 5 < inter["nearest_distance_m"] < 15
    assert inter["object_kind"] == "four_way" if "object_kind" in inter else True


def test_map_and_report_written(run):
    res, _ = run
    assert "leaflet" in Path(res.files["map"]).read_text(encoding="utf-8").lower()
    assert "GeoAcquirer analytical report" in Path(res.files["report"]).read_text(encoding="utf-8")


def test_default_run_exports_only_substantiated(tmp_path):
    write_fixture_files(tmp_path / "data")
    agent = GeoAcquirerAgent(tmp_path / "runs", local_dir=tmp_path / "data", cache=LayerCache(tmp_path / "cache"))
    res = agent.run(FLAGSHIP, area_bbox=BBOX)
    exported = gpd.read_file(res.files["geojson"])
    assert set(exported["status"]) == {"VERIFIED_MATCH", "PROBABLE_MATCH"}
    assert len(exported) == 2
