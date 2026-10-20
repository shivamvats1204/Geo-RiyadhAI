
import pytest
from shapely.geometry import Polygon, box

from geoacquirer.providers.osm import parse_overpass
from geoacquirer.providers.registry import score_sources
from geoacquirer.providers.local import scan_inventory, guess_entity_type
from geoacquirer.security import ProviderDisabled
from geoacquirer.spatial import build_junctions
from geoacquirer.testing import write_fixture_files


def test_unauthorized_providers_are_disabled(monkeypatch):
    for k in ("GEOACQUIRER_ENABLE_SUHAIL", "GEOACQUIRER_SUHAIL_AUTHORIZATION_REF", "GEOACQUIRER_ENABLE_URBI", "GEOACQUIRER_URBI_AUTHORIZATION_REF"):
        monkeypatch.delenv(k, raising=False)
    from geoacquirer.providers.suhail import SuhailProvider
    from geoacquirer.providers.urbi import UrbiProvider
    with pytest.raises(ProviderDisabled):
        SuhailProvider()
    with pytest.raises(ProviderDisabled):
        UrbiProvider()
    scores = {s.provider_id: s for s in score_sources("parcel", set())}
    assert scores["suhail"].eligible is False
    elig = {s.provider_id for s in score_sources("parking", set()) if s.eligible}
    assert elig == {"local_files", "overture"} and "urbi" not in elig  # urbi stays disabled without recorded authorization


def test_enabling_requires_recorded_authorization_and_key(monkeypatch):
    monkeypatch.setenv("GEOACQUIRER_ENABLE_URBI", "1")
    from geoacquirer.providers.urbi import UrbiProvider
    with pytest.raises(ProviderDisabled):  # flag alone is not enough
        UrbiProvider()
    monkeypatch.setenv("GEOACQUIRER_URBI_AUTHORIZATION_REF", "contract-123")
    monkeypatch.delenv("GEOACQUIRER_URBI_API_KEY", raising=False)
    from geoacquirer.security import SecurityError
    with pytest.raises(SecurityError):  # and no embedded fallback key exists
        UrbiProvider()


def test_no_hardcoded_credentials_in_source():
    import pathlib, re
    root = pathlib.Path(__file__).parents[1] / "src"
    pat = re.compile(r"\bgsk_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}|api[_-]?key\s*=\s*[\"'][A-Za-z0-9]{8,}[\"']", re.I)  # generic key shapes; never put a real key in a test
    offenders = [str(p) for p in root.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_suhail_tile_math_and_polygon_roundtrip():
    mvt = pytest.importorskip("mapbox_vector_tile")
    from geoacquirer.providers.suhail import decode_tile_parcels, merge_parts, tile_px_to_lonlat, EXTENT
    from geoacquirer.acquisition.grid import tile_xy
    lon, lat = 46.6753, 24.7136
    z = 15
    x, y = tile_xy(lon, lat, z)
    # corners of the tile in lon/lat
    w, n = tile_px_to_lonlat(0, 0, z, x, y)
    e, s = tile_px_to_lonlat(EXTENT, EXTENT, z, x, y)
    assert w < lon < e and s < lat < n
    # encode a square parcel (tile pixels, y-down) and decode it back
    poly = Polygon([(1000, 1000), (1400, 1000), (1400, 1400), (1000, 1400)])
    data = mvt.encode([{"name": "parcels", "features": [{"geometry": poly.wkt, "properties": {"parcel_id": "P1", "parcel_objectid": "9"}}]}],
                      default_options={"y_coord_down": True})
    rows = decode_tile_parcels(data, z, x, y)
    assert len(rows) == 1 and rows[0]["parcel_id"] == "P1"
    g = rows[0]["geometry"]
    assert g.geom_type == "Polygon" and w < g.centroid.x < e and s < g.centroid.y < n
    # expected centroid from the pixel centre
    cx, cy = tile_px_to_lonlat(1200, 1200, z, x, y)
    assert abs(g.centroid.x - cx) < 1e-6 and abs(g.centroid.y - cy) < 1e-6
    # parts clipped by a tile border are unioned
    left, right = box(0, 0, 1, 1), box(1, 0, 2, 1)
    m = merge_parts([{"parcel_id": "A", "parcel_objectid": "1", "geometry": left}, {"parcel_id": "A", "parcel_objectid": "1", "geometry": right}])
    assert len(m) == 1 and m.geometry.iloc[0].area == pytest.approx(2.0) and m.n_tile_parts.iloc[0] == 2


def test_suhail_rules_parser_reports_explicit_statuses():
    from geoacquirer.providers.suhail import parse_rules_response
    rows = parse_rules_response({"data": [{"id": 7, "landuse": "commercial", "maxBuildingHeight": 20, "extra": 1}]}, {"parcel_objectid": "9"})
    assert rows[0]["api_status"] == "SUCCESS" and rows[0]["landuse"] == "commercial" and "extra" not in rows[0]
    assert parse_rules_response({"data": []}, {"parcel_objectid": "9"})[0]["api_status"] == "NO_RULES"


def test_urbi_parsers():
    from geoacquirer.providers.urbi import extract_geo, parse_full_schedule, item_to_row
    assert extract_geo("POINT(46.8199 24.8229)") == (24.8229, 46.8199)
    assert extract_geo("POINT(-0.5 -1.25)") == (-1.25, -0.5)  # the notebook regex dropped negatives
    assert extract_geo(None) == (None, None)
    assert parse_full_schedule({"is_24x7": True}) == "24 Hours Daily"
    s = parse_full_schedule({"Mon": {"working_hours": [{"from": "08:00", "to": "12:00"}, {"from": "14:00", "to": "18:00"}]}})
    assert s == "Mon: 08:00-12:00, 14:00-18:00"
    row = item_to_row({"id": "123_456", "geometry": {"centroid": "POINT(46.7 24.7)"}, "capacity": {"total": 8}, "schedule": {"is_24x7": True}})
    assert row["parking_id"] == "123" and row["capacity"] == 8 and row["capacity_is_reported_not_live"] is True


def test_osm_bridge_does_not_create_junction_but_shared_node_does():
    js = {"elements": [
        # two streets sharing node 2 -> crossroad
        {"type": "way", "id": 1, "nodes": [1, 2, 3], "tags": {"highway": "residential", "name": "A"},
         "geometry": [{"lat": 24.7000, "lon": 46.6990}, {"lat": 24.7000, "lon": 46.7000}, {"lat": 24.7000, "lon": 46.7010}]},
        {"type": "way", "id": 2, "nodes": [4, 2, 5], "tags": {"highway": "residential", "name": "B"},
         "geometry": [{"lat": 24.6990, "lon": 46.7000}, {"lat": 24.7000, "lon": 46.7000}, {"lat": 24.7010, "lon": 46.7000}]},
        # a bridge crossing street A at a different place without a shared node
        {"type": "way", "id": 3, "nodes": [6, 7], "tags": {"highway": "tertiary", "bridge": "yes", "name": "Bridge"},
         "geometry": [{"lat": 24.6995, "lon": 46.7005}, {"lat": 24.7005, "lon": 46.7005}]},
        {"type": "node", "id": 10, "lat": 24.7001, "lon": 46.7001, "tags": {"shop": "supermarket", "name": "S"}},
    ]}
    L = parse_overpass(js)
    assert len(L["road"]) == 5 and len(L["poi"]) == 1 and L["poi"].iloc[0].category_path == ["shopping", "supermarket"]
    from geoacquirer.providers.overture import connectors_frame, segments_frame
    from geoacquirer.spatial import to_metric, utm_crs_for
    crs = utm_crs_for(46.7, 24.7)
    seg = segments_frame(to_metric(L["road"], crs))
    con = connectors_frame(to_metric(L["connector"], crs))
    j = build_junctions(seg, con)
    assert len(j) == 1 and j.iloc[0].kind == "four_way"  # the bridge adds none


def test_local_inventory_detects_types_and_polygons(tmp_path):
    write_fixture_files(tmp_path)
    inv = {i.entity_type: i for i in scan_inventory(tmp_path)}
    assert set(inv) == {"poi", "road", "connector", "building"}
    assert inv["building"].has_polygons and not inv["poi"].has_polygons
    assert guess_entity_type({"parcel_objectid", "lon"}) == "parcel"
