import numpy as np
import geopandas as gpd
from shapely.geometry import box

from riyadhai.citystore import CityStore, landuse_class


def _layer(tmp_path):
    rng = np.random.default_rng(0)
    geoms = [box(x, y, x + 0.0005, y + 0.0005) for x, y in zip(rng.uniform(46.0, 46.6, 400), rng.uniform(24.0, 24.6, 400))]
    geoms.append(box(46.05, 24.30, 46.45, 24.32))            # a long, thin 'street' crossing many tiles
    geoms.append(box(46.20, 24.10, 46.21, 24.11).buffer(0.0))
    g = gpd.GeoDataFrame({"urbi_id": [str(i) for i in range(len(geoms))]}, geometry=geoms, crs=4326)
    city = CityStore(tmp_path / "city")
    city.write("street", g.iloc[:200], 0)
    city.write("street", g.iloc[200:], 1)
    city.build_index("street")
    return city, g


def test_tiled_query_equals_brute_force_including_oversized_features(tmp_path):
    city, g = _layer(tmp_path)
    rng = np.random.default_rng(1)
    for _ in range(60):
        x, y = rng.uniform(45.95, 46.55), rng.uniform(23.95, 24.55)
        bb = (x, y, x + rng.uniform(0.005, 0.15), y + rng.uniform(0.005, 0.15))
        got = set(city.query("street", bb)["urbi_id"]) if len(city.query("street", bb)) else set()
        exp = set(g.urbi_id[g.geometry.intersects(box(*bb))])
        assert got == exp, (bb, got ^ exp)


def test_long_feature_found_from_a_tile_far_from_its_representative_point(tmp_path):
    city, g = _layer(tmp_path)
    long_id = str(len(g) - 2)
    far = (46.07, 24.305, 46.08, 24.315)                       # near the street's west end; its centre is ~0.2 degrees away
    assert long_id in set(city.query("street", far)["urbi_id"])
    ext = city.extent("street")
    assert ext["n_large"] >= 1 and ext["small_reach_deg"] < 0.02


def test_row_counts_preserved_and_empty_box(tmp_path):
    city, g = _layer(tmp_path)
    assert city.index("street")["rows"] == len(g)
    assert city.query("street", (10.0, 10.0, 10.1, 10.1)).empty


def test_landuse_keyword_mapping():
    assert landuse_class("سكني / تجاري / مكتبي") == "mixed" and landuse_class("سكني (فلل)") == "residential"
    assert landuse_class("تجاري") == "commercial" and landuse_class("صناعي") == "industrial"
    assert landuse_class(None) == "unknown" and landuse_class("مناطق عسكرية") == "other"
