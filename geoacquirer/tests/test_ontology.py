

def test_category_mask_on_empty_series_is_boolean_and_filters_a_frame():
    import geopandas as gpd, pandas as pd
    from geoacquirer.ontology import category_mask
    g = gpd.GeoDataFrame({"category_path": pd.Series([], dtype=object)}, geometry=[], crs=4326)
    m = category_mask(g["category_path"], ["shop"])
    assert m.dtype == bool
    assert isinstance(g[m], gpd.GeoDataFrame)  # used to degrade to a plain DataFrame with no .geometry
