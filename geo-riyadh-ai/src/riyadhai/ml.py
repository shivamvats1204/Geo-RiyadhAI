"""Spatial machine learning: building land-use (residential vs commercial) classification and retail clustering.

Labels are Overture/OSM `subtype` values, which are crowd-sourced and noisy, not cadastral truth. Evaluation uses spatial
block cross-validation (1 km blocks held out together) so neighbouring buildings never straddle train and test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoacquirer.ontology import category_mask

from .provenance import data_date
from .store import AreaDataset

LABELS = {"residential": 0, "commercial": 1}
TOP_CATS = ["shopping", "food_and_drink", "services_and_business", "health_care", "education", "community_and_government", "lifestyle_services"]
FEATURE_GROUPS = {
    "shape": ["area_m2", "perimeter_m", "compactness", "rect_aspect"],
    "neighbourhood": ["n_buildings_50m", "built_density_100m"],
    "poi": ["poi_total_100m", "poi_total_250m"] + [f"poi_{c}_100m" for c in TOP_CATS],
    "road": ["dist_primary_m", "dist_secondary_m", "dist_junction_m", "road_class_30m"],
}
ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]


def _counts(pts: gpd.GeoSeries, objs: gpd.GeoSeries, r: float) -> np.ndarray:
    out = np.zeros(len(pts), dtype=np.int32)
    if len(objs):
        idx = objs.sindex.query(pts.values, predicate="dwithin", distance=r)
        np.add.at(out, idx[0], 1)
    return out


def _nearest(pts, objs, cap=3000.0):
    d = np.full(len(pts), cap)
    if len(objs):
        p, dist = objs.sindex.nearest(pts.values, return_all=False, max_distance=cap, return_distance=True)
        d[p[0]] = dist
    return d


def _rect_sides(g) -> tuple[float, float]:
    if g.geom_type != "Polygon":
        return 1.0, 1.0
    c = np.array(g.exterior.coords)
    return float(np.hypot(*(c[1] - c[0]))), float(np.hypot(*(c[2] - c[1])))


def building_features(ds: AreaDataset, b: Optional[gpd.GeoDataFrame] = None, context: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
    b = ds.layers["building"] if b is None else b
    geom = b.geometry
    cen = geom.centroid
    area = geom.area.to_numpy()
    perim = geom.length.to_numpy()
    mrr = geom.minimum_rotated_rectangle()
    sides = np.array([_rect_sides(g) for g in mrr])
    rect_w, rect_l = np.maximum(sides.min(axis=1), 1e-6), np.maximum(sides.max(axis=1), 1e-6)
    X = pd.DataFrame({"building_id": b["building_id"].to_numpy(), "area_m2": area, "perimeter_m": perim,
                      "compactness": 4 * np.pi * area / np.maximum(perim ** 2, 1e-9), "rect_aspect": rect_l / rect_w})
    allb = context if context is not None else ds.layers["building"]
    allc = allb.geometry.centroid
    X["n_buildings_50m"] = _counts(cen, allc, 50.0) - 1
    X["built_density_100m"] = np.array([0.0] * len(X))
    idx = allc.sindex.query(cen.values, predicate="dwithin", distance=100.0)
    dens = np.zeros(len(X)); np.add.at(dens, idx[0], allb.geometry.area.to_numpy()[idx[1]])
    X["built_density_100m"] = dens / (np.pi * 100.0 ** 2)
    poi = ds.layers["poi"]
    X["poi_total_100m"] = _counts(cen, poi.geometry, 100.0)
    X["poi_total_250m"] = _counts(cen, poi.geometry, 250.0)
    for c in TOP_CATS:
        sub = poi[poi["category_path"].apply(lambda p: p is not None and len(p) > 0 and p[0] == c)]
        X[f"poi_{c}_100m"] = _counts(cen, sub.geometry, 100.0)
    road = ds.layers["road"]
    X["dist_primary_m"] = _nearest(cen, road[road["class"].isin(["primary", "trunk", "motorway"])].geometry)
    X["dist_secondary_m"] = _nearest(cen, road[road["class"].isin(["secondary"])].geometry)
    X["dist_junction_m"] = _nearest(cen, ds.layers["junction"].geometry)
    near = road[~road["class"].isin(["service"])]
    sc = np.zeros(len(X))
    pr = near.sindex.query(cen.values, predicate="dwithin", distance=30.0)
    rank = {"primary": 1.0, "secondary": .8, "tertiary": .6, "residential": .4}
    cls = near["class"].to_numpy()
    for i, j in zip(*pr):
        sc[i] = max(sc[i], rank.get(cls[j], .2))
    X["road_class_30m"] = sc
    if "floors" in b.columns:
        X["floors"] = pd.to_numeric(b["floors"], errors="coerce").to_numpy()
    X["x"], X["y"] = cen.x.to_numpy(), cen.y.to_numpy()
    return X


def labelled_dataset(ds: AreaDataset, max_rows: Optional[int] = None, seed: int = 0):
    b = ds.layers["building"]
    lab = b["subtype"].map(LABELS)
    sel = b[lab.notna()].copy()
    if max_rows and len(sel) > max_rows:
        sel = sel.sample(max_rows, random_state=seed)
    X = building_features(ds, sel)
    y = sel["subtype"].map(LABELS).astype(int).to_numpy()
    groups = (np.floor(X["x"] / 1000).astype(int).astype(str) + "_" + np.floor(X["y"] / 1000).astype(int).astype(str)).to_numpy()
    return X, y, groups


def _lgbm(seed=0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, class_weight="balanced", random_state=seed, verbose=-1, n_jobs=4)


def spatial_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, features: list[str], model_factory, n_splits: int = 5) -> dict:
    oof = np.zeros(len(y))
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = model_factory()
        m.fit(X.iloc[tr][features], y[tr])
        p = m.predict_proba(X.iloc[te][features])[:, 1]
        oof[te] = p
        folds.append({"fold": k, "n_test": int(len(te)), "pr_auc": float(average_precision_score(y[te], p)) if y[te].sum() else None})
    pred = (oof >= 0.5).astype(int)
    return {"balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4), "f1_commercial": round(float(f1_score(y, pred)), 4),
            "pr_auc": round(float(average_precision_score(y, oof)), 4), "roc_auc": round(float(roc_auc_score(y, oof)), 4),
            "fold_pr_auc": [f["pr_auc"] for f in folds], "ece_uncalibrated": round(float(_ece(y, oof)), 4), "oof": oof}


def _ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return ece


def run_experiment(ds: AreaDataset, max_rows: Optional[int] = 30000, seed: int = 0, out_dir: Optional[Path] = None) -> dict:
    t0 = time.time()
    X, y, groups = labelled_dataset(ds, max_rows, seed)
    res = {"dataset": {"n_labelled": int(len(y)), "commercial_share": round(float(y.mean()), 4), "spatial_blocks_1km": int(len(set(groups))),
                       "label_source": "Overture building `subtype` (OSM-derived; noisy, not cadastral)",
                       "release": next(iter(ds.provenance.values()), {}).get("dataset_version")}}
    models = {}
    base = spatial_cv(X, y, groups, ["area_m2"], lambda: DummyClassifier(strategy="prior"))
    models["baseline_majority_class"] = base
    lr = spatial_cv(X, y, groups, FEATURE_GROUPS["shape"], lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")))
    models["baseline_logistic_shape_only"] = lr
    rule = ((X["poi_shopping_100m"] > 0).astype(int)).to_numpy()
    models["baseline_rule_shopping_poi_within_100m"] = {"balanced_accuracy": round(float(balanced_accuracy_score(y, rule)), 4), "f1_commercial": round(float(f1_score(y, rule)), 4),
                                                         "pr_auc": round(float(average_precision_score(y, rule)), 4)}
    full = spatial_cv(X, y, groups, ALL_FEATURES, _lgbm)
    models["lightgbm_all_features"] = full
    # random (non-spatial) split for contrast: shows how much leakage inflates the score
    from sklearn.model_selection import KFold
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = _lgbm(); m.fit(X.iloc[tr][ALL_FEATURES], y[tr]); oof[te] = m.predict_proba(X.iloc[te][ALL_FEATURES])[:, 1]
    models["lightgbm_random_split_for_contrast_only"] = {"pr_auc": round(float(average_precision_score(y, oof)), 4), "roc_auc": round(float(roc_auc_score(y, oof)), 4),
                                                         "balanced_accuracy": round(float(balanced_accuracy_score(y, (oof >= .5).astype(int))), 4)}
    abl = {}
    for g, cols in FEATURE_GROUPS.items():
        feats = [c for c in ALL_FEATURES if c not in cols]
        r = spatial_cv(X, y, groups, feats, _lgbm)
        abl[f"without_{g}"] = {k: r[k] for k in ("pr_auc", "roc_auc", "balanced_accuracy", "f1_commercial")}
    res["models_spatial_cv"] = {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in models.items()}
    res["ablations_spatial_cv"] = abl
    res["limitations"] = ["labels are noisy OSM-derived tags", "spatial CV is blocked at 1 km; performance in an unseen city is not measured",
                          "probabilities are NOT calibrated (see ece_uncalibrated)"]
    # final model + predictions for unlabelled buildings
    final = _lgbm(seed).fit(X[ALL_FEATURES], y)
    imp = dict(sorted(zip(ALL_FEATURES, final.booster_.feature_importance("gain").round(1).tolist()), key=lambda kv: -kv[1]))
    res["top_feature_gain"] = dict(list(imp.items())[:10])
    res["seconds"] = round(time.time() - t0, 1)
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        final.booster_.save_model(str(out_dir / "building_landuse_lgbm.txt"))
        (out_dir / "building_landuse_metrics.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def predict_unlabelled(ds: AreaDataset, model_path: Path, limit: Optional[int] = None) -> gpd.GeoDataFrame:
    import lightgbm as lgb
    b = ds.layers["building"]
    un = b[b["subtype"].isna()]
    if limit:
        un = un.sample(min(limit, len(un)), random_state=0)
    X = building_features(ds, un)
    booster = lgb.Booster(model_file=str(model_path))
    out = gpd.GeoDataFrame({"building_id": X["building_id"], "commercial_score": booster.predict(X[ALL_FEATURES]), "is_model_output": True},
                           geometry=un.geometry.centroid.values, crs=ds.crs)
    return out


def retail_clusters(ds: AreaDataset, min_cluster_size: int = 15, min_samples: int = 5) -> tuple[gpd.GeoDataFrame, dict]:
    """HDBSCAN on shop POIs: an unsupervised description of density, not a prediction."""
    poi = ds.layers["poi"]
    shops = poi[category_mask(poi["category_path"], ["shop"])].reset_index(drop=True)
    XY = np.column_stack([shops.geometry.x, shops.geometry.y])
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(XY)
    shops["cluster"] = lab
    cl = shops[shops["cluster"] >= 0].dissolve(by="cluster", aggfunc={"poi_id": "count"}).rename(columns={"poi_id": "n_shops"})
    cl["geometry"] = cl.geometry.convex_hull
    cl = cl.reset_index()
    info = {"algorithm": "HDBSCAN", "min_cluster_size": min_cluster_size, "min_samples": min_samples, "n_shops": int(len(shops)), "n_clusters": int(cl.shape[0]),
            "noise_share": round(float((lab < 0).mean()), 4), "kind": "unsupervised density description (not a predictive model)"}
    if cl.shape[0] > 1 and (lab >= 0).sum() > cl.shape[0]:
        info["silhouette_non_noise"] = round(float(silhouette_score(XY[lab >= 0], lab[lab >= 0])), 4)
    return cl, info


# ------------------------------------------------------------------------------------------------ Urbi-labelled variant
def labelled_dataset_urbi(ds: AreaDataset, max_rows: Optional[int] = 40000, seed: int = 0):
    ub = ds.layers["urbi_building"]
    lab = ub["purpose_class"].map(LABELS)
    sel = ub[lab.notna()].copy()
    sel["building_id"] = sel["urbi_id"]
    if max_rows and len(sel) > max_rows:
        sel = sel.sample(max_rows, random_state=seed)
    X = building_features(ds, sel, context=ub)
    y = sel["purpose_class"].map(LABELS).astype(int).to_numpy()
    groups = (np.floor(X["x"] / 1000).astype(int).astype(str) + "_" + np.floor(X["y"] / 1000).astype(int).astype(str)).to_numpy()
    return X, y, groups


def run_experiment_urbi(ds: AreaDataset, max_rows: Optional[int] = 40000, seed: int = 0, out_dir: Optional[Path] = None) -> dict:
    """Same protocol as run_experiment, with Urbi `purpose` labels (about 10x more labelled buildings than the OSM subtype)."""
    t0 = time.time()
    X, y, groups = labelled_dataset_urbi(ds, max_rows, seed)
    groups_f = {**FEATURE_GROUPS, "floors": ["floors"]}
    all_f = [c for g in groups_f.values() for c in g]
    res = {"dataset": {"n_labelled": int(len(y)), "commercial_share": round(float(y.mean()), 4), "spatial_blocks_1km": int(len(set(groups))),
                       "label_source": "Urbi/2GIS building `purpose_name` mapped by keyword to residential vs commercial (operator-curated; not cadastral)",
                       "release": f"urbi live-api {data_date()}"}}
    models = {}
    models["baseline_majority_class"] = spatial_cv(X, y, groups, ["area_m2"], lambda: DummyClassifier(strategy="prior"))
    models["baseline_logistic_shape_only"] = spatial_cv(X, y, groups, FEATURE_GROUPS["shape"], lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")))
    rule = ((X["poi_shopping_100m"] > 0).astype(int)).to_numpy()
    models["baseline_rule_shopping_poi_within_100m"] = {"balanced_accuracy": round(float(balanced_accuracy_score(y, rule)), 4), "f1_commercial": round(float(f1_score(y, rule)), 4),
                                                         "pr_auc": round(float(average_precision_score(y, rule)), 4)}
    models["lightgbm_all_features"] = spatial_cv(X, y, groups, all_f, _lgbm)
    models["lightgbm_without_floors"] = spatial_cv(X, y, groups, [c for c in all_f if c != "floors"], _lgbm)
    abl = {}
    for g, cols in groups_f.items():
        r = spatial_cv(X, y, groups, [c for c in all_f if c not in cols], _lgbm)
        abl[f"without_{g}"] = {k: r[k] for k in ("pr_auc", "roc_auc", "balanced_accuracy", "f1_commercial")}
    res["models_spatial_cv"] = {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in models.items()}
    res["ablations_spatial_cv"] = abl
    res["limitations"] = ["labels are keyword-mapped Urbi purposes; 'commercial' includes offices/hotels", "1 km block CV in one city area", "probabilities are uncalibrated"]
    final = _lgbm(seed).fit(X[all_f], y)
    imp = dict(sorted(zip(all_f, final.booster_.feature_importance("gain").round(1).tolist()), key=lambda kv: -kv[1]))
    res["top_feature_gain"] = dict(list(imp.items())[:10])
    res["seconds"] = round(time.time() - t0, 1)
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "building_landuse_urbi_metrics.json").write_text(json.dumps(res, indent=2, default=str))
    return res
