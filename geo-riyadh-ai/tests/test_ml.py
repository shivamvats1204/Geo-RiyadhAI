import numpy as np
from sklearn.model_selection import GroupKFold

from riyadhai import ml as ML


def test_feature_table_and_labels(world):
    _, ds, truth, ctx = world
    X, y, groups = ML.labelled_dataset(ds)
    assert set(np.unique(y)) <= {0, 1} and len(X) == len(y) == len(groups)
    assert set(ML.ALL_FEATURES) <= set(X.columns)
    assert (X["area_m2"] > 0).all() and X["compactness"].between(0, 1.01).all()
    # buildings are 20 x 20 m boxes in the synthetic area
    assert X["area_m2"].between(399, 401).all()


def test_spatial_blocks_never_straddle_train_and_test(world):
    _, ds, truth, ctx = world
    X, y, groups = ML.labelled_dataset(ds)
    for tr, te in GroupKFold(n_splits=3).split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "a 1 km block appears in both train and test (spatial leakage)"


def test_spatial_cv_reports_all_metrics_and_beats_nothing_when_labels_are_random(world):
    _, ds, truth, ctx = world
    X, y, groups = ML.labelled_dataset(ds)
    r = ML.spatial_cv(X, y, groups, ML.ALL_FEATURES, lambda: ML._lgbm(0), n_splits=3)
    assert {"balanced_accuracy", "f1_commercial", "pr_auc", "roc_auc", "ece_uncalibrated", "fold_pr_auc"} <= set(r)
    # synthetic commercial labels are planted at random, so a spatial model must not look good: a high score here would mean leakage
    assert r["roc_auc"] < 0.75


def test_retail_clusters_are_labelled_descriptive(world):
    _, ds, truth, ctx = world
    cl, info = ML.retail_clusters(ds, min_cluster_size=8, min_samples=3)
    assert info["kind"].startswith("unsupervised") and info["n_clusters"] >= 1 and 0 <= info["noise_share"] < 1
