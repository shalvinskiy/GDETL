"""XGBoost, one model per horizon (direct strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def _params(seed: int, classifier: bool) -> dict:
    p = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
    )
    if classifier:
        p["objective"] = "binary:logistic"
        p["eval_metric"] = "logloss"
    else:
        p["objective"] = "count:poisson"
    return p


def fit_predict_regressor(train_df, test_df, feature_cols, target_col, seed=42) -> np.ndarray:
    train = train_df.dropna(subset=[target_col])
    model = xgb.XGBRegressor(**_params(seed, classifier=False))
    model.fit(train[feature_cols], train[target_col].astype(float))
    return np.maximum(model.predict(test_df[feature_cols]), 0.0)


def fit_predict_classifier(train_df, test_df, feature_cols, target_col, seed=42) -> np.ndarray:
    train = train_df.dropna(subset=[target_col])
    y = train[target_col].astype(int)
    if y.nunique() < 2:
        rate = float(y.mean()) if len(y) else 0.0
        return np.full(len(test_df), rate)
    model = xgb.XGBClassifier(**_params(seed, classifier=True))
    model.fit(train[feature_cols], y)
    return model.predict_proba(test_df[feature_cols])[:, 1]


def fit_full_regressor(train_df, feature_cols, target_col, seed=42):
    train = train_df.dropna(subset=[target_col])
    model = xgb.XGBRegressor(**_params(seed, classifier=False))
    model.fit(train[feature_cols], train[target_col].astype(float))
    return model


def fit_full_classifier(train_df, feature_cols, target_col, seed=42):
    train = train_df.dropna(subset=[target_col])
    model = xgb.XGBClassifier(**_params(seed, classifier=True))
    model.fit(train[feature_cols], train[target_col].astype(int))
    return model
