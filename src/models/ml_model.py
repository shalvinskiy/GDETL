"""ML benchmark: XGBoost, trained with the DIRECT multi-horizon strategy
(one independently-fit model per horizon h in {1,2,3,4}) rather than
recursive forecasting, to avoid compounding one-step errors and because our
horizon is short (<=4 weeks) -- see README "target" section for the
direct-vs-recursive decision.

XGBoost natively handles missing values (NaN) in features, so the small
number of NaNs produced by rolling windows / early history is left as-is
rather than imputed (avoids inventing information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def _model_params(seed: int, classifier: bool) -> dict:
    base = dict(
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
        base["objective"] = "binary:logistic"
        base["eval_metric"] = "logloss"
    else:
        base["objective"] = "count:poisson"  # target is a non-negative event count
    return base


def fit_predict_regressor(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, seed: int = 42
) -> np.ndarray:
    train = train_df.dropna(subset=[target_col])
    X_train, y_train = train[feature_cols], train[target_col].astype(float)
    X_test = test_df[feature_cols]
    model = xgb.XGBRegressor(**_model_params(seed, classifier=False))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.maximum(preds, 0.0)


def fit_predict_classifier(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, seed: int = 42
) -> np.ndarray:
    train = train_df.dropna(subset=[target_col])
    X_train, y_train = train[feature_cols], train[target_col].astype(int)
    X_test = test_df[feature_cols]
    if y_train.nunique() < 2:
        # degenerate fold (no positive examples yet) -> fall back to empirical rate
        rate = float(y_train.mean()) if len(y_train) else 0.0
        return np.full(len(test_df), rate)
    model = xgb.XGBClassifier(**_model_params(seed, classifier=True))
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def fit_full_regressor(train_df: pd.DataFrame, feature_cols: list[str], target_col: str, seed: int = 42):
    train = train_df.dropna(subset=[target_col])
    model = xgb.XGBRegressor(**_model_params(seed, classifier=False))
    model.fit(train[feature_cols], train[target_col].astype(float))
    return model


def fit_full_classifier(train_df: pd.DataFrame, feature_cols: list[str], target_col: str, seed: int = 42):
    train = train_df.dropna(subset=[target_col])
    model = xgb.XGBClassifier(**_model_params(seed, classifier=True))
    model.fit(train[feature_cols], train[target_col].astype(int))
    return model
