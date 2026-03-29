from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor


@dataclass(frozen=True)
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(mae=mae, mse=mse, rmse=rmse, r2=r2)


def tune_xgb_time_series(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> GridSearchCV:
    base = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )

    # Keep a small grid so it runs on a laptop, but still "grid search + time-aware CV".
    param_grid = {
        "min_child_weight": [1, 5],
        "gamma": [0.0, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs
