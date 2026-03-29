from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .modeling import compute_metrics


@dataclass(frozen=True)
class AblationRow:
    feature_removed: str
    n_train: int
    n_test: int
    mae: float
    mse: float
    rmse: float
    r2: float
    delta_mae: float
    delta_rmse: float
    delta_r2: float


def drop_one_feature_ablation(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_model: XGBRegressor,
    baseline_metrics,
    features: list[str],
    out_csv: Path,
) -> pd.DataFrame:
    """
    Retrain XGBoost with one feature removed at a time and report metric deltas vs baseline.

    Notes:
    - Uses the same hyperparameters as the already-tuned baseline model.
    - Re-fits the scaler for each ablation run to keep preprocessing consistent.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[AblationRow] = []
    base_params = base_model.get_params()

    for f in features:
        if f not in X_train.columns:
            continue

        Xtr = X_train.drop(columns=[f])
        Xte = X_test.drop(columns=[f])

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr.values)
        Xte_s = scaler.transform(Xte.values)

        model = XGBRegressor(**base_params)
        model.fit(Xtr_s, y_train.to_numpy())
        pred = model.predict(Xte_s)

        m = compute_metrics(y_test.to_numpy(), pred)

        rows.append(
            AblationRow(
                feature_removed=f,
                n_train=int(len(Xtr)),
                n_test=int(len(Xte)),
                mae=m.mae,
                mse=m.mse,
                rmse=m.rmse,
                r2=m.r2,
                delta_mae=m.mae - float(baseline_metrics.mae),
                delta_rmse=m.rmse - float(baseline_metrics.rmse),
                delta_r2=m.r2 - float(baseline_metrics.r2),
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values("delta_rmse", ascending=False)
    df.to_csv(out_csv, index=False)
    return df

