from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ConformalBand:
    alpha: float
    qhat: float
    coverage: float
    avg_width: float
    lower: np.ndarray
    upper: np.ndarray


def _split_conformal_qhat(abs_residuals: np.ndarray, alpha: float) -> float:
    n = len(abs_residuals)
    if n == 0:
        return float("nan")
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q = float(min(max(q, 0.0), 1.0))
    return float(np.quantile(abs_residuals, q, method="higher"))


def fit_split_conformal_bands(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    xgb_params: dict,
    alphas: tuple[float, ...] = (0.1, 0.05),
    calibration_fraction: float = 0.2,
) -> dict[float, ConformalBand]:
    """
    Fit split conformal intervals for XGBoost regression.
    Uses the last `calibration_fraction` of the training sequence as calibration set.
    """
    n = len(X_train)
    n_cal = max(20, int(np.floor(calibration_fraction * n)))
    n_cal = min(n_cal, max(20, n - 20))
    if n < 60:
        return {}

    split = n - n_cal
    X_fit, y_fit = X_train[:split], y_train[:split]
    X_cal, y_cal = X_train[split:], y_train[split:]

    model = XGBRegressor(**xgb_params)
    model.fit(X_fit, y_fit)

    cal_pred = model.predict(X_cal)
    test_pred = model.predict(X_test)
    abs_res = np.abs(y_cal - cal_pred)

    out: dict[float, ConformalBand] = {}
    for alpha in alphas:
        qhat = _split_conformal_qhat(abs_res, alpha)
        lower = test_pred - qhat
        upper = test_pred + qhat
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        avg_width = float(np.mean(upper - lower))
        out[alpha] = ConformalBand(
            alpha=float(alpha),
            qhat=float(qhat),
            coverage=coverage,
            avg_width=avg_width,
            lower=lower,
            upper=upper,
        )
    return out


def save_interval_table(
    *,
    index: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: dict[float, ConformalBand],
    out_csv: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": pd.Index(index).astype(str),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    for alpha, band in bands.items():
        level = int((1 - alpha) * 100)
        df[f"lower_{level}"] = band.lower
        df[f"upper_{level}"] = band.upper

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_prediction_intervals(
    *,
    interval_df: pd.DataFrame,
    out_png: Path,
    title: str,
    max_points: int = 400,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df = interval_df.copy()
    if len(df) > max_points:
        df = df.iloc[-max_points:].copy()

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, df["y_true"].to_numpy(), label="Actual", linewidth=1.5)
    ax.plot(x, df["y_pred"].to_numpy(), label="Predicted", linewidth=1.5)

    for level in (90, 95):
        lo = f"lower_{level}"
        up = f"upper_{level}"
        if lo in df.columns and up in df.columns:
            alpha = 0.2 if level == 90 else 0.12
            ax.fill_between(x, df[lo].to_numpy(), df[up].to_numpy(), alpha=alpha, label=f"{level}% PI")

    ax.set_title(title)
    ax.set_xlabel("Test points (latest window)")
    ax.set_ylabel("Target")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

