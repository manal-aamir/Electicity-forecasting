from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from .data import download_uci_dataset, interpolate_missing, load_power_consumption_txt, resample_scale
from .features import add_cyclical_time_features, add_lag_features, hw_components_one_step_ahead_features
from .modeling import compute_metrics, tune_xgb_time_series
from .ablation import drop_one_feature_ablation
from .error_analysis import plot_slice_rmse_bar, slice_rmse_table
from .xai import run_lime_explanations, run_pdp, run_permutation_feature_importance, run_shap_summary


def _ensure_output_dirs(project_root: Path) -> dict[str, Path]:
    base = project_root / "outputs"
    d = {
        "base": base,
        "models": base / "models",
        "plots": base / "plots",
        "explanations": base / "explanations",
        "tables": base / "tables",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _train_test_split_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(np.floor((1.0 - test_size) * n))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def _next_step_supervised(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Predict y(t) using features at time t-1 by shifting target backward.
    This aligns with lagged HW component features and avoids using same-timestep residuals.
    """
    y = df[target_col].shift(-1)
    X = df.drop(columns=[target_col])
    # Drop last row (no next target) and any rows with NaNs from lagging.
    m = y.notna()
    X, y = X.loc[m], y.loc[m]
    X = X.dropna()
    y = y.loc[X.index]
    return X, y


def run_experiment(
    project_root: Path,
    scales: list[str],
    targets: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    shap_samples: int = 500,
    lime_samples: int = 1,
    target_lags: list[int] | None = None,
    run_ablation: bool = False,
    ablation_top_n: int = 10,
    allow_download: bool = True,
) -> None:
    out = _ensure_output_dirs(project_root)

    if allow_download:
        txt_path = download_uci_dataset(project_root)
    else:
        txt_path = (project_root / "data" / "raw" / "household_power_consumption.txt")
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {txt_path}. Run without --no_download to fetch it."
            )

    df_1min = load_power_consumption_txt(txt_path)
    df_1min = interpolate_missing(df_1min, method="time")

    # Normalize column names to match paper-like naming.
    df_1min = df_1min.rename(
        columns={
            "Global_active_power": "global_active_power",
            "Global_reactive_power": "global_reactive_power",
            "Voltage": "voltage",
            "Global_intensity": "global_intensity",
            "Sub_metering_1": "sub_metering_1",
            "Sub_metering_2": "sub_metering_2",
            "Sub_metering_3": "sub_metering_3",
        }
    )

    metrics_rows: list[dict] = []

    for scale in scales:
        df = resample_scale(df_1min, scale=scale)
        time_feats = add_cyclical_time_features(df.index)
        df = pd.concat([df, time_feats], axis=1)

        for target in targets:
            if target not in df.columns:
                raise ValueError(f"Target {target!r} not found in columns: {list(df.columns)}")

            # Chronological split
            train_df, test_df = _train_test_split_time(df, test_size=test_size)

            # ADF test on training target (paper reports stationarity check)
            adf_stat, adf_p, *_ = adfuller(train_df[target].astype(float).to_numpy())
            (out["tables"] / f"adf_{scale}_{target}.json").write_text(
                json.dumps({"scale": scale, "target": target, "adf_stat": float(adf_stat), "p_value": float(adf_p)}),
                encoding="utf-8",
            )

            # Holt-Winters component augmentation (lagged by 1 step)
            hw_feats = hw_components_one_step_ahead_features(
                y_train=train_df[target],
                y_full_index=df.index,
                scale=scale,
            )
            df_aug = pd.concat([df, hw_feats], axis=1)
            if target_lags:
                df_aug = add_lag_features(df_aug, cols=[target], lags=target_lags)

            # Supervised next-step dataset
            X_all, y_all = _next_step_supervised(df_aug, target_col=target)
            # Split AFTER shifting/lagging so X/y lengths stay consistent.
            n_all = len(X_all)
            cut = int(np.floor((1.0 - test_size) * n_all))
            X_train, y_train = X_all.iloc[:cut], y_all.iloc[:cut]
            X_test, y_test = X_all.iloc[cut:], y_all.iloc[cut:]

            # Standardize (fit on train only)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train.values)
            X_test_s = scaler.transform(X_test.values)

            # XGBoost tuning with time-aware CV
            gs = tune_xgb_time_series(X_train_s, y_train.to_numpy(), random_state=random_state)
            model = gs.best_estimator_

            # Evaluate
            y_pred = model.predict(X_test_s)
            m = compute_metrics(y_test.to_numpy(), y_pred)

            # Feature ablation: remove one feature -> retrain -> measure delta accuracy
            if run_ablation:
                # pick a manageable set: top-N by XGBoost gain
                booster = model.get_booster()
                score = booster.get_score(importance_type="gain")
                # score keys are feature indices like "f0", "f1", ...
                scored = []
                for k, v in score.items():
                    if k.startswith("f") and k[1:].isdigit():
                        idx = int(k[1:])
                        if 0 <= idx < len(X_train.columns):
                            scored.append((X_train.columns[idx], float(v)))
                scored.sort(key=lambda t: t[1], reverse=True)
                selected = [name for name, _ in scored[: max(1, ablation_top_n)]]
                if not selected:
                    selected = list(X_train.columns)[: max(1, ablation_top_n)]

                drop_one_feature_ablation(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    base_model=model,
                    baseline_metrics=m,
                    features=selected,
                    out_csv=out["tables"] / f"ablation_drop1_{scale}_{target}.csv",
                )

            # Error decomposition on test set (when does the model fail?)
            ts_test = pd.DatetimeIndex(X_test.index)
            for kind in ("hour", "dow", "month"):
                slice_df = slice_rmse_table(
                    timestamps=ts_test,
                    y_true=y_test.to_numpy(),
                    y_pred=y_pred,
                    slice_kind=kind,
                )
                slice_df.insert(0, "scale", scale)
                slice_df.insert(1, "target", target)

                slice_csv = out["tables"] / f"error_slice_{kind}_{scale}_{target}.csv"
                slice_df.to_csv(slice_csv, index=False)

                plot_slice_rmse_bar(
                    slice_df,
                    title=f"Test RMSE by {slice_df['slice_name'].iloc[0]} ({scale}, {target})",
                    out_png=out["plots"] / f"error_slice_{kind}_{scale}_{target}.png",
                )

            metrics_rows.append(
                {
                    "scale": scale,
                    "target": target,
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    **asdict(m),
                    "best_params": json.dumps(gs.best_params_),
                }
            )

            # Persist model bundle
            bundle_path = out["models"] / f"hw_xgb_{scale}_{target}.joblib"
            joblib.dump(
                {
                    "scale": scale,
                    "target": target,
                    "feature_names": list(X_train.columns),
                    "scaler": scaler,
                    "model": model,
                    "best_params": gs.best_params_,
                },
                bundle_path,
            )

            # XAI outputs
            feat_names = list(X_train.columns)

            run_shap_summary(
                model=model,
                X_test=X_test_s,
                feature_names=feat_names,
                out_png=out["plots"] / f"shap_summary_{scale}_{target}.png",
                max_samples=shap_samples,
            )

            run_permutation_feature_importance(
                model=model,
                X_test=X_test_s,
                y_test=y_test.to_numpy(),
                feature_names=feat_names,
                out_csv=out["tables"] / f"pfi_{scale}_{target}.csv",
            )

            # PDPs on a few high-signal features (time cycles + HW seasonal/level if present)
            pdp_candidates = [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "hw_seasonal_lag1",
                "hw_level_lag1",
            ]
            run_pdp(
                model=model,
                X_train=X_train_s,
                feature_names=feat_names,
                features=[f for f in pdp_candidates if f in feat_names],
                out_png=out["plots"] / f"pdp_{scale}_{target}.png",
            )

            run_lime_explanations(
                model=model,
                X_train=X_train_s,
                X_test=X_test_s,
                feature_names=feat_names,
                out_dir=out["explanations"] / f"lime_{scale}_{target}",
                num_instances=lime_samples,
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "scale"])

    # Write "latest" for convenience, and append to a history file for comparisons.
    metrics_latest = out["tables"] / "metrics_latest.csv"
    metrics_history = out["tables"] / "metrics_history.csv"
    metrics_df.to_csv(metrics_latest, index=False)

    stamped = metrics_df.copy()
    stamped.insert(0, "run_utc", datetime.now(timezone.utc).isoformat())
    if metrics_history.exists():
        prev = pd.read_csv(metrics_history)
        pd.concat([prev, stamped], axis=0, ignore_index=True).to_csv(metrics_history, index=False)
    else:
        stamped.to_csv(metrics_history, index=False)
