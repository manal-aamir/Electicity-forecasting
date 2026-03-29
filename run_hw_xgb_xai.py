from __future__ import annotations

import argparse
from pathlib import Path

from src.hw_xgb_xai.pipeline import run_experiment


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HW+XGB+XAI multi-scale forecasting.")
    p.add_argument(
        "--scales",
        nargs="+",
        default=["hourly", "daily", "weekly", "monthly", "quarterly"],
        choices=["hourly", "daily", "weekly", "monthly", "quarterly"],
    )
    p.add_argument(
        "--target",
        nargs="+",
        default=["global_active_power"],
        help="One or more targets: global_active_power, sub_metering_1, sub_metering_2, sub_metering_3",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--shap_samples", type=int, default=500)
    p.add_argument("--lime_samples", type=int, default=1)
    p.add_argument(
        "--target_lags",
        nargs="*",
        type=int,
        default=[],
        help="Add lag features of the target (e.g. --target_lags 1 24 168).",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Run drop-one feature ablation (retrain per removed feature).",
    )
    p.add_argument(
        "--ablation_top_n",
        type=int,
        default=10,
        help="How many features to ablate (chosen by XGBoost gain).",
    )
    p.add_argument("--no_download", action="store_true", help="Do not download dataset (must already exist).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    run_experiment(
        project_root=project_root,
        scales=args.scales,
        targets=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        shap_samples=args.shap_samples,
        lime_samples=args.lime_samples,
        target_lags=args.target_lags if args.target_lags else None,
        run_ablation=args.ablation,
        ablation_top_n=args.ablation_top_n,
        allow_download=not args.no_download,
    )


if __name__ == "__main__":
    main()
