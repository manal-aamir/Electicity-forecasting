from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.hw_xgb_xai.pipeline import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_PLOTS = PROJECT_ROOT / "outputs" / "plots"
OUT_EXPL = PROJECT_ROOT / "outputs" / "explanations"

SCALES = ["hourly", "daily", "weekly", "monthly", "quarterly"]
TARGETS = ["global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3"]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _plot_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not generated yet: `{path.name}`")


def main() -> None:
    st.set_page_config(page_title="HW+XGB+XAI Dashboard", layout="wide")
    st.title("Interactive POC: HW + XGBoost + XAI")
    st.caption(
        "Run forecasting experiments and inspect metrics, uncertainty bands, "
        "explainability, error decomposition, and feature ablation in one interface."
    )

    with st.sidebar:
        st.header("Run Configuration")
        scales = st.multiselect("Scales", SCALES, default=["hourly"])
        targets = st.multiselect("Targets", TARGETS, default=["global_active_power"])
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        shap_samples = st.number_input("SHAP samples", min_value=50, max_value=5000, value=500, step=50)
        lime_samples = st.number_input("LIME samples", min_value=1, max_value=20, value=1, step=1)
        target_lags_raw = st.text_input("Target lags (comma-separated)", value="")
        run_ablation = st.checkbox("Run feature ablation", value=False)
        ablation_top_n = st.number_input("Ablation top-N", min_value=1, max_value=50, value=10, step=1)

        lags: list[int] = []
        if target_lags_raw.strip():
            try:
                lags = [int(x.strip()) for x in target_lags_raw.split(",") if x.strip()]
            except ValueError:
                st.error("Invalid lag list. Example: 1,24,168")
                lags = []

        run_btn = st.button("Run Experiment", type="primary", use_container_width=True)

    if run_btn:
        if not scales or not targets:
            st.error("Select at least one scale and one target.")
        else:
            try:
                with st.spinner("Running pipeline... this may take a few minutes"):
                    run_experiment(
                        project_root=PROJECT_ROOT,
                        scales=scales,
                        targets=targets,
                        test_size=float(test_size),
                        shap_samples=int(shap_samples),
                        lime_samples=int(lime_samples),
                        target_lags=lags if lags else None,
                        run_ablation=run_ablation,
                        ablation_top_n=int(ablation_top_n),
                        allow_download=True,
                    )
                st.success("Experiment finished. Outputs refreshed.")
            except ValueError as e:
                st.error(
                    f"{e}\n\nTip: for coarse scales (weekly/monthly/quarterly), "
                    "use smaller lags (e.g., 1,2,4) or leave lag field empty."
                )
                st.stop()

    st.subheader("Metrics")
    metrics = _read_csv(OUT_TABLES / "metrics_latest.csv")
    if metrics is None:
        st.warning("No metrics yet. Click 'Run Experiment' first.")
        return

    sel_scale = st.selectbox("View scale", options=SCALES, index=0)
    sel_target = st.selectbox("View target", options=TARGETS, index=0)

    st.dataframe(metrics, use_container_width=True)
    row = metrics[(metrics["scale"] == sel_scale) & (metrics["target"] == sel_target)]
    if not row.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{float(row.iloc[0]['mae']):.4f}")
        c2.metric("RMSE", f"{float(row.iloc[0]['rmse']):.4f}")
        c3.metric("R²", f"{float(row.iloc[0]['r2']):.4f}")
        cov95 = row.iloc[0].get("coverage_95", None)
        if cov95 is not None and pd.notna(cov95):
            c4.metric("95% Coverage", f"{float(cov95):.3f}")
        else:
            c4.metric("95% Coverage", "N/A")
    else:
        st.info("No row for selected scale/target in latest run.")

    tab_unc, tab_xai, tab_advanced, tab_error, tab_ablation = st.tabs(
        ["Uncertainty", "XAI Outputs", "Advanced Insights", "Error Decomposition", "Feature Impact"]
    )

    with tab_unc:
        _plot_if_exists(
            OUT_PLOTS / f"intervals_{sel_scale}_{sel_target}.png",
            "Prediction intervals (90% / 95%)",
        )
        intervals = _read_csv(OUT_TABLES / f"intervals_{sel_scale}_{sel_target}.csv")
        if intervals is not None:
            st.markdown("**Interval table (latest rows)**")
            st.dataframe(intervals.tail(20), use_container_width=True)
        else:
            st.info("Interval table not found for current selection.")

    with tab_xai:
        c1, c2 = st.columns(2)
        with c1:
            _plot_if_exists(
                OUT_PLOTS / f"shap_summary_{sel_scale}_{sel_target}.png",
                "SHAP summary",
            )
        with c2:
            _plot_if_exists(
                OUT_PLOTS / f"pdp_{sel_scale}_{sel_target}.png",
                "Partial dependence plot",
            )

        pfi = _read_csv(OUT_TABLES / f"pfi_{sel_scale}_{sel_target}.csv")
        if pfi is not None:
            st.markdown("**Permutation Feature Importance (top 15)**")
            st.dataframe(pfi.head(15), use_container_width=True)
        else:
            st.info("PFI table not found for current selection.")

        lime_path = OUT_EXPL / f"lime_{sel_scale}_{sel_target}" / "lime_instance_0.html"
        if lime_path.exists():
            st.markdown(f"LIME explanation file: `{lime_path}`")
        else:
            st.info("LIME explanation file not found for current selection.")

    with tab_advanced:
        st.markdown("**SHAP interaction dependence plots**")
        inter_dir = OUT_PLOTS / f"shap_interactions_{sel_scale}_{sel_target}"
        if inter_dir.exists():
            images = sorted(inter_dir.glob("*.png"))
            if images:
                for p in images:
                    st.image(str(p), caption=p.name, use_container_width=True)
            else:
                st.info("No SHAP interaction plots found yet.")
        else:
            st.info("No SHAP interaction directory found yet.")

        st.markdown("**Counterfactual explanations (target: 20% lower prediction)**")
        cfs = _read_csv(OUT_TABLES / f"counterfactuals_{sel_scale}_{sel_target}.csv")
        if cfs is not None:
            st.dataframe(cfs, use_container_width=True)
        else:
            st.info("Counterfactual table not found for this selection.")

        st.markdown("**Model confidence vs LIME stability**")
        _plot_if_exists(
            OUT_PLOTS / f"lime_stability_vs_uncertainty_{sel_scale}_{sel_target}.png",
            "LIME stability vs uncertainty",
        )
        stabs = _read_csv(OUT_TABLES / f"lime_stability_{sel_scale}_{sel_target}.csv")
        if stabs is not None:
            st.dataframe(stabs, use_container_width=True)
        else:
            st.info("LIME stability table not found for this selection.")

    with tab_error:
        c1, c2, c3 = st.columns(3)
        with c1:
            _plot_if_exists(
                OUT_PLOTS / f"error_slice_hour_{sel_scale}_{sel_target}.png",
                "RMSE by hour",
            )
        with c2:
            _plot_if_exists(
                OUT_PLOTS / f"error_slice_dow_{sel_scale}_{sel_target}.png",
                "RMSE by day-of-week",
            )
        with c3:
            _plot_if_exists(
                OUT_PLOTS / f"error_slice_month_{sel_scale}_{sel_target}.png",
                "RMSE by month",
            )

    with tab_ablation:
        abl = _read_csv(OUT_TABLES / f"ablation_drop1_{sel_scale}_{sel_target}.csv")
        if abl is None:
            st.info("No ablation file for this selection. Re-run with 'Run feature ablation' enabled.")
        else:
            st.markdown("**Drop-one feature ablation (sorted by delta RMSE)**")
            st.dataframe(abl, use_container_width=True)


if __name__ == "__main__":
    main()

