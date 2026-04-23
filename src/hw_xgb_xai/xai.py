from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def run_shap_summary(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    out_png: Path,
    max_samples: int = 500,
) -> None:
    Xs = X_test
    if Xs.shape[0] > max_samples:
        Xs = Xs[:max_samples]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    plt.figure()
    shap.summary_plot(shap_values, features=Xs, feature_names=feature_names, show=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_lime_explanations(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    num_instances: int = 1,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )

    paths: list[Path] = []
    k = min(num_instances, X_test.shape[0])
    for i in range(k):
        exp = explainer.explain_instance(
            data_row=X_test[i],
            predict_fn=model.predict,
            num_features=min(10, len(feature_names)),
        )
        out_path = out_dir / f"lime_instance_{i}.html"
        out_path.write_text(exp.as_html(), encoding="utf-8")
        paths.append(out_path)
    return paths


def run_permutation_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    out_csv: Path,
    n_repeats: int = 5,
) -> pd.DataFrame:
    r = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def run_pdp(
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    features: Iterable[str],
    out_png: Path,
) -> None:
    feat_idx = [feature_names.index(f) for f in features if f in feature_names]
    if not feat_idx:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=feat_idx,
        feature_names=feature_names,
        ax=ax,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def run_shap_interaction_dependence_plots(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    max_samples: int = 500,
    top_n: int = 3,
) -> list[Path]:
    """
    Create SHAP dependence plots with interaction coloring for top features.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    Xs = X_test[:max_samples] if len(X_test) > max_samples else X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][: max(1, min(top_n, len(feature_names)))]

    preferred_interactions = ["sub_metering_3", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    out_paths: list[Path] = []
    for i, idx in enumerate(top_idx):
        f_name = feature_names[idx]
        inter = None
        for cand in preferred_interactions:
            if cand in feature_names and cand != f_name:
                inter = cand
                break
        if inter is None:
            inter = feature_names[int(top_idx[(i + 1) % len(top_idx)])] if len(top_idx) > 1 else "auto"

        plt.figure()
        shap.dependence_plot(
            ind=f_name,
            shap_values=shap_values,
            features=Xs,
            feature_names=feature_names,
            interaction_index=inter,
            show=False,
        )
        out_path = out_dir / f"shap_interaction_{i+1}_{f_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        out_paths.append(out_path)

    return out_paths


def run_lime_stability_vs_uncertainty(
    *,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    width95: np.ndarray,
    out_csv: Path,
    out_png: Path,
    n_instances: int = 8,
    n_repeats: int = 5,
    num_features: int = 8,
) -> pd.DataFrame:
    """
    Quantify LIME stability (across random seeds) and compare with interval width (uncertainty).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    k = int(min(n_instances, len(X_test), len(width95)))
    if k <= 1:
        df = pd.DataFrame(columns=["instance_idx", "lime_stability", "interval_width_95"])
        df.to_csv(out_csv, index=False)
        return df

    rows = []
    for i in range(k):
        ws = []
        for r in range(n_repeats):
            explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=feature_names,
                mode="regression",
                discretize_continuous=True,
                random_state=42 + r,
            )
            exp = explainer.explain_instance(
                data_row=X_test[i],
                predict_fn=model.predict,
                num_features=min(num_features, len(feature_names)),
            )
            w = np.zeros(len(feature_names), dtype=float)
            for name, val in exp.as_list():
                # LIME names often start with feature text; match by startswith
                for j, fn in enumerate(feature_names):
                    if name.startswith(fn):
                        w[j] = float(val)
                        break
            nrm = np.linalg.norm(w)
            if nrm > 0:
                w = w / nrm
            ws.append(w)

        sims = []
        for a in range(len(ws)):
            for b in range(a + 1, len(ws)):
                sims.append(float(np.dot(ws[a], ws[b])))
        stability = float(np.mean(sims)) if sims else float("nan")
        rows.append(
            {
                "instance_idx": i,
                "lime_stability": stability,
                "interval_width_95": float(width95[i]),
            }
        )

    df = pd.DataFrame(rows)
    rho, p = spearmanr(df["interval_width_95"], df["lime_stability"], nan_policy="omit")
    df["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
    df["spearman_p"] = float(p) if np.isfinite(p) else np.nan
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["interval_width_95"], df["lime_stability"], alpha=0.8)
    ax.set_xlabel("95% interval width (uncertainty)")
    ax.set_ylabel("LIME stability (mean pairwise cosine)")
    ax.set_title(f"LIME stability vs uncertainty (rho={rho:.3f}, p={p:.3g})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return df
