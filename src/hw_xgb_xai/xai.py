from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
