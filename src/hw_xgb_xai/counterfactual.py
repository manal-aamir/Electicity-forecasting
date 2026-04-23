from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_counterfactual_recourse(
    *,
    model,
    scaler,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_pred_test: np.ndarray,
    reduction_target: float = 0.2,
    n_instances: int = 5,
    n_trials: int = 800,
    random_state: int = 42,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Generate practical counterfactual suggestions:
    "What should change to reduce prediction by X%?"
    """
    rng = np.random.default_rng(random_state)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    feats = list(X_train.columns)
    lo = X_train.quantile(0.05).to_numpy()
    hi = X_train.quantile(0.95).to_numpy()

    def pred(x_row: np.ndarray) -> float:
        xs = scaler.transform(x_row.reshape(1, -1))
        return float(model.predict(xs)[0])

    rows = []
    k = min(n_instances, len(X_test))
    for i in range(k):
        x0 = X_test.iloc[i].to_numpy(dtype=float)
        base_pred = float(y_pred_test[i])
        target = base_pred * (1.0 - reduction_target)

        best = x0.copy()
        best_score = abs(base_pred - target)
        best_pred = base_pred

        for _ in range(n_trials):
            cand = x0.copy()
            # mutate random subset of features
            n_mut = int(rng.integers(1, max(2, len(feats) // 4)))
            idxs = rng.choice(len(feats), size=n_mut, replace=False)
            for j in idxs:
                cand[j] = rng.uniform(lo[j], hi[j])

            p = pred(cand)
            dist = float(np.mean(np.abs((cand - x0) / (np.abs(x0) + 1e-6))))
            score = abs(p - target) + 0.2 * dist
            if score < best_score:
                best_score = score
                best = cand
                best_pred = p

        delta = best - x0
        changed = sorted(
            [(feats[j], float(x0[j]), float(best[j]), float(delta[j])) for j in range(len(feats)) if abs(delta[j]) > 1e-9],
            key=lambda t: abs(t[3]),
            reverse=True,
        )
        top_changes = changed[:5]

        row = {
            "instance_idx": i,
            "base_prediction": base_pred,
            "counterfactual_prediction": best_pred,
            "target_prediction": target,
            "achieved_reduction_pct": 100.0 * (base_pred - best_pred) / (abs(base_pred) + 1e-9),
            "n_features_changed": len(changed),
        }
        for k2 in range(5):
            if k2 < len(top_changes):
                f, v0, v1, dv = top_changes[k2]
                row[f"change_{k2+1}"] = f"{f}: {v0:.4f} -> {v1:.4f} (Δ {dv:+.4f})"
            else:
                row[f"change_{k2+1}"] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

