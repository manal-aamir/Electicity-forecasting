from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SliceRMSE:
    slice_name: str
    bucket: int
    n: int
    rmse: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def slice_rmse_table(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    slice_kind: str,
) -> pd.DataFrame:
    """
    Compute RMSE in buckets defined by a time slice.

    slice_kind:
      - "hour": 0..23
      - "dow": 0..6 (Mon=0)
      - "month": 1..12
    """
    if len(timestamps) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("timestamps, y_true, y_pred must have same length")

    if slice_kind == "hour":
        buckets = pd.Series(timestamps.hour, index=timestamps, name="bucket")
        full_range = range(24)
        slice_name = "hour_of_day"
    elif slice_kind == "dow":
        buckets = pd.Series(timestamps.dayofweek, index=timestamps, name="bucket")
        full_range = range(7)
        slice_name = "day_of_week"
    elif slice_kind == "month":
        buckets = pd.Series(timestamps.month, index=timestamps, name="bucket")
        full_range = range(1, 13)
        slice_name = "month"
    else:
        raise ValueError(f"Unknown slice_kind: {slice_kind}")

    df = pd.DataFrame(
        {"bucket": buckets.to_numpy(), "y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)},
        index=timestamps,
    )

    rows: list[SliceRMSE] = []
    g = df.groupby("bucket", sort=True)
    for b in full_range:
        if b in g.groups:
            part = df.loc[g.groups[b]]
            rows.append(SliceRMSE(slice_name=slice_name, bucket=int(b), n=int(len(part)), rmse=_rmse(part.y_true, part.y_pred)))
        else:
            rows.append(SliceRMSE(slice_name=slice_name, bucket=int(b), n=0, rmse=float("nan")))

    return pd.DataFrame([r.__dict__ for r in rows])


def plot_slice_rmse_bar(
    slice_df: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
) -> None:
    """
    Expects columns: bucket, rmse, n (and optionally slice_name).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = slice_df["bucket"].to_numpy()
    y = slice_df["rmse"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel("bucket")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.3)

    # Show counts as a light annotation (small font) when it fits.
    for xi, yi, ni in zip(x, y, slice_df["n"].to_numpy()):
        if np.isfinite(yi) and ni > 0:
            ax.text(xi, yi, str(int(ni)), ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

