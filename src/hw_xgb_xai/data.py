from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import pandas as pd


UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
)


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path
    raw_zip: Path
    raw_txt: Path


def get_dataset_paths(project_root: Path) -> DatasetPaths:
    raw_dir = project_root / "data" / "raw"
    raw_zip = raw_dir / "uci_household_power_consumption.zip"
    raw_txt = raw_dir / "household_power_consumption.txt"
    return DatasetPaths(raw_dir=raw_dir, raw_zip=raw_zip, raw_txt=raw_txt)


def download_uci_dataset(project_root: Path) -> Path:
    paths = get_dataset_paths(project_root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    if not paths.raw_zip.exists():
        with urlopen(UCI_ZIP_URL) as r:
            content = r.read()
        paths.raw_zip.write_bytes(content)

    if not paths.raw_txt.exists():
        with zipfile.ZipFile(paths.raw_zip, "r") as zf:
            members = zf.namelist()
            # Typical file name: "household_power_consumption.txt"
            txt_candidates = [m for m in members if m.endswith(".txt")]
            if not txt_candidates:
                raise RuntimeError(f"No .txt found in zip members: {members}")
            name = txt_candidates[0]
            paths.raw_txt.write_bytes(zf.read(name))

    return paths.raw_txt


def load_power_consumption_txt(txt_path: Path) -> pd.DataFrame:
    # UCI format: semicolon-separated, missing as "?"
    df = pd.read_csv(
        txt_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # Combine date+time to datetime index
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    df.insert(0, "datetime", dt)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def interpolate_missing(df: pd.DataFrame, method: Literal["time", "linear"] = "time") -> pd.DataFrame:
    # Time interpolation requires DatetimeIndex
    out = df.copy()
    out = out.interpolate(method=method, limit_direction="both")
    out = out.ffill().bfill()
    return out


def resample_scale(df_1min: pd.DataFrame, scale: str) -> pd.DataFrame:
    """
    Returns aggregated dataframe at a time scale.

    Implementation detail:
    - For power-like channels, mean is reasonable.
    - For sub-metering (energy in Wh per minute), sum is reasonable.
    The paper is not explicit on aggregation; this keeps units intuitive.
    """
    rule = {
        "hourly": "1h",
        "daily": "1D",
        "weekly": "1W",
        # pandas 2.2+ deprecates "M"/"Q" in favor of "ME"/"QE"
        "monthly": "1ME",
        "quarterly": "1QE",
    }[scale]

    sub_cols = [c for c in df_1min.columns if c.lower().startswith("sub_metering")]
    other_cols = [c for c in df_1min.columns if c not in sub_cols]

    agg = {}
    for c in other_cols:
        agg[c] = "mean"
    for c in sub_cols:
        agg[c] = "sum"

    return df_1min.resample(rule).agg(agg).dropna()
