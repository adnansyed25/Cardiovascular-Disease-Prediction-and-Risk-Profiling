from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cardiac import config

RAW_COLUMNS = [
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
]


@dataclass
class CleaningReport:
    start_rows: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)

    def log(self, step: str, removed: int) -> None:
        self.steps.append((step, int(removed)))

    @property
    def end_rows(self) -> int:
        return self.start_rows - sum(n for _, n in self.steps)

    def __str__(self) -> str:
        lines = [f"Start rows: {self.start_rows:,}"]
        lines += [f"  - {name}: removed {n:,}" for name, n in self.steps]
        lines.append(f"Final rows: {self.end_rows:,}")
        return "\n".join(lines)


def load_raw(path: str | Path = config.RAW_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for where to get it."
        )
    df = pd.read_csv(path, sep=None, engine="python")
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df[RAW_COLUMNS]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_years"] = (out["age"] / 365.25).round(1)
    out["bmi"] = (out["weight"] / (out["height"] / 100) ** 2).round(1)
    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
    out["mean_arterial_pressure"] = ((out["ap_hi"] + 2 * out["ap_lo"]) / 3).round(1)
    out["is_male"] = (out["gender"] == 2).astype(int)
    return out


def clean(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    report = report if report is not None else CleaningReport()
    report.start_rows = len(df)

    n = len(df)
    df = df.dropna()
    report.log("missing values", n - len(df))

    n = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
    report.log("duplicate records", n - len(df))

    df = add_features(df)

    for col, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        n = len(df)
        df = df[df[col].between(lo, hi)]
        report.log(f"{col} outside [{lo}, {hi}]", n - len(df))

    n = len(df)
    df = df[df["ap_hi"] > df["ap_lo"]]
    report.log("systolic <= diastolic", n - len(df))

    return df.reset_index(drop=True)


def load_clean(path: str | Path = config.RAW_CSV, verbose: bool = False) -> pd.DataFrame:
    report = CleaningReport()
    df = clean(load_raw(path), report)
    if verbose:
        print(report)
    return df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURES].copy(), df[config.TARGET].astype(int)


def make_synthetic(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age_years = rng.uniform(30, 65, n)
    ap_hi = rng.normal(128, 15, n).round()
    ap_lo = (ap_hi - rng.normal(45, 8, n)).round()
    chol = rng.choice([1, 2, 3], n, p=[0.75, 0.14, 0.11])
    logit = 0.06 * (ap_hi - 128) + 0.05 * (age_years - 50) + 0.5 * (chol - 1)
    cardio = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "id": np.arange(n),
        "age": (age_years * 365.25).astype(int),
        "gender": rng.choice([1, 2], n),
        "height": rng.normal(165, 8, n).round(),
        "weight": rng.normal(74, 12, n).round(1),
        "ap_hi": ap_hi.astype(int),
        "ap_lo": ap_lo.astype(int),
        "cholesterol": chol,
        "gluc": rng.choice([1, 2, 3], n, p=[0.85, 0.07, 0.08]),
        "smoke": rng.choice([0, 1], n, p=[0.9, 0.1]),
        "alco": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "active": rng.choice([0, 1], n, p=[0.2, 0.8]),
        "cardio": cardio,
    })
