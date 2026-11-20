from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cardiac import config, plots
from cardiac.plots import plt
from cardiac.tabular import data

LEVELS = {1: "Normal", 2: "Above normal", 3: "Well above"}


def run(data_path: Path = config.RAW_CSV, out_dir: Path = config.FIGURES_DIR) -> pd.DataFrame:
    out_dir = Path(out_dir)
    df = data.load_clean(data_path, verbose=True)

    cols = ["age_years", "bmi", "ap_hi", "ap_lo"]
    titles = ["Age (years)", "BMI (kg/m²)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, title in zip(axes.flat, cols, titles):
        for val, color, lab in [(0, plots.PALETTE[0], "No CVD"), (1, plots.PALETTE[1], "CVD")]:
            ax.hist(df.loc[df.cardio == val, col], bins=40, alpha=0.6, color=color, label=lab, density=True)
        ax.set_title(title)
        ax.legend(fontsize=8)
    plots.save(fig, out_dir / "eda_distributions.png")

    cats = {"cholesterol": "Cholesterol", "gluc": "Glucose", "smoke": "Smoker",
            "alco": "Alcohol", "active": "Physically active", "is_male": "Male"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, (col, title) in zip(axes.flat, cats.items()):
        rate = df.groupby(col)["cardio"].mean()
        labels = [LEVELS.get(i, str(i)) if col in ("cholesterol", "gluc") else ("Yes" if i else "No")
                  for i in rate.index]
        bars = ax.bar(labels, rate.values, color=plots.PALETTE[0])
        ax.bar_label(bars, labels=[f"{v * 100:.0f}%" for v in rate.values], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("CVD rate")
    axes[1, 0].set_ylabel("CVD rate")
    plots.save(fig, out_dir / "eda_risk_factors.png")

    bands = pd.cut(df["age_years"], bins=np.arange(25, 70, 5))
    rate = df.groupby(bands, observed=True)["cardio"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([str(b) for b in rate.index], rate.values, marker="o", color=plots.PALETTE[1])
    ax.set_ylabel("CVD rate")
    ax.set_xlabel("Age band (years)")
    ax.set_title("CVD prevalence rises with age")
    ax.grid(alpha=0.3)
    plots.save(fig, out_dir / "eda_age_trend.png")

    corr = df[config.FEATURES + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlations")
    plots.save(fig, out_dir / "eda_correlation.png")

    summary = df[config.FEATURES + [config.TARGET]].describe().T
    print(summary.round(2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=config.RAW_CSV)
    p.add_argument("--out", type=Path, default=config.FIGURES_DIR)
    a = p.parse_args(argv)
    run(a.data, a.out)


if __name__ == "__main__":
    main()
