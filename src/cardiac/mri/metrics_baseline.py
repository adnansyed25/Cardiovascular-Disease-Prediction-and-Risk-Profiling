from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiac import config, plots

FEATURES = ["EDV_LV", "ESV_LV", "EF_LV", "EDV_RV", "ESV_RV", "EF_RV", "MYO_MASS_ED", "MYO_MASS_ES",
            "EDV_LV_index", "EDV_RV_index", "MYO_MASS_index", "RV_LV_ratio"]


def run(metrics_csv: Path = config.PROCESSED_DIR / "acdc_heart_metrics.csv",
        out_dir: Path = config.REPORTS_DIR) -> dict:
    df = pd.read_csv(metrics_csv)
    df = df[df["group"].isin(config.ACDC_CLASSES)]
    feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feats)
    X, y = df[feats].to_numpy(), df["group"].map(config.ACDC_CLASSES.index).to_numpy()

    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=config.RANDOM_STATE),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    results = {}
    for name, model in candidates.items():
        pred = cross_val_predict(model, X, y, cv=cv)
        results[name] = {"cv_accuracy": accuracy_score(y, pred), "pred": pred}
        print(f"{name:<20} 5-fold CV accuracy {results[name]['cv_accuracy']:.3f}")

    best = max(results, key=lambda k: results[k]["cv_accuracy"])
    pred = results[best]["pred"]
    labels = list(range(len(config.ACDC_CLASSES)))
    cm = confusion_matrix(y, pred, labels=labels)
    out_dir = Path(out_dir)
    plots.confusion_single(cm, config.ACDC_CLASSES, f"Heart-metrics {best} (5-fold CV)",
                           out_dir / "figures" / "mri_metrics_baseline_confusion.png")
    print(classification_report(y, pred, labels=labels, target_names=config.ACDC_CLASSES, zero_division=0))

    summary = {"n_patients": int(len(y)), "features": feats,
               "cv_accuracy": {k: v["cv_accuracy"] for k, v in results.items()}, "best": best}
    (out_dir / "mri_metrics_baseline.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path, default=config.PROCESSED_DIR / "acdc_heart_metrics.csv")
    p.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    a = p.parse_args(argv)
    np.random.seed(config.RANDOM_STATE)
    run(a.metrics, a.out)


if __name__ == "__main__":
    main()
