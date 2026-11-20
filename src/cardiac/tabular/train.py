from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split

from cardiac import config, plots
from cardiac.tabular import data, explain, models

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluate(model, X, y) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def fit_model(name: str, X_train, y_train, cv, n_iter: int, tune: bool):
    pipe = models.build(name)
    if tune and models.search_space(name):
        search = RandomizedSearchCV(
            pipe, models.search_space(name), n_iter=n_iter, scoring="roc_auc", cv=cv,
            n_jobs=-1, random_state=config.RANDOM_STATE, refit=True,
        )
        search.fit(X_train, y_train)
        idx = search.best_index_
        return (search.best_estimator_, float(search.cv_results_["mean_test_score"][idx]),
                float(search.cv_results_["std_test_score"][idx]), search.best_params_)
    scores = cross_val_score(clone(pipe), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)
    return pipe, float(scores.mean()), float(scores.std()), {}


def run(data_path: Path = config.RAW_CSV, model_names: list[str] | None = None, n_iter: int = 15,
        cv_folds: int = 5, test_size: float = 0.2, sample: int | None = None, tune: bool = True,
        out_dir: Path = config.REPORTS_DIR, df: pd.DataFrame | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = data.load_clean(data_path, verbose=True)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=config.RANDOM_STATE)
    X, y = data.get_xy(df)
    print(f"Using {len(X):,} patients | positive rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    model_names = model_names or models.available_models()
    rows, rocs, cms, fitted = [], {}, {}, {}
    for name in model_names:
        t0 = time.time()
        est, cv_auc, cv_std, params = fit_model(name, X_train, y_train, cv, n_iter, tune)
        test = evaluate(est, X_test, y_test)
        train_acc = accuracy_score(y_train, est.predict(X_train))
        label = models.display_name(name)
        rows.append({"model": label, "key": name, "cv_roc_auc": cv_auc, "cv_roc_auc_std": cv_std,
                     "train_accuracy": train_acc, **{f"test_{k}": v for k, v in test.items()},
                     "best_params": json.dumps({k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                                                for k, v in params.items()}),
                     "seconds": round(time.time() - t0, 1)})
        fpr, tpr, _ = roc_curve(y_test, est.predict_proba(X_test)[:, 1])
        rocs[label] = (fpr, tpr, test["roc_auc"])
        cms[label] = confusion_matrix(y_test, est.predict(X_test))
        fitted[name] = est
        print(f"{label:<24} CV AUC {cv_auc:.3f}±{cv_std:.3f} | test acc {test['accuracy']:.3f} "
              f"AUC {test['roc_auc']:.3f} | {time.time() - t0:.0f}s")

    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    best_key = results.loc[0, "key"]
    best = fitted[best_key]
    best_label = models.display_name(best_key)

    plots.roc_curves(rocs, "ROC curves: held-out test set", fig_dir / "tabular_roc.png")
    plots.confusion_grid(cms, ["No CVD", "CVD"], fig_dir / "tabular_confusion.png")
    plots.metric_bars({r["model"]: {m: r[f"test_{m}"] for m in METRICS} for r in rows},
                      METRICS, "Test-set metrics by model", fig_dir / "tabular_metrics.png")
    importance = explain.permutation(best, X_test, y_test, best_label, fig_dir / "tabular_importance.png")
    try:
        explain.shap_summary(best, X_test, fig_dir / "tabular_shap.png")
    except Exception as e:
        print(f"SHAP plot skipped ({type(e).__name__}: {e})")

    results.to_csv(out_dir / "tabular_results.csv", index=False)
    importance.to_csv(out_dir / "tabular_feature_importance.csv", index=False)
    _write_markdown(results, importance, best_label, len(X_train), len(X_test), out_dir / "tabular_results.md")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": config.FEATURES, "name": best_label},
                config.MODELS_DIR / "best_tabular_model.joblib")
    print(f"\nBest by CV: {best_label}. Results written to {out_dir}")
    return results


def _write_markdown(results, importance, best_label, n_train, n_test, path: Path) -> None:
    cols = ["model", "cv_roc_auc", "train_accuracy", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc"]
    t = results[cols].copy()
    for c in cols[1:]:
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in t.values)
    top = ", ".join(importance["feature"].head(5))
    path.write_text(
        f"# Clinical model results\n\nTrain: {n_train:,} patients | Test: {n_test:,} patients "
        f"(stratified 80/20 split, 5-fold CV for tuning).\n\n{header}{body}\n"
        f"Best model (by CV ROC-AUC): **{best_label}**. Top features: {top}.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.RAW_CSV)
    p.add_argument("--models", nargs="+", choices=models.available_models())
    p.add_argument("--n-iter", type=int, default=15, help="random-search iterations per model")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--sample", type=int, help="subsample N rows (for quick experiments)")
    p.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    p.add_argument("--quick", action="store_true", help="= --sample 5000 --no-tune --cv 3")
    p.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    a = p.parse_args(argv)
    if a.quick:
        a.sample, a.no_tune, a.cv = a.sample or 5000, True, 3
    np.random.seed(config.RANDOM_STATE)
    run(a.data, a.models, a.n_iter, a.cv, sample=a.sample, tune=not a.no_tune, out_dir=a.out)


if __name__ == "__main__":
    main()
