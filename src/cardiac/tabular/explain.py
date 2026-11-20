from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cardiac import config, plots


def permutation(model, X: pd.DataFrame, y: pd.Series, title: str, path: Path,
                n_repeats: int = 5, max_rows: int = 5000) -> pd.DataFrame:
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=config.RANDOM_STATE)
        y = y.loc[X.index]
    res = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    table = (pd.DataFrame({"feature": X.columns, "importance": res.importances_mean,
                           "std": res.importances_std})
             .sort_values("importance", ascending=False)
             .reset_index(drop=True))
    plots.horizontal_importance(table["feature"], table["importance"], table["std"],
                                f"Permutation importance (drop in ROC-AUC): {title}", path)
    return table


def shap_summary(model, X: pd.DataFrame, path: Path, max_rows: int = 1000) -> Path | None:
    try:
        import shap
    except ImportError:
        print("shap not installed: skipping SHAP plot (pip install shap)")
        return None
    import matplotlib.pyplot as plt

    X = X.sample(min(max_rows, len(X)), random_state=config.RANDOM_STATE)
    clf = model.named_steps["clf"]
    Xt = X
    if "pre" in model.named_steps:
        Xt = pd.DataFrame(model.named_steps["pre"].transform(X),
                          columns=model.named_steps["pre"].get_feature_names_out(), index=X.index)
    try:
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Xt)
    except Exception:
        background = shap.sample(Xt, 100, random_state=config.RANDOM_STATE)
        explainer = shap.Explainer(lambda d: clf.predict_proba(d)[:, 1], background)
        values = explainer(Xt).values
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    shap.summary_plot(values, Xt, show=False)
    fig = plt.gcf()
    return plots.save(fig, path)
