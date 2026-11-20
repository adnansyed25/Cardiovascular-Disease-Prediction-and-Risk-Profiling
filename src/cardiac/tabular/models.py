from __future__ import annotations

from typing import Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardiac import config

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _scaled(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler(), config.NUMERIC_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _passthrough(estimator) -> Pipeline:
    return Pipeline([("clf", estimator)])


RS = config.RANDOM_STATE

_REGISTRY: dict[str, tuple[str, Callable[[], Pipeline], dict]] = {
    "naive_bayes": (
        "Naive Bayes",
        lambda: _scaled(GaussianNB()),
        {"clf__var_smoothing": loguniform(1e-12, 1e-6)},
    ),
    "logistic_regression": (
        "Logistic Regression",
        lambda: _scaled(LogisticRegression(max_iter=2000)),
        {"clf__C": loguniform(1e-3, 1e2)},
    ),
    "knn": (
        "k-Nearest Neighbours",
        lambda: _scaled(KNeighborsClassifier(n_jobs=-1)),
        {"clf__n_neighbors": randint(15, 151), "clf__weights": ["uniform", "distance"]},
    ),
    "random_forest": (
        "Random Forest",
        lambda: _passthrough(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)),
        {
            "clf__max_depth": randint(6, 16),
            "clf__min_samples_leaf": randint(5, 60),
            "clf__max_features": ["sqrt", 0.5],
        },
    ),
    "hist_gradient_boosting": (
        "Hist Gradient Boosting",
        lambda: _passthrough(HistGradientBoostingClassifier(random_state=RS)),
        {
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__max_leaf_nodes": randint(8, 48),
            "clf__min_samples_leaf": randint(20, 200),
            "clf__l2_regularization": loguniform(1e-3, 10),
        },
    ),
}

if HAS_XGBOOST:
    _REGISTRY["xgboost"] = (
        "XGBoost",
        lambda: _passthrough(
            XGBClassifier(
                n_estimators=400, tree_method="hist", eval_metric="logloss",
                n_jobs=-1, random_state=RS,
            )
        ),
        {
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__max_depth": randint(3, 8),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__min_child_weight": randint(1, 20),
        },
    )


def available_models() -> list[str]:
    return list(_REGISTRY)


def display_name(name: str) -> str:
    return _REGISTRY[name][0]


def build(name: str) -> Pipeline:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from: {available_models()}")
    return _REGISTRY[name][1]()


def search_space(name: str) -> dict:
    return _REGISTRY[name][2]
