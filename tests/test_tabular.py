import numpy as np
import pandas as pd
import pytest

from cardiac import config
from cardiac.tabular import data, models, train


@pytest.fixture(scope="module")
def raw():
    return data.make_synthetic(600, seed=1)


def test_features_are_derived_correctly(raw):
    df = data.add_features(raw)
    row = df.iloc[0]
    assert row["age_years"] == pytest.approx(raw.iloc[0]["age"] / 365.25, abs=0.06)
    assert row["bmi"] == pytest.approx(raw.iloc[0]["weight"] / (raw.iloc[0]["height"] / 100) ** 2, abs=0.06)
    assert row["pulse_pressure"] == raw.iloc[0]["ap_hi"] - raw.iloc[0]["ap_lo"]
    assert set(df["is_male"]) <= {0, 1}


def test_clean_removes_impossible_values(raw):
    bad = raw.copy()
    bad.loc[0, "ap_hi"] = 16020
    bad.loc[1, "ap_lo"] = -70
    bad.loc[2, ["ap_hi", "ap_lo"]] = [80, 120]
    bad = pd.concat([bad, bad.iloc[[3]].assign(id=9999)])
    report = data.CleaningReport()
    out = data.clean(bad, report)
    assert len(out) == report.end_rows
    assert {0, 1, 2}.isdisjoint(set(out["id"]))
    assert (out["ap_hi"] > out["ap_lo"]).all()
    for col, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        assert out[col].between(lo, hi).all()


def test_ordinal_encoding_preserves_order(raw):
    X, _ = data.get_xy(data.clean(raw))
    assert set(X["cholesterol"].unique()) <= {1, 2, 3}


def test_load_raw_errors_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_raw(tmp_path / "missing.csv")
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        data.load_raw(tmp_path / "bad.csv")


def test_semicolon_csv_supported(tmp_path, raw):
    path = tmp_path / "semi.csv"
    raw.to_csv(path, sep=";", index=False)
    assert len(data.load_raw(path)) == len(raw)


@pytest.mark.parametrize("name", models.available_models())
def test_every_model_trains_and_predicts(name, raw):
    X, y = data.get_xy(data.clean(raw))
    m = models.build(name).fit(X, y)
    proba = m.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_training_pipeline_end_to_end(tmp_path, raw, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    df = data.clean(raw)
    res = train.run(model_names=["logistic_regression", "naive_bayes"], n_iter=2, cv_folds=3,
                    out_dir=tmp_path, df=df)
    assert len(res) == 2
    assert res["test_roc_auc"].between(0, 1).all()
    for f in ["tabular_results.csv", "tabular_results.md", "figures/tabular_roc.png",
              "figures/tabular_confusion.png", "figures/tabular_importance.png"]:
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "models" / "best_tabular_model.joblib").exists()
