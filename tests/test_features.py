import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import prepare_features


def test_expected_transformations_and_no_input_mutation(valid_applicant):
    raw = pd.DataFrame([valid_applicant])
    original = raw.copy(deep=True)
    X = prepare_features(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert X.shape == (1, 18)
    assert X.loc[0, "AGE_YEARS"] == 40.0
    assert X.loc[0, "YEARS_ID_PUBLISH"] == 10.0
    assert X.loc[0, "CODE_GENDER"] == 1
    assert X.loc[0, "FLAG_OWN_CAR"] == 0
    assert X.loc[0, "FLAG_OWN_REALTY"] == 1
    assert X.loc[0, "EDUCATION_LEVEL"] == 3
    assert X.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(406597 / 202501)
    assert X.loc[0, "ANNUITY_INCOME_RATIO"] == pytest.approx(24700 / 202501)
    assert X.loc[0, "CREDIT_GOODS_RATIO"] == pytest.approx(406597 / 351001)
    assert "TARGET" not in X.columns
    assert "DAYS_BIRTH" not in X.columns


@pytest.mark.parametrize("employment", [365243, None])
def test_employment_sentinel_and_null_remain_missing(valid_applicant, employment):
    raw = pd.DataFrame([{**valid_applicant, "DAYS_EMPLOYED": employment}])
    assert pd.isna(prepare_features(raw).loc[0, "YEARS_EMPLOYED"])


def test_missing_values_are_not_filled_using_other_rows(valid_applicant):
    first = {**valid_applicant, "EXT_SOURCE_3": None, "AMT_ANNUITY": None}
    X = prepare_features(pd.DataFrame([first, valid_applicant]))
    assert pd.isna(X.loc[0, "EXT_SOURCE_3"])
    assert pd.isna(X.loc[0, "ANNUITY_INCOME_RATIO"])
    # Changing a different row must not change the first row's features.
    changed = {**valid_applicant, "AMT_INCOME_TOTAL": 9999999}
    other = prepare_features(pd.DataFrame([first, changed]))
    pd.testing.assert_series_equal(X.iloc[0], other.iloc[0])


def test_zero_denominators_do_not_produce_infinity(valid_applicant):
    raw = pd.DataFrame([{
        **valid_applicant, "AMT_INCOME_TOTAL": 0, "AMT_GOODS_PRICE": 0,
    }])
    assert np.isfinite(prepare_features(raw).to_numpy()).all()


def test_data_loader_uses_requested_file_and_keeps_missing(tmp_path, valid_applicant):
    from scripts.prepare_data import load_and_prepare_data
    path = tmp_path / "custom.csv"
    pd.DataFrame([
        {**valid_applicant, "EXT_SOURCE_3": None, "TARGET": 1},
        {**valid_applicant, "TARGET": 0},
    ]).to_csv(path, index=False)
    X, y = load_and_prepare_data(path)
    assert list(y) == [1, 0]
    assert "TARGET" not in X.columns
    assert pd.isna(X.loc[0, "EXT_SOURCE_3"])
