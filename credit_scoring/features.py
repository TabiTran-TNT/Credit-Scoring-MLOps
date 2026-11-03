import numpy as np
import pandas as pd

SELECTED_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "NAME_EDUCATION_TYPE",
]


def prepare_features(df):
    """Apply the same row-wise transformations in training and serving.

    No medians or other dataset statistics are calculated here.
    Ratios are calculated before the pipeline's median imputation.
    """
    X = df.loc[:, SELECTED_FEATURES].copy()
    numeric_columns = [
        column for column in SELECTED_FEATURES
        if column not in {
            "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "NAME_EDUCATION_TYPE",
        }
    ]
    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="raise").astype(float)

    X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)
    X["AGE_YEARS"] = (-X["DAYS_BIRTH"] / 365.25).round(1)
    X["YEARS_EMPLOYED"] = (-X["DAYS_EMPLOYED"] / 365.25).round(1)
    X["YEARS_ID_PUBLISH"] = (-X["DAYS_ID_PUBLISH"] / 365.25).round(1)
    X = X.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH"])

    # Preserve the tutorial's mapping to match the existing trained model.
    X["CODE_GENDER"] = X["CODE_GENDER"].map({"M": 0, "F": 1}).fillna(0).astype(int)
    X["FLAG_OWN_CAR"] = X["FLAG_OWN_CAR"].map({"N": 0, "Y": 1}).astype(int)
    X["FLAG_OWN_REALTY"] = X["FLAG_OWN_REALTY"].map({"N": 0, "Y": 1}).astype(int)
    education_map = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3,
        "Academic degree": 4,
    }
    X["EDUCATION_LEVEL"] = X["NAME_EDUCATION_TYPE"].map(education_map).fillna(1).astype(int)
    X = X.drop(columns=["NAME_EDUCATION_TYPE"])

    X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / (X["AMT_INCOME_TOTAL"] + 1)
    X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / (X["AMT_INCOME_TOTAL"] + 1)
    X["CREDIT_GOODS_RATIO"] = X["AMT_CREDIT"] / (X["AMT_GOODS_PRICE"] + 1)
    return X
