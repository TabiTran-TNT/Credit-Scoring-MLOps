"""Clearly synthetic data for CI and installation checks, not model evaluation."""
import numpy as np
import pandas as pd


def synthetic_applications(rows=240, seed=42):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({
        'SK_ID_CURR': np.arange(rows),
        'EXT_SOURCE_1': rng.uniform(0, 1, rows), 'EXT_SOURCE_2': rng.uniform(0, 1, rows),
        'EXT_SOURCE_3': rng.uniform(0, 1, rows),
        'DAYS_BIRTH': rng.integers(-24000, -8000, rows),
        'DAYS_EMPLOYED': rng.integers(-5000, -1, rows),
        'DAYS_ID_PUBLISH': rng.integers(-4000, -1, rows),
        'AMT_INCOME_TOTAL': rng.uniform(80000, 400000, rows),
        'AMT_CREDIT': rng.uniform(100000, 900000, rows),
        'AMT_ANNUITY': rng.uniform(10000, 50000, rows),
        'AMT_GOODS_PRICE': rng.uniform(100000, 800000, rows),
        'CODE_GENDER': rng.choice(['F', 'M'], rows), 'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y', 'CNT_CHILDREN': rng.integers(0, 4, rows),
        'NAME_EDUCATION_TYPE': 'Higher education',
    })
    raw['TARGET'] = (raw['EXT_SOURCE_1'] < .4).astype(int)
    raw.loc[::11, 'EXT_SOURCE_3'] = np.nan
    return raw
