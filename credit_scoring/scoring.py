"""Input normalization shared by offline scoring and service checks."""
import numpy as np
import pandas as pd
from app.schemas import LoanApplication
from credit_scoring.features import prepare_features


def score_frame(raw, model, columns, model_id, model_version=None):
    payloads = raw.astype(object).where(pd.notna(raw), None).to_dict('records')
    fields = list(LoanApplication.model_fields)
    normalized = [LoanApplication.model_validate({key: row[key] for key in fields}).model_dump()
                  for row in payloads]
    features = prepare_features(pd.DataFrame(normalized)).loc[:, columns]
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame({
        'row_id': raw['SK_ID_CURR'].to_numpy() if 'SK_ID_CURR' in raw else raw.index.to_numpy(),
        'prediction': (probabilities >= .5).astype(int),
        'probability_of_default': probabilities,
        'risk_category': np.where(probabilities < .3, 'Low', np.where(probabilities < .6, 'Medium', 'High')),
        'model_id': model_id, 'model_version': model_version,
    })
