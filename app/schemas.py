from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoanApplication(BaseModel):
    """Raw application fields; names match the source CSV.

    Nullable fields must be present, but may contain JSON null.
    The fitted pipeline handles those missing numeric values.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    EXT_SOURCE_1: float | None = Field(..., ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(..., ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(..., ge=0, le=1)
    DAYS_BIRTH: int = Field(..., ge=-36525, le=-6570)
    DAYS_EMPLOYED: int | None = Field(..., ge=-36525)
    DAYS_ID_PUBLISH: int = Field(..., le=0)
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float | None = Field(..., gt=0)
    AMT_GOODS_PRICE: float | None = Field(..., gt=0)
    CODE_GENDER: Literal["M", "F", "XNA"]
    FLAG_OWN_CAR: Literal["N", "Y"]
    FLAG_OWN_REALTY: Literal["N", "Y"]
    CNT_CHILDREN: int = Field(..., ge=0)
    NAME_EDUCATION_TYPE: Literal[
        "Lower secondary",
        "Secondary / secondary special",
        "Incomplete higher",
        "Higher education",
        "Academic degree",
    ]

    @field_validator("DAYS_EMPLOYED")
    @classmethod
    def validate_employment_days(cls, value):
        if value is not None and value > 0 and value != 365243:
            raise ValueError("Use non-positive days, null, or sentinel 365243.")
        return value


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability_of_default: float = Field(ge=0, le=1)
    risk_category: Literal["Low", "Medium", "High"]


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
