import pandas as pd
import numpy as np
np.random.seed(4201)
from probability_of_death.model.train_model import preprocessing
from probability_of_death.config import (DROP_FEATURES,
                                         ID_FEATURES,
                                         ENG_AGE,
                                         ENG_QUART_DAY,
                                         ENG_DEMOGRAPHIC,
                                         ENGINEERED_MORTALITY_PROXIES,
                                         ENGINEERED_NUMERICAL_FEATURES,
                                         ENGINEERED_CATEGORICAL_FEATURES,
                                         MEAN_MORTALITY,
                                         MAX_MORTALITY,
                                         COUNT_COMORBIDITIES,
)

from tests.utils.synthetic_data import (
make_synthetic_patient_data,
make_synthetic_comorbidities_data,
random_dates
)


def test_preprocessing():
    df = make_synthetic_patient_data()
    comorbidities_df = make_synthetic_comorbidities_data()

    df, mapping = preprocessing(df, comorbidities_df)

    # Assertions
    # New features created - inlcudes mortality features (enocder_demographics)
    for feature in ENGINEERED_MORTALITY_PROXIES + ENGINEERED_NUMERICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES:
        assert feature in df.columns, f"feature {feature} present in dataframe"

    # Features dropped
    for feature in DROP_FEATURES:
        assert feature not in df.columns, f"Column {feature} was not dropped"
    # Age
    assert pd.api.types.is_numeric_dtype(df[ENG_AGE])
    assert df[ENG_AGE].between(0, 90).all()
    # Quarter-of-day
    assert df[ENG_QUART_DAY].dtype.name == "category"

    # Demographic missing
    assert set(df[ENG_DEMOGRAPHIC].unique()).issubset({0, 1})

    # Mortality proxies:
    for col in ENGINEERED_MORTALITY_PROXIES:
        assert pd.api.types.is_numeric_dtype(df[col])
    assert df[MAX_MORTALITY].between(0, 1).all()
    assert df[MEAN_MORTALITY].between(0, 1).all()
    assert df[COUNT_COMORBIDITIES].min() >= 0
    assert pd.api.types.is_integer_dtype(df[COUNT_COMORBIDITIES])