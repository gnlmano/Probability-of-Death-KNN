import pytest
from tests.utils.synthetic_data import (
make_synthetic_patient_data,
make_synthetic_comorbidities_data,
random_dates
)
from probability_of_death.model.train_model import preprocessing

# Creates a prerprocssed train data with mapping
@pytest.fixture
def preprocessed_training_data():
    df_raw = make_synthetic_patient_data()
    df_comorbidities = make_synthetic_comorbidities_data()
    df_processed, mapping = preprocessing(df_raw, df_comorbidities)

    return df_processed, mapping

