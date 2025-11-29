import numpy as np
import pandas as pd
import joblib
import pytest

from tests.utils.synthetic_data import (
make_synthetic_patient_data,
make_synthetic_comorbidities_data
)
from probability_of_death.config import (
TRAIN_NUMERICAL_FEATURES,
TRAIN_CATEGORICAL_FEATURES,
TARGET,
MODEL_VERSION,
RAW_DATA_PATH,
COMORBIDITY_DATA_PATH
)
from probability_of_death.model.train_model import  (
train_model_after_preprocessing,
train_model,
preprocessing
)

def test_train_model_after_preprocessing(preprocessed_training_data):

    df_train, _ = preprocessed_training_data
    model = train_model_after_preprocessing(df_train)
    X = df_train[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES]
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:,1]

    assert model is not None, "Model doesn't not exist"
    assert hasattr(model, "predict"), "Model cannot predict"
    assert(len(y_pred) == len(X)), "Model output length doesn't match"
    assert set(pd.Series(y_pred).unique()).issubset({0,1}), "Model prediction are outside valid ranges"
    assert pd.Series(y_pred_proba).between(0,1).all(), "Model probability prediction are outside valide ranges"

# Test that the model is being saved correctly + the saved model can be loaded and make predictions
# Create fixture for saved model and comorbidities path
@pytest.fixture
def trained_model_and_mapping(tmp_path, preprocessed_training_data):
    df_processed, mapping = preprocessed_training_data

    # Get the trained model and save model and mapping accordingly.
    model = train_model_after_preprocessing(df_processed)
    model_path = tmp_path / "model.pkl"
    mapping_path = tmp_path / "mapping.pkl"

    joblib.dump(model, model_path)
    joblib.dump(mapping, mapping_path)

    return model_path, mapping_path, df_processed

def test_train_model(tmp_path, monkeypatch, preprocessed_training_data):
    df_train = make_synthetic_patient_data()
    df_comorbidities = make_synthetic_comorbidities_data()

    train_csv = tmp_path / "train.csv"
    comorbidities_csv = tmp_path / "comorbidities.csv"

    df_train.to_csv(train_csv, index=False)
    df_comorbidities.to_csv(comorbidities_csv, index=False)

    monkeypatch.setattr("probability_of_death.model.train_model.RAW_DATA_PATH", str(train_csv))
    monkeypatch.setattr("probability_of_death.model.train_model.COMORBIDITY_DATA_PATH", str(comorbidities_csv))
    monkeypatch.setattr("probability_of_death.model.train_model.MODEL_OUTPUT_PATH", str(tmp_path))

    train_model()

    model_path = tmp_path / f"model_{MODEL_VERSION}.pkl"
    mapping_path = tmp_path / f"icd_mapping_{MODEL_VERSION}.pkl"

    assert model_path.exists(), "Model doesn't exist"
    assert mapping_path.exists(), "Mapping doesn't exist"

    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict"), "Model cannot predict"

    # Load saved model
    loaded_model = joblib.load(model_path)

    # Test that model can predict on new data
    df_test = make_synthetic_patient_data()
    df_comorbidities_test = make_synthetic_comorbidities_data()

    df_test, mapping = preprocessing(df_test, df_comorbidities_test)
    df_test = df_test[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES]

    predictions = loaded_model.predict(df_test)
    prediction_proba = loaded_model.predict_proba(df_test)[:, 1]

    assert len(predictions) == len(df_test), "Prediction length doesn't match"
    assert set(predictions).issubset({0,1}), "Prediction values are outside valid ranges"
    assert pd.Series(prediction_proba).between(0, 1).all(), "Prediction probability values are outside valide ranges"