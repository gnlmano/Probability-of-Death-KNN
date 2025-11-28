import numpy as np
import pandas as pd
import joblib
import pytest

from probability_of_death.config import (TRAIN_NUMERICAL_FEATURES,
                                         TRAIN_CATEGORICAL_FEATURES,
      )
from probability_of_death.model.train_model import  train_model_after_preprocessing

def test_model_training(preprocessed_training_data):

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


# Test that the model is being save correctly + the saved model can be loaded and make predictions
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

def test_model_saving(trained_model_and_mapping):
    model_path, mapping_path, _ = trained_model_and_mapping

    assert model_path.exists(), "Model doesn't not exist"
    assert mapping_path.exists(), "Mapping doesn't exist"

    loaded_model = joblib.load(model_path)
    loaded_mapping = joblib.load(mapping_path)

    assert hasattr(loaded_model, "predict"), "Model cannot predict"
    assert isinstance(loaded_mapping, pd.DataFrame), "Mapping is not a dataframe"

def test_model_predict(trained_model_and_mapping):
    model_path, _, df_processed = trained_model_and_mapping

    loaded_model = joblib.load(model_path)
    X = df_processed[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES]

    prediction = loaded_model.predict(X)
    prediction_proba = loaded_model.predict_proba(X)[:,1]

    assert len(prediction) == len(X), "Prediction length doesn't match"
    assert set(prediction).issubset({0,1}), "Prediction values are outside valide ranges"
    assert pd.Series(prediction_proba).between(0,1).all(), "Prediction probability values are outside valide ranges"




















