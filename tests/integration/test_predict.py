import numpy as np
import pandas as pd
import joblib
from jedi.common import monkeypatch
from pathlib import Path

from tests.utils.synthetic_data import (
make_synthetic_patient_data,
make_synthetic_comorbidities_data
)
from probability_of_death.model.train_model import (
preprocessing,
train_model_after_preprocessing
)
from probability_of_death.config import TARGET


def test_predict_model(tmp_path, monkeypatch):
    # Create a testing test set
    np.random.seed(4201)
    df_test = make_synthetic_patient_data()
    df_test["icustay_id"] = np.random.randint(low=0, high=9999, size=len(df_test))
    test_csv = tmp_path / "test.csv"
    df_test.to_csv(test_csv, index=False)

    # Create a testin train set
    df_train = make_synthetic_patient_data()
    df_comorbidities = make_synthetic_comorbidities_data()

    # Create and save mapping
    _, mapping = preprocessing(df_train, df_comorbidities)
    mapping_path = tmp_path / "mapping.csv"
    joblib.dump(mapping, mapping_path)

    # Train the model
    df_train_processed, _ = preprocessing(df_train, df_comorbidities)
    model = train_model_after_preprocessing(df_train_processed)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    # Patch model load paths
    monkeypatch.setattr("probability_of_death.model.predict_model.TEST_DATA_PATH", str(test_csv))
    monkeypatch.setattr("probability_of_death.model.predict_model.ICD_MAPPING_PATH", str(mapping_path))
    monkeypatch.setattr("probability_of_death.model.predict_model.MODEL_PATH", str(model_path))
    monkeypatch.setattr("probability_of_death.model.predict_model.MODEL_VERSION", "test_version")

    from probability_of_death.model.predict_model import predict_model
    submission_df = predict_model()

    assert "icustay_id" in submission_df.columns, "ICU stay id not present"
    assert TARGET in submission_df.columns, "HOSPITAL_EXPIRE_FLAG not present"
    assert submission_df[TARGET].between(0,1).all(), "Prediction probabilities outside valid range"

    output_file = Path("predictions/test_version.csv")
    assert output_file.exists(), "Prediction file was not created"






