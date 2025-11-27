import os
import pandas as pd
import joblib
from category_encoders.target_encoder import TargetEncoder
from src.probability_of_death.config import (TEST_DATA_PATH,
                                             TRAIN_CATEGORICAL_FEATURES, TRAIN_NUMERICAL_FEATURES,
                                             MODEL_VERSION, ICD_MAPPING_PATH,
                                             MODEL_PATH)
from src.probability_of_death.feature_engineering.feature_engineering import  create_basic_features, encoder_demographics, apply_icd9_mapping
from src.probability_of_death.preprocessing.preprocessor import change_feature_names

def predict_model():
    ########################
    ### 1. PREPROCESSING ###
    ########################

    # Load in train data
    df = pd.read_csv(TEST_DATA_PATH)
    print("✅ TRAINING DATA LOADED")

    # Create basic features
    df = create_basic_features(df)
    print("✅ AGE AND QUARTER-OF-DAY CREATED")

    # Create demographic feature
    df = encoder_demographics(df)
    print("✅ DEMOGRAPHIC FEATURES CENSORED AND INTERIM DROPPED")

    # Change ICD column names (extract first three chars)
    df = change_feature_names(df)
    print("✅ ICD9 CODES SHORTENED - TRAIN")

    # Encode ICD9 codes
    mappings = joblib.load(ICD_MAPPING_PATH)
    df = apply_icd9_mapping(df, mappings)
    print("✅ ICD9 CODE MAPPED")

    X = df[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES]
    print("✅ TEST DATA CREATED")

    model = joblib.load(MODEL_PATH)
    print("✅ MODEL LOADED")

    y_test = model.predict_proba(X)

    submission = pd.DataFrame({
        "icustay_id": df["icustay_id"],
         "HOSPITAL_EXPIRE_FLAG": y_test[:,1]
    })
    os.makedirs("predictions", exist_ok=True)
    submission.to_csv(f"predictions/{MODEL_VERSION}.csv", index=False)

    print(f"✅ PREDICTION COMPLETE, SAVED TO predictions")
    return submission


if __name__ == "__main__":
    predict_model()


