import os
import sys
import pandas as pd
import joblib
from category_encoders.target_encoder import TargetEncoder
from probability_of_death.config import (TEST_DATA_PATH,TRAIN_CATEGORICAL_FEATURES_LESS_DIAGNOSIS,
                                             TRAIN_NUMERICAL_FEATURES, DIAGNOSIS,
                                             MODEL_VERSION, ICD_MAPPING_PATH,
                                             MODEL_PATH)
from probability_of_death.feature_engineering.feature_engineering import  create_basic_features, encoder_demographics, apply_icd9_mapping
from probability_of_death.preprocessing.preprocessor import change_feature_names
import logging
logging.basicConfig(
                    level=logging.INFO,
                    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers = [logging.StreamHandler(sys.stdout),
                                logging.FileHandler("testing.log")])
logger = logging.getLogger(__name__)



def predict_model():
    ########################
    ### 1. PREPROCESSING ###
    ########################
    try:
        # Load in train data
        df = pd.read_csv(TEST_DATA_PATH)
        logger.info("TRAINING DATA LOADED")

        # Create basic features
        df = create_basic_features(df)
        logger.info("AGE AND QUARTER-OF-DAY CREATED")

        # Create demographic feature
        df = encoder_demographics(df)
        logger.info("DEMOGRAPHIC FEATURES CENSORED AND INTERIM DROPPED")

        # Change ICD column names (extract first three chars)
        df = change_feature_names(df)
        logger.info("ICD9 CODES SHORTENED - TRAIN")

        # Encode ICD9 codes
        mappings = joblib.load(ICD_MAPPING_PATH)
        df = apply_icd9_mapping(df, mappings)
        logger.info("ICD9 CODE MAPPED")

        X = df[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES_LESS_DIAGNOSIS + [DIAGNOSIS]]
        logger.info("TEST DATA WITH FEATURES SELECTED")

        model = joblib.load(MODEL_PATH)
        logger.info("MODEL LOADED")

        y_test = model.predict_proba(X)

        submission = pd.DataFrame({
            "icustay_id": df["icustay_id"],
             "HOSPITAL_EXPIRE_FLAG": y_test[:,1]
        })
        os.makedirs("predictions", exist_ok=True)
        submission.to_csv(f"predictions/{MODEL_VERSION}.csv", index=False)

        logger.info(f"PREDICTION COMPLETE, SAVED TO predictions")
        return submission
    except Exception as e:
        logger.error(f"PIPELINE FAILED TO PREDICT {e}")


if __name__ == "__main__":
    predict_model()


