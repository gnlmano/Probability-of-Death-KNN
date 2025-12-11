import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, accuracy_score, precision_score
from category_encoders.target_encoder import TargetEncoder
import xgboost as xgb
from probability_of_death.config import (RAW_DATA_PATH, DROP_FEATURES, DIAGNOSIS,
                                             COMORBIDITY_DATA_PATH, TARGET, ID_FEATURES,
                                             TRAIN_NUMERICAL_FEATURES,
                                             XGB_PARAMS, MODEL_VERSION, MODEL_OUTPUT_PATH,
                                         TRAIN_CATEGORICAL_FEATURES_LESS_DIAGNOSIS)
from probability_of_death.feature_engineering.feature_engineering import  create_basic_features, encoder_demographics, encoder_icd9_codes
from probability_of_death.preprocessing.preprocessor import drop_features, change_feature_names, change_comorbidities_icd9code

import logging

logging.basicConfig(
                    level=logging.INFO,
                    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers = [logging.StreamHandler(sys.stdout),
                                logging.FileHandler("training.log")])
logger = logging.getLogger(__name__)

# Preprocessing function
def preprocessing(df, comorbidities):
    ########################
    ### 1. PREPROCESSING ####
    ########################

    # Create basic features
    df = create_basic_features(df)
    logger.info("AGE AND QUARTER-OF-DAY CREATED")
    # Create demographic feature
    df = encoder_demographics(df)
    logger.info("DEMOGRAPHIC FEATURES CENSORED AND INTERIM DROPPED")
    # Drop unused features
    df = drop_features(df, DROP_FEATURES)
    logger.info("UNUSED FEATURES DROPPED")
    # Change ICD column names (extract first three chars)
    df = change_feature_names(df)
    logger.info("ICD9 CODES SHORTENED - TRAIN")
    # Shorten comorbid ICD codes
    comorbidities = change_comorbidities_icd9code(comorbidities)
    logger.info("ICD9 CODES SHORTENED - COMORBIDITIES")
    # Encode ICD9 comorbidities data
    df, mapping, global_mean = encoder_icd9_codes(df, comorbidities)
    logger.info("ICD9 CODE ENCODED")
    # Drop ID features
    df = df.drop(ID_FEATURES, axis=1)
    logger.info("ID FEATURES DROPPED")
    logger.info("PREPROCESSING COMPLETE")

    return df, mapping

def train_model_after_preprocessing(df):

    # TRAINING INPUTS
    X = df[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES_LESS_DIAGNOSIS + [DIAGNOSIS]]
    y = df[TARGET]

    ########################
    ### 2. TRANSFORMERS ####
    ########################
    diagnosis_encoder = TargetEncoder(
        cols=[DIAGNOSIS],
        smoothing=0.25,
        handle_missing='value',
        handle_unknown='value'
    )
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, TRAIN_NUMERICAL_FEATURES + [DIAGNOSIS]),
            ("cat", categorical_transformer, TRAIN_CATEGORICAL_FEATURES_LESS_DIAGNOSIS)
        ],
        remainder='drop'
    )
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model_pipeline = Pipeline([
        ("diagnosis_target_encoder", diagnosis_encoder),
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    logger.info("PIPELINE CREATED")
    ########################
    ### 3. TRAIN MODEL ###
    ########################
    model_pipeline.fit(X, y)
    logger.info("MODEL TRAINED")
    # Check model error
    y_hat_in = model_pipeline.predict(X)
    logger.info("MODEL RUNS")
    logger.info(
        f'1. Model Accuracy:{accuracy_score(y, y_hat_in)} \n'
        f'2. Model Recall:{recall_score(y, y_hat_in)} \n'
        f'3. Model Precision: {precision_score(y, y_hat_in)}'
    )
    return model_pipeline

def train_model():
    try:
        df_train_raw = pd.read_csv(RAW_DATA_PATH)
        logger.info("TRAINING DATA LOADED")

        df_comorbidites_raw = pd.read_csv(COMORBIDITY_DATA_PATH)
        logger.info("COMORBIDITIES DATA LOADED")
        df_train_processed, mapping = preprocessing(df_train_raw, df_comorbidites_raw)
        logger.info("PREPROCESSING")
        model_pipeline = train_model_after_preprocessing(df_train_processed)
        logger.info("MODEL TRAINING")
        ########################
        ### 2. SAVE MODEL ######
        ########################
        logger.info("SAVING MODEL")
        model_path = f"{MODEL_OUTPUT_PATH}/model_{MODEL_VERSION}.pkl"
        mapping_path = f"{MODEL_OUTPUT_PATH}/icd_mapping_{MODEL_VERSION}.pkl"

        joblib.dump(model_pipeline, model_path)
        joblib.dump(mapping, mapping_path)

        logger.info("SAVED MODEL")
    except Exception as e:
        logger.error(f"PIPELINE FAILED TO TRAIN {e}")
        raise


if  __name__ == "__main__":
    train_model()