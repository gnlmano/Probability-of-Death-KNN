import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, accuracy_score, precision_score
from category_encoders.target_encoder import TargetEncoder
import xgboost as xgb
from src.config import (RAW_DATA_PATH, TEST_DATA_PATH, DROP_FEATURES,
                        COMORBIDITY_DATA_PATH, TARGET,ID_FEATURES,
                        TRAIN_CATEGORICAL_FEATURES, TRAIN_NUMERICAL_FEATURES,
                        XGB_PARAMS, MODEL_VERSION, MODEL_OUTPUT_PATH)
from src.feature_engineering.feature_engineering import  create_basic_features, encoder_demographics, encoder_icd9_codes
from src.preprocessing.preprocessor import drop_features, change_feature_names, change_comorbidities_icd9code

def train_model():
    ########################
    ### 1. PREPROCESSING ###
    ########################

    # Load in train data
    df = pd.read_csv(RAW_DATA_PATH)
    print("✅ TRAINING DATA LOADED")

    # Create basic features
    df = create_basic_features(df)
    print("✅ AGE AND QUARTER-OF-DAY CREATED")

    # Create demographic feature
    df = encoder_demographics(df)
    print("✅ DEMOGRAPHIC FEATURES CENSORED AND INTERIM DROPPED")

    #  Drop unused features
    df = drop_features(df, DROP_FEATURES)
    print("✅ UNUSED FEATURES DROPPED")

    # Change ICD column names (extract first three chars)
    df = change_feature_names(df)
    print("✅ ICD9 CODES SHORTENED - TRAIN")

    # Change ICD9 features (extract first three chars) from comorbidities
    comorbidities = pd.read_csv(COMORBIDITY_DATA_PATH)
    comorbidities = change_comorbidities_icd9code(comorbidities)
    print("✅ ICD9 CODES SHORTENED - COMORBIDITIES")

    # Create target encoding
    test = pd.read_csv(TEST_DATA_PATH)
    # Encode ICD9 codes
    df, mapping = encoder_icd9_codes(df, comorbidities)

    print("✅ ICD9 CODE ENCODED")

    # Drop ID features
    df = df.drop(ID_FEATURES, axis = 1)
    print("✅ ID FEATURES DROPPED")

    X = df[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES]
    y = df[TARGET]

    ########################
    ### 2. TRANSFORMERS ###
    ########################
    # Encode feature "DIAGNOSOS"
    diagnosis_encoder = TargetEncoder(
        cols=['DIAGNOSIS'],
        smoothing=0.25,  # reduces overfitting
        handle_missing='value',
        handle_unknown='value'
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")),
               ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # COMBINE PREPROCESSORS
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, TRAIN_NUMERICAL_FEATURES),
        ("cat", categorical_transformer, TRAIN_CATEGORICAL_FEATURES)],
        remainder='drop')

    model = xgb.XGBClassifier(**XGB_PARAMS)

    model_pipeline = Pipeline(steps=[
        ("diagnosis_target_encoder", diagnosis_encoder),
        ('preprocessor', preprocessor),
        ('classifier', model)])
    print("✅ PIPELINE CREATED")


    ########################
    ### 3. TRAIN MODEL ###
    ########################
    model_pipeline.fit(X, y)
    print("✅ MODEL TRAINED")

    # Check model error
    y_hat_in = model_pipeline.predict(X)
    print("✅ MODEL RUNS")
    print(f'1. Model Accuracy:{accuracy_score(y, y_hat_in)} \n2. Model Recall:{recall_score(y, y_hat_in)} \n3. Model Precision: {precision_score(y, y_hat_in)}')

    ########################
    ### 3. SAVE MODEL ###
    ########################
    print("✅ SAVING MODEL")
    model_path = f"{MODEL_OUTPUT_PATH}/model_{MODEL_VERSION}.pkl"
    mapping_path = f"{MODEL_OUTPUT_PATH}/icd_mapping_{MODEL_VERSION}.pkl"
    preprocessor_path = f"{MODEL_OUTPUT_PATH}/preprocessor_{MODEL_VERSION}.pkl"

    joblib.dump(model_pipeline, model_path)
    joblib.dump(mapping, mapping_path)
    joblib.dump(preprocessor, preprocessor_path)
    print("✅ SAVED MODEL")

    return model_pipeline

if  __name__ == "__main__":
    train_model()