RAW_DATA_PATH =  "data/mimic_train.csv"
TEST_DATA_PATH = "data/mimic_test_death.csv"
COMORBIDITY_DATA_PATH = "data/extra_data/MIMIC_diagnoses.csv"
DROP_FEATURES = [ 'DOD', 'DISCHTIME', 'DEATHTIME', 'LOS', 'ADMITTIME', 'DOB', 'INSURANCE', 'GENDER', 'MARITAL_STATUS', 'ETHNICITY', 'RELIGION']
ID_FEATURES = ['subject_id', 'hadm_id', 'icustay_id']
TARGET = 'HOSPITAL_EXPIRE_FLAG'
RAW_NUMERICAL_COLS = [
    'HeartRate_Min',
 'HeartRate_Max',
 'HeartRate_Mean',
 'SysBP_Min',
 'SysBP_Max',
 'SysBP_Mean',
 'DiasBP_Min',
 'DiasBP_Max',
 'DiasBP_Mean',
 'MeanBP_Min',
 'MeanBP_Max',
 'MeanBP_Mean',
 'RespRate_Min',
 'RespRate_Max',
 'RespRate_Mean',
 'TempC_Min',
 'TempC_Max',
 'TempC_Mean',
 'SpO2_Min',
 'SpO2_Max',
 'SpO2_Mean',
 'Glucose_Min',
 'Glucose_Max',
 'Glucose_Mean',
 'Diff']
RAW_CATEGORICAL_COLS = [
 'ADMISSION_TYPE',
 'RELIGION',
 'MARITAL_STATUS',
 'ETHNICITY',
 'DIAGNOSIS',
 'ICD9_diagnosis',
 'FIRST_CAREUNIT']
ADMIT_TIME = 'ADMITTIME'
DOB = 'DOB'

SUBJECT_ID = 'subject_id'
HADM_ID = 'hadm_id'
ICD9_CODE = 'icd9_code'

RELIGION =  'RELIGION'
ETHNICITY = 'ETHNICITY'
MARITAL_STATUS = 'MARITAL_STATUS'

ICD9_DIAGNOSIS_COLS = ['ICD9_diagnosis', 'DIAGNOSIS']

ENGINEERED_MORTALITY_PROXIES = ['max_mortality', 'mean_mortality', 'count_comorbidities']
ENGINEERED_NUMERICAL_FEATURES = ['AGE_AT_ADMIT']
ENGINEERED_CATEGORICAL_FEATURES = ['QUART_DAY']

# DEFINE ENGINEERED FEATURES
ENG_AGE = 'AGE_AT_ADMIT'
ENG_QUART_DAY = 'QUART_DAY'
ENG_DEMOGRAPHIC = 'demographic_info_missing'

OOF_ENCODING_SPLITS = 5
RANDOM_STATE = 42


RELIGION_MISSING = [
    "NOT SPECIFIED",
    "UNOBTAINABLE"
]

MARITAL_MISSING = [
    "UNKNOWN (DEFAULT)"
]

ETHNICITY_MISSING = [
    "UNABLE TO OBTAIN",
    "UNKNOWN/NOT SPECIFIED"
]

# FEATURES FOR TRAINING
TRAIN_NUMERICAL_FEATURES = RAW_NUMERICAL_COLS + ENGINEERED_MORTALITY_PROXIES + ENGINEERED_NUMERICAL_FEATURES
print(TRAIN_NUMERICAL_FEATURES)
TRAIN_CATEGORICAL_FEATURES = ['ADMISSION_TYPE', 'FIRST_CAREUNIT', 'DIAGNOSIS'] + ENGINEERED_CATEGORICAL_FEATURES

XGB_PARAMS = {"n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,}

MODEL_VERSION = "1.0"
MODEL_OUTPUT_PATH = 'models'
ICD_MAPPING_PATH = f"models/icd_mapping_{MODEL_VERSION}.pkl"
MODEL_PATH = f"models/model_{MODEL_VERSION}.pkl"