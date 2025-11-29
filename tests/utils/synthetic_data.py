import pandas as pd
import numpy as np

from probability_of_death.config import (
C_SUBJECT_ID,
C_HADM_ID,
C_SEQ_NUM ,
C_ICD9_CODE,
RAW_NUMERICAL_COLS_USED,
RAW_CATEGORICAL_COLS_USED
)

def random_dates(start, end, n):
    np.random.seed(4201)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    return start + pd.to_timedelta(
        np.random.randint(0, (end - start).days, n), unit="D"
    )

def make_synthetic_patient_data(n=10):
    base_dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rand_seconds = np.random.randint(0, 24*60*60, size=n)
    admit_times = base_dates + pd.to_timedelta(rand_seconds, unit="s")

    df = pd.DataFrame({
        'DOD': np.arange(n),
        'DISCHTIME': np.arange(n),
        'DEATHTIME': np.arange(n),
        'LOS': np.arange(n),
        "subject_id": np.arange(n),
        "hadm_id": np.arange(100, 100 + n),
        "ADMITTIME": admit_times,
        "RELIGION": ["NONE", "CATHOLIC", None, "JEWISH", "HINDU", "NOT SPECIFIED", "CATHOLIC", None, "UNOBTAINABLE", "HINDU"],
        "MARITAL_STATUS": ["UNKNOWN (DEFAULT)", None, "MARRIED", "DIVORCED", "WIDOWED", "LAF", None, "UNOBTAINABLE", "DIVORCED", "UNKNOWN (DEFAULT)"],
        "ETHNICITY": ["UNABLE TO OBTAIN", "WHITE", None, "BLACK", "HISPANIC", "UNABLE TO OBTAIN", "UNKNOWN/NOT SPECIFIED", None, "UNKNOWN/NOT SPECIFIED", "HISPANIC"],
        "INSURANCE": ["YES", "NO", "YES", "poe", "YES", "NO", "YES", "poe", "YES", "NO"],
        "GENDER": ["Male", None, "Female", "Declined", "None", "Male", None, "Female", None, "Declined)"],
        "ICD9_diagnosis": ["44323", "54345", None, "AB", "004", 2234, "54345", None, 4566, 23],
        "HOSPITAL_EXPIRE_FLAG": [1,0,0,0,1,1,0,0,1,0],
        "icustay_id": np.random.randint(10000,99999,size = n)
    })
    df["DOB"] = random_dates("1920-01-01", "2000-12-31", 10)

    # For the other features, I use random values
    for feature in RAW_NUMERICAL_COLS_USED:
        if feature not in df.columns:
            df[feature] = np.random.randn(n)
    for feature in RAW_CATEGORICAL_COLS_USED:
        if feature not in df.columns:
            df[feature] = np.random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])

    return df

def generate_fake_icd_codes(n):
    base_codes = ["44323", "54345", "25000", "41401", "V3000", "AB123"]
    return np.random.choice(base_codes, size=n)

def make_synthetic_comorbidities_data(n_patients=10, codes_per_patient=3):
    total = n_patients * codes_per_patient

    subject_ids = np.repeat(np.arange(n_patients), codes_per_patient)
    hadm_ids = np.repeat(np.arange(100, 100 + n_patients), codes_per_patient)

    df = pd.DataFrame({
        C_SUBJECT_ID: subject_ids,
        C_HADM_ID: hadm_ids,
        C_SEQ_NUM: np.tile(np.arange(codes_per_patient), n_patients),
        C_ICD9_CODE: generate_fake_icd_codes(total)
    })

    return df