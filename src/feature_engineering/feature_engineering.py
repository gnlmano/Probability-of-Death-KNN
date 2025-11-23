import pandas as pd
import numpy as np
from src.config import (TARGET, ENG_QUART_DAY, ENG_AGE, ADMIT_TIME, DOB, SUBJECT_ID, HADM_ID, ICD9_CODE,
                        OOF_ENCODING_SPLITS, RANDOM_STATE, RELIGION_MISSING, MARITAL_MISSING, ETHNICITY_MISSING,
                        ENG_DEMOGRAPHIC,RELIGION, ETHNICITY, MARITAL_STATUS, TARGET, OOF_ENCODING_SPLITS,
                        RANDOM_STATE, SUBJECT_ID, HADM_ID, ICD9_CODE, MORTALITY_PROXY, COUNT_COMORBIDITIES, MEAN_MORTALITY,
                        MAX_MORTALITY)

# CREATE AGE, QUART OF DAY, AND CAP AGE >90
def create_basic_features(df):
    # CREATE AGE FEATURE -> AGE ABOVE 90 ARE CENSORED - CAST TO 90
    df[ENG_AGE] = pd.to_datetime(df[ADMIT_TIME]).dt.year - pd.to_datetime(df[DOB]).dt.year
    df[ENG_AGE] = df[ENG_AGE].apply(lambda x: 90 if x > 90 else x)
    # CREATE QUARTER OF DAY FOR ADMIT TIME
    df[ENG_QUART_DAY] = np.ceil(pd.to_datetime(df[ADMIT_TIME]).dt.hour / 6).astype("category")

    return df

# ENCODE DEMOGRAPHIC FEATURES
def encoder_demographics(df):
    """
    The function creates a new feature called "demographic_info_missing".
    This is to avoid using demographic features in predicting mortality.
    This avoids ethical issues in using such features to predict mortality.
    """

    df = df.copy()

    religion_missing = RELIGION_MISSING
    marital_missing = MARITAL_MISSING
    ethnicity_missing = ETHNICITY_MISSING

    df["religion_missing"] = df[RELIGION].astype(str).str.upper().isin(religion_missing) | df[RELIGION].isna()
    df["marital_missing"] = df[MARITAL_STATUS].astype(str).str.upper().isin(marital_missing) | df[MARITAL_STATUS].isna()
    df["ethnicity_missing"] = df[ETHNICITY].astype(str).str.upper().isin(ethnicity_missing) | df[ETHNICITY].isna()

    df[ENG_DEMOGRAPHIC] = (
        df[["religion_missing", "marital_missing", "ethnicity_missing"]]
        .any(axis=1)
        .astype(int)
    )
    df = df.drop(columns = ['religion_missing', 'marital_missing', 'ethnicity_missing'], axis = 1)

    return df


def encoder_icd9_codes(
    train_df,
    comorbidity_df,
    target_col = TARGET,
    subject_col= SUBJECT_ID,
    hadm_col= HADM_ID,
    icd_col= ICD9_CODE):

    comorbidity_df = comorbidity_df.copy()
    comorbidity_df = comorbidity_df.merge(
        train_df[[target_col, subject_col, hadm_col]],
        on=[subject_col, hadm_col],
        how="inner"
    )

    comorbidity_df[MORTALITY_PROXY] = comorbidity_df.groupby(icd_col)[target_col].transform("mean")
    icd9_mapping = comorbidity_df.groupby([subject_col, hadm_col]).agg(
        **{
            MAX_MORTALITY: (MORTALITY_PROXY, "max"),
            MEAN_MORTALITY: (MORTALITY_PROXY, "mean"),
            COUNT_COMORBIDITIES: (MORTALITY_PROXY, "count"),
        })

    train_df = train_df.merge(
        icd9_mapping,
        on=[subject_col, hadm_col],
        how="left")

    train_df[MAX_MORTALITY].fillna(0, inplace=True)
    train_df[MEAN_MORTALITY].fillna(0, inplace=True)
    train_df[COUNT_COMORBIDITIES].fillna(0, inplace=True)


    return train_df, icd9_mapping

def apply_icd9_mapping(df,
                       mapping,
                       subject_col= SUBJECT_ID,
                       hadm_col= HADM_ID
                       ):
    df = df.merge(mapping, on=[subject_col, hadm_col], how="left")

    df[MAX_MORTALITY].fillna(0, inplace=True)
    df[MEAN_MORTALITY].fillna(0, inplace=True)
    df[COUNT_COMORBIDITIES].fillna(0, inplace=True)

    return df