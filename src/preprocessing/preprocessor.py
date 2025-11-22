import numpy as np
import pandas as pd
from src.config import DROP_FEATURES, ICD9_CODE


# DROP FIXED COLUMNS
def drop_features(df, columns):
    df = df.copy()
    df = df.drop(columns=DROP_FEATURES)

    return df

# CHANGE COLUMN NAMES, EXTRACT FIRST THREE CHARS IN ICD9CODES
def change_feature_names(df):
    df = df.copy()
    df['ICD9_diagnosis'] = df['ICD9_diagnosis'].str[:3].astype("str")
    return df

# FUNCTION TO CONVERT ICD9 CODES IN THE COMORBIDITIES DATA
def change_comorbidities_icd9code(df):
    df = df.copy()
    df = df.rename(columns = str.lower)
    df[ICD9_CODE] = df[ICD9_CODE].str[:3]
    df = df.dropna()

    return df
