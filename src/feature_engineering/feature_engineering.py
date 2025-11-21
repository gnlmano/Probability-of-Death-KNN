import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.config import (TARGET, ENG_QUART_DAY, ENG_AGE, ADMIT_TIME, DOB, SUBJECT_ID, HADM_ID, ICD9_CODE,
                        OOF_ENCODING_SPLITS, RANDOM_STATE, RELIGION_MISSING, MARITAL_MISSING, ETHNICITY_MISSING,
                        ENG_DEMOGRAPHIC,RELIGION, ETHNICITY, MARITAL_STATUS, TARGET, OOF_ENCODING_SPLITS,
                        RANDOM_STATE, SUBJECT_ID, HADM_ID, ICD9_CODE)

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

# TARGET ENCODER FOR ICD9 CODES
def encoder_icd9_codes(
    train_df,
    test_df,
    comorbidity_df,
    target_col = TARGET,
    subject_col= SUBJECT_ID,
    hadm_col= HADM_ID,
    icd_col= ICD9_CODE,
    n_splits = OOF_ENCODING_SPLITS,
    random_state = RANDOM_STATE
):
    """
    Computes out-of-fold ICD9 mortality-based features for training data, and a global-encoded dataset for comorbidities proxy

    Returns:
        train_features: df with OOF max_mortality, mean_mortality, count_comorbidities
        test_features:  df with global-encoded max_mortality, mean_mortality, count_comorbidities
        icd_mortality_global:  mapping icd9_code to mortality rate (for deployment)
    """

    # Get dataframes
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    y = train[target_col].values  # Needed for folds

    # Empty arrays for OOF features
    oof_max = np.zeros(len(train))
    oof_mean = np.zeros(len(train))
    oof_count = np.zeros(len(train))


    # K-fold loop

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(train):

        fold_train = train.loc[train_idx, [subject_col, hadm_col, target_col]]
        fold_val = train.loc[val_idx, [subject_col, hadm_col]]
        # Merge comorbidities with fold-train labels only
        fold_com = comorbidity_df.merge(
            fold_train,
            on=[subject_col, hadm_col],
            how="inner"
        )
        # Compute ICD9 mortality from fold-train only
        icd_mortality_fold = fold_com.groupby(icd_col)[target_col].mean()

        # Compute per-patient mortality features for validation fold
        fold_val_com = comorbidity_df.merge(
            fold_val,
            on=[subject_col, hadm_col],
            how="inner"
        )

        # Map ICD9 → mortality proxies
        fold_val_com["mortality_proxy"] = fold_val_com[icd_col].map(icd_mortality_fold)

        # Aggregate per patient
        agg = fold_val_com.groupby([subject_col, hadm_col]).agg(
            max_mortality=("mortality_proxy", "max"),
            mean_mortality=("mortality_proxy", "mean"),
            count_comorbidities=(icd_col, "count")
        ).reset_index()

        # Merge onto validation rows
        merged = fold_val.merge(agg, on=[subject_col, hadm_col], how="left")

        # Store results
        oof_max[val_idx] = merged["max_mortality"].fillna(0)
        oof_mean[val_idx] = merged["mean_mortality"].fillna(0)
        oof_count[val_idx] = merged["count_comorbidities"].fillna(0)

    # Enocder - global data for test set.
    # Use full train labels
    com_full = comorbidity_df.merge(
        train[[subject_col, hadm_col, target_col]],
        on=[subject_col, hadm_col],
        how="inner"
    )
    icd_mortality_global = com_full.groupby(icd_col)[target_col].mean()

    # Apply to test set
    test_com = comorbidity_df.merge(
        test[[subject_col, hadm_col]],
        on=[subject_col, hadm_col],
        how="inner"
    )
    test_com["mortality_proxy"] = test_com[icd_col].map(icd_mortality_global)

    test_agg = test_com.groupby([subject_col, hadm_col]).agg(
        max_mortality=("mortality_proxy", "max"),
        mean_mortality=("mortality_proxy", "mean"),
        count_comorbidities=(icd_col, "count")
    ).reset_index()

    test_features = test.merge(test_agg, on=[subject_col, hadm_col], how="left")
    test_features[["max_mortality", "mean_mortality"]] = \
        test_features[["max_mortality", "mean_mortality"]].fillna(0)
    test_features["count_comorbidities"] = test_features["count_comorbidities"].fillna(0)

    # Build train fetures ouptut
    train_features = train.copy()
    train_features["max_mortality"] = oof_max
    train_features["mean_mortality"] = oof_mean
    train_features["count_comorbidities"] = oof_count

    return {"train_df" : train_features,
            "test_df" : test_features,
            "comorbidities_df" : icd_mortality_global}



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

    comorbidity_df['mortality_proxy'] = comorbidity_df.groupby(icd_col)[target_col].transform("mean")
    icd9_mapping = comorbidity_df.groupby([subject_col, hadm_col]).agg(
        max_mortality = ("mortality_proxy", "max"),
        mean_mortality = ("mortality_proxy", "mean"),
        count_comorbidities = ("mortality_proxy", "count"))

    train_df = train_df.merge(
        icd9_mapping,
        on=[subject_col, hadm_col],
        how="inner")

    return train_df, icd9_mapping