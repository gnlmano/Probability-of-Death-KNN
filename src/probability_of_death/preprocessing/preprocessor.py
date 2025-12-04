from probability_of_death.config import DROP_FEATURES, ICD9_CODE, ICD9_DIAGNOSIS
import logging
logger = logging.getLogger(__name__)

# DROP FIXED COLUMNS
def drop_features(df, columns):
    df = df.copy()
    df = df.drop(columns=DROP_FEATURES)
    logger.info("Features dropped")

    return df

# CHANGE COLUMN NAMES, EXTRACT FIRST THREE CHARS IN ICD9CODES
def change_feature_names(df):
    df = df.copy()
    df[ICD9_DIAGNOSIS] = df[ICD9_DIAGNOSIS].astype("string").str[:3]
    logger.info("ICD9 codes shortened in train")
    return df

# FUNCTION TO CONVERT ICD9 CODES IN THE COMORBIDITIES DATA
def change_comorbidities_icd9code(df):
    df = df.copy()
    df = df.rename(columns = str.lower)
    df = df.dropna(subset=[ICD9_CODE])
    df[ICD9_CODE] = df[ICD9_CODE].astype("string").str[:3]
    logger.info("ICD9 codes shortened in comorbidities")

    return df

