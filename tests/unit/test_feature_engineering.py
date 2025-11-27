import pandas as pd
from probability_of_death.config import (
    ENG_QUART_DAY, ENG_AGE, ADMIT_TIME, DOB,
    RELIGION_MISSING, ETHNICITY_MISSING,
    ENG_DEMOGRAPHIC, RELIGION, ETHNICITY, MARITAL_STATUS, TARGET,
    SUBJECT_ID, HADM_ID, ICD9_CODE, COUNT_COMORBIDITIES, MEAN_MORTALITY,
    MAX_MORTALITY
)

from probability_of_death.feature_engineering.feature_engineering import (
    create_basic_features,
    encoder_demographics,
    encoder_icd9_codes,
    apply_icd9_mapping
)


# TEST CREATING NEW COLUMNS

def test_create_basic_features():

    df = pd.DataFrame(
        {
            ADMIT_TIME: [
                "2020-01-01 02:00:00",  # hour 2 should return Q1
                "2020-01-01 13:00:00",  # hour 13 should return Q4
            ],
            DOB: [
                "1930-01-01",  # 2020 - 1930 = 90
                "1920-01-01",  # 2020 - 1920 = 100 should cap to 90
            ]
        }
    )

    test_result = create_basic_features(df.copy())

    # columns created
    assert ENG_AGE in test_result.columns, "create_basic_features: age failed"
    assert ENG_QUART_DAY in test_result.columns, "create_basic_features: quarter of day failed"

    # correct values
    assert test_result.loc[0, ENG_AGE] == 90, "create_basic_features: age calculation failed"  # exactly 90
    assert test_result.loc[1, ENG_AGE] == 90, "create_basic_features: age clipping failed"  # capped to 90
    assert test_result.loc[0, ENG_QUART_DAY] == 1, "create_basic_features: quarter of day wrong 1"  # Q1
    assert test_result.loc[1, ENG_QUART_DAY] == 3, "create_basic_features: quarter of day wrong 2"  # Q3

    # correct dtype
    assert str(test_result[ENG_QUART_DAY].dtype) == "category", "create_basic_features: wrong dtype"

# TEST DEMOGRAPHICS ENCODER

def test_encoder_demographics():
    df = pd.DataFrame({
        RELIGION: ["CHRISTIAN", RELIGION_MISSING[0], None],
        MARITAL_STATUS: ["MARRIED", None, "SINGLE"],
        ETHNICITY: ["ASIAN", "WHITE", ETHNICITY_MISSING[0]]
    })

    # Expected return values
    # patient 1 - 0
    # patient 2 - 1
    # patient 3 - 1
    expected_return_values = [0,1,1]
    result = encoder_demographics(df.copy())

    # No columns dropped
    assert RELIGION in result.columns, "encoder_demographics: religion dropped"
    assert MARITAL_STATUS in result.columns, "encoder_demographics: marital status dropped"
    assert ETHNICITY in result.columns, "encoder_demographics: ethnicity dropped"

    # No temp columns reamin
    assert "religion_missing" not in result.columns, "encoder_demographics: religion temp remains"
    assert "marital_status" not in result.columns, "encoder_demographics: marital status temp remains"
    assert "ethnicity_missing" not in result.columns, "encoder_demographics: ethnicity temp remains"

    # Function returns expected values and dtype
    assert result[ENG_DEMOGRAPHIC].to_list() ==  expected_return_values, "encoder_demographics: wrong expected values"
    assert result[ENG_DEMOGRAPHIC].dtype == int, "encoder_demographics: wrong dtype"


def test_encoder_icd9_codes():
    train_df = pd.DataFrame({
        SUBJECT_ID: [1, 2],
        HADM_ID: [10, 20],
        TARGET: [1, 0],
    })

    comorbidity_df = pd.DataFrame({
        SUBJECT_ID: [1, 1],
        HADM_ID: [10, 10],
        ICD9_CODE: ["250.00", "401.9"]
    })

    # Expected return values
    # ICD 250.00 -> mortality mean = 1
    # ICD 401.9  -> mortality mean = 1

    # patient 1
    # MAX = 1
    # MEAN = 1
    # COUNT = 2

    # patient 2
    # MAX = 0
    # MEAN = 0
    # COUNT = 0

    result, icd9_map = encoder_icd9_codes(train_df.copy(), comorbidity_df.copy())

    # function returns expected values icd9 mappiong
    merged_proxy = icd9_map.reset_index()
    assert merged_proxy[MAX_MORTALITY].iloc[0] == 1.0, "encoder_icd9_codes: max mortality mapping"
    assert merged_proxy[MEAN_MORTALITY].iloc[0] == 1.0, "encoder_icd9_codes: mean mortality mapping"
    assert merged_proxy[COUNT_COMORBIDITIES].iloc[0] == 2, "encoder_icd9_codes: count comorbidities mapping"

    # mapping should only have 1 row:
    assert len(icd9_map) == 1, "encoder_icd9_codes: icd9 mapping correct length"

    # new features created
    assert MAX_MORTALITY in result.columns, "encoder_icd9_codes: max mortality created"
    assert MEAN_MORTALITY in result.columns, "encoder_icd9_codes: mean mortality created"
    assert COUNT_COMORBIDITIES in result.columns, "encoder_icd9_codes: count comorbidities created"

    # correct expected values
    row1 = result[result[HADM_ID] == 10].iloc[0]
    assert row1[MAX_MORTALITY] == 1, "encoder_icd9_codes: wrong max mortality patient 1"
    assert row1[MEAN_MORTALITY] == 1, "encoder_icd9_codes: wrong mean mortality patient 1"
    assert row1[COUNT_COMORBIDITIES] == 2, "encoder_icd9_codes: wrong count comorbidities patient 1"

    row2 = result[result[HADM_ID] == 20].iloc[0]
    assert row2[MAX_MORTALITY] == 0, "encoder_icd9_codes: wrong max mortality patient 2"
    assert row2[MEAN_MORTALITY] == 0, "encoder_icd9_codes: wrong mean mortality patient 2"
    assert row2[COUNT_COMORBIDITIES] == 0, "encoder_icd9_codes: wrong count comorbidities patient 2"



def test_apply_icd9_mapping():
    df = pd.DataFrame(
        {
            SUBJECT_ID: [1, 2, 3],
            HADM_ID: [10, 20, 30],
        }
    )
    mapping = pd.DataFrame({
        SUBJECT_ID: [1, 2],
        HADM_ID: [10, 20],
        MAX_MORTALITY: [0.91, 0.22],
        MEAN_MORTALITY: [0.77, 0.34],
        COUNT_COMORBIDITIES: [3, 1],
    })

    # Expected return values
    # Admission 10 -> 0.91, 0.77, 3
    # Admission 20 -> 0.22, 0.34, 1
    # Admission 30 -> 0, 0, 0

    result = apply_icd9_mapping(df.copy(), mapping)

    for col in [MAX_MORTALITY, MEAN_MORTALITY, COUNT_COMORBIDITIES]:
        assert col in result.columns, "apply_icd9_mapping: column doesn't exist"

    row10 = result[result[HADM_ID] == 10].iloc[0]
    assert row10[MAX_MORTALITY] == 0.91, "apply_icd9_mapping: pateint 10- wrong max mortality"
    assert row10[MEAN_MORTALITY] == 0.77, "apply_icd9_mapping: pateint 10- wrong mean mortality"
    assert row10[COUNT_COMORBIDITIES] == 3, "apply_icd9_mapping: pateint 10- wrong count comorbidities"

    row20 = result[result[HADM_ID] == 20].iloc[0]
    assert row20[MAX_MORTALITY] == 0.22, "apply_icd9_mapping: pateint 20- wrong max mortality"
    assert row20[MEAN_MORTALITY] == 0.34, "apply_icd9_mapping: pateint 20- wrong mean mortality"
    assert row20[COUNT_COMORBIDITIES] == 1, "apply_icd9_mapping: pateint 20- wrong count comorbidities"

    row30 = result[result[HADM_ID] == 30].iloc[0]
    assert row30[MAX_MORTALITY] == 0, "apply_icd9_mapping: pateint 30- wrong max mortality"
    assert row30[MEAN_MORTALITY] == 0, "apply_icd9_mapping: pateint 30- wrong mean mortality"
    assert row30[COUNT_COMORBIDITIES] == 0, "apply_icd9_mapping: pateint 30- wrong count comorbidities"




