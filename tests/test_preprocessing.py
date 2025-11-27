import pandas as pd
import src.preprocessing.preprocessor as fe
from src.config import DROP_FEATURES, ICD9_CODE, ICD9_DIAGNOSIS
from src.preprocessing.preprocessor import (
    drop_features,
    change_feature_names,
    change_comorbidities_icd9code
)

def test_drop_features(monkeypatch):
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }

    )

    test_drop = ["a", "c"]
    monkeypatch.setattr(fe, "DROP_FEATURES", test_drop) #monkey patch (module_name, global_var_name, name_of_replcaement)

    result = drop_features(df.copy(), test_drop)

    assert "a" not in result.columns, "drop_features: col a exists"
    assert "c" not in result.columns, "drop_features: col c exists"
    assert "b" in result.columns, "drop_features: col b does not exist"


def test_change_feature_names(monkeypatch):

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": ["3233rs", 409221,  "TEEV$%", None]
        }
    )

    test_col = "a"
    monkeypatch.setattr(fe, "ICD9_DIAGNOSIS", test_col)

    result = change_feature_names(df.copy())

    assert result.loc[result['id'] == 1, "a"].iloc[0] == "323", "change_features: patient 1 wrong ICD9"
    assert result.loc[result['id'] == 2, "a"].iloc[0] == "409", "change_features: patient 2 wrong ICD9"
    assert result.loc[result['id'] == 3, "a"].iloc[0] == "TEE", "change_features: patient 3 wrong ICD9"
    assert pd.isna(result.loc[result['id'] == 4, "a"].iloc[0]), "change_features: patient 4 wrong ICD9"


def test_change_comorbidities_icd9code(monkeypatch):
    df = pd.DataFrame(
        {
            "FEATURE1": [1,2,3,4,5],
            "FEAT_2": [2,3,4,5,6],
            "ICD_FEATURE": ["44323", "54345", "grgrg", "AB", None]

        }
    )
    test_col = "icd_feature"
    monkeypatch.setattr(fe, "ICD9_CODE", test_col)

    result = change_comorbidities_icd9code(df.copy())

    assert result.loc[result['feature1'] == 1, test_col].iloc[0] == "443", "change_comorbidities_icd9code: patient 1 wrong ICD9"
    assert result.loc[result['feature1'] == 2, test_col].iloc[0] == "543", "change_comorbidities_icd9code: patient 2 wrong ICD9"
    assert result.loc[result['feature1'] == 3, test_col].iloc[0] == "grg", "change_comorbidities_icd9code: patient 3 wrong ICD9"
    assert result.loc[result['feature1'] == 4, test_col].iloc[0] == "AB", "change_comorbidities_icd9code: patient 4 wrong ICD9"

    assert not (result['feature1'] == 5).any(),"change_comorbidities_icd9code: patient 5 should have been dropped"
    assert result[test_col].isna().sum() == 0, "change_comorbidities_icd9code: missing ICD codes were not removed"