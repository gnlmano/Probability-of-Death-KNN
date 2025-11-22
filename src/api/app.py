import pandas as pd
import joblib
import random
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.config import(
TEST_DATA_PATH,
MODEL_PATH,
ICD_MAPPING_PATH,
TRAIN_NUMERICAL_FEATURES,
TRAIN_CATEGORICAL_FEATURES
)
from src.feature_engineering.feature_engineering import (
create_basic_features,
encoder_demographics,
apply_icd9_mapping
)
from src.preprocessing.preprocessor import (
change_feature_names
)

# Load test data at start
test = pd.read_csv(TEST_DATA_PATH)

app = FastAPI(
    title = "ICU Mortality Prediction API",
    description = "Predicts mortality risk for in-hospital triage use",
    version = "1.0"
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>ICU Mortality Prediction API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f7f7f7;
                    padding: 40px;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
                }
                a {
                    display: block;
                    margin: 10px 0;
                    font-size: 18px;
                    color: #007acc;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to ICU Mortality Prediction API</h1>
                <p>Available Endpoints:</p>

                <a href="/health">/health – API Health Status</a>
                <a href="/sampling">/sampling – Raw JSON sample + prediction</a>
                <a href="/sampling/html">/sampling/html – Visual sample + prediction</a>
            </div>
        </body>
    </html>
    """
@app.get("/health")
def health_check():
    return {"status": "ok"}


def predict_api(df):
    """
    Applies pre-processing, similar to the predict model steps
    """
    df = df.copy()

    # Feature engineering
    df = create_basic_features(df)
    df = encoder_demographics(df)
    df = change_feature_names(df)

    # Merge with ICD9 mappings
    mapping = joblib.load(ICD_MAPPING_PATH)
    df = apply_icd9_mapping(df, mapping)

    # Prepare final df to predict on
    df = df[TRAIN_NUMERICAL_FEATURES + TRAIN_CATEGORICAL_FEATURES]

    return df

@app.get("/sampling")
def get_random_sample_and_predict():
    """
    Randomly samples a test-set observation and returns predicted mortality risk.
    """

    idx = random.randint(0, len(test)- 1)
    sample = test.iloc[[idx]]

    X = predict_api(sample)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Get prediction
    prob = float(model.predict_proba(X)[:, 1][0])

    input_raw = sample.to_dict(orient="records")

    return {
        "sample_index": idx,
        "input_features": input_raw,
        "predicted_mortality_risk": prob
    }

# Simple UI interface to show data and predciton
def dataframe_to_vertical_html(df):
    rows = ""
    for col, val in df.iloc[0].items():
        rows += f"""
        <tr>
            <th>{col}</th>
            <td>{val}</td>
        </tr>
        """
    return f"""
    <table class="vtable">
        <tbody>
            {rows}
        </tbody>
    </table>
    """

@app.get("/sampling/html", response_class=HTMLResponse)
def get_sample_html():
    idx = random.randint(0, len(test) - 1)
    sample = test.iloc[[idx]]

    X = predict_api(sample)
    model = joblib.load(MODEL_PATH)
    prob = float(model.predict_proba(X)[:, 1][0])

    # Convert sample row to vertical table
    table_html = dataframe_to_vertical_html(sample)

    return f"""
    <html>
    <head>
        <title>Random ICU Prediction</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f7f7f7;
                padding: 40px;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                max-width: 800px;
                margin: auto;
                box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #222;
            }}
            .vtable {{
                width: 100%;
                border-collapse: collapse;
                font-size: 15px;
            }}
            .vtable th {{
                text-align: left;
                padding: 8px;
                background: #efefef;
                width: 40%;
                border: 1px solid #ddd;
            }}
            .vtable td {{
                padding: 8px;
                border: 1px solid #ddd;
                background: white;
            }}
            .pred-box {{
                margin-top: 20px;
                background: #fff;
                padding: 15px;
                font-size: 32px;
                font-weight: bold;
                color: #007acc;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Random ICU Sample Prediction</h1>
        <p style="color:#666; font-size:15px; margin-top:-10px;">
            Demographic features are <strong>excluded</strong> from the prediction model to avoid ethical bias.
        </p>

        <p><strong>Row:</strong> {idx}</p>

            <h2>Patient Features</h2>
            {table_html}

            <h2>Predicted Mortality Risk</h2>
            <div class="pred-box">{prob:.3f}</div>
        </div>
    </body>
    </html>
    """


