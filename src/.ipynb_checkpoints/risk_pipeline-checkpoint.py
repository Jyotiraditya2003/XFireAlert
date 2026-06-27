import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

# Loading model and scaler

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Features
FEATURES = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI"
]



# Fire Risk Prediction

def predict_risk(weather_dict):
    """
    weather_dict example:
    {
        "Temperature": 35,
        "RH": 22,
        "Ws": 15,
        "Rain": 0.0,
        "FFMC": 88.5,
        "DMC": 26.7,
        "DC": 94.3,
        "ISI": 5.2,
        "BUI": 28.6
    }
    """

    # DataFrame
    df = pd.DataFrame([weather_dict], columns=FEATURES)

    # Scale
    df_scaled = scaler.transform(df)

    # Probability
    prob = model.predict_proba(df_scaled)[0][1]

    return float(prob)

# SHAP Explainability

explainer = shap.TreeExplainer(model)

def explain_risk(weather_dict):

    df = pd.DataFrame([weather_dict], columns=FEATURES)

    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=FEATURES)

    shap_values = explainer.shap_values(df_scaled)

    values = shap_values[1][0]

    contributions = dict(zip(FEATURES, values))

    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return contributions