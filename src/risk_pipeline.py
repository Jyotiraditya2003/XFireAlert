import os
import pandas as pd
import joblib
import shap

# Ensuring environment compatibility using os.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "risk_pipeline.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

pipeline = joblib.load(PIPELINE_PATH)
FEATURES = joblib.load(FEATURE_PATH)

def predict_risk(weather_dict):
    df = pd.DataFrame([weather_dict])
    df = df[FEATURES]
    probability = pipeline.predict_proba(df)[0][1]
    return float(probability)

def explain_risk(weather_dict):
    df = pd.DataFrame([weather_dict])
    df = df[FEATURES]

    # 1. Extract the final model (XGBoost is always the last step)
    xgb_model = pipeline.steps[-1][1]

    # 2. THE BULLETPROOF TRANSFORMATION
    # This automatically applies all scalers/imputers, regardless of what they are named
    if len(pipeline.steps) > 1:
        X_transformed = pipeline[:-1].transform(df)
    else:
        X_transformed = df

    # 3. Calculate SHAP on the correctly transformed data
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    # 4. Safely extract the values for this specific prediction
    if isinstance(shap_values, list):
        # Multi-class format
        values = shap_values[1][0] 
    else:
        # Binary or Regression format
        if len(shap_values.shape) == 3: 
             values = shap_values[0, :, 1]
        else:
             values = shap_values[0]

    # 5. Map back to dictionary
    contributions = {f: float(v) for f, v in zip(FEATURES, values)}
    
    # Sort by absolute magnitude so the UI shows the most impactful items first
    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return contributions