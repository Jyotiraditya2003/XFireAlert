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
    # Ensure a completely fresh data frame is built directly from the UI input
    df = pd.DataFrame([weather_dict])
    df = df[FEATURES]

    # 1. Extract the final XGBoost model
    xgb_model = pipeline.steps[-1][1]

    # 2. Complete the step-wise transformations cleanly
    if len(pipeline.steps) > 1:
        X_transformed = pipeline[:-1].transform(df)
    else:
        X_transformed = df

    # 3. Native XGBoost SHAP calculation bypasses library wrapper discrepancies
    import xgboost as xgb
    
    # Check if the transformed input is a DataFrame or a Numpy Array
    if isinstance(X_transformed, pd.DataFrame):
        dmatrix = xgb.DMatrix(X_transformed)
    else:
        # If transformers dropped column names, re-assign them to map shapes correctly
        dmatrix = xgb.DMatrix(pd.DataFrame(X_transformed, columns=FEATURES))
        
    # Native tree booster yields an exact instance contribution array: [features + 1 bias value]
    booster = xgb_model.get_booster()
    native_shap = booster.predict(dmatrix, pred_contribs=True)[0]
    
    # Extract feature values, leaving out the final array bias term
    values = native_shap[:-1]

    # 4. Explicit dictionary mapping
    contributions = {f: float(v) for f, v in zip(FEATURES, values)}
    
    # Sort by absolute magnitude so the UI highlights shifting values
    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return contributions