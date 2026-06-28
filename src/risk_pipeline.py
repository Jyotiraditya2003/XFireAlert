import os
import pandas as pd
import joblib
import shap

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "risk_pipeline.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# --- LOAD GLOBALLY ---
pipeline = joblib.load(PIPELINE_PATH)
FEATURES = joblib.load(FEATURE_PATH)

def predict_risk(weather_dict):
    """Predicts the final fire probability score."""
    df = pd.DataFrame([weather_dict])
    # Enforce exact column order from training
    df = df[FEATURES]
    
    probability = pipeline.predict_proba(df)[0][1]
    return float(probability)

def explain_risk(weather_dict):
    """Generates dynamic SHAP values for the given weather input."""
    df = pd.DataFrame([weather_dict])
    df = df[FEATURES]

    # 1. Manually separate the scaler and the model from the pipeline
    scaler = pipeline.named_steps["scaler"]
    xgb_model = pipeline.named_steps["model"]

    # 2. Transform the raw UI data through the scaler
    X_scaled = scaler.transform(df)

    # 3. Create the Explainer directly on the XGBoost model
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)

    # 4. Safely extract the 1D array of feature impacts
    if isinstance(shap_values, list):
        # Legacy SHAP format [class_0_array, class_1_array]
        vals = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        # 3D SHAP format (n_samples, n_features, n_classes)
        vals = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2:
        # Standard format (n_samples, n_features)
        vals = shap_values[0]
    else:
        vals = shap_values

    # 5. Map the values back to their feature names
    contributions = {feat: float(val) for feat, val in zip(FEATURES, vals)}
    
    # Sort from highest absolute impact to lowest
    contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    return contributions