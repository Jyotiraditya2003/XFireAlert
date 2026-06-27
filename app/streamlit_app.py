import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

FEATURE_TRANSLATIONS = {
    "Temperature": "Current Temperature Profile",
    "RH": "Relative Humidity Levels",
    "Ws": "Wind Speed Conditions",
    "Rain": "Precipitation/Rainfall",
    "FFMC": "Surface Vegetation Dryness (FFMC)",
    "DMC": "Shallow Soil Moisture (DMC)",
    "DC": "Deep Drought Index (DC)",
    "ISI": "Wind Spread Potential (ISI)",
    "BUI": "Heavy Fuel Combustibility (BUI)"
}

# allow src imports
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_DIR / "src"
sys.path.insert(0, str(SRC_PATH))

import streamlit as st
from PIL import Image

from risk_pipeline import predict_risk, explain_risk
from image_pipeline import predict_image, generate_gradcam, classify_fire
from vgg_pipeline import predict_vgg
from vgg_gradcam import generate_vgg_gradcam
from yolo_pipeline import detect_fire
from fusion_engine import fuse_predictions, get_alert_level
from database import log_alert
from model_router import choose_model

st.set_page_config(page_title="XFireAlert", layout="wide")

st.title("XFireAlert — Hybrid Forest Fire Prediction System")
tab1, tab2 = st.tabs(["Fire Monitor", "Know Your AI"])
with tab1:
    use_weather = st.sidebar.checkbox("Use Weather Sensor Data", value=True)
    # WEATHER INPUTS
    p_risk = None
    explanation = None

    if use_weather:

        st.sidebar.header("🌡 Weather Conditions")

        temp = st.sidebar.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, value=35.0, step=0.1)
        rh = st.sidebar.number_input("Relative Humidity (%)", min_value=5.0, max_value=100.0, value=30.0, step=0.1)
        ws = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        rain = st.sidebar.number_input("Rain (mm)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        ffmc = st.sidebar.number_input("FFMC", min_value=50.0, max_value=100.0, value=85.0, step=0.1)
        dmc = st.sidebar.number_input("DMC", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        dc = st.sidebar.number_input("DC", min_value=0.0, max_value=600.0, value=100.0, step=1.0)
        isi = st.sidebar.number_input("ISI", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        bui = st.sidebar.number_input("BUI", min_value=0.0, max_value=100.0, value=30.0, step=0.1)

        weather = {
            "Temperature": temp,
            "RH": rh,
            "Ws": ws,
            "Rain": rain,
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui
        }

    # MODEL SELECTOR
    st.sidebar.header("Vision Model Control")

    model_choice = st.sidebar.selectbox(
        "Select Fire Detection Model",
        [
            "AUTO (Recommended)",
            "AgniDrishti Lite",
            "AgniDrishti Pro",
            "AgniNetra Sentinel"
        ]
    )

    uploaded_file = st.file_uploader("Upload a forest image", type=["jpg","jpeg","png"])
    predict_button = st.sidebar.button("Run Prediction")

    # RUN PREDICTION
    if predict_button:

        # WEATHER
        if use_weather:
            p_risk = predict_risk(weather)
            explanation = explain_risk(weather)
        else:
            p_risk = None
            explanation = None

        p_det = None

        # IMAGE
        if uploaded_file is not None:

            img = Image.open(uploaded_file)
            img_path = "temp.jpg"
            img.save(img_path)

            st.image(img, caption="Uploaded Image", use_container_width=True)

            # AUTO MODE
            if model_choice == "AUTO (Recommended)":

                # Step 1: scout using MobileNet
                mobilenet_prob = None
                if uploaded_file is not None:
                    mobilenet_prob = predict_image(img_path)
                selected_model = choose_model(p_risk, mobilenet_prob)
                if selected_model == "weather_only":
                    st.warning("No image provided — using weather-based wildfire prediction only.")

                if selected_model != "weather_only":
                    st.info("Auto Router selected: **{}** model".format(selected_model.upper()))
                else:
                    st.info("No image available — operating in Weather Prediction Mode")

                # MobileNet chosen
                if selected_model == "mobilenet":
                    p_det = mobilenet_prob
                    st.metric("Fire Probability (Scout)", round(p_det,3))
                    st.write("Classification:", classify_fire(p_det))

                    heatmap = generate_gradcam(img_path)
                    st.image(heatmap, caption="MobileNet Explainability", use_container_width=True)

                # VGG verification
                elif selected_model == "vgg":
                    p_det = predict_vgg(img_path)
                    st.metric("Fire Probability (Verification)", round(p_det,3))
                    st.write("Classification:", classify_fire(p_det))

                    heatmap = generate_vgg_gradcam(img_path)
                    st.image(heatmap, caption="VGG16 Verification Heatmap", use_container_width=True)

                # YOLO localization
                elif selected_model == "yolo":
                    p_det, boxed = detect_fire(img_path)
                    st.metric("Fire Confidence (Localization)", round(p_det,3))
                    st.image(boxed, caption="YOLOv8 Fire Localization", use_container_width=True)

            # MANUAL: MobileNet
            elif model_choice == "AgniDrishti Lite":
                p_det = predict_image(img_path)
                st.metric("Fire Probability", round(p_det,3))
                st.write("Classification:", classify_fire(p_det))

                heatmap = generate_gradcam(img_path)
                st.image(heatmap, caption="MobileNet Heatmap", use_container_width=True)

            # MANUAL: VGG
            elif model_choice == "AgniDrishti Pro":
                p_det = predict_vgg(img_path)
                st.metric("Fire Probability", round(p_det,3))
                st.write("Classification:", classify_fire(p_det))

                heatmap = generate_vgg_gradcam(img_path)
                st.image(heatmap, caption="VGG16 Heatmap", use_container_width=True)

            # MANUAL: YOLO
            else:
                p_det, boxed = detect_fire(img_path)
                st.metric("YOLO Confidence", round(p_det,3))
                st.image(boxed, caption="YOLOv8 Fire Localization", use_container_width=True)

        else:
            st.warning("No image uploaded — using weather-only prediction.")

        # FUSION
        p_final = fuse_predictions(p_risk, p_det)
        alert = get_alert_level(p_final)

        st.header("Final Alert Decision")
        if p_final is not None:
            st.metric("Final Fire Probability", p_final)
        else:
            st.metric("Final Fire Probability", "N/A")
        st.subheader("ALERT LEVEL: {}".format(alert))

        # WEATHER EXPLAIN
        if p_risk is not None:
            st.header("Weather Risk Explanation")
            st.metric("Weather Fire Risk", "{}%".format(round(p_risk * 100, 1)))

            if explanation is not None:
                
                # 1. THE CLASSIC SHAP BAR CHART
                st.subheader("Feature Impact Analysis (SHAP)")
                st.caption("Red bars increase fire probability. Blue bars decrease fire probability.")
                
                # Create a matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Extract data from our dictionary
                features = list(explanation.keys())
                values = list(explanation.values())
                
                # Classic SHAP Colors: Crimson for positive, dodgerblue for negative
                colors = ['#ff0051' if val > 0 else '#008bfb' for val in values]
                
                # Create horizontal bars
                bars = ax.barh(features, values, color=colors)
                
                # Formatting the chart to look clean and professional
                ax.axvline(x=0, color='black', linewidth=1.2) # Center line
                ax.invert_yaxis() # Put the most important features at the top
                ax.set_xlabel("SHAP Value (Impact on Model Output)")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Render the plot in Streamlit
                st.pyplot(fig)

                st.markdown("---")

                # 2. RAW VALUES TABLE (Optional, but good for research demo)
                with st.expander("View Exact SHAP Mathematical Values"):
                    shap_df = pd.DataFrame(
                        [{"Feature": k, "SHAP Score": "{:+.4f}".format(v)} for k, v in explanation.items()]
                    )
                    st.dataframe(shap_df, hide_index=True, use_container_width=True)

                st.markdown("---")

                # 3. HUMAN-READABLE TRANSLATION (NLP)
                st.subheader("What does this mean?")
                
                danger_factors = []
                safe_factors = []
                
                for feature, shap_val in explanation.items():
                    if shap_val > 0.05:  
                        danger_factors.append((feature, shap_val))
                    elif shap_val < -0.05:
                        safe_factors.append((feature, shap_val))
                        
                danger_factors = sorted(danger_factors, key=lambda x: x[1], reverse=True)[:3]
                safe_factors = sorted(safe_factors, key=lambda x: abs(x[1]), reverse=True)[:3]

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("🚨 **Increasing the Danger:**")
                    if not danger_factors:
                        st.write("No major environmental threats detected.")
                    for feat, val in danger_factors:
                        human_name = FEATURE_TRANSLATIONS.get(feat, feat)
                        st.error("**{}** ({:+.2f}) is pushing the fire risk UP.".format(human_name, val))
                        
                with col2:
                    st.markdown("🛡️ **Keeping it Safe:**")
                    if not safe_factors:
                        st.write("No mitigating weather conditions right now.")
                    for feat, val in safe_factors:
                        # Simplified to map directly to our neutral translations
                        human_name = FEATURE_TRANSLATIONS.get(feat, feat)
                        st.success("**{}** ({:+.2f}) is helping keep the risk DOWN.".format(human_name, val))

        else:
            st.info("Weather data not available — system operating in Camera-Only Mode")

        # DATABASE
        top_features = None
        if explanation is not None:
            top_features = dict(list(explanation.items())[:5])

        log_alert(p_risk, p_det, p_final, alert, top_features)

with tab2:

    st.header("Understanding the Intelligence Behind XFireAlert")
    st.subheader("AgniDrishti Lite")
    st.write("""
    • Model Used: MobileNetV2     
    • Lightweight convolutional neural network  
    • Quickly scans incoming images  
    • Detects early signs like smoke, brightness, and color changes  
    • Used as a first-response detection model
    """)

    st.info("Role: Fast detection when immediate response is needed.")

    st.markdown("---")

    st.subheader("AgniDrishti Pro")
    st.write("""
    • Model Used: VGG16  
    • Deep neural network with stronger feature extraction  
    • Confirms whether fire evidence is real  
    • Reduces false alarms caused by fog, clouds, or sunlight
    """)

    st.info("Role: Confirms suspicious cases detected by MobileNet.")

    st.markdown("---")

    st.subheader("AgniNetra Sentinel")
    st.write("""
    • Model Used: YOLOv8         
    • Object detection network  
    • Locates exact position of visible flames  
    • Draws bounding boxes around fire
    """)

    st.info("Role: Provides physical visual evidence of fire location.")

    st.markdown("---")

    st.subheader("Weather Intelligence")
    st.write("""
    Uses meteorological parameters like temperature, humidity, wind speed, and fuel moisture indices.
    This predicts whether environmental conditions support wildfire ignition.
    """)

    st.info("Role: Predicts fire risk even before visible fire appears.")

    st.markdown("---")

    st.subheader("AUTO Mode Decision Logic")

    st.write("""
    The system automatically selects the best model:

    • High smoke + high weather risk → Verification (VGG16)  
    • Visible flames → Localization (YOLOv8)  
    • Weak evidence → Scout (MobileNetV2)
    """)

    st.success("This creates a robust AI decision-support wildfire monitoring system.")