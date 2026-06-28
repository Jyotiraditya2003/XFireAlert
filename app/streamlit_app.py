import os
import sys
import time
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ==========================================
# DYNAMIC THEME GENERATION
# ==========================================
streamlit_dir = ".streamlit"
config_path = os.path.join(streamlit_dir, "config.toml")

if not os.path.exists(streamlit_dir):
    os.makedirs(streamlit_dir)

theme_settings = """
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
"""

with open(config_path, "w") as f:
    f.write(theme_settings)

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
st.set_page_config(page_title="XFireAlert", layout="wide")

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

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_DIR / "src"
sys.path.insert(0, str(SRC_PATH))

# --- IMPORT PIPELINES ---
from risk_pipeline import predict_risk, explain_risk
from image_pipeline import predict_image, generate_gradcam, classify_fire
from vgg_pipeline import predict_vgg
from vgg_gradcam import generate_vgg_gradcam
from yolo_pipeline import detect_fire
from fusion_engine import fuse_predictions, get_alert_level
from database import log_alert, get_all_alerts, get_alert_stats, DB_AVAILABLE
from model_router import choose_model

# ==========================================
# APP HEADER
# ==========================================
st.title("XFireAlert")
st.markdown("### Hybrid Forest Fire Prediction & Monitoring System")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Fire Monitor Dashboard", "Know Your AI", "Alert History"])

with tab1:
    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    with st.sidebar:
        st.header("System Controls")
        use_weather = st.checkbox("Enable Weather Sensors", value=True)
        
        st.markdown("---")
        st.subheader("Vision Model Control")
        model_choice = st.selectbox(
            "Select Fire Detection Model",
            [
                "AUTO (Recommended)",
                "AgniDrishti Lite (MobileNet)",
                "AgniDrishti Pro (VGG16)",
                "AgniNetra Sentinel (YOLOv8)"
            ]
        )
        
        uploaded_file = st.file_uploader("Upload Drone/Satellite Imagery", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        predict_button = st.button("Run Analysis", type="primary", use_container_width=True)

    # ==========================================
    # MAIN PAGE INPUTS (GRID LAYOUT)
    # ==========================================
    p_risk = None
    explanation = None
    weather = {}

    if use_weather:
        with st.expander("Configure Live Weather Sensor Data", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, value=35.0, step=0.1)
                rain = st.number_input("Rain (mm)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
                dc = st.number_input("Deep Drought Index (DC)", min_value=0.0, max_value=600.0, value=100.0, step=1.0)
                
            with col2:
                rh = st.number_input("Relative Humidity (%)", min_value=5.0, max_value=100.0, value=30.0, step=0.1)
                ffmc = st.number_input("Vegetation Dryness (FFMC)", min_value=50.0, max_value=100.0, value=85.0, step=0.1)
                isi = st.number_input("Wind Spread Potential (ISI)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
                
            with col3:
                ws = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
                dmc = st.number_input("Soil Moisture (DMC)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
                bui = st.number_input("Combustibility (BUI)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)

            weather = {
                "Temperature": temp, "RH": rh, "Ws": ws, "Rain": rain,
                "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi, "BUI": bui
            }

    # ==========================================
    # EXECUTION & ANIMATED STATUS
    # ==========================================
    if predict_button:
        st.markdown("---")
        
        with st.status("Initiating XFireAlert Hybrid Engine...", expanded=True) as status:
            p_det = None
            
            if use_weather:
                st.write("Fetching and analyzing meteorological data...")
                time.sleep(0.5)
                p_risk = predict_risk(weather)
                explanation = explain_risk(weather)
            
            if uploaded_file is not None:
                st.write("Processing imagery through Vision Neural Networks...")
                img = Image.open(uploaded_file)
                img_path = "temp.jpg"
                img.save(img_path)
                time.sleep(0.5)
                
                if model_choice == "AUTO (Recommended)":
                    st.write("Auto-Router actively selecting best model...")
                    mobilenet_prob = predict_image(img_path)
                    selected_model = choose_model(p_risk, mobilenet_prob)
                    
                    if selected_model == "mobilenet":
                        st.write("Rapid Scout (MobileNet) selected.")
                        p_det = mobilenet_prob
                        heatmap = generate_gradcam(img_path)
                    elif selected_model == "vgg":
                        st.write("Verification Engine (VGG16) selected.")
                        p_det = predict_vgg(img_path)
                        heatmap = generate_vgg_gradcam(img_path)
                    elif selected_model == "yolo":
                        st.write("Localization Engine (YOLOv8) selected.")
                        p_det, boxed = detect_fire(img_path)
                
                elif model_choice == "AgniDrishti Lite (MobileNet)":
                    p_det = predict_image(img_path)
                    heatmap = generate_gradcam(img_path)
                elif model_choice == "AgniDrishti Pro (VGG16)":
                    p_det = predict_vgg(img_path)
                    heatmap = generate_vgg_gradcam(img_path)
                else: 
                    p_det, boxed = detect_fire(img_path)

            st.write("Fusing multimodal predictions...")
            time.sleep(0.5)
            p_final = fuse_predictions(p_risk, p_det)
            alert = get_alert_level(p_final)
            
            top_features = dict(list(explanation.items())[:5]) if explanation else None
            log_alert(p_risk, p_det, p_final, alert, top_features)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

        # ==========================================
        # DETAILED RESULTS DASHBOARD (VERTICAL FLOW)
        # ==========================================
        
        st.success(f"**SYSTEM DECISION: {alert}**")

        with st.container(border=True):
            st.subheader("Global Risk Assessment")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Final Fire Probability", f"{round(p_final * 100, 1)}%" if p_final else "N/A")
            with m2:
                st.metric("Weather Risk Index", f"{round(p_risk * 100, 1)}%" if p_risk else "N/A")
            with m3:
                st.metric("Visual Detection Confidence", f"{round(p_det * 100, 1)}%" if p_det else "N/A")

        if uploaded_file is not None:
            with st.container(border=True):
                st.subheader("Vision Analysis Results")
                if model_choice == "AUTO (Recommended)":
                    st.caption(f"**Model Triggered:** {selected_model.upper()}")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.image(img, caption="Original Imagery", use_container_width=True)
                with v_col2:
                    if 'boxed' in locals():
                        st.image(boxed, caption="YOLOv8 Localization bounding boxes", use_container_width=True)
                    elif 'heatmap' in locals():
                        st.image(heatmap, caption="AI Attention Heatmap (Explainability)", use_container_width=True)

        if p_risk is not None and explanation is not None:
            with st.container(border=True):
                st.subheader("Environmental Risk Breakdown")
                
                features = list(explanation.keys())
                values = list(explanation.values())
                colors = ['#ff4b4b' if val > 0 else '#008bfb' for val in values]
                
                fig = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in values],
                    textposition="auto"
                ))
                
                fig.update_layout(
                    title="SHAP Value Impact (Red = Increases Risk, Blue = Decreases Risk)",
                    xaxis_title="Impact on Model Output",
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Exact SHAP Mathematical Values"):
                    shap_df = pd.DataFrame(
                        [{"Feature": k, "SHAP Score": "{:+.4f}".format(v)} for k, v in explanation.items()]
                    )
                    st.dataframe(shap_df, hide_index=True, use_container_width=True)

                st.markdown("---")
                st.markdown("##### Feature Impact Summary")
                danger_factors = [(k, v) for k, v in explanation.items() if v > 0.05]
                safe_factors = [(k, v) for k, v in explanation.items() if v < -0.05]
                
                danger_factors = sorted(danger_factors, key=lambda x: x[1], reverse=True)[:3]
                safe_factors = sorted(safe_factors, key=lambda x: abs(x[1]), reverse=True)[:3]

                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    st.markdown("**Increasing the Danger**")
                    if not danger_factors:
                        st.write("No major environmental threats detected.")
                    for feat, val in danger_factors:
                        human_name = FEATURE_TRANSLATIONS.get(feat, feat)
                        st.error(f"**{human_name}** is pushing the fire risk UP.")
                
                with h_col2:
                    st.markdown("**Mitigating Factors**")
                    if not safe_factors:
                        st.write("No mitigating weather conditions right now.")
                    for feat, val in safe_factors:
                        human_name = FEATURE_TRANSLATIONS.get(feat, feat)
                        st.success(f"**{human_name}** is helping keep the risk DOWN.")

# ==========================================
# TAB 2: DOCUMENTATION (GRID LAYOUT, ORIGINAL TEXT)
# ==========================================
with tab2:
    st.header("Understanding the Intelligence Behind XFireAlert")
    st.markdown("---")

    doc_col1, doc_col2 = st.columns(2)

    with doc_col1:
        with st.container(border=True):
            st.subheader("AgniDrishti Lite")
            st.write("""
            • Model Used: MobileNetV2     
            • Lightweight convolutional neural network  
            • Quickly scans incoming images  
            • Detects early signs like smoke, brightness, and color changes  
            • Used as a first-response detection model
            """)
            st.info("Role: Fast detection when immediate response is needed.")

        with st.container(border=True):
            st.subheader("AgniNetra Sentinel")
            st.write("""
            • Model Used: YOLOv8         
            • Object detection network  
            • Locates exact position of visible flames  
            • Draws bounding boxes around fire
            """)
            st.info("Role: Provides physical visual evidence of fire location.")

    with doc_col2:
        with st.container(border=True):
            st.subheader("AgniDrishti Pro")
            st.write("""
            • Model Used: VGG16  
            • Deep neural network with stronger feature extraction  
            • Confirms whether fire evidence is real  
            • Reduces false alarms caused by fog, clouds, or sunlight
            """)
            st.info("Role: Confirms suspicious cases detected by MobileNet.")

        with st.container(border=True):
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

# ==========================================
# TAB 3: ALERT HISTORY
# ==========================================
with tab3:
    st.header("Alert History & Analytics")
    st.markdown("---")
    
    if not DB_AVAILABLE:
        st.warning("MongoDB not connected. Configure .env file to enable alert history.")
        st.code("MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/...")
    else:
        stats = get_alert_stats()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alerts", stats["total_alerts"])
            with col2:
                st.metric("High Alerts", stats["high_alerts"], delta="Critical")
            with col3:
                st.metric("Moderate Warnings", stats["moderate_alerts"])
            with col4:
                st.metric("Low Risk", stats["low_alerts"])
            
            st.markdown("---")
        
        st.subheader("Recent Alerts")
        
        alerts = get_all_alerts(limit=50)
        
        if alerts:
            df = pd.DataFrame(alerts)
            df = df[["timestamp", "alert_level", "final_probability", "weather_risk", "vision_confidence"]]
            df.columns = ["Timestamp", "Alert Level", "Final Prob", "Weather Risk", "Vision Conf"]
            df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            
            def color_alert(val):
                if val == "HIGH ALERT":
                    return "background-color: #ff4b4b; color: white"
                elif val == "MODERATE WARNING":
                    return "background-color: #ffa500; color: black"
                elif val == "LOW RISK":
                    return "background-color: #00ff00; color: black"
                return ""
            
            styled_df = df.style.applymap(color_alert, subset=["Alert Level"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No alerts recorded yet. Run an analysis to create your first alert.")