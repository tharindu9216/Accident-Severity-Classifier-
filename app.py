import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
from pathlib import Path

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="centered")
st.title("🚦 Road Accident Severity Classifier")
st.markdown("**Best Model:** Cascade Random Forest (3-stage + Tuned)")

# ================== YOUR FILE ID ==================
FILE_ID = "1ivkwAA26cU4TeDsHg9TKOB3l8vGUhmCZ"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "artifacts/cascade_rf_model.pkl"

def ensure_model_present():
    path = Path(MODEL_PATH)
    if path.exists() and path.stat().st_size > 100_000:
        st.success("✅ Model loaded from cache")
        return True

    os.makedirs("artifacts", exist_ok=True)
    st.info("📥 Downloading large model (~703 MB) from Google Drive... This may take 1-2 minutes.")

    try:
        gdown.download(MODEL_URL, str(path), quiet=False, fuzzy=True)
        if path.exists() and path.stat().st_size > 100_000:
            st.success("✅ Model downloaded successfully!")
            return True
    except Exception as e:
        st.warning(f"Download attempt failed: {e}")

    st.error("❌ Could not download the model. Please check Google Drive sharing settings.")
    st.stop()
    return False

ensure_model_present()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()

# Safe extraction of label encoder and classes
le = bundle.get('label_encoder')
if le is None or not hasattr(le, 'classes_'):
    st.error("LabelEncoder not found in model bundle.")
    st.stop()

# Force classes to be string for safety
le.classes_ = np.array([str(c) for c in le.classes_])

preproc = bundle['preprocessor']

# ================== PREDICTION FUNCTION ==================
def predict_severity(input_dict):
    # Safe defaults for all expected columns
    default_row = {
        'Day_of_Week': 'Monday',
        'Junction_Control': 'Give way or uncontrolled',
        'Junction_Detail': 'Not at junction or within 20 metres',
        'Light_Conditions': 'Daylight',
        'Carriageway_Hazards': 'None',
        'Road_Surface_Conditions': 'Dry',
        'Road_Type': 'Single carriageway',
        'Urban_or_Rural_Area': 'Urban',
        'Weather_Conditions': 'Fine no high winds',
        'Vehicle_Type': 'Car',
        'Number_of_Vehicles': 2,
        'Number_of_Casualties': 1,
        'Speed_limit': 30,
        'High_Speed': 0,
    }
    
    default_row.update(input_dict)
    input_df = pd.DataFrame([default_row])

    # Transform
    X = preproc.transform(input_df)
    X_dense = X.toarray() if hasattr(X, 'toarray') else X

    # Cascade prediction
    proba1 = bundle['stage1'].predict_proba(X_dense)
    X2 = np.hstack([X_dense, proba1])

    proba2 = bundle['stage2'].predict_proba(X2)
    X3 = np.hstack([X2, proba2])

    stage3 = bundle.get('stage3_tuned', bundle.get('stage3'))
    proba3 = stage3.predict_proba(X3)
    pred = stage3.predict(X3)[0]

    # Safe inverse transform
    try:
        severity = le.inverse_transform([pred])[0]
    except Exception:
        # Fallback if label is unknown
        severity = "Unknown"
    
    probs = dict(zip(le.classes_, proba3[0].round(4)))
    return severity, probs

# ================== INPUT FORM ==================
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        junction_control = st.selectbox("Junction Control", 
            ["Give way or uncontrolled", "Auto traffic signal", "Stop sign", 
             "Authorised person", "Not at junction or within 20 metres"])
        junction_detail = st.selectbox("Junction Detail", 
            ["Not at junction or within 20 metres", "T or staggered junction", "Crossroads", 
             "Roundabout", "Slip road", "Other junction"])
        light = st.selectbox("Light Conditions", 
            ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", 
             "Darkness - no lighting", "Dusk or dawn"])
        weather = st.selectbox("Weather Conditions", 
            ["Fine no high winds", "Raining no high winds", "Snowing no high winds", 
             "Fine with high winds", "Raining with high winds", "Fog or mist", "Other"])

    with col2:
        road_type = st.selectbox("Road Type", 
            ["Single carriageway", "Dual carriageway", "One way street", "Roundabout", "Slip road"])
        road_surface = st.selectbox("Road Surface Conditions", 
            ["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood over 3cm deep"])
        urban_rural = st.selectbox("Urban or Rural", ["Urban", "Rural"])
        vehicle_type = st.selectbox("Vehicle Type", 
            ["Car", "Taxi/Private hire car", "Motorcycle over 500cc", "Bus or coach", "Van / Goods vehicle", "Other"])
        speed = st.slider("Speed Limit (mph)", 20, 70, 30, step=10)
        num_vehicles = st.number_input("Number of Vehicles", 1, 20, 2)
        num_casualties = st.number_input("Number of Casualties", 1, 20, 1)

    submitted = st.form_submit_button("🚀 Predict Severity")

if submitted:
    input_data = {
        'Day_of_Week': day,
        'Junction_Control': junction_control,
        'Junction_Detail': junction_detail,
        'Light_Conditions': light,
        'Weather_Conditions': weather,
        'Road_Type': road_type,
        'Road_Surface_Conditions': road_surface,
        'Urban_or_Rural_Area': urban_rural,
        'Vehicle_Type': vehicle_type,
        'Speed_limit': speed,
        'Number_of_Vehicles': num_vehicles,
        'Number_of_Casualties': num_casualties,
        'High_Speed': 1 if speed >= 60 else 0,
        'Carriageway_Hazards': 'None',
    }
    
    severity, probs = predict_severity(input_data)
    
    if severity == "Fatal":
        st.error(f"**Predicted Severity: {severity}**")
    elif severity == "Serious":
        st.warning(f"**Predicted Severity: {severity}**")
    else:
        st.success(f"**Predicted Severity: {severity}**")
    
    st.write("**Probability Distribution**")
    for k, v in probs.items():
        st.write(f"• {k}: **{v:.1%}**")

st.caption("Powered by Cascade Random Forest • Trained on UK Road Accident Data")