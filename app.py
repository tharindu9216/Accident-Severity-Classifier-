import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="centered")
st.title("🚦 Road Accident Severity Classifier")
st.markdown("**Best Model:** Cascade Random Forest (3-stage + Tuned)")

@st.cache_resource
def load_model():
    return joblib.load("artifacts/cascade_rf_model.pkl")

bundle = load_model()
le = bundle['label_encoder']

def predict_severity(input_df):
    preproc = bundle['preprocessor']
    X = preproc.transform(input_df)
    X_dense = X.toarray() if hasattr(X, 'toarray') else X

    # Stage 1
    proba1 = bundle['stage1'].predict_proba(X_dense)
    X2 = np.hstack([X_dense, proba1])

    # Stage 2
    proba2 = bundle['stage2'].predict_proba(X2)
    X3 = np.hstack([X2, proba2])

    # Stage 3 (use tuned version if available)
    stage3 = bundle.get('stage3_tuned', bundle['stage3'])
    proba3 = stage3.predict_proba(X3)
    pred = stage3.predict(X3)[0]

    severity = le.inverse_transform([pred])[0]
    probs = dict(zip(le.classes_, proba3[0].round(4)))
    return severity, probs

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        junction_control = st.selectbox("Junction Control", ["Give way or uncontrolled", "Auto traffic signal", "Stop sign", "Authorised person", "Not at junction or within 20 metres"])
        light = st.selectbox("Light Conditions", ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting", "Dusk or dawn"])
        weather = st.selectbox("Weather Conditions", ["Fine no high winds", "Raining no high winds", "Snowing no high winds", "Fine with high winds", "Raining with high winds", "Fog or mist", "Other"])
    with col2:
        road_type = st.selectbox("Road Type", ["Single carriageway", "Dual carriageway", "One way street", "Roundabout", "Slip road"])
        urban_rural = st.selectbox("Urban or Rural", ["Urban", "Rural"])
        speed = st.slider("Speed Limit (mph)", 20, 70, 30, step=10)
        num_vehicles = st.number_input("Number of Vehicles", 1, 20, 2)
        num_casualties = st.number_input("Number of Casualties", 1, 20, 1)

    submitted = st.form_submit_button("Predict Severity")

if submitted:
    input_data = {
        'Day_of_Week': day,
        'Junction_Control': junction_control,
        'Light_Conditions': light,
        'Weather_Conditions': weather,
        'Road_Type': road_type,
        'Urban_or_Rural_Area': urban_rural,
        'Speed_limit': speed,
        'Number_of_Vehicles': num_vehicles,
        'Number_of_Casualties': num_casualties,
        'High_Speed': 1 if speed >= 60 else 0,
        # Add any missing columns with most common value if needed
    }
    input_df = pd.DataFrame([input_data])
    
    severity, probs = predict_severity(input_df)
    
    if severity == "Fatal":
        st.error(f"**Predicted Severity: {severity}**")
    elif severity == "Serious":
        st.warning(f"**Predicted Severity: {severity}**")
    else:
        st.success(f"**Predicted Severity: {severity}**")
    
    st.write("**Probability Distribution**")
    for k, v in probs.items():
        st.write(f"• {k}: **{v:.1%}**")