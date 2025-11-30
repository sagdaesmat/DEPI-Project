import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ----- Config -----
TITLE = "InnerSight — Mental Health Risk Intelligence"
SUBTITLE = "This tool provides an AI-powered screening estimate. It is not a clinical diagnosis."

PIPELINE_FILE = "innersight_pipeline.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
SHAP_IMAGE = "shap_summary.png"

st.set_page_config(page_title="InnerSight", layout="centered")

# Header
st.title(TITLE)
st.write(SUBTITLE)
st.write("---")

# Load artifacts
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(PIPELINE_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    return pipeline, label_encoder, feature_names

pipeline, label_encoder, feature_names = load_artifacts()

# Form
with st.form("assessment_form"):
    st.header("Basic Information")
    age = st.number_input("Age", 12, 100, 30)
    gender = st.selectbox("Gender", ["male","female","non-binary","prefer not to say"])
    employment = st.selectbox("Employment Status", ["employed","self-employed","student","unemployed","other"])
    work_env = st.selectbox("Work Environment", ["on-site","remote","hybrid"])
    mental_history = st.selectbox("Previous Mental Health Diagnosis?", ["no","yes"])
    seeks_treatment = st.selectbox("Currently Seeking Treatment?", ["no","yes"])

    st.header("Lifestyle & Symptoms")
    stress = st.slider("Stress Level (0–10)", 0, 10, 5)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    activity = st.slider("Physical Activity Days per Week", 0, 7, 2)

    st.subheader("Symptom Questionnaire (0=never, 3=always)")
    d1 = st.selectbox("Little interest in activities?", [0,1,2,3])
    d2 = st.selectbox("Feeling down or hopeless?", [0,1,2,3])
    d3 = st.selectbox("Trouble concentrating?", [0,1,2,3])

    a1 = st.selectbox("Feeling nervous or on edge?", [0,1,2,3])
    a2 = st.selectbox("Unable to stop worrying?", [0,1,2,3])
    a3 = st.selectbox("Irritability?", [0,1,2,3])

    s1 = st.slider("Support: I have someone to talk to", 0, 4, 3)
    s2 = st.slider("Support: I feel supported by family/friends", 0, 4, 3)

    p1 = st.slider("I complete tasks effectively", 0, 4, 3)
    p2 = st.slider("Productivity has been impacted", 0, 4, 2)

    submitted = st.form_submit_button("Get Assessment")

def scale(value, min_v, max_v):
    return (value-min_v)/(max_v-min_v)*100

if submitted:
    depression = scale(d1 + d2 + d3, 0, 9)
    anxiety = scale(a1 + a2 + a3, 0, 9)
    social = scale(s1 + s2, 0, 8)
    productivity = scale(p1 + (4-p2), 0, 8)

    raw = {
        "age": age,
        "stress_level": stress,
        "sleep_hours": sleep,
        "physical_activity_days": activity,
        "depression_score": depression,
        "anxiety_score": anxiety,
        "social_support_score": social,
        "productivity_score": productivity,
        "mental_health_history": 1 if mental_history=="yes" else 0,
        "seeks_treatment": 1 if seeks_treatment=="yes" else 0,
        "gender": gender,
        "employment_status": employment,
        "work_environment": work_env
    }

    input_df = pd.DataFrame([raw])

    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Risk Level: {label.upper()}")
    st.write(f"Probabilities: Low {proba[0]:.2f}, Medium {proba[1]:.2f}, High {proba[2]:.2f}")

    if os.path.exists(SHAP_IMAGE):
        st.subheader("Global Feature Importance (SHAP)")
        st.image(SHAP_IMAGE, use_column_width=True)
