import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ==========================
# UI COLORS (Readable)
# ==========================
BG = "#f7fdf9"         # خلفية فاتحة جدًا
ACCENT = "#b4e4ce"     # Mint Green هادي
TEXT = "#1e3d32"       # غامق وواضح جدًا
CARD_BG = "#ffffff"    # كرت أبيض
BORDER = "#2f5b49"     # أخضر غامق

# ==========================
# Streamlit Config
# ==========================
st.set_page_config(
    page_title="InnerSight — Mental Health Risk",
    layout="wide",
)

# ==========================
# CSS Styling
# ==========================
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BG};
        }}

        h1, h2, h3, h4, h5 {{
            color: {TEXT} !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        .result-card {{
            background-color: {CARD_BG};
            padding: 20px;
            border-radius: 12px;
            border: 2px solid {BORDER};
            margin-top: 20px;
        }}

        .stSelectbox > div > div {{
            color: {TEXT} !important;
        }}

        .stSlider > div > div {{
            color: {TEXT} !important;
        }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# HEADER
# ==========================
st.markdown(f"<h1>InnerSight — Mental Health Risk Intelligence</h1>", unsafe_allow_html=True)
st.write("AI-powered mental health screening (not a medical diagnosis).")
st.write("---")

# ==========================
# LOAD ARTIFACTS
# ==========================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("innersight_pipeline.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return pipeline, label_encoder, feature_names

pipeline, label_encoder, feature_names = load_artifacts()

# ==========================
# FORM INPUTS
# ==========================
with st.form("assessment_form"):
    st.subheader("Basic Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 12, 100, 30)
        gender = st.selectbox("Gender", ["male", "non-binary", "prefer not to say"])
        employment = st.selectbox("Employment Status", ["self-employed", "student", "unemployed"])

    with col2:
        work_env = st.selectbox("Work Environment", ["on-site", "remote"])
        mental_history = st.selectbox("Previous Mental Health Diagnosis?", ["no", "yes"])
        seeks_treatment = st.selectbox("Currently Seeking Treatment?", ["no", "yes"])

    st.subheader("Lifestyle")
    stress = st.slider("Stress Level (0–10)", 0, 10, 5)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    activity = st.slider("Physical Activity Days per Week", 0, 7, 2)

    st.subheader("Symptoms (0 = never, 3 = always)")

    d1 = st.selectbox("Little interest or pleasure in activities", [0, 1, 2, 3])
    d2 = st.selectbox("Feeling down or hopeless", [0, 1, 2, 3])
    d3 = st.selectbox("Trouble concentrating", [0, 1, 2, 3])

    a1 = st.selectbox("Feeling nervous or on edge", [0, 1, 2, 3])
    a2 = st.selectbox("Unable to stop worrying", [0, 1, 2, 3])
    a3 = st.selectbox("Irritability", [0, 1, 2, 3])

    s1 = st.slider("I have someone to talk to", 0, 4, 3)
    s2 = st.slider("I feel supported by others", 0, 4, 3)

    p1 = st.slider("I complete tasks effectively", 0, 4, 3)
    p2 = st.slider("My productivity has decreased", 0, 4, 2)

    submitted = st.form_submit_button("Get Assessment")

# ==========================
# SCALING
# ==========================
def scale(v, a, b):
    return (v - a) / (b - a) * 100

# ==========================
# PREDICTION
# ==========================
if submitted:
    depression = scale(d1+d2+d3, 0, 9)
    anxiety = scale(a1+a2+a3, 0, 9)
    social = scale(s1+s2, 0, 8)
    productivity = scale(p1 + (4-p2), 0, 8)

    row = {
        "age": age,
        "stress_level": stress,
        "sleep_hours": sleep,
        "physical_activity_days": activity,
        "depression_score": depression,
        "anxiety_score": anxiety,
        "social_support_score": social,
        "productivity_score": productivity,
        "gender": gender,
        "employment_status": employment,
        "work_environment": work_env,
        "mental_health_history": 1 if mental_history == "yes" else 0,
        "seeks_treatment": 1 if seeks_treatment == "yes" else 0,
    }

    df = pd.DataFrame([row])

    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0]
    label = label_encoder.inverse_transform([pred])[0]

    st.markdown("<h3>Assessment Result</h3>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card">
            <p><strong>Predicted Risk Level:</strong> {label.upper()}</p>
            <p>Low: {proba[0]:.2f} &nbsp;|&nbsp; Medium: {proba[1]:.2f} &nbsp;|&nbsp; High: {proba[2]:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("---")

    if os.path.exists("shap_summary.png"):
        st.subheader("Global Feature Importance (SHAP)")
        st.image("shap_summary.png", caption="", use_column_width=True)

