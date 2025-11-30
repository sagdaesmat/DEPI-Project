import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ==========================
# Custom Mint-Green Styling
# ==========================
MINT = "#d4f1e4"
DARK = "#2b4c3f"
WHITE = "#ffffff"

st.set_page_config(
    page_title="InnerSight — Mental Health Risk",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {MINT};
        }}

        h1, h2, h3, h4 {{
            color: {DARK} !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: {MINT};
        }}

        .result-card {{
            padding: 20px;
            border-radius: 12px;
            background-color: {WHITE};
            border: 2px solid {DARK};
        }}

        .shap-img {{
            border-radius: 10px;
            border: 1px solid {DARK};
        }}

        .css-1cpxqw2, .css-ffhzg2, .stSelectbox > div > div {{
            color: {DARK} !important;
        }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Header
# ==========================
st.markdown("<h1>InnerSight — Mental Health Risk Intelligence</h1>", unsafe_allow_html=True)
st.write("This tool provides an AI-based screening estimate for mental health risk.")

st.write("---")

# ==========================
# Load Artifacts
# ==========================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("innersight_pipeline.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return pipeline, label_encoder, feature_names

pipeline, label_encoder, feature_names = load_artifacts()

# ==========================
# Input Form
# ==========================
with st.form("assessment_form"):
    st.subheader("Basic Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 12, 100, 30)
        gender = st.selectbox("Gender", ["male", "female", "non-binary"])
        employment = st.selectbox("Employment Status", ["employed", "self-employed", "student", "unemployed"])

    with col2:
        work_env = st.selectbox("Work Environment", ["on-site", "remote", "hybrid"])
        mental_history = st.selectbox("Previous Mental Health Diagnosis?", ["no", "yes"])
        seeks_treatment = st.selectbox("Currently Seeking Treatment?", ["no", "yes"])

    st.subheader("Lifestyle")
    stress = st.slider("Stress Level (0–10)", 0, 10, 5)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    activity = st.slider("Physical Activity Days per Week", 0, 7, 2)

    st.subheader("Symptom Questionnaire (0=never, 3=always)")

    d1 = st.selectbox("Little interest or pleasure in activities", [0, 1, 2, 3])
    d2 = st.selectbox("Feeling down or hopeless", [0, 1, 2, 3])
    d3 = st.selectbox("Trouble concentrating", [0, 1, 2, 3])

    a1 = st.selectbox("Feeling nervous, anxious, or on edge", [0, 1, 2, 3])
    a2 = st.selectbox("Not being able to stop worrying", [0, 1, 2, 3])
    a3 = st.selectbox("Becoming easily annoyed or irritable", [0, 1, 2, 3])

    s1 = st.slider("I have someone to talk to when stressed", 0, 4, 3)
    s2 = st.slider("I feel supported by family/friends", 0, 4, 3)

    p1 = st.slider("I can complete tasks effectively", 0, 4, 3)
    p2 = st.slider("My productivity has been impacted", 0, 4, 2)

    submitted = st.form_submit_button("Get Assessment")

# ==========================
# Utility Scaling Function
# ==========================
def scale(v, a, b):
    return (v - a) / (b - a) * 100

# ==========================
# Prediction
# ==========================
if submitted:
    depression = scale(d1 + d2 + d3, 0, 9)
    anxiety = scale(a1 + a2 + a3, 0, 9)
    social = scale(s1 + s2, 0, 8)
    productivity = scale(p1 + (4 - p2), 0, 8)

    row = {
        "age": age,
        "stress_level": stress,
        "sleep_hours": sleep,
        "physical_activity_days": activity,
        "depression_score": depression,
        "anxiety_score": anxiety,
        "social_support_score": social,
        "productivity_score": productivity,
        "mental_health_history": 1 if mental_history == "yes" else 0,
        "seeks_treatment": 1 if seeks_treatment == "yes" else 0,
        "gender": gender,
        "employment_status": employment,
        "work_environment": work_env
    }

    df_input = pd.DataFrame([row])

    pred = pipeline.predict(df_input)[0]
    proba = pipeline.predict_proba(df_input)[0]
    label = label_encoder.inverse_transform([pred])[0]

    st.markdown("<h3>Assessment Result</h3>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write(f"**Predicted Risk Level:** {label.upper()}")
        st.write(f"Low: {proba[0]:.2f}   |   Medium: {proba[1]:.2f}   |   High: {proba[2]:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")

    if os.path.exists("shap_summary.png"):
        st.subheader("Global Feature Importance (SHAP)")
        st.image("shap_summary.png", use_column_width=True, caption="")

