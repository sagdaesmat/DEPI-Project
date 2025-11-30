# InnerSight — Mental Health Risk Intelligence System

InnerSight is an end-to-end, explainable AI system that predicts mental health risk levels (Low, Medium, High) based on behavioral, psychological, and lifestyle features.

## Features
- Fully engineered machine learning pipeline
- Multiple models evaluated with stratified 5-fold cross-validation
- Hyperparameter tuning for Random Forest & XGBoost
- Stacking ensemble for improved generalization
- SHAP explainability for transparency
- Final calibrated pipeline deployed via Streamlit
- Clean UI for interactive assessments

## Files
- `innersight_pipeline.pkl` — final ML pipeline (preprocessing + calibrated model)
- `feature_names.pkl` — ordered feature names
- `label_encoder.pkl` — mapping for target labels
- `shap_summary.png` — SHAP global importance
- `app.py` — main Streamlit application
- `requirements.txt` — environment dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
