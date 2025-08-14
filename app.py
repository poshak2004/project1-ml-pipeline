import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")
st.title("📚 Student Exam Score Predictor")
st.write("End-to-end ML demo: preprocess → train → deploy (regression)")

MODEL_PATH = os.path.join("models", "model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run `python src/preprocess.py` then `python src/train.py`.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("single_infer"):
    st.subheader("🔮 Predict a single student's exam score")
    study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=16.0, value=2.0, step=0.5)
    attendance_pct = st.slider("Attendance %", min_value=50, max_value=100, value=85, step=1)
    sleep_hours = st.slider("Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    past_grade = st.selectbox("Past Grade", ["A","B","C","D"])
    internet_access = st.selectbox("Internet Access at Home", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
    parent_education = st.selectbox("Parent Education", ["HS","UG","PG"])
    part_time_job = st.selectbox("Part-Time Job", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    test_prep = st.selectbox("Test Prep Completed", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
    distractions_hours = st.slider("Daily Distraction Hours (phone/games)", 0.0, 10.0, 2.0, 0.5)
    submitted = st.form_submit_button("Predict")
    if submitted:
        X = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_pct": attendance_pct,
            "sleep_hours": sleep_hours,
            "past_grade": past_grade,
            "internet_access": internet_access,
            "parent_education": parent_education,
            "part_time_job": part_time_job,
            "test_prep": test_prep,
            "distractions_hours": distractions_hours
        }])
        pred = model.predict(X)[0]
        st.success(f"Estimated Exam Score: **{pred:.1f} / 100**")
        st.progress(min(1.0, max(0.0, pred/100.0)))

st.divider()

st.subheader("📁 Batch Predict from CSV")
st.write("Upload a CSV with the same feature columns (no target columns needed).")
uploaded = st.file_uploader("student_features.csv", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in ["study_hours","attendance_pct","sleep_hours","past_grade","internet_access","parent_education","part_time_job","test_prep","distractions_hours"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        preds = model.predict(df)
        out = df.copy()
        out["pred_exam_score"] = np.round(preds, 1)
        st.dataframe(out.head(20))
        st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")
