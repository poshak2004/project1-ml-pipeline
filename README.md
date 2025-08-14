# 📚 Student Exam Score Predictor — End-to-End ML Pipeline

**Goal:** Showcase a complete ML lifecycle: data → preprocess → train → evaluate → deploy (Streamlit).  
**Use Case:** Estimate a student's exam score (0–100) from study habits, attendance, sleep, prior grades, etc.

## 🧱 Project Structure
```text
.
├── data/
│   ├── raw/                # input CSV lives here
│   └── processed/          # train/test splits get saved here
├── models/                 # trained model + metrics
├── notebooks/              # EDA and experiments (optional)
├── reports/figures/        # charts
├── src/
│   ├── preprocess.py       # train/test split
│   └── train.py            # pipelines, training, metrics
├── app.py                  # Streamlit UI for inference
├── requirements.txt
└── README.md
```

## 🚀 Quickstart
```bash
# 1) Create & activate venv (recommended)
python -m venv .venv && source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate                          # Windows PowerShell

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Replace data/raw/student_performance.csv with your real dataset
#    Columns expected:
#    study_hours, attendance_pct, sleep_hours, past_grade(A/B/C/D),
#    internet_access(0/1), parent_education(HS/UG/PG),
#    part_time_job(0/1), test_prep(0/1), distractions_hours, exam_score, pass_fail

# 4) Preprocess (creates data/processed/train.csv & test.csv)
python src/preprocess.py

# 5) Train models (saves models/model.pkl & models/metrics.json)
python src/train.py

# 6) Run the app
streamlit run app.py
```

## 📊 Metrics
Training compares **Ridge** vs **RandomForest** and saves:
- `models/model.pkl` — best pipeline (with preprocessing)
- `models/metrics.json` — MAE / RMSE / R² per model

## 📝 Notes
- This repo ships with a **synthetic dataset** at `data/raw/student_performance.csv` so it runs out-of-the-box.
- Swap in a real dataset later to make the project resume-ready.
- Add your **EDA notebook** in `notebooks/` (correlations, feature importance).

## 📸 Screenshots
Add screenshots of Streamlit UI here (and maybe a short GIF).

## 🧠 Future Improvements
- Hyperparameter search (Optuna)
- Add classification head (pass/fail) and calibration
- Model monitoring for data drift
- Dockerize for deployment
- CI checks with pre-commit

## 🧾 License
MIT
