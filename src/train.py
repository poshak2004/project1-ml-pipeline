import os
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths & Directories
# -----------------------------
PROC_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Load Processed Data
# -----------------------------
def load_processed():
    """Load processed training and test datasets."""
    train_path = os.path.join(PROC_DIR, "train.csv")
    test_path = os.path.join(PROC_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Run src/preprocess.py first to generate processed data.")
    return pd.read_csv(train_path), pd.read_csv(test_path)

# -----------------------------
# Feature / Target Split
# -----------------------------
def get_features_target(df: pd.DataFrame):
    """Split dataframe into features (X) and target (y)."""
    y = df["exam_score"]
    X = df.drop(columns=["exam_score", "pass_fail"])
    return X, y

# -----------------------------
# Build ML Pipeline
# -----------------------------
def build_pipeline(model):
    """Create preprocessing + model pipeline."""
    numeric_features = [
        "study_hours", "attendance_pct", "sleep_hours", "distractions_hours",
        "internet_access", "part_time_job", "test_prep"
    ]
    categorical_features = ["past_grade", "parent_education"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return Pipeline(steps=[("pre", preprocessor), ("model", model)])

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(y_true, y_pred):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5# works in all sklearn versions
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load data
    train_df, test_df = load_processed()
    X_train, y_train = get_features_target(train_df)
    X_test, y_test = get_features_target(test_df)

    # Candidate models
    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    results = {}
    best_model_name, best_score = None, float("-inf")

    # Train & Evaluate
    for name, model in candidates.items():
        print(f"\nTraining: {name}")
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        metrics = evaluate(y_test, preds)
        results[name] = metrics
        print(f"Metrics: {metrics}")

        if metrics["R2"] > best_score:
            best_score = metrics["R2"]
            best_model_name = name
            best_pipe = pipe

    # Save best model
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "model.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Best Model: {best_model_name} (R² = {best_score:.3f}) saved to {MODELS_DIR}/model.pkl")
    print(f"📊 All metrics saved to {MODELS_DIR}/metrics.json")
