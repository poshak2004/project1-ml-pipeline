import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join("data", "raw", "student_performance.csv")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw data at {path}.")
    return pd.read_csv(path)

def split_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_path = os.path.join(PROC_DIR, "train.csv")
    test_path = os.path.join(PROC_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train -> {train_path} ({len(train_df)} rows)")
    print(f"Saved test  -> {test_path} ({len(test_df)} rows)")

if __name__ == "__main__":
    df = load_data(RAW_PATH)
    split_save(df)
