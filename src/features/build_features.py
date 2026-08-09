import pandas as pd
import numpy as np
from datetime import datetime

NOW = datetime(2026, 7, 29)

df = pd.read_csv("data/labeled/memories_labeled.csv", parse_dates=["created_at", "last_accessed_at"])

df["days_since_created"] = (NOW - df["created_at"]).dt.days
df["days_since_accessed"] = (NOW - df["last_accessed_at"]).dt.days

feedback_map = {"positive": 1, "none": 0, "negative": -1}
df["feedback_encoded"] = df["feedback"].map(feedback_map)

df = pd.get_dummies(df, columns=["type"], prefix="type")

df["access_count_log"] = np.log1p(df["access_count"])

df = df.dropna(subset=["human_label", "human_score"])
df = df[df["human_label"].astype(str).str.strip() != ""]

feature_cols = ["days_since_created", "days_since_accessed", "feedback_encoded", "access_count_log"] \
    + [c for c in df.columns if c.startswith("type_")]

final_df = df[["memory_id", "text"] + feature_cols + ["human_score", "human_label"]]

if __name__ == "__main__":
    final_df.to_csv("data/processed/training_features.csv", index=False)
    print(f"Saved {len(final_df)} labeled rows with {len(feature_cols)} features.")
    print(final_df.head())