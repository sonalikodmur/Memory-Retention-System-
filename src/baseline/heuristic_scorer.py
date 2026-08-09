import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("data/raw/synthetic_memories.csv", parse_dates=["created_at", "last_accessed_at"])

NOW = datetime(2026, 7, 29)

def recency_score(last_accessed_at, half_life_days=30):
    days_since = (NOW - last_accessed_at).days
    return 0.5 ** (days_since / half_life_days)

def frequency_score(access_count):
    return np.log1p(access_count) / np.log1p(30)

def feedback_score(feedback):
    return {"positive": 1.0, "negative": 0.0, "none": 0.5}[feedback]

def type_weight(mem_type):
    return {
        "task": 1.0,
        "goal": 0.9,
        "preference": 0.8,
        "fact": 0.6,
        "event": 0.3,
    }[mem_type]

def heuristic_score(row):
    r = recency_score(row["last_accessed_at"])
    f = frequency_score(row["access_count"])
    fb = feedback_score(row["feedback"])
    t = type_weight(row["type"])

    score = (0.4 * r) + (0.25 * f) + (0.2 * fb) + (0.15 * t)
    return round(score, 3)


if __name__ == "__main__":
    df["heuristic_score"] = df.apply(heuristic_score, axis=1)
    df.to_csv("data/processed/memories_with_heuristic_score.csv", index=False)
    print(df[["text", "type", "access_count", "feedback", "heuristic_score"]].head(10))