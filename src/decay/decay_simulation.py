import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

model = joblib.load("src/model/saved_models/retention_model.pkl")

FEATURE_COLS = [
    "days_since_created", "days_since_accessed", "feedback_encoded", "access_count_log",
    "type_event", "type_fact", "type_goal", "type_preference", "type_task"
]

def build_feature_row(memory_type, days_since_created, days_since_accessed, feedback_encoded, access_count):
    row = {col: 0 for col in FEATURE_COLS}
    row["days_since_created"] = days_since_created
    row["days_since_accessed"] = days_since_accessed
    row["feedback_encoded"] = feedback_encoded
    row["access_count_log"] = np.log1p(access_count)
    row[f"type_{memory_type}"] = 1
    return pd.DataFrame([row])[FEATURE_COLS]



def simulate_memory(memory_type, feedback_encoded, initial_access_count, num_days=90, access_days=None):
    access_days = access_days or []
    days_since_created = 0
    days_since_accessed = 0
    access_count = initial_access_count

    history = []
    for day in range(num_days):
        if day in access_days:
            days_since_accessed = 0
            access_count += 1
        else:
            days_since_accessed += 1
        days_since_created += 1

        features = build_feature_row(memory_type, days_since_created, days_since_accessed, feedback_encoded, access_count)
        score = model.predict(features)[0]
        history.append({"day": day, "score": score, "was_accessed": day in access_days})

    return pd.DataFrame(history)


if __name__ == "__main__":
    examples = {
        "goal_with_occasional_access": simulate_memory("goal", feedback_encoded=0, initial_access_count=2, num_days=90, access_days=[10, 45]),
        "event_never_revisited": simulate_memory("event", feedback_encoded=0, initial_access_count=0, num_days=90, access_days=[]),
        "preference_frequently_used": simulate_memory("preference", feedback_encoded=1, initial_access_count=5, num_days=90, access_days=[5, 15, 25, 35, 45, 55, 65, 75]),
        "fact_negative_feedback": simulate_memory("fact", feedback_encoded=-1, initial_access_count=1, num_days=90, access_days=[]),
    }

    all_results = []
    for name, df in examples.items():
        df["memory_name"] = name
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("data/processed/decay_simulation_results.csv", index=False)

    for name, df in examples.items():
        print(f"\n{name}: day 0 score={df['score'].iloc[0]:.3f}, day 89 score={df['score'].iloc[-1]:.3f}")