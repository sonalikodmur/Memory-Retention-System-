import pandas as pd
from datetime import datetime

full_df = pd.read_csv("data/processed/memories_with_heuristic_score.csv")
already_labeled = pd.read_csv("data/labeled/memories_labeled.csv")

unused = full_df[~full_df["memory_id"].isin(already_labeled["memory_id"])]
goals = unused[unused["type"] == "goal"].copy()

goals["last_accessed_at"] = pd.to_datetime(goals["last_accessed_at"])
NOW = datetime(2026, 7, 29)
goals["days_since_accessed"] = (NOW - goals["last_accessed_at"]).dt.days

bins = [(0, 20), (20, 45), (45, 75), (75, 200)]
samples = []

for low, high in bins:
    bucket = goals[(goals["days_since_accessed"] >= low) & (goals["days_since_accessed"] < high)]
    n = min(8, len(bucket))
    if n > 0:
        samples.append(bucket.sample(n=n, random_state=3))

goal_sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

goal_sample["human_label"] = ""
goal_sample["human_score"] = ""
goal_sample["notes"] = ""

if __name__ == "__main__":
    goal_sample.to_csv("data/labeled/goal_spread_to_label.csv", index=False)
    print(f"Saved {len(goal_sample)} goal rows spread across recency bins.")
    print(goal_sample[["text", "days_since_accessed"]].sort_values("days_since_accessed"))