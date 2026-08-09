import pandas as pd

full_df = pd.read_csv("data/processed/memories_with_heuristic_score.csv")
already_labeled = pd.read_csv("data/labeled/memories_labeled.csv")

unused = full_df[~full_df["memory_id"].isin(already_labeled["memory_id"])]

target_types = unused[unused["type"].isin(["fact", "goal", "preference"])].copy()

target_types["last_accessed_at"] = pd.to_datetime(target_types["last_accessed_at"])
import datetime
NOW = datetime.datetime(2026, 7, 29)
target_types["days_since_accessed"] = (NOW - target_types["last_accessed_at"]).dt.days

likely_delete = target_types[
    (target_types["days_since_accessed"] > 60) | (target_types["feedback"] == "negative")
]

sample_size = min(60, len(likely_delete))
new_sample = likely_delete.sample(n=sample_size, random_state=7).copy()

new_sample["human_label"] = ""
new_sample["human_score"] = ""
new_sample["notes"] = ""

if __name__ == "__main__":
    new_sample.to_csv("data/labeled/additional_memories_to_label.csv", index=False)
    print(f"Saved {len(new_sample)} candidate rows (fact/goal/preference, likely delete) for labeling.")
    print(new_sample["type"].value_counts())