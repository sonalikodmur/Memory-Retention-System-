import pandas as pd
from datetime import datetime

full_df = pd.read_csv("data/processed/memories_with_heuristic_score.csv")
already_labeled = pd.read_csv("data/labeled/memories_labeled.csv")

unused = full_df[~full_df["memory_id"].isin(already_labeled["memory_id"])]

durable_keywords = ["allergic", "hometown", "mother tongue", "left-handed", "glasses", "phone number", "birthday", "studied"]

facts = unused[unused["type"] == "fact"].copy()
facts = facts[~facts["text"].str.lower().str.contains("|".join(durable_keywords))]

facts["last_accessed_at"] = pd.to_datetime(facts["last_accessed_at"])
NOW = datetime(2026, 7, 29)
facts["days_since_accessed"] = (NOW - facts["last_accessed_at"]).dt.days

likely_delete_facts = facts[(facts["days_since_accessed"] > 40) | (facts["feedback"] == "negative")]

sample_size = min(35, len(likely_delete_facts))
new_sample = likely_delete_facts.sample(n=sample_size, random_state=11).copy()

new_sample["human_label"] = ""
new_sample["human_score"] = ""
new_sample["notes"] = ""

if __name__ == "__main__":
    new_sample.to_csv("data/labeled/additional_facts_to_label.csv", index=False)
    print(f"Saved {len(new_sample)} generic fact candidates for labeling.")
    print(new_sample[["text", "feedback", "days_since_accessed"]].head(10))

    