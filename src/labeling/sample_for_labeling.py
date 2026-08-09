import pandas as pd

df = pd.read_csv("data/processed/memories_with_heuristic_score.csv")

sample = df.sample(n=270, random_state=42).copy()

sample["human_label"] = ""      # you will fill: keep / delete
sample["human_score"] = ""      # you will fill: a number 0.0 to 1.0
sample["notes"] = ""            # optional: why you decided this

if __name__ == "__main__":
    sample.to_csv("data/labeled/memories_to_label.csv", index=False)
    print(f"Saved {len(sample)} rows to data/labeled/memories_to_label.csv for manual labeling.")