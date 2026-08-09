import pandas as pd
df = pd.read_csv("data/labeled/memories_labeled.csv")
print("Total rows:", len(df))
print("Unique memory_ids:", df["memory_id"].nunique())
dupes = df[df["memory_id"].duplicated(keep=False)].sort_values("memory_id")
print("\nDuplicate memory_id rows:")
print(dupes[["memory_id", "text", "human_score"]])