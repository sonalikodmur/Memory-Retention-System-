import pandas as pd

df = pd.read_csv("data/labeled/memories_labeled.csv")

# average human_score across duplicate memory_ids, keep everything else from first occurrence
agg = df.groupby("memory_id").agg({
    "text": "first", "type": "first", "created_at": "first", "last_accessed_at": "first",
    "access_count": "first", "feedback": "first",
    "heuristic_score": "first",
    "human_score": "mean",
    "notes": "first"
}).reset_index()

agg["human_score"] = agg["human_score"].round(2)
agg["human_label"] = agg["human_score"].apply(lambda s: "keep" if s >= 0.5 else "delete")

agg.to_csv("data/labeled/memories_labeled.csv", index=False)
print(f"Resolved to {len(agg)} unique memories (was {len(df)})")
print(f"Duplicates fixed: {len(df) - len(agg)}")