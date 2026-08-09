import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("data/raw/synthetic_memories.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(df["text"].tolist(), convert_to_tensor=True)
similarity_matrix = util.cos_sim(embeddings, embeddings)

THRESHOLD = 0.6
candidates = []

n = len(df)
for i in range(n):
    for j in range(i + 1, n):
        score = similarity_matrix[i][j].item()
        if score >= THRESHOLD:
            candidates.append({
                "i": i, "j": j,
                "text_a": df.iloc[i]["text"], "text_b": df.iloc[j]["text"],
                "type_a": df.iloc[i]["type"], "type_b": df.iloc[j]["type"],
                "created_a": df.iloc[i]["created_at"], "created_b": df.iloc[j]["created_at"],
                "similarity": round(score, 3)
            })

candidates_df = pd.DataFrame(candidates)
CONTRADICTION_KEYWORDS = [
    ("prefers", "prefers"),
    ("dislikes", "loves"),
    ("vegetarian", "non-vegetarian"),
    ("dark mode", "light mode"),
    ("tea", "coffee"),
    ("python", "java"),
]

COMPLETION_MARKERS = ["now", "bought", "got", "completed", "fluent"]
IN_PROGRESS_MARKERS = ["wants to", "is learning", "is preparing", "is saving", "next year"]

def classify_pair(row):
    text_a, text_b = row["text_a"].lower(), row["text_b"].lower()
    sim = row["similarity"]

    if text_a == text_b:
        return "DUPLICATE"

    if sim < 0.95:
        has_completion_a = any(m in text_a for m in COMPLETION_MARKERS)
        has_completion_b = any(m in text_b for m in COMPLETION_MARKERS)
        has_progress_a = any(m in text_a for m in IN_PROGRESS_MARKERS)
        has_progress_b = any(m in text_b for m in IN_PROGRESS_MARKERS)

        if (has_progress_a and has_completion_b) or (has_progress_b and has_completion_a):
            return "EVOLVING"

        for kw1, kw2 in CONTRADICTION_KEYWORDS:
            if kw1 in text_a and kw2 in text_b:
                return "CONFLICT"
            if kw1 in text_b and kw2 in text_a:
                return "CONFLICT"

    if sim >= 0.95:
        return "DUPLICATE"

    if sim >= 0.85:
        return "NEAR_DUPLICATE"

    return "RELATED"

candidates_df["action"] = candidates_df.apply(classify_pair, axis=1)
print(candidates_df["action"].value_counts())
print(f"Found {len(candidates_df)} candidate pairs above similarity {THRESHOLD}")

if __name__ == "__main__":
    candidates_df.to_csv("data/processed/merge_candidates.csv", index=False)
    print("\nSample of each category:")
    for action in ["DUPLICATE", "CONFLICT", "EVOLVING", "RELATED"]:
        subset = candidates_df[candidates_df["action"] == action].drop_duplicates(subset=["text_a", "text_b"])
        print(f"\n--- {action} ---")
        print(subset[["text_a", "text_b", "similarity"]].head(4))