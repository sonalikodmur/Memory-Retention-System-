import pandas as pd

# Load CSV file
df = pd.read_csv("memory_retention_big_dataset_1000.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

print(df.describe())
print(df[["frequency_of_use","recency","retention_score"]].describe())
print(df["action"].value_counts())

print(df.groupby("action")["retention_score"].mean())
print(df.groupby("memory_type")["retention_score"].mean())

reshaped_df = df[[
    "memory_id",
    "memory_text",
    "retention_score",
    "action"
]]
#RESHAPING THE DATA 
print(reshaped_df.head())

df_indexed = df.set_index("memory_id")

print(df_indexed.head())

long_df = df.melt(
    id_vars=["memory_id", "memory_text"],
    value_vars=["frequency_of_use", "recency", "retention_score"],
    var_name="feature",
    value_name="value"
)
#FILTERING THE DATA
print(long_df.head(10))

high_retention = df[df["retention_score"] >= 0.70]

print(high_retention)

keep_data = df[df["action"] == "KEEP"]

print(keep_data.head())

filtered_data = df[
    (df["frequency_of_use"] >= 10) &
    (df["retention_score"] >= 0.70)
]

print(filtered_data)

selected_actions = df[
    df["action"].isin(["KEEP", "COMPRESS"])
]

print(selected_actions.head())

action_df = pd.DataFrame({
    "action": ["KEEP", "COMPRESS", "MERGE", "DELETE"],
    "action_description": [
        "Keep memory permanently",
        "Summarize memory",
        "Combine with similar memory",
        "Remove memory"
    ]
})

#MERGING THE DATASET
print(action_df)
merged_df = pd.merge(
    df,
    action_df,
    on="action",
    how="left"
)

print(merged_df.head())

#MISSING VALUES
print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())
#clean the missing value row
df_clean = df.dropna()

print(df_clean.shape)

#fill with data
df["retention_score"] = df["retention_score"].fillna(
    df["retention_score"].mean()
)

#using median 
df["frequency_of_use"] = df["frequency_of_use"].fillna(
    df["frequency_of_use"].median()
)

#fill missing categories values
df["action"] = df["action"].fillna("UNKNOWN")
df["user_feedback"] = df["user_feedback"].fillna("None")

#FINAL DATASET
print("\nFinal Dataset:")
print(df.head())