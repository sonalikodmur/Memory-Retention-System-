import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. Load training features
# ============================================================

features_df = pd.read_csv(
    "data/processed/training_features.csv"
)

# Save original columns BEFORE merging additional labeled data.
# These are the columns that belong to the ML feature dataset.
original_feature_cols = features_df.columns.tolist()


# ============================================================
# 2. Load labeled data
# ============================================================

raw_labeled = pd.read_csv(
    "data/labeled/memories_labeled.csv",
    parse_dates=["created_at", "last_accessed_at"]
)

features_df = features_df.merge(
    raw_labeled[
        [
            "memory_id",
            "type",
            "created_at",
            "last_accessed_at",
            "access_count",
            "feedback"
        ]
    ],
    on="memory_id",
    how="left"
)


# ============================================================
# 3. Load trained ML model
# ============================================================

model = joblib.load(
    "src/model/saved_models/retention_model.pkl"
)


# ============================================================
# 4. Select ONLY features used by the ML model
# ============================================================

feature_cols = [
    c for c in original_feature_cols
    if c not in [
        "memory_id",
        "text",
        "human_score",
        "human_label"
    ]
]

print("=== ML Features ===")
print(feature_cols)


# ============================================================
# 5. Generate ML predictions
# ============================================================

features_df["ml_score"] = model.predict(
    features_df[feature_cols]
)


# ============================================================
# 6. Load heuristic scores
# ============================================================

heuristic_df = pd.read_csv(
    "data/processed/memories_with_heuristic_score.csv"
)

heuristic_df["created_at"] = pd.to_datetime(
    heuristic_df["created_at"]
)

heuristic_df["last_accessed_at"] = pd.to_datetime(
    heuristic_df["last_accessed_at"]
)


# ============================================================
# 7. Match ML results with heuristic results
# ============================================================

merge_keys = [
    "text",
    "type",
    "created_at",
    "last_accessed_at",
    "access_count",
    "feedback"
]

comparison = features_df.merge(
    heuristic_df[
        merge_keys + ["heuristic_score"]
    ],
    on=merge_keys,
    how="left"
)

print(f"\nRow count check: features_df={len(features_df)}, comparison={len(comparison)}")
# ============================================================
# 8. Check for unmatched rows
# ============================================================

dropped = comparison[
    comparison["heuristic_score"].isna()
]

if len(dropped) > 0:
    print(
        f"Dropping {len(dropped)} row(s) with "
        "corrupted/missing heuristic score "
        "(likely an Excel auto-format issue):"
    )

    print(
        dropped[
            ["memory_id", "text"]
        ]
    )


comparison = comparison.dropna(
    subset=["heuristic_score"]
)


# ============================================================
# 9. Calculate ML metrics
# ============================================================

mae_ml = mean_absolute_error(
    comparison["human_score"],
    comparison["ml_score"]
)

rmse_ml = mean_squared_error(
    comparison["human_score"],
    comparison["ml_score"]
) ** 0.5


# ============================================================
# 10. Calculate heuristic metrics
# ============================================================

mae_heuristic = mean_absolute_error(
    comparison["human_score"],
    comparison["heuristic_score"]
)

rmse_heuristic = mean_squared_error(
    comparison["human_score"],
    comparison["heuristic_score"]
) ** 0.5


# ============================================================
# 11. Print overall results
# ============================================================

print("\n=== Heuristic baseline ===")
print(
    f"MAE: {mae_heuristic:.3f}  "
    f"RMSE: {rmse_heuristic:.3f}"
)

print("\n=== ML model ===")
print(
    f"MAE: {mae_ml:.3f}  "
    f"RMSE: {rmse_ml:.3f}"
)


# ============================================================
# 12. Calculate row-level errors
# ============================================================

comparison["ml_error"] = (
    comparison["human_score"]
    - comparison["ml_score"]
).abs()

comparison["heuristic_error"] = (
    comparison["human_score"]
    - comparison["heuristic_score"]
).abs()

# Positive = ML performed better
# Negative = heuristic performed better
comparison["ml_advantage"] = (
    comparison["heuristic_error"]
    - comparison["ml_error"]
)


# ============================================================
# 13. Recover memory type
# ============================================================

type_cols = [
    c for c in comparison.columns
    if c.startswith("type_")
]

if type_cols:
    comparison["type"] = (
        comparison[type_cols]
        .idxmax(axis=1)
        .str.replace(
            "type_",
            "",
            regex=False
        )
    )


# ============================================================
# 14. Save comparison results
# ============================================================

if __name__ == "__main__":

    comparison.to_csv(
        "evaluation/results/model_comparison.csv",
        index=False
    )

    print(
        "\nResults saved to: "
        "evaluation/results/model_comparison.csv"
    )


    # ========================================================
    # 15. Rows where ML helped most
    # ========================================================

    print(
        "\n=== Rows where ML helped most "
        "(vs heuristic) ==="
    )

    print(
        comparison
        .sort_values(
            "ml_advantage",
            ascending=False
        )[
            [
                "text",
                "type",
                "human_score",
                "heuristic_score",
                "ml_score"
            ]
        ]
        .head(5)
    )


    # ========================================================
    # 16. Rows where heuristic performed better
    # ========================================================

    print(
        "\n=== Rows where heuristic actually "
        "did better ==="
    )

    print(
        comparison
        .sort_values(
            "ml_advantage",
            ascending=True
        )[
            [
                "text",
                "type",
                "human_score",
                "heuristic_score",
                "ml_score"
            ]
        ]
        .head(5)
    )