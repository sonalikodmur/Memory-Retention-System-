import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv("data/processed/training_features.csv")

feature_cols = [c for c in df.columns if c not in ["memory_id", "text", "human_score", "human_label"]]
X = df[feature_cols]
y = df["human_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5

print(f"Test MAE: {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")

importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

if __name__ == "__main__":
    joblib.dump(model, "src/model/saved_models/retention_model.pkl")
    print("\nModel saved to src/model/saved_models/retention_model.pkl")