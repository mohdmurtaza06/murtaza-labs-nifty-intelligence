import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ==================================================
# LOAD DATA
# ==================================================

data = pd.read_csv(
    DATA_DIR / "nifty50_features.csv",
    index_col="Date",
    parse_dates=True
)

# ==================================================
# FEATURES
# ==================================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "Previous_Return",
    "MA_5",
    "MA_20",
    "MA_50",
    "Volatility_10",
    "Volume_Change"
]

model_data = data[features + ["Target"]].copy()

# Convert to numeric
for column in features:
    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce"
    )

model_data["Target"] = pd.to_numeric(
    model_data["Target"],
    errors="coerce"
)

# Remove infinity
model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove missing rows
model_data = model_data.dropna()

# ==================================================
# X / Y
# ==================================================

X = model_data[features]
y = model_data["Target"].astype(int)

# ==================================================
# TIME-BASED SPLIT
# ==================================================

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nTesting period:")
print(X_test.index.min(), "to", X_test.index.max())

# ==================================================
# XGBOOST
# ==================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training complete!")

# ==================================================
# PREDICTION
# ==================================================

predictions = model.predict(X_test)

# Probability of UP
probabilities = model.predict_proba(X_test)[:, 1]

# ==================================================
# RESULTS
# ==================================================

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("XGBOOST RESULTS")
print("================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)