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
    DATA_DIR / "nifty50_better_features.csv",
    index_col="Date",
    parse_dates=True
)

print("Dataset:", data.shape)

# ==================================================
# FEATURES
# ==================================================

features = [
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    "Price_to_MA5",
    "Price_to_MA20",
    "Price_to_MA50",

    "High_Low_Range",
    "Open_Close_Return",

    "Volatility_10",
    "Volatility_20",

    "Volume_Change",
    "Relative_Volume",

    "RSI_14",

    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    "BB_Position",

    "ATR_Percent"
]

# ==================================================
# CLEAN
# ==================================================

model_data = data[features + ["Target"]].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()

print("Clean dataset:", model_data.shape)

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

print("\n================================")
print("TRAIN / TEST")
print("================================")

print("Training:", len(X_train))
print("Testing :", len(X_test))

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

probabilities = model.predict_proba(X_test)[:, 1]

# ==================================================
# RESULTS
# ==================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("BETTER FEATURE XGBOOST")
print("================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(
    ascending=False
)

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(importance)