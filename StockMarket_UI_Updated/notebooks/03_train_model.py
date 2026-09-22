import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

print("Original dataset:", data.shape)

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

# Keep only features + target
model_data = data[features + ["Target"]].copy()

# Convert everything to numeric
for column in features:
    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce"
    )

model_data["Target"] = pd.to_numeric(
    model_data["Target"],
    errors="coerce"
)

# Replace infinity with NaN
model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows containing missing values
before = len(model_data)

model_data = model_data.dropna()

after = len(model_data)

print("Rows removed because of missing/infinite values:", before - after)
print("Clean dataset:", model_data.shape)

# ==================================================
# X AND Y
# ==================================================

X = model_data[features]
y = model_data["Target"].astype(int)

# ==================================================
# TIME-BASED TRAIN/TEST SPLIT
# ==================================================

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n================================")
print("TRAIN / TEST SPLIT")
print("================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nTesting period:")
print(X_test.index.min(), "to", X_test.index.max())

# ==================================================
# RANDOM FOREST
# ==================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete!")

# ==================================================
# PREDICTIONS
# ==================================================

predictions = model.predict(X_test)

# ==================================================
# EVALUATION
# ==================================================

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))