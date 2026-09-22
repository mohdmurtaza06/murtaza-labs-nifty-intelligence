import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# ==================================================
# LOAD DATA
# ==================================================

data = pd.read_csv(
    DATA_DIR / "nifty50_external_features.csv",
    index_col="Date",
    parse_dates=True
)


# ==================================================
# TECHNICAL FEATURES
# ==================================================

technical_features = [
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
# EXTERNAL FEATURES
# ==================================================

external_features = [
    "India_VIX_Return_1D",
    "India_VIX_Return_5D",

    "NIFTY_Bank_Return_1D",
    "NIFTY_Bank_Return_5D",

    "SP500_Return_1D",
    "SP500_Return_5D",

    "USD_INR_Return_1D",
    "USD_INR_Return_5D",

    "Gold_Return_1D",
    "Gold_Return_5D",

    "Crude_Oil_Return_1D"
]


# ==================================================
# ALL FEATURES
# ==================================================

all_features = (
    technical_features
    + external_features
)


# ==================================================
# CLEAN DATA
# ==================================================

model_data = data[
    all_features + ["Target", "Close"]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()

model_data["Target"] = (
    model_data["Target"]
    .astype(int)
)


# ==================================================
# TIME-BASED SPLIT
# ==================================================

split_index = int(
    len(model_data) * 0.8
)

train = model_data.iloc[:split_index]
test = model_data.iloc[split_index:]


# ==================================================
# FUNCTION TO TRAIN MODEL
# ==================================================

def train_xgboost(features, name):

    X_train = train[features]
    y_train = train["Target"]

    X_test = test[features]
    y_test = test["Target"]

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

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\n{name}")
    print("=" * 40)

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return model, accuracy


# ==================================================
# MODEL A
# ==================================================

technical_model, technical_accuracy = train_xgboost(
    technical_features,
    "TECHNICAL ONLY"
)


# ==================================================
# MODEL B
# ==================================================

external_model, external_accuracy = train_xgboost(
    all_features,
    "TECHNICAL + EXTERNAL"
)


# ==================================================
# COMPARISON
# ==================================================

improvement = (
    external_accuracy
    - technical_accuracy
)


print("\n")
print("========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    f"\nTechnical only: "
    f"{technical_accuracy * 100:.2f}%"
)

print(
    f"Technical + external: "
    f"{external_accuracy * 100:.2f}%"
)

print(
    f"\nAccuracy change: "
    f"{improvement * 100:+.2f} percentage points"
)


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.Series(
    external_model.feature_importances_,
    index=all_features
).sort_values(
    ascending=False
)

print("\n")
print("========================================")
print("TOP EXTERNAL MODEL FEATURES")
print("========================================")

print(
    importance.head(15)
)
