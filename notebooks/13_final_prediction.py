import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = DATA_DIR / "nifty50_external_features.csv"

print("=" * 60)
print("NIFTY 50 FINAL PREDICTION MODEL")
print("=" * 60)

print("\nLoading dataset...")

data = pd.read_csv(
    DATA_FILE,
    index_col="Date",
    parse_dates=True
)

print(f"Dataset loaded: {data.shape}")


# ============================================================
# TECHNICAL FEATURES
# ============================================================

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


# ============================================================
# EXTERNAL FEATURES
# ============================================================

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


# ============================================================
# ALL FEATURES
# ============================================================

all_features = (
    technical_features
    + external_features
)


print("\nNumber of features:", len(all_features))


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in all_features
    if feature not in data.columns
]

if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Some required features are missing from the dataset."
    )


print("All required features found.")


# ============================================================
# CLEAN DATA
# ============================================================

model_data = data[
    all_features + ["Target", "Close"]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# TRAINING DATA
# ============================================================

training_data = model_data.dropna(
    subset=all_features + ["Target"]
).copy()

training_data["Target"] = (
    training_data["Target"]
    .astype(int)
)


print("\nTraining samples:", len(training_data))


# ============================================================
# FINAL MODEL
# ============================================================

print("\nTraining final XGBoost model...")

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


model.fit(
    training_data[all_features],
    training_data["Target"]
)

print("Training complete!")


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR /
    "nifty50_final_xgboost.pkl"
)

joblib.dump(
    model,
    model_file
)

print("\nModel saved to:")
print(model_file)


# ============================================================
# FIND LATEST USABLE ROW
# ============================================================

prediction_data = model_data.dropna(
    subset=all_features
).copy()

latest_date = prediction_data.index[-1]

latest_row = prediction_data.iloc[-1]


# ============================================================
# PREDICTION
# ============================================================

X_latest = pd.DataFrame(
    [latest_row[all_features].values],
    columns=all_features
)


prediction = model.predict(
    X_latest
)[0]


probabilities = model.predict_proba(
    X_latest
)[0]


probability_down = probabilities[0]

probability_up = probabilities[1]


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("LATEST NIFTY 50 PREDICTION")
print("=" * 60)

print("\nData date:")
print(latest_date.strftime("%Y-%m-%d"))

print("\nLatest NIFTY Close:")
print(f"{latest_row['Close']:,.2f}")


print("\nPrediction:")

if prediction == 1:

    print("🟢 UP")

else:

    print("🔴 DOWN")


print("\nProbability UP:")
print(f"{probability_up * 100:.2f}%")


print("\nProbability DOWN:")
print(f"{probability_down * 100:.2f}%")


# ============================================================
# CONFIDENCE
# ============================================================

confidence = max(
    probability_up,
    probability_down
)


print("\nModel confidence:")
print(f"{confidence * 100:.2f}%")


# ============================================================
# SIMPLE SIGNAL
# ============================================================

print("\nSignal:")

if probability_up >= 0.60:

    print("🟢 BULLISH")

elif probability_down >= 0.60:

    print("🔴 BEARISH")

else:

    print("🟡 NEUTRAL")


# ============================================================
# TOP FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=all_features
)

importance = importance.sort_values(
    ascending=False
)


print("\n")
print("=" * 60)
print("TOP 10 FEATURES")
print("=" * 60)

for feature, value in importance.head(10).items():

    print(
        f"{feature:<30} "
        f"{value:.4f}"
    )


# ============================================================
# SAVE PREDICTION
# ============================================================

prediction_result = pd.DataFrame({

    "Date": [latest_date],

    "Close": [
        latest_row["Close"]
    ],

    "Prediction": [
        "UP" if prediction == 1
        else "DOWN"
    ],

    "Probability_UP": [
        probability_up
    ],

    "Probability_DOWN": [
        probability_down
    ],

    "Confidence": [
        confidence
    ]

})


prediction_file = (
    DATA_DIR /
    "latest_prediction.csv"
)

prediction_result.to_csv(
    prediction_file,
    index=False
)


print("\nPrediction saved to:")
print(prediction_file)

print("\n")
print("=" * 60)
print("FINAL PREDICTION COMPLETE")
print("=" * 60)