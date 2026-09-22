import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


DATA_FILE = DATA_DIR / "live_market_data.csv"
MODEL_FILE = MODEL_DIR / "nifty50_final_xgboost.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("NIFTY 50 LIVE PREDICTION")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")


# ============================================================
# LOAD LIVE DATA
# ============================================================

print("\nLoading live market data...")

data = pd.read_csv(
    DATA_FILE,
    index_col="Date",
    parse_dates=True
)

print(
    f"Dataset loaded: {data.shape}"
)


# ============================================================
# GET MODEL FEATURES
# ============================================================

print("\nChecking model features...")

if hasattr(model, "feature_names_in_"):

    required_features = list(
        model.feature_names_in_
    )

else:

    print(
        "WARNING: Model does not contain feature names."
    )

    raise ValueError(
        "Cannot safely determine model features."
    )


print(
    f"Number of model features: "
    f"{len(required_features)}"
)


# ============================================================
# CHECK MISSING FEATURES
# ============================================================

missing_features = [
    feature
    for feature in required_features
    if feature not in data.columns
]


if missing_features:

    print("\nMISSING FEATURES:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "\nLive dataset is missing features "
        "required by the trained model."
    )


print("All required features found!")


# ============================================================
# SELECT LATEST ROW
# ============================================================

latest = data.iloc[-1:].copy()


# ============================================================
# PREPARE INPUT
# ============================================================

X_latest = latest[
    required_features
].copy()


# ============================================================
# CLEAN DATA
# ============================================================

X_latest = X_latest.replace(
    [np.inf, -np.inf],
    np.nan
)


if X_latest.isna().any().any():

    print(
        "\nWARNING: Missing values detected."
    )

    print(
        X_latest.isna().sum()[
            X_latest.isna().sum() > 0
        ]
    )

    raise ValueError(
        "Latest row contains missing features."
    )


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    X_latest
)[0]


probabilities = model.predict_proba(
    X_latest
)[0]


prob_down = probabilities[0]

prob_up = probabilities[1]


# ============================================================
# SIGNAL
# ============================================================

if prediction == 1:

    signal = "BULLISH"
    direction = "UP"

else:

    signal = "BEARISH"
    direction = "DOWN"


confidence = max(
    prob_up,
    prob_down
)


# ============================================================
# LATEST MARKET INFORMATION
# ============================================================

latest_date = latest.index[-1]

latest_close = latest["Close"].iloc[0]


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("LATEST NIFTY 50 PREDICTION")
print("=" * 60)

print(
    f"\nData date:"
)

print(
    latest_date.strftime("%Y-%m-%d")
)


print(
    f"\nLatest NIFTY Close:"
)

print(
    f"{latest_close:,.2f}"
)


print(
    "\nPrediction:"
)

print(
    f"{direction}"
)


print(
    f"\nProbability UP:"
)

print(
    f"{prob_up * 100:.2f}%"
)


print(
    f"\nProbability DOWN:"
)

print(
    f"{prob_down * 100:.2f}%"
)


print(
    f"\nModel confidence:"
)

print(
    f"{confidence * 100:.2f}%"
)


print(
    "\nSignal:"
)

print(
    f"{signal}"
)


# ============================================================
# SAVE RESULT
# ============================================================

result = pd.DataFrame({

    "Date": [
        latest_date
    ],

    "NIFTY_Close": [
        latest_close
    ],

    "Prediction": [
        direction
    ],

    "Probability_UP": [
        prob_up
    ],

    "Probability_DOWN": [
        prob_down
    ],

    "Confidence": [
        confidence
    ],

    "Signal": [
        signal
    ]

})


output_file = (
    DATA_DIR /
    "live_prediction.csv"
)


result.to_csv(
    output_file,
    index=False
)


print(
    "\nPrediction saved to:"
)

print(
    output_file
)


print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)
