import json
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

PRODUCTION_PREDICTION_FILE = DATA_DIR / "latest_production_prediction.json"
LEGACY_PREDICTION_FILE = DATA_DIR / "live_prediction.csv"

TRACKER_FILE = DATA_DIR / "prediction_history.csv"


# ============================================================
# LOAD CURRENT PREDICTION
# ============================================================

print("=" * 60)
print("PREDICTION TRACKER")
print("=" * 60)

print("\nLoading latest prediction...")

if PRODUCTION_PREDICTION_FILE.exists():
    with PRODUCTION_PREDICTION_FILE.open(encoding="utf-8") as file:
        production = json.load(file)

    required = [
        "prediction_date", "nifty_close", "prediction", "probability_up",
        "probability_down", "confidence",
    ]
    missing = [key for key in required if key not in production]
    if missing:
        raise ValueError(
            "Production prediction is missing required fields: "
            + ", ".join(missing)
        )

    prediction = str(production["prediction"]).upper()
    current = pd.DataFrame([{
        "Date": production["prediction_date"],
        "NIFTY_Close": production["nifty_close"],
        "Prediction": prediction,
        "Probability_UP": production["probability_up"],
        "Probability_DOWN": production["probability_down"],
        "Confidence": production["confidence"],
        "Signal": "BULLISH" if prediction == "UP" else "BEARISH",
    }])
    source = "production"
elif LEGACY_PREDICTION_FILE.exists():
    current = pd.read_csv(LEGACY_PREDICTION_FILE)
    source = "legacy fallback"
else:
    raise FileNotFoundError(
        "No production or legacy prediction file was found."
    )

current["Date"] = pd.to_datetime(
    current["Date"]
)

print(f"Latest prediction loaded ({source}).")


# ============================================================
# CHECK EXISTING HISTORY
# ============================================================

if TRACKER_FILE.exists():

    history = pd.read_csv(
        TRACKER_FILE
    )

    history["Prediction_Date"] = pd.to_datetime(
        history["Prediction_Date"]
    )

else:

    history = pd.DataFrame(
        columns=[
            "Prediction_Date",
            "NIFTY_Close",
            "Prediction",
            "Probability_UP",
            "Probability_DOWN",
            "Confidence",
            "Signal",
            "Actual_Next_Day",
            "Correct"
        ]
    )


# ============================================================
# CREATE NEW RECORD
# ============================================================

prediction_date = current["Date"].iloc[0]

new_record = pd.DataFrame({

    "Prediction_Date": [
        prediction_date
    ],

    "NIFTY_Close": [
        current["NIFTY_Close"].iloc[0]
    ],

    "Prediction": [
        current["Prediction"].iloc[0]
    ],

    "Probability_UP": [
        current["Probability_UP"].iloc[0]
    ],

    "Probability_DOWN": [
        current["Probability_DOWN"].iloc[0]
    ],

    "Confidence": [
        current["Confidence"].iloc[0]
    ],

    "Signal": [
        current["Signal"].iloc[0]
    ],

    "Actual_Next_Day": [
        None
    ],

    "Correct": [
        None
    ]
})


# ============================================================
# AVOID DUPLICATES
# ============================================================

if not history.empty:

    already_exists = (
        history["Prediction_Date"]
        == prediction_date
    ).any()

else:

    already_exists = False


if already_exists:

    print(
        "\nPrediction for this date already exists."
    )

else:

    history = pd.concat(
        [
            history,
            new_record
        ],
        ignore_index=True
    )

    print(
        "\nNew prediction added."
    )


# ============================================================
# SAVE HISTORY
# ============================================================

history.to_csv(
    TRACKER_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\nPrediction history:")

print(
    history.tail(10).to_string(
        index=False
    )
)


print("\nSaved to:")

print(
    TRACKER_FILE
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
