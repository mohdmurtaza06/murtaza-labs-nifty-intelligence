import json
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

RESEARCH_PREDICTION_FILE = DATA_DIR / "latest_research_prediction.json"

TRACKER_FILE = DATA_DIR / "prediction_history.csv"


# ============================================================
# LOAD CURRENT PREDICTION
# ============================================================

print("=" * 60)
print("RESEARCH PREDICTION TRACKER")
print("=" * 60)

print("\nLoading latest research-only prediction...")

if RESEARCH_PREDICTION_FILE.exists():
    with RESEARCH_PREDICTION_FILE.open(encoding="utf-8") as file:
        production = json.load(file)

    required = [
        "prediction_date", "nifty_close", "prediction", "probability_up",
        "probability_down", "confidence", "target_horizon",
    ]
    missing = [key for key in required if key not in production]
    if missing:
        raise ValueError(
            "Research prediction is missing required fields: "
            + ", ".join(missing)
        )

    if production.get("trading_mode") != "RESEARCH_ONLY":
        raise ValueError("Prediction artifact is not explicitly marked research-only.")
    if production.get("live_orders_enabled") is not False:
        raise ValueError("Research prediction must explicitly keep live orders disabled.")

    prediction = str(production["prediction"]).upper()
    if prediction not in {"UP", "DOWN"}:
        raise ValueError(f"Unsupported prediction label: {prediction!r}")
    current = pd.DataFrame([{
        "Date": production["prediction_date"],
        "NIFTY_Close": production["nifty_close"],
        "Prediction": prediction,
        "Probability_UP": production["probability_up"],
        "Probability_DOWN": production["probability_down"],
        "Confidence": production["confidence"],
        "Target_Horizon": production["target_horizon"],
        "Signal": "BULLISH" if prediction == "UP" else "BEARISH",
    }])
    source = "research candidate"
else:
    raise FileNotFoundError(
        "No research prediction artifact found. Run 15_live_prediction_production.py "
        "after creating a candidate model. Legacy prediction files are not accepted."
    )

current["Date"] = pd.to_datetime(
    current["Date"]
)

print(f"Latest research prediction loaded ({source}).")


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

    # Historical rows created before target horizon was recorded cannot be
    # silently mixed with new execution assumptions.
    if "Target_Horizon" not in history.columns:
        history["Target_Horizon"] = "legacy_unknown"
    else:
        history["Target_Horizon"] = history["Target_Horizon"].fillna(
            "legacy_unknown"
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
            "Target_Horizon",
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

    "Target_Horizon": [
        current["Target_Horizon"].iloc[0]
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
        (
            history["Prediction_Date"] == prediction_date
        )
        & (
            history["Target_Horizon"] == current["Target_Horizon"].iloc[0]
        )
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
