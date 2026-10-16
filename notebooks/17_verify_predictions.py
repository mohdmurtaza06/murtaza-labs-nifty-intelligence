import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TRACKER_FILE = DATA_DIR / "prediction_history.csv"
EXPECTED_TARGET_HORIZON = "next_session_open_to_close_direction"


def parse_bool(value):
    """Parse CSV booleans, including numeric 1.0/0.0 values from pandas."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


print("=" * 60)
print("PREDICTION VERIFICATION")
print("=" * 60)

if not TRACKER_FILE.exists():
    raise FileNotFoundError(f"Prediction history not found: {TRACKER_FILE}")

history = pd.read_csv(TRACKER_FILE, parse_dates=["Prediction_Date"])
if history.empty:
    print("Prediction history is empty. Nothing to verify.")
    raise SystemExit(0)

# Older records did not preserve the target/execution horizon. Mark them
# unsupported, rather than scoring them against a different outcome definition.
if "Target_Horizon" not in history.columns:
    history["Target_Horizon"] = "legacy_unknown"
else:
    history["Target_Horizon"] = history["Target_Horizon"].fillna("legacy_unknown")

for column in (
    "Actual_Next_Session_Date",
    "Actual_Open",
    "Actual_Close",
    "Actual_Open_Close_Return",
):
    if column not in history.columns:
        history[column] = np.nan

if "Actual_Next_Day" not in history.columns:
    history["Actual_Next_Day"] = np.nan
if "Correct" not in history.columns:
    history["Correct"] = np.nan

legacy_unresolved = (
    history["Actual_Next_Day"].isna()
    & (history["Target_Horizon"] != EXPECTED_TARGET_HORIZON)
)
history.loc[legacy_unresolved, "Actual_Next_Day"] = "UNSUPPORTED_TARGET_HORIZON"
history.loc[legacy_unresolved, "Correct"] = np.nan

unresolved = history[
    history["Actual_Next_Day"].isna()
    & (history["Target_Horizon"] == EXPECTED_TARGET_HORIZON)
].copy()

print(f"Predictions found: {len(history)}")
print(f"Unresolved supported-horizon predictions: {len(unresolved)}")
print(f"Legacy/unsupported records excluded: {int(legacy_unresolved.sum())}")

if unresolved.empty:
    history.to_csv(TRACKER_FILE, index=False)
    print("No supported-horizon predictions need verification.")
    raise SystemExit(0)

print("Downloading NIFTY data needed to resolve the next-session outcomes...")
nifty = yf.download(
    "^NSEI",
    start=(
        unresolved["Prediction_Date"].min() - pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d"),
    end=(pd.Timestamp.today() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False,
)

if nifty.empty:
    raise RuntimeError("NIFTY outcome download returned no rows; history was not updated.")

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
required_columns = {"Open", "Close"}
if not required_columns.issubset(nifty.columns):
    raise RuntimeError("NIFTY data does not contain both Open and Close; history was not updated.")

for index, row in unresolved.iterrows():
    prediction_date = pd.Timestamp(row["Prediction_Date"]).tz_localize(None)
    future_dates = nifty.index[nifty.index > prediction_date]
    if len(future_dates) == 0:
        print(f"No next-session data yet for {prediction_date.date()}. Leaving unresolved.")
        continue

    next_date = future_dates[0]
    next_open = float(nifty.loc[next_date, "Open"])
    next_close = float(nifty.loc[next_date, "Close"])
    if not np.isfinite(next_open) or not np.isfinite(next_close) or next_open <= 0:
        print(f"Invalid open/close data for {next_date.date()}. Leaving unresolved.")
        continue

    actual_return = next_close / next_open - 1.0
    if actual_return > 0:
        actual = "UP"
    elif actual_return < 0:
        actual = "DOWN"
    else:
        actual = "FLAT"

    prediction = str(row["Prediction"]).upper()
    correct = None if actual == "FLAT" else prediction == actual

    history.loc[index, "Actual_Next_Session_Date"] = next_date.strftime("%Y-%m-%d")
    history.loc[index, "Actual_Open"] = next_open
    history.loc[index, "Actual_Close"] = next_close
    history.loc[index, "Actual_Open_Close_Return"] = actual_return
    history.loc[index, "Actual_Next_Day"] = actual
    history.loc[index, "Correct"] = correct

    print(
        f"{prediction_date.date()} -> {next_date.date()} | "
        f"open={next_open:.2f} close={next_close:.2f} | "
        f"actual={actual} prediction={prediction} correct={correct}"
    )

history.to_csv(TRACKER_FILE, index=False)

score_history = history[
    history["Target_Horizon"] == EXPECTED_TARGET_HORIZON
].copy()
resolved = score_history[score_history["Correct"].notna()].copy()
parsed_correct = resolved["Correct"].map(parse_bool).dropna()
unsupported_resolved = history[
    (history["Target_Horizon"] != EXPECTED_TARGET_HORIZON)
    & history["Correct"].notna()
]

print()
print("=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)
print(f"Total predictions: {len(history)}")
print(f"Resolved supported-horizon records: {len(resolved)}")
print(f"Legacy/unsupported resolved records excluded: {len(unsupported_resolved)}")
print(f"Supported verified observations: {len(parsed_correct)}")
if not parsed_correct.empty:
    accuracy = parsed_correct.mean()
    print(f"Verified directional accuracy: {accuracy * 100:.2f}%")
    print(f"Correct predictions: {int(parsed_correct.sum())}")
    print(f"Wrong predictions: {int((~parsed_correct).sum())}")
else:
    print("No supported-horizon verified observations yet.")

print(f"History saved to: {TRACKER_FILE}")
print("=" * 60)
