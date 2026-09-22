import pandas as pd
import yfinance as yf
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

TRACKER_FILE = DATA_DIR / "prediction_history.csv"


# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================

print("=" * 60)
print("PREDICTION VERIFICATION")
print("=" * 60)

print("\nLoading prediction history...")

history = pd.read_csv(
    TRACKER_FILE,
    parse_dates=["Prediction_Date"]
)

print(
    f"Predictions found: {len(history)}"
)


# ============================================================
# FIND UNRESOLVED PREDICTIONS
# ============================================================

unresolved = history[
    history["Actual_Next_Day"].isna()
].copy()


if unresolved.empty:

    print(
        "\nNo unresolved predictions."
    )

    print(
        "All predictions have already been verified."
    )

    exit()


print(
    f"\nUnresolved predictions: "
    f"{len(unresolved)}"
)


# ============================================================
# DOWNLOAD NIFTY DATA
# ============================================================

print(
    "\nDownloading latest NIFTY data..."
)

nifty = yf.download(
    "^NSEI",
    start=(
        unresolved["Prediction_Date"].min()
        - pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d"),
    end=(
        pd.Timestamp.today()
        + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False
)


# ============================================================
# CLEAN COLUMN FORMAT
# ============================================================

if isinstance(nifty.columns, pd.MultiIndex):

    nifty.columns = nifty.columns.get_level_values(0)


nifty.index = pd.to_datetime(
    nifty.index
).tz_localize(None)


# ============================================================
# VERIFY EACH PREDICTION
# ============================================================

for index, row in unresolved.iterrows():

    prediction_date = (
        row["Prediction_Date"]
    )

    prediction_close = (
        row["NIFTY_Close"]
    )

    prediction = row["Prediction"]


    # --------------------------------------------------------
    # FIND NEXT AVAILABLE TRADING DAY
    # --------------------------------------------------------

    future_dates = nifty.index[
        nifty.index > prediction_date
    ]


    if len(future_dates) == 0:

        print(
            f"\nNo future trading data available "
            f"for {prediction_date.date()}"
        )

        continue


    next_date = future_dates[0]

    next_close = float(
        nifty.loc[next_date, "Close"]
    )


    # --------------------------------------------------------
    # DETERMINE ACTUAL MOVEMENT
    # --------------------------------------------------------

    if next_close > prediction_close:

        actual = "UP"

    elif next_close < prediction_close:

        actual = "DOWN"

    else:

        actual = "FLAT"


    # --------------------------------------------------------
    # CHECK PREDICTION
    # --------------------------------------------------------

    if actual == "FLAT":

        correct = None

    else:

        correct = (
            prediction == actual
        )


    # --------------------------------------------------------
    # UPDATE RECORD
    # --------------------------------------------------------

    history.loc[
        index,
        "Actual_Next_Day"
    ] = actual


    history.loc[
        index,
        "Correct"
    ] = correct


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"Prediction date: "
        f"{prediction_date.date()}"
    )

    print(
        f"Prediction: "
        f"{prediction}"
    )

    print(
        f"Prediction close: "
        f"{prediction_close:,.2f}"
    )

    print(
        f"Next trading day: "
        f"{next_date.date()}"
    )

    print(
        f"Next close: "
        f"{next_close:,.2f}"
    )

    print(
        f"Actual movement: "
        f"{actual}"
    )

    if correct is True:

        print("Result: CORRECT")

    elif correct is False:

        print("Result: WRONG")

    else:

        print("Result: FLAT")


# ============================================================
# SAVE UPDATED HISTORY
# ============================================================

history.to_csv(
    TRACKER_FILE,
    index=False
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

resolved = history[
    history["Correct"].notna()
].copy()


print("\n")
print("=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)


print(
    f"\nTotal predictions: "
    f"{len(history)}"
)

print(
    f"Resolved predictions: "
    f"{len(resolved)}"
)


if len(resolved) > 0:

    accuracy = (
        resolved["Correct"]
        .astype(bool)
        .mean()
    )

    print(
        f"\nVerified accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nCorrect predictions: "
        f"{resolved['Correct'].astype(bool).sum()}"
    )

    print(
        "Wrong predictions: "
        f"{(~resolved['Correct'].astype(bool)).sum()}"
    )

else:

    print(
        "\nNo predictions can be verified yet."
    )


print(
    "\nHistory saved to:"
)

print(
    TRACKER_FILE
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)