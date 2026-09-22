import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


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
# CLEAN DATA
# ==================================================

model_data = data[
    features + ["Target", "Close"]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()

model_data["Target"] = model_data["Target"].astype(int)


# ==================================================
# WALK-FORWARD SETTINGS
# ==================================================

# First 60% is initial training history
initial_train_size = int(len(model_data) * 0.60)

# Remaining data is tested in yearly-ish chunks
test_size = 252

results = []

start = initial_train_size


# ==================================================
# WALK-FORWARD LOOP
# ==================================================

while start < len(model_data):

    train_end = start
    test_end = min(
        start + test_size,
        len(model_data)
    )

    train_data = model_data.iloc[:train_end]
    test_data = model_data.iloc[start:test_end].copy()

    X_train = train_data[features]
    y_train = train_data["Target"]

    X_test = test_data[features]
    y_test = test_data["Target"]

    print("\n========================================")
    print("WALK-FORWARD PERIOD")
    print("========================================")

    print(
        "Training:",
        train_data.index.min(),
        "to",
        train_data.index.max()
    )

    print(
        "Testing:",
        test_data.index.min(),
        "to",
        test_data.index.max()
    )

    # ==================================================
    # MODEL
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

    model.fit(X_train, y_train)

    # ==================================================
    # PREDICTIONS
    # ==================================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    test_data["Prediction"] = predictions

    # ==================================================
    # NEXT-DAY RETURN
    # ==================================================

    test_data["Next_Return"] = (
        test_data["Close"].shift(-1)
        / test_data["Close"]
        - 1
    )

    test_data = test_data.dropna(
        subset=["Next_Return"]
    )

    # ==================================================
    # STRATEGY
    # ==================================================

    test_data["Strategy_Return"] = (
        test_data["Prediction"]
        * test_data["Next_Return"]
    )

    # Transaction costs
    transaction_cost = 0.0005

    test_data["Position_Change"] = (
        test_data["Prediction"]
        .diff()
        .abs()
        .fillna(0)
    )

    test_data["Trading_Cost"] = (
        test_data["Position_Change"]
        * transaction_cost
    )

    test_data["Net_Return"] = (
        test_data["Strategy_Return"]
        - test_data["Trading_Cost"]
    )

    # ==================================================
    # PERIOD PERFORMANCE
    # ==================================================

    strategy_return = (
        (1 + test_data["Net_Return"]).prod()
        - 1
    )

    buy_hold_return = (
        (1 + test_data["Next_Return"]).prod()
        - 1
    )

    results.append({
        "Test_Start": test_data.index.min(),
        "Test_End": test_data.index.max(),
        "Accuracy": accuracy,
        "Strategy_Return": strategy_return,
        "Buy_Hold_Return": buy_hold_return,
        "Trades": int(
            test_data["Position_Change"].sum()
        )
    })

    start = test_end


# ==================================================
# RESULTS TABLE
# ==================================================

results_df = pd.DataFrame(results)

print("\n\n========================================")
print("WALK-FORWARD RESULTS")
print("========================================")

print(
    results_df.to_string(index=False)
)


# ==================================================
# OVERALL RESULTS
# ==================================================

total_strategy = (
    (1 + results_df["Strategy_Return"]).prod()
    - 1
)

total_buy_hold = (
    (1 + results_df["Buy_Hold_Return"]).prod()
    - 1
)

average_accuracy = (
    results_df["Accuracy"].mean()
)

total_trades = (
    results_df["Trades"].sum()
)


print("\n========================================")
print("OVERALL WALK-FORWARD RESULTS")
print("========================================")

print(
    f"\nAverage accuracy: "
    f"{average_accuracy * 100:.2f}%"
)

print(
    f"\nStrategy cumulative return: "
    f"{total_strategy * 100:.2f}%"
)

print(
    f"\nBuy & Hold cumulative return: "
    f"{total_buy_hold * 100:.2f}%"
)

print(
    f"\nTotal position changes: "
    f"{total_trades}"
)


# ==================================================
# SAVE
# ==================================================

output_file = (
    DATA_DIR / "walk_forward_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)