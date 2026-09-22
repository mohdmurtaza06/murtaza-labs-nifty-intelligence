import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier


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
# FEATURES
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

all_features = technical_features + external_features


# ==================================================
# CLEAN
# ==================================================

model_data = data[
    all_features + ["Target", "Close"]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()

model_data["Target"] = model_data["Target"].astype(int)


# ==================================================
# SETTINGS
# ==================================================

initial_train_size = int(
    len(model_data) * 0.60
)

test_size = 252

# Simple research assumption
TRANSACTION_COST = 0.0005


# ==================================================
# MODEL FUNCTION
# ==================================================

def create_model():

    return XGBClassifier(
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


# ==================================================
# WALK FORWARD
# ==================================================

results = []

start = initial_train_size

while start < len(model_data):

    train_end = start
    test_end = min(
        start + test_size,
        len(model_data)
    )

    train_data = model_data.iloc[:train_end].copy()
    test_data = model_data.iloc[start:test_end].copy()

    print("\n========================================")
    print("TEST PERIOD")
    print("========================================")

    print(
        test_data.index.min(),
        "to",
        test_data.index.max()
    )

    # ------------------------------------------------
    # TECHNICAL MODEL
    # ------------------------------------------------

    technical_model = create_model()

    technical_model.fit(
        train_data[technical_features],
        train_data["Target"]
    )

    technical_prediction = technical_model.predict(
        test_data[technical_features]
    )

    # ------------------------------------------------
    # EXTERNAL MODEL
    # ------------------------------------------------

    external_model = create_model()

    external_model.fit(
        train_data[all_features],
        train_data["Target"]
    )

    external_prediction = external_model.predict(
        test_data[all_features]
    )

    # ------------------------------------------------
    # NEXT DAY RETURN
    # ------------------------------------------------

    test_data["Next_Return"] = (
        test_data["Close"].shift(-1)
        / test_data["Close"]
        - 1
    )

    test_data = test_data.dropna(
        subset=["Next_Return"]
    )

    # ------------------------------------------------
    # POSITIONS
    # ------------------------------------------------

    test_data["Technical_Position"] = (
        technical_prediction[:len(test_data)]
    )

    test_data["External_Position"] = (
        external_prediction[:len(test_data)]
    )

    # ------------------------------------------------
    # RAW RETURNS
    # ------------------------------------------------

    test_data["Technical_Return"] = (
        test_data["Technical_Position"]
        * test_data["Next_Return"]
    )

    test_data["External_Return"] = (
        test_data["External_Position"]
        * test_data["Next_Return"]
    )

    # ------------------------------------------------
    # TRANSACTION COSTS
    # ------------------------------------------------

    test_data["Technical_Change"] = (
        test_data["Technical_Position"]
        .diff()
        .abs()
        .fillna(0)
    )

    test_data["External_Change"] = (
        test_data["External_Position"]
        .diff()
        .abs()
        .fillna(0)
    )

    test_data["Technical_Cost"] = (
        test_data["Technical_Change"]
        * TRANSACTION_COST
    )

    test_data["External_Cost"] = (
        test_data["External_Change"]
        * TRANSACTION_COST
    )

    # ------------------------------------------------
    # NET RETURNS
    # ------------------------------------------------

    test_data["Technical_Net"] = (
        test_data["Technical_Return"]
        - test_data["Technical_Cost"]
    )

    test_data["External_Net"] = (
        test_data["External_Return"]
        - test_data["External_Cost"]
    )

    # ------------------------------------------------
    # PERIOD RETURNS
    # ------------------------------------------------

    technical_return = (
        (1 + test_data["Technical_Net"]).prod()
        - 1
    )

    external_return = (
        (1 + test_data["External_Net"]).prod()
        - 1
    )

    buy_hold_return = (
        (1 + test_data["Next_Return"]).prod()
        - 1
    )

    # ------------------------------------------------
    # SHARPE
    # ------------------------------------------------

    technical_mean = (
        test_data["Technical_Net"].mean()
    )

    external_mean = (
        test_data["External_Net"].mean()
    )

    technical_std = (
        test_data["Technical_Net"].std()
    )

    external_std = (
        test_data["External_Net"].std()
    )

    technical_sharpe = (
        technical_mean / technical_std
        * np.sqrt(252)
        if technical_std != 0
        else 0
    )

    external_sharpe = (
        external_mean / external_std
        * np.sqrt(252)
        if external_std != 0
        else 0
    )

    # ------------------------------------------------
    # STORE
    # ------------------------------------------------

    results.append({
        "Test_Start":
            test_data.index.min(),

        "Test_End":
            test_data.index.max(),

        "Technical_Return":
            technical_return,

        "External_Return":
            external_return,

        "Buy_Hold_Return":
            buy_hold_return,

        "Technical_Sharpe":
            technical_sharpe,

        "External_Sharpe":
            external_sharpe,

        "Technical_Trades":
            int(test_data["Technical_Change"].sum()),

        "External_Trades":
            int(test_data["External_Change"].sum())
    })

    start = test_end


# ==================================================
# RESULTS
# ==================================================

results_df = pd.DataFrame(results)

print("\n\n========================================")
print("WALK-FORWARD BACKTEST")
print("========================================")

print(
    results_df.to_string(index=False)
)


# ==================================================
# COMPOUND RETURNS
# ==================================================

technical_total = (
    (1 + results_df["Technical_Return"])
    .prod()
    - 1
)

external_total = (
    (1 + results_df["External_Return"])
    .prod()
    - 1
)

buy_hold_total = (
    (1 + results_df["Buy_Hold_Return"])
    .prod()
    - 1
)


# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("OVERALL PERFORMANCE")
print("========================================")

print(
    f"\nTechnical strategy: "
    f"{technical_total * 100:.2f}%"
)

print(
    f"External strategy: "
    f"{external_total * 100:.2f}%"
)

print(
    f"Buy & Hold: "
    f"{buy_hold_total * 100:.2f}%"
)

print(
    "\nAverage technical Sharpe:",
    f"{results_df['Technical_Sharpe'].mean():.3f}"
)

print(
    "Average external Sharpe:",
    f"{results_df['External_Sharpe'].mean():.3f}"
)

print(
    "\nTotal technical trades:",
    results_df["Technical_Trades"].sum()
)

print(
    "Total external trades:",
    results_df["External_Trades"].sum()
)


# ==================================================
# SAVE
# ==================================================

output_file = (
    DATA_DIR /
    "walk_forward_backtest_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)