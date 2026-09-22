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
# CLEAN
# ==================================================

model_data = data[features + ["Target", "Close"]].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()

# ==================================================
# X / Y
# ==================================================

X = model_data[features]
y = model_data["Target"].astype(int)

# ==================================================
# TIME SPLIT
# ==================================================

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_data = model_data.iloc[split_index:].copy()

# ==================================================
# TRAIN MODEL
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

print("Training XGBoost...")

model.fit(X_train, y_train)

print("Training complete!")

# ==================================================
# PREDICTIONS
# ==================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

test_data["Prediction"] = predictions
test_data["Probability_UP"] = probabilities

# ==================================================
# FUTURE RETURN
# ==================================================

# Return from today's close to tomorrow's close

test_data["Next_Return"] = (
    test_data["Close"].shift(-1)
    / test_data["Close"]
    - 1
)

# Remove final test row because it has no tomorrow
test_data = test_data.dropna(
    subset=["Next_Return"]
)

# ==================================================
# STRATEGY
# ==================================================

# 1 = invested in NIFTY
# 0 = cash

test_data["Position"] = test_data["Prediction"]

# Daily strategy return
test_data["Strategy_Return"] = (
    test_data["Position"]
    * test_data["Next_Return"]
)

# ==================================================
# TRANSACTION COST
# ==================================================

# Assumed round-trip trading friction
# This is a simple research assumption, not a live brokerage quote.

TRANSACTION_COST = 0.0005

# Detect changes in position
test_data["Position_Change"] = (
    test_data["Position"]
    .diff()
    .abs()
    .fillna(0)
)

test_data["Trading_Cost"] = (
    test_data["Position_Change"]
    * TRANSACTION_COST
)

# Net strategy return
test_data["Net_Strategy_Return"] = (
    test_data["Strategy_Return"]
    - test_data["Trading_Cost"]
)

# ==================================================
# BUY AND HOLD
# ==================================================

test_data["Buy_Hold_Return"] = (
    test_data["Next_Return"]
)

# ==================================================
# CUMULATIVE RETURNS
# ==================================================

test_data["Strategy_Equity"] = (
    1 + test_data["Net_Strategy_Return"]
).cumprod()

test_data["Buy_Hold_Equity"] = (
    1 + test_data["Buy_Hold_Return"]
).cumprod()

# ==================================================
# PERFORMANCE METRICS
# ==================================================

strategy_total_return = (
    test_data["Strategy_Equity"].iloc[-1] - 1
)

buy_hold_total_return = (
    test_data["Buy_Hold_Equity"].iloc[-1] - 1
)

# Annualized return
days = len(test_data)

strategy_annual_return = (
    (1 + strategy_total_return)
    ** (252 / days)
    - 1
)

buy_hold_annual_return = (
    (1 + buy_hold_total_return)
    ** (252 / days)
    - 1
)

# Volatility
strategy_volatility = (
    test_data["Net_Strategy_Return"].std()
    * np.sqrt(252)
)

# Sharpe ratio
strategy_sharpe = (
    strategy_annual_return / strategy_volatility
    if strategy_volatility != 0
    else 0
)

# Maximum drawdown
running_max = (
    test_data["Strategy_Equity"]
    .cummax()
)

drawdown = (
    test_data["Strategy_Equity"]
    / running_max
    - 1
)

max_drawdown = drawdown.min()

# Number of trades
trades = int(
    test_data["Position_Change"].sum()
)

# Win rate
active_days = (
    test_data[test_data["Position"] == 1]
)

if len(active_days) > 0:
    win_rate = (
        active_days["Next_Return"] > 0
    ).mean()
else:
    win_rate = 0

# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("BACKTEST RESULTS")
print("========================================")

print("\nTest period:")
print(
    test_data.index.min(),
    "to",
    test_data.index.max()
)

print("\nStrategy total return:")
print(
    f"{strategy_total_return * 100:.2f}%"
)

print("\nBuy & Hold total return:")
print(
    f"{buy_hold_total_return * 100:.2f}%"
)

print("\nStrategy annualized return:")
print(
    f"{strategy_annual_return * 100:.2f}%"
)

print("\nBuy & Hold annualized return:")
print(
    f"{buy_hold_annual_return * 100:.2f}%"
)

print("\nStrategy Sharpe ratio:")
print(
    f"{strategy_sharpe:.3f}"
)

print("\nMaximum drawdown:")
print(
    f"{max_drawdown * 100:.2f}%"
)

print("\nNumber of position changes:")
print(trades)

print("\nWin rate while invested:")
print(
    f"{win_rate * 100:.2f}%"
)

print("\nTransaction cost assumption:")
print(
    f"{TRANSACTION_COST * 100:.03f}% per position change"
)

# ==================================================
# SAVE RESULTS
# ==================================================

output_file = (
    DATA_DIR / "backtest_results.csv"
)

test_data.to_csv(output_file)

print("\nResults saved to:")
print(output_file)