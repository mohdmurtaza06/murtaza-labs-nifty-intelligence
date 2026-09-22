import pandas as pd
import numpy as np
from pathlib import Path

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

input_file = DATA_DIR / "nifty50.csv"
output_file = DATA_DIR / "nifty50_better_features.csv"

# ==================================================
# LOAD DATA
# ==================================================

data = pd.read_csv(input_file, skiprows=[1, 2])

# First column contains Date
data = data.rename(columns={"Price": "Date"})

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)

data = data.dropna(subset=["Date"])
data = data.set_index("Date")

# ==================================================
# CLEAN DATA
# ==================================================

columns = ["Open", "High", "Low", "Close", "Volume"]

for column in columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.dropna()

# ==================================================
# RETURN FEATURES
# ==================================================

data["Return_1D"] = data["Close"].pct_change()

data["Return_5D"] = (
    data["Close"].pct_change(5)
)

data["Return_10D"] = (
    data["Close"].pct_change(10)
)

data["Return_20D"] = (
    data["Close"].pct_change(20)
)

# ==================================================
# MOVING AVERAGE RELATIONSHIPS
# ==================================================

data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_20"] = data["Close"].rolling(20).mean()
data["MA_50"] = data["Close"].rolling(50).mean()

# How far price is from moving averages
data["Price_to_MA5"] = (
    data["Close"] / data["MA_5"] - 1
)

data["Price_to_MA20"] = (
    data["Close"] / data["MA_20"] - 1
)

data["Price_to_MA50"] = (
    data["Close"] / data["MA_50"] - 1
)

# ==================================================
# PRICE ACTION
# ==================================================

# Intraday range
data["High_Low_Range"] = (
    (data["High"] - data["Low"])
    / data["Close"]
)

# Open-to-close movement
data["Open_Close_Return"] = (
    (data["Close"] - data["Open"])
    / data["Open"]
)

# ==================================================
# VOLATILITY
# ==================================================

data["Volatility_10"] = (
    data["Return_1D"].rolling(10).std()
)

data["Volatility_20"] = (
    data["Return_1D"].rolling(20).std()
)

# ==================================================
# VOLUME FEATURES
# ==================================================

data["Volume_Change"] = (
    data["Volume"].pct_change()
)

data["Volume_MA20"] = (
    data["Volume"].rolling(20).mean()
)

data["Relative_Volume"] = (
    data["Volume"] / data["Volume_MA20"]
)

# ==================================================
# RSI
# ==================================================

rsi = RSIIndicator(
    close=data["Close"],
    window=14
)

data["RSI_14"] = rsi.rsi()

# ==================================================
# MACD
# ==================================================

macd = MACD(
    close=data["Close"],
    window_fast=12,
    window_slow=26,
    window_sign=9
)

data["MACD"] = macd.macd()
data["MACD_Signal"] = macd.macd_signal()
data["MACD_Histogram"] = macd.macd_diff()

# ==================================================
# BOLLINGER BANDS
# ==================================================

bb = BollingerBands(
    close=data["Close"],
    window=20,
    window_dev=2
)

data["BB_High"] = bb.bollinger_hband()
data["BB_Low"] = bb.bollinger_lband()
data["BB_Middle"] = bb.bollinger_mavg()

# Position inside Bollinger Bands
data["BB_Position"] = (
    (data["Close"] - data["BB_Low"])
    / (data["BB_High"] - data["BB_Low"])
)

# ==================================================
# ATR
# ==================================================

atr = AverageTrueRange(
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    window=14
)

data["ATR_14"] = atr.average_true_range()

data["ATR_Percent"] = (
    data["ATR_14"] / data["Close"]
)

# ==================================================
# TARGET
# ==================================================

future_close = data["Close"].shift(-1)

data["Target"] = (
    future_close > data["Close"]
).astype("float")

# Last day has no future price
data.loc[future_close.isna(), "Target"] = np.nan

# ==================================================
# CLEAN
# ==================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna()

data["Target"] = data["Target"].astype(int)

# ==================================================
# SAVE
# ==================================================

data.to_csv(output_file)

# ==================================================
# DISPLAY
# ==================================================

print("\n========================================")
print("BETTER FEATURE DATASET CREATED")
print("========================================")

print("\nDataset shape:")
print(data.shape)

print("\nFeatures:")

for column in data.columns:
    print(" -", column)

print("\nTarget distribution:")
print(data["Target"].value_counts())

print("\nDate range:")
print(data.index.min(), "to", data.index.max())

print("\nSaved to:")
print(output_file)