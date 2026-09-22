import pandas as pd
from pathlib import Path

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

input_file = DATA_DIR / "nifty50.csv"
output_file = DATA_DIR / "nifty50_features.csv"

print("Loading:", input_file)

# ==================================================
# LOAD DATA
# ==================================================

# Yahoo Finance puts an extra Ticker row in this CSV
data = pd.read_csv(input_file, skiprows=[1, 2])

# The first column contains the dates
data = data.rename(columns={"Price": "Date"})

# Convert Date
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# Remove invalid dates
data = data.dropna(subset=["Date"])

# Set Date as index
data = data.set_index("Date")

# ==================================================
# CLEAN NUMERIC DATA
# ==================================================

columns = ["Open", "High", "Low", "Close", "Volume"]

for col in columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove missing values
data = data.dropna()

# ==================================================
# FEATURE ENGINEERING
# ==================================================

# Daily percentage return
data["Return"] = data["Close"].pct_change()

# Previous day's return
data["Previous_Return"] = data["Return"].shift(1)

# Moving averages
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["MA_20"] = data["Close"].rolling(window=20).mean()
data["MA_50"] = data["Close"].rolling(window=50).mean()

# 10-day volatility
data["Volatility_10"] = data["Return"].rolling(window=10).std()

# Volume percentage change
data["Volume_Change"] = data["Volume"].pct_change()

# ==================================================
# TARGET
# ==================================================

# Tomorrow's closing price > today's closing price
# 1 = UP
# 0 = DOWN

future_close = data["Close"].shift(-1)

data["Target"] = (future_close > data["Close"]).astype("float")

# Last row has no future price, so remove it
data.loc[future_close.isna(), "Target"] = None
# Remove rows containing NaN
data = data.dropna()

# ==================================================
# SAVE
# ==================================================

data.to_csv(output_file)

# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("DATASET CREATED SUCCESSFULLY")
print("========================================")

print("\nDataset shape:")
print(data.shape)

print("\nColumns:")
for column in data.columns:
    print(" -", column)

print("\nTarget distribution:")
print(data["Target"].value_counts())

print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())

print("\nSaved to:")
print(output_file)