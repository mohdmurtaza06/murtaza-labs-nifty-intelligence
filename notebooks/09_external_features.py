import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "nifty50_better_features.csv"
OUTPUT_FILE = DATA_DIR / "nifty50_external_features.csv"


# ==================================================
# LOAD OUR EXISTING NIFTY DATA
# ==================================================

nifty = pd.read_csv(
    INPUT_FILE,
    index_col="Date",
    parse_dates=True
)

print("NIFTY dataset:", nifty.shape)


# ==================================================
# EXTERNAL MARKET DATA
# ==================================================

tickers = {
    "India_VIX": "^INDIAVIX",
    "NIFTY_Bank": "^NSEBANK",
    "SP500": "^GSPC",
    "USD_INR": "USDINR=X",
    "Gold": "GC=F",
    "Crude_Oil": "CL=F"
}


external = pd.DataFrame()


for name, ticker in tickers.items():

    print(f"\nDownloading {name} ({ticker})...")

    df = yf.download(
        ticker,
        start="2015-01-01",
        end="2026-01-01",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        print(f"WARNING: No data received for {name}")
        continue

    # Handle Yahoo Finance multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    close.name = name

    external = pd.concat(
        [external, close],
        axis=1
    )


# ==================================================
# CREATE EXTERNAL FEATURES
# ==================================================

for column in external.columns:

    # Daily return
    external[f"{column}_Return_1D"] = (
        external[column].pct_change()
    )

    # 5-day return
    external[f"{column}_Return_5D"] = (
        external[column].pct_change(5)
    )


# ==================================================
# KEEP ONLY RETURNS
# ==================================================

external_features = external[
    [
        col
        for col in external.columns
        if "Return" in col
    ]
].copy()
# ==================================================
# PREVENT LOOK-AHEAD BIAS
# ==================================================

# External market information is shifted by one
# NIFTY trading day so that only information
# available before the prediction is used.

external_features = external_features.shift(1)


# ==================================================
# ALIGN WITH NIFTY
# ==================================================

data = nifty.join(
    external_features,
    how="inner"
)


# ==================================================
# CLEAN
# ==================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna()


# ==================================================
# SAVE
# ==================================================

data.to_csv(OUTPUT_FILE)


# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("EXTERNAL FEATURE DATASET")
print("========================================")

print("\nDataset shape:")
print(data.shape)

print("\nExternal features:")

for column in external_features.columns:
    print(" -", column)

print("\nDate range:")
print(
    data.index.min(),
    "to",
    data.index.max()
)

print("\nSaved to:")
print(OUTPUT_FILE)