import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PERIOD = "2y"


# ============================================================
# TICKERS
# ============================================================

tickers = {

    "NIFTY": "^NSEI",

    "INDIA_VIX": "^INDIAVIX",

    "NIFTY_BANK": "^NSEBANK",

    "SP500": "^GSPC",

    "USD_INR": "INR=X",

    "GOLD": "GC=F",

    "CRUDE_OIL": "CL=F"
}


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 60)
print("DOWNLOADING LIVE MARKET DATA")
print("=" * 60)     


datasets = {}


for name, ticker in tickers.items():

    print(f"\nDownloading {name} ({ticker})...")

    df = yf.download(
        ticker,
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:

        print(f"WARNING: No data for {name}")

        continue

    # Handle yfinance multi-level columns
    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    datasets[name] = df

    print(
        f"{name}: "
        f"{df.index.min().date()} → "
        f"{df.index.max().date()}"
    )


# ============================================================
# NIFTY DATA
# ============================================================

nifty = datasets["NIFTY"].copy()

nifty = nifty[
    ["Open", "High", "Low", "Close", "Volume"]
].copy()


# ============================================================
# TECHNICAL FEATURES
# ============================================================

nifty["Return_1D"] = (
    nifty["Close"].pct_change(1)
)

nifty["Return_5D"] = (
    nifty["Close"].pct_change(5)
)

nifty["Return_10D"] = (
    nifty["Close"].pct_change(10)
)

nifty["Return_20D"] = (
    nifty["Close"].pct_change(20)
)


# ------------------------------------------------------------
# MOVING AVERAGES
# ------------------------------------------------------------

nifty["MA5"] = (
    nifty["Close"].rolling(5).mean()
)

nifty["MA20"] = (
    nifty["Close"].rolling(20).mean()
)

nifty["MA50"] = (
    nifty["Close"].rolling(50).mean()
)


nifty["Price_to_MA5"] = (
    nifty["Close"] / nifty["MA5"] - 1
)

nifty["Price_to_MA20"] = (
    nifty["Close"] / nifty["MA20"] - 1
)

nifty["Price_to_MA50"] = (
    nifty["Close"] / nifty["MA50"] - 1
)


# ============================================================
# PRICE FEATURES
# ============================================================

nifty["High_Low_Range"] = (
    nifty["High"] - nifty["Low"]
) / nifty["Close"]


nifty["Open_Close_Return"] = (
    nifty["Close"] - nifty["Open"]
) / nifty["Open"]


# ============================================================
# VOLATILITY
# ============================================================

nifty["Volatility_10"] = (
    nifty["Return_1D"]
    .rolling(10)
    .std()
)

nifty["Volatility_20"] = (
    nifty["Return_1D"]
    .rolling(20)
    .std()
)


# ============================================================
# VOLUME
# ============================================================

nifty["Volume_Change"] = (
    nifty["Volume"].pct_change()
)


volume_ma20 = (
    nifty["Volume"]
    .rolling(20)
    .mean()
)

nifty["Relative_Volume"] = (
    nifty["Volume"] / volume_ma20
)


# ============================================================
# RSI
# ============================================================

delta = nifty["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

nifty["RSI_14"] = (
    100 - (100 / (1 + rs))
)


# ============================================================
# MACD
# ============================================================

ema12 = (
    nifty["Close"]
    .ewm(span=12, adjust=False)
    .mean()
)

ema26 = (
    nifty["Close"]
    .ewm(span=26, adjust=False)
    .mean()
)

nifty["MACD"] = (
    ema12 - ema26
)

nifty["MACD_Signal"] = (
    nifty["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

nifty["MACD_Histogram"] = (
    nifty["MACD"]
    - nifty["MACD_Signal"]
)


# ============================================================
# BOLLINGER BANDS
# ============================================================

bb_middle = (
    nifty["Close"]
    .rolling(20)
    .mean()
)

bb_std = (
    nifty["Close"]
    .rolling(20)
    .std()
)

bb_high = (
    bb_middle + 2 * bb_std
)

bb_low = (
    bb_middle - 2 * bb_std
)

nifty["BB_Position"] = (
    (nifty["Close"] - bb_low)
    / (bb_high - bb_low)
)


# ============================================================
# ATR
# ============================================================

previous_close = (
    nifty["Close"].shift(1)
)

tr1 = (
    nifty["High"] - nifty["Low"]
)

tr2 = (
    abs(nifty["High"] - previous_close)
)

tr3 = (
    abs(nifty["Low"] - previous_close)
)

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

atr14 = (
    true_range
    .rolling(14)
    .mean()
)

nifty["ATR_Percent"] = (
    atr14 / nifty["Close"]
)


# ============================================================
# EXTERNAL FEATURES
# ============================================================

external = pd.DataFrame(
    index=nifty.index
)


external_map = {

    "INDIA_VIX":
        "India_VIX",

    "NIFTY_BANK":
        "NIFTY_Bank",

    "SP500":
        "SP500",

    "USD_INR":
        "USD_INR",

    "GOLD":
        "Gold",

    "CRUDE_OIL":
        "Crude_Oil"
}


for source, prefix in external_map.items():

    if source not in datasets:

        continue

    series = datasets[source]["Close"]

    series = series.reindex(
        external.index
    )

    external[
        f"{prefix}_Return_1D"
    ] = series.pct_change(1)

    external[
        f"{prefix}_Return_5D"
    ] = series.pct_change(5)


# ============================================================
# IMPORTANT:
# PREVENT LOOK-AHEAD
# ============================================================

external = external.shift(1)


# ============================================================
# COMBINE
# ============================================================

features = pd.concat(
    [
        nifty,
        external
    ],
    axis=1
)


# ============================================================
# CLEAN
# ============================================================

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    DATA_DIR /
    "live_market_data.csv"
)

features.to_csv(
    output_file
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("LIVE DATA CREATED")
print("=" * 60)

print(
    "\nDataset shape:",
    features.shape
)

print(
    "\nLatest available date:",
    features.index[-1]
)

print(
    "\nLatest NIFTY close:",
    f"{features['Close'].iloc[-1]:,.2f}"
)

print(
    "\nSaved to:"
)

print(output_file)

print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)