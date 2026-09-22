import sys
import json
import subprocess
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
sys.path.insert(0, str(BASE / "src"))

from model_features_ultimate import build_features

MODEL_PATH = MODELS / "nifty50_production_model.pkl"
LIVE_PATH = DATA / "live_market_data.csv"
EXT_PATH = DATA / "nifty50_external_features.csv"
OUTPUT_JSON = DATA / "latest_production_prediction.json"
OUTPUT_CSV = DATA / "latest_production_prediction.csv"


def run_live_data_update():
    script = BASE / "notebooks" / "14_live_data.py"

    if not script.exists():
        print("WARNING: 14_live_data.py not found.")
        return False

    print("Updating live market data...")

    import os

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script)],
        cwd=str(BASE),
        capture_output=True,
        text=False,
        env=env,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")

    if stdout:
        print(stdout[-4000:])

    if result.returncode != 0:
        print("WARNING: live data update failed.")
        if stderr:
            print(stderr[-3000:])
        return False

    return True


def load_raw():
    ext = pd.read_csv(EXT_PATH, parse_dates=["Date"])
    ext = ext.drop_duplicates("Date").set_index("Date").sort_index()

    live = pd.read_csv(LIVE_PATH, parse_dates=["Date"])
    live = live.drop_duplicates("Date").set_index("Date").sort_index()

    if len(ext):
        newer = live[live.index > ext.index.max()]
        raw = pd.concat([ext, newer], axis=0, sort=False)
    else:
        raw = live

    return raw.sort_index()


def add_safe_cross_market_features(df):
    df = df.copy()

    groups = {
        "INDIA_VIX": ("India_VIX_Return_1D", "India_VIX_Return_5D", 0),
        "BANKNIFTY": ("NIFTY_Bank_Return_1D", "NIFTY_Bank_Return_5D", 0),
        "SP500": ("SP500_Return_1D", "SP500_Return_5D", 1),
        "USDINR": ("USD_INR_Return_1D", "USD_INR_Return_5D", 1),
        "GOLD": ("Gold_Return_1D", "Gold_Return_5D", 1),
        "CRUDE": ("Crude_Oil_Return_1D", "Crude_Oil_Return_5D", 1),
    }

    detected = []

    for prefix, (c1, c5, lag) in groups.items():
        if c1 not in df.columns:
            continue

        r1 = pd.to_numeric(df[c1], errors="coerce")
        r5 = (
            pd.to_numeric(df[c5], errors="coerce")
            if c5 in df.columns
            else r1.rolling(5).sum()
        )

        if lag:
            r1 = r1.shift(lag)
            r5 = r5.shift(lag)

        df[f"PROD_{prefix}_RET1"] = r1
        df[f"PROD_{prefix}_RET5"] = r5
        df[f"PROD_{prefix}_MOM20"] = r1.rolling(20).sum()
        df[f"PROD_{prefix}_VOL20"] = r1.rolling(20).std()
        df[f"PROD_{prefix}_VOL60"] = r1.rolling(60).std()
        detected.append(prefix)

    risk_on = [
        df[c]
        for c in ["PROD_SP500_RET5", "PROD_BANKNIFTY_RET5"]
        if c in df
    ]
    risk_off = [
        df[c]
        for c in ["PROD_INDIA_VIX_RET5", "PROD_USDINR_RET5"]
        if c in df
    ]

    if risk_on:
        df["PROD_RISK_ON"] = pd.concat(risk_on, axis=1).mean(axis=1)

    if risk_off:
        df["PROD_RISK_OFF"] = pd.concat(risk_off, axis=1).mean(axis=1)

    if "PROD_RISK_ON" in df and "PROD_RISK_OFF" in df:
        df["PROD_RISK_SCORE"] = (
            df["PROD_RISK_ON"] - df["PROD_RISK_OFF"]
        )

    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    df["PROD_NIFTY_VOL20"] = ret.rolling(20).std() * np.sqrt(252)
    df["PROD_NIFTY_VOL60"] = ret.rolling(60).std() * np.sqrt(252)
    df["PROD_VOL_REGIME"] = (
        df["PROD_NIFTY_VOL20"]
        / df["PROD_NIFTY_VOL60"].replace(0, np.nan)
    )

    df["PROD_TREND_20_50"] = ma20 / ma50.replace(0, np.nan) - 1
    df["PROD_TREND_50_200"] = ma50 / ma200.replace(0, np.nan) - 1
    df["PROD_PRICE_200"] = close / ma200.replace(0, np.nan) - 1

    df["PROD_TREND_REGIME"] = (
        np.sign(df["PROD_TREND_20_50"].fillna(0))
        + np.sign(df["PROD_TREND_50_200"].fillna(0))
        + np.sign(df["PROD_PRICE_200"].fillna(0))
    )

    df["PROD_VOLATILITY_REGIME"] = np.sign(
        df["PROD_VOL_REGIME"].fillna(1) - 1
    )

    df["PROD_DAY_OF_WEEK"] = df.index.dayofweek
    df["PROD_MONTH_START"] = (df.index.day <= 3).astype(int)

    month_end = df.index.to_period("M").to_timestamp("M")
    df["PROD_MONTH_END_DISTANCE"] = (month_end - df.index).days
    df["PROD_MONTH_END"] = (
        df["PROD_MONTH_END_DISTANCE"] <= 3
    ).astype(int)

    return df, detected


def get_indicator(row, names):
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value):
                return float(value)
    return None


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Production model not found: {MODEL_PATH}\n"
        "Run 27_build_production_model.py first."
    )

run_live_data_update()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
medians = pd.Series(bundle["medians"], dtype=float)

raw = load_raw()
raw_enriched, detected = add_safe_cross_market_features(raw)

features = build_features(raw_enriched)

prod_cols = [
    c for c in raw_enriched.columns
    if c.startswith("PROD_")
]

extra = (
    raw_enriched[prod_cols]
    .apply(pd.to_numeric, errors="coerce")
    .reindex(features.index)
)

features = pd.concat([features, extra], axis=1)
features = features.loc[:, ~features.columns.duplicated()]

missing_features = [
    c for c in feature_cols
    if c not in features.columns
]

if missing_features:
    raise RuntimeError(
        "Feature mismatch. Missing production features:\n"
        + "\n".join(missing_features[:50])
    )

latest_date = features.index.max()
latest = features.loc[[latest_date], feature_cols].copy()

# Match training-time imputation exactly.
latest = latest.replace([np.inf, -np.inf], np.nan)
latest = latest.fillna(medians.reindex(feature_cols))
latest = latest.fillna(0.0)

prob = model.predict_proba(latest)[0]
prob_down = float(prob[0])
prob_up = float(prob[1])

prediction = "UP" if prob_up >= prob_down else "DOWN"
# Persist all probability values as fractions (0-1), matching the legacy
# prediction/history contract. Presentation layers convert them to percent.
confidence = max(prob_up, prob_down)

raw_latest = raw_enriched.loc[latest_date]

close = float(raw_latest["Close"])
previous_close = (
    float(raw_enriched["Close"].iloc[-2])
    if len(raw_enriched) >= 2
    else None
)

day_change_pct = (
    ((close / previous_close) - 1) * 100
    if previous_close
    else None
)

rsi = get_indicator(
    latest.iloc[0],
    ["RSI_14", "RSI", "rsi_14"],
)

macd = get_indicator(
    latest.iloc[0],
    ["MACD", "MACD_Line"],
)

india_vix = get_indicator(
    raw_latest,
    ["India_VIX", "India_VIX_Close", "VIX"],
)

banknifty_ret = get_indicator(
    raw_latest,
    ["NIFTY_Bank_Return_1D"],
)

sp500_ret = get_indicator(
    raw_latest,
    ["SP500_Return_1D"],
)

usdinr_ret = get_indicator(
    raw_latest,
    ["USD_INR_Return_1D"],
)

gold_ret = get_indicator(
    raw_latest,
    ["Gold_Return_1D"],
)

crude_ret = get_indicator(
    raw_latest,
    ["Crude_Oil_Return_1D"],
)

trend_regime = get_indicator(
    latest.iloc[0],
    ["PROD_TREND_REGIME"],
)

vol_regime = get_indicator(
    latest.iloc[0],
    ["PROD_VOL_REGIME"],
)

risk_score = get_indicator(
    latest.iloc[0],
    ["PROD_RISK_SCORE"],
)

result = {
    "prediction_date": str(latest_date.date()),
    "prediction": prediction,
    "probability_up": round(prob_up, 6),
    "probability_down": round(prob_down, 6),
    "confidence": round(confidence, 6),
    "nifty_close": round(close, 2),
    "day_change_pct": round(day_change_pct, 4) if day_change_pct is not None else None,
    "rsi_14": round(rsi, 4) if rsi is not None else None,
    "macd": round(macd, 6) if macd is not None else None,
    "india_vix": india_vix,
    "banknifty_return_1d": banknifty_ret,
    "sp500_return_1d": sp500_ret,
    "usdinr_return_1d": usdinr_ret,
    "gold_return_1d": gold_ret,
    "crude_return_1d": crude_ret,
    "trend_regime": trend_regime,
    "volatility_regime": vol_regime,
    "risk_score": risk_score,
    "model": bundle.get("architecture", "CATBOOST_PRODUCTION"),
    "model_trained_through": bundle.get("trained_through"),
    "feature_count": len(feature_cols),
    "detected_sources": detected,
}

OUTPUT_JSON.write_text(
    json.dumps(result, indent=2, default=float),
    encoding="utf-8",
)

pd.DataFrame([result]).to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print("=" * 72)
print("LIVE PRODUCTION PREDICTION")
print("=" * 72)
print(f"Date             : {result['prediction_date']}")
print(f"NIFTY 50         : {result['nifty_close']:.2f}")
if result["day_change_pct"] is not None:
    print(f"Day change       : {result['day_change_pct']:+.2f}%")
print("-" * 72)
print(f"PREDICTION       : {prediction}")
print(f"UP probability   : {prob_up * 100:.2f}%")
print(f"DOWN probability : {prob_down * 100:.2f}%")
print(f"Confidence       : {confidence * 100:.2f}%")
print("-" * 72)

if rsi is not None:
    print(f"RSI(14)          : {rsi:.2f}")
if macd is not None:
    print(f"MACD             : {macd:.6f}")
if trend_regime is not None:
    print(f"Trend regime     : {trend_regime:+.0f}")
if vol_regime is not None:
    print(f"Volatility ratio : {vol_regime:.4f}")
if risk_score is not None:
    print(f"Risk score       : {risk_score:+.6f}")

print("-" * 72)
print(f"Features used    : {len(feature_cols)}")
print(f"Model trained to : {bundle.get('trained_through')}")
print(f"JSON output      : {OUTPUT_JSON}")
print(f"CSV output       : {OUTPUT_CSV}")
print("=" * 72)
