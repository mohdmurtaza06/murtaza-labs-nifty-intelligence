import sys
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
sys.path.insert(0, str(BASE / "src"))

from model_features_ultimate import build_features

EXT_PATH = DATA / "nifty50_external_features.csv"
LIVE_PATH = DATA / "live_market_data.csv"


def load_data():
    ext = pd.read_csv(EXT_PATH, parse_dates=["Date"])
    ext = ext.drop_duplicates("Date").set_index("Date").sort_index()

    live = pd.read_csv(LIVE_PATH, parse_dates=["Date"])
    live = live.drop_duplicates("Date").set_index("Date").sort_index()

    # Prefer the prepared external dataset, then add any newer live NIFTY rows.
    if len(ext):
        newer = live[live.index > ext.index.max()]
        raw = pd.concat([ext, newer], axis=0, sort=False)
    else:
        raw = live

    return raw.sort_index()


def add_safe_cross_market_features(df):
    """
    The external dataset contains return features rather than raw price series.

    Same-session Indian variables are usable at/near the NIFTY close:
      India VIX, NIFTY Bank

    Overseas / timing-sensitive variables are shifted one session so the
    production model cannot accidentally consume information that was not
    available when the NIFTY prediction was made:
      S&P 500, USD/INR, Gold, Crude Oil
    """
    df = df.copy()

    groups = {
        "INDIA_VIX": {
            "r1": "India_VIX_Return_1D",
            "r5": "India_VIX_Return_5D",
            "lag": 0,
        },
        "BANKNIFTY": {
            "r1": "NIFTY_Bank_Return_1D",
            "r5": "NIFTY_Bank_Return_5D",
            "lag": 0,
        },
        "SP500": {
            "r1": "SP500_Return_1D",
            "r5": "SP500_Return_5D",
            "lag": 1,
        },
        "USDINR": {
            "r1": "USD_INR_Return_1D",
            "r5": "USD_INR_Return_5D",
            "lag": 1,
        },
        "GOLD": {
            "r1": "Gold_Return_1D",
            "r5": "Gold_Return_5D",
            "lag": 1,
        },
        "CRUDE": {
            "r1": "Crude_Oil_Return_1D",
            "r5": "Crude_Oil_Return_5D",
            "lag": 1,
        },
    }

    detected = {}

    for prefix, cfg in groups.items():
        if cfg["r1"] not in df.columns:
            continue

        detected[prefix] = cfg["r1"]

        r1 = pd.to_numeric(df[cfg["r1"]], errors="coerce")
        r5 = (
            pd.to_numeric(df[cfg["r5"]], errors="coerce")
            if cfg["r5"] in df.columns
            else r1.rolling(5).sum()
        )

        if cfg["lag"]:
            r1 = r1.shift(cfg["lag"])
            r5 = r5.shift(cfg["lag"])

        df[f"PROD_{prefix}_RET1"] = r1
        df[f"PROD_{prefix}_RET5"] = r5
        df[f"PROD_{prefix}_MOM20"] = r1.rolling(20).sum()
        df[f"PROD_{prefix}_VOL20"] = r1.rolling(20).std()
        df[f"PROD_{prefix}_VOL60"] = r1.rolling(60).std()

    # Cross-market risk scores.
    risk_on = [
        df[c]
        for c in [
            "PROD_SP500_RET5",
            "PROD_BANKNIFTY_RET5",
        ]
        if c in df.columns
    ]

    risk_off = [
        df[c]
        for c in [
            "PROD_INDIA_VIX_RET5",
            "PROD_USDINR_RET5",
        ]
        if c in df.columns
    ]

    if risk_on:
        df["PROD_RISK_ON"] = pd.concat(risk_on, axis=1).mean(axis=1)

    if risk_off:
        df["PROD_RISK_OFF"] = pd.concat(risk_off, axis=1).mean(axis=1)

    if "PROD_RISK_ON" in df and "PROD_RISK_OFF" in df:
        df["PROD_RISK_SCORE"] = (
            df["PROD_RISK_ON"] - df["PROD_RISK_OFF"]
        )

    # NIFTY volatility/trend regime.
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

    # Calendar context.
    df["PROD_DAY_OF_WEEK"] = df.index.dayofweek
    df["PROD_MONTH_START"] = (df.index.day <= 3).astype(int)
    month_end = df.index.to_period("M").to_timestamp("M")
    df["PROD_MONTH_END_DISTANCE"] = (month_end - df.index).days
    df["PROD_MONTH_END"] = (
        df["PROD_MONTH_END_DISTANCE"] <= 3
    ).astype(int)

    return df, detected


raw = load_data()
raw_enriched, detected_sources = add_safe_cross_market_features(raw)

# Build the original technical feature space.
features = build_features(raw_enriched)

# Add only production-safe regime/cross-market features.
added = [
    c for c in raw_enriched.columns
    if c.startswith("PROD_")
]

extra = (
    raw_enriched[added]
    .apply(pd.to_numeric, errors="coerce")
    .reindex(features.index)
)

features = pd.concat([features, extra], axis=1)
features = features.loc[:, ~features.columns.duplicated()]

close = pd.to_numeric(
    raw_enriched["Close"], errors="coerce"
).reindex(features.index)

next_return = close.shift(-1) / close - 1
target = (next_return > 0).astype(int)

feature_cols = [
    c for c in features.columns
    if c not in {"Target", "Next_Return"}
    and features[c].notna().mean() >= 0.75
]

X = (
    features[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
)

mask = target.notna() & X.notna().any(axis=1)
X = X.loc[mask]
y = target.loc[mask]

# Conservative CatBoost configuration used for the production candidate.
model = CatBoostClassifier(
    iterations=500,
    depth=4,
    learning_rate=0.02,
    l2_leaf_reg=20,
    random_strength=1.5,
    bagging_temperature=1.0,
    loss_function="Logloss",
    verbose=False,
    random_seed=42,
    thread_count=-1,
    allow_writing_files=False,
)

medians = X.median(numeric_only=True)
X_train = X.fillna(medians)

model.fit(X_train, y)

bundle = {
    "model": model,
    "feature_cols": feature_cols,
    "medians": medians.to_dict(),
    "trained_through": str(X.index.max().date()),
    "training_rows": int(len(X)),
    "feature_count": int(len(feature_cols)),
    "target": "next_trading_day_direction",
    "detected_sources": detected_sources,
    "safe_lagged_sources": [
        "SP500",
        "USDINR",
        "GOLD",
        "CRUDE",
    ],
    "same_session_sources": [
        "INDIA_VIX",
        "BANKNIFTY",
    ],
    "architecture": "CATBOOST_PRODUCTION_SAFE_REGIME_CROSS_MARKET",
}

model_path = MODELS / "nifty50_production_model.pkl"
meta_path = MODELS / "nifty50_production_meta.json"

joblib.dump(bundle, model_path)

meta = {
    "architecture": bundle["architecture"],
    "trained_through": bundle["trained_through"],
    "training_rows": bundle["training_rows"],
    "feature_count": bundle["feature_count"],
    "target": bundle["target"],
    "detected_sources": detected_sources,
    "safe_lagged_sources": bundle["safe_lagged_sources"],
    "same_session_sources": bundle["same_session_sources"],
    "note": (
        "Production feature timing is constrained so overseas/timing-sensitive "
        "market returns are lagged before they enter the model."
    ),
}

meta_path.write_text(
    json.dumps(meta, indent=2, default=str),
    encoding="utf-8",
)

print("=" * 72)
print("PRODUCTION MODEL BUILT")
print("=" * 72)
print(f"Training rows : {len(X)}")
print(f"Features      : {len(feature_cols)}")
print(f"Trained thru  : {X.index.max().date()}")
print("Model         : CatBoost")
print("Architecture  : Production-safe regime + cross-market")
print()
print("Detected sources:")
for k, v in detected_sources.items():
    print(f"  {k:10s}: {v}")

print()
print("Timing protection:")
print("  Same session : India VIX, NIFTY Bank")
print("  Lagged       : S&P 500, USD/INR, Gold, Crude Oil")
print()
print(f"Saved model   : {model_path}")
print(f"Saved metadata: {meta_path}")
