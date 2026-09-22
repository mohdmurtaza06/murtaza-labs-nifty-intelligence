import sys
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
sys.path.insert(0, str(BASE / "src"))

from model_features_ultimate import build_features


EXT_PATH = DATA / "nifty50_external_features.csv"
LIVE_PATH = DATA / "live_market_data.csv"

ext = (
    pd.read_csv(EXT_PATH, parse_dates=["Date"])
    .drop_duplicates("Date")
    .set_index("Date")
    .sort_index()
)

live = (
    pd.read_csv(LIVE_PATH, parse_dates=["Date"])
    .drop_duplicates("Date")
    .set_index("Date")
    .sort_index()
)

raw = pd.concat(
    [ext, live[live.index > ext.index.max()]],
    axis=0,
    sort=False,
).sort_index()


def find_col(columns, names):
    lower = {str(c).lower(): c for c in columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def find_contains(columns, tokens):
    for c in columns:
        s = str(c).lower()
        if all(token.lower() in s for token in tokens):
            return c
    return None


def add_return_features(df, source_col, prefix):
    s = pd.to_numeric(df[source_col], errors="coerce")
    r1 = s.pct_change(1)
    r5 = s.pct_change(5)
    r20 = s.pct_change(20)

    df[f"{prefix}_RET1"] = r1
    df[f"{prefix}_RET5"] = r5
    df[f"{prefix}_RET20"] = r20
    df[f"{prefix}_VOL20"] = r1.rolling(20).std() * np.sqrt(252)
    df[f"{prefix}_MOM_ACCEL"] = r5 - r20 / 4.0
    return df


def add_cross_market_regime(df):
    """
    Adds only backward-looking transformations.

    The script does NOT download new data. It uses the cross-market series
    already present in nifty50_external_features.csv. This avoids changing
    the historical sample while we test whether regime-aware transformations
    improve generalization.
    """
    cols = list(df.columns)
    added = []

    mappings = {
        "SP500": [
            "sp500",
            "s&p500",
            "sp_500",
            "spx",
        ],
        "NASDAQ": [
            "nasdaq",
            "nasdaq_composite",
            "ixic",
        ],
        "BANKNIFTY": [
            "banknifty",
            "nifty_bank",
            "niftybank",
            "nifty_bank_index",
        ],
        "VIX": [
            "india_vix",
            "indiavix",
            "vix",
        ],
        "USDINR": [
            "usd_inr",
            "usdinr",
            "usd/inr",
        ],
        "CRUDE": [
            "crude_oil",
            "crude",
            "brent",
            "wti",
        ],
        "GOLD": [
            "gold",
            "gold_price",
            "xauusd",
        ],
        "BOND": [
            "us10y",
            "us_10y",
            "10y_yield",
            "bond_yield",
        ],
    }

    found = {}

    for key, aliases in mappings.items():
        c = find_col(cols, aliases)
        if c is None:
            # Fall back to contains matching, but avoid accidental matches
            # on already-derived return columns.
            for alias in aliases:
                c = next(
                    (
                        x for x in cols
                        if alias.lower() in str(x).lower()
                        and "return" not in str(x).lower()
                        and "ret" not in str(x).lower()
                    ),
                    None,
                )
                if c is not None:
                    break
        if c is not None:
            found[key] = c
            before = set(df.columns)
            df = add_return_features(df, c, key)
            added.extend([x for x in df.columns if x not in before])

    # Relative regime relationships.
    nifty = find_col(
        df.columns,
        ["Close", "NIFTY_Close", "Nifty_Close", "close"],
    )

    if nifty is None:
        nifty = find_contains(df.columns, ["close"])

    if nifty is not None:
        nret = pd.to_numeric(df[nifty], errors="coerce").pct_change()

        # Market stress / risk-on composite.
        risk_on = []
        for key in ["SP500", "NASDAQ", "BANKNIFTY"]:
            c = f"{key}_RET5"
            if c in df:
                risk_on.append(df[c])

        risk_off = []
        for key in ["VIX", "USDINR", "CRUDE"]:
            c = f"{key}_RET5"
            if c in df:
                risk_off.append(df[c])

        if risk_on:
            df["RISK_ON_SCORE"] = pd.concat(risk_on, axis=1).mean(axis=1)
            added.append("RISK_ON_SCORE")

        if risk_off:
            df["RISK_OFF_SCORE"] = pd.concat(risk_off, axis=1).mean(axis=1)
            added.append("RISK_OFF_SCORE")

        if "RISK_ON_SCORE" in df and "RISK_OFF_SCORE" in df:
            df["RISK_REGIME_SCORE"] = (
                df["RISK_ON_SCORE"] - df["RISK_OFF_SCORE"]
            )
            added.append("RISK_REGIME_SCORE")

        # NIFTY volatility regime.
        df["NIFTY_VOL20"] = nret.rolling(20).std() * np.sqrt(252)
        df["NIFTY_VOL60"] = nret.rolling(60).std() * np.sqrt(252)
        df["VOL_REGIME_RATIO"] = (
            df["NIFTY_VOL20"] /
            df["NIFTY_VOL60"].replace(0, np.nan)
        )
        added.extend(["NIFTY_VOL20", "NIFTY_VOL60", "VOL_REGIME_RATIO"])

        # Trend regime using only information available at the close.
        ma20 = pd.to_numeric(df[nifty], errors="coerce").rolling(20).mean()
        ma50 = pd.to_numeric(df[nifty], errors="coerce").rolling(50).mean()
        ma200 = pd.to_numeric(df[nifty], errors="coerce").rolling(200).mean()

        df["REGIME_TREND_20_50"] = ma20 / ma50.replace(0, np.nan) - 1
        df["REGIME_TREND_50_200"] = ma50 / ma200.replace(0, np.nan) - 1
        df["REGIME_PRICE_200"] = (
            pd.to_numeric(df[nifty], errors="coerce") /
            ma200.replace(0, np.nan) - 1
        )
        added.extend([
            "REGIME_TREND_20_50",
            "REGIME_TREND_50_200",
            "REGIME_PRICE_200",
        ])

        # Volatility/trend state buckets represented numerically.
        trend_score = (
            np.sign(df["REGIME_TREND_20_50"].fillna(0))
            + np.sign(df["REGIME_TREND_50_200"].fillna(0))
            + np.sign(df["REGIME_PRICE_200"].fillna(0))
        )
        vol_score = np.sign(
            df["VOL_REGIME_RATIO"].fillna(1) - 1
        )

        df["TREND_REGIME"] = trend_score
        df["VOLATILITY_REGIME"] = vol_score
        added.extend(["TREND_REGIME", "VOLATILITY_REGIME"])

    # Calendar regime, still fully known before the next session.
    idx = df.index
    df["DAY_OF_WEEK"] = idx.dayofweek
    df["MONTH_END_DISTANCE"] = (
        idx.to_period("M").to_timestamp("M") - idx
    ).days
    df["MONTH_START"] = (idx.day <= 3).astype(int)
    df["MONTH_END"] = (df["MONTH_END_DISTANCE"] <= 3).astype(int)
    added.extend([
        "DAY_OF_WEEK",
        "MONTH_END_DISTANCE",
        "MONTH_START",
        "MONTH_END",
    ])

    return df, found, sorted(set(added))


raw_enriched, found_sources, added_features = add_cross_market_regime(raw.copy())

features = build_features(raw_enriched)
# Explicitly append the newly engineered regime/cross-market features.
# build_features() does not automatically preserve arbitrary columns.
extra_features = (
    raw_enriched[added_features]
    .apply(pd.to_numeric, errors="coerce")
    .reindex(features.index)
)

features = pd.concat(
    [features, extra_features],
    axis=1
)

# Remove accidental duplicate feature names.
features = features.loc[:, ~features.columns.duplicated()]

close_col = find_col(
    raw_enriched.columns,
    ["Close", "NIFTY_Close", "Nifty_Close", "close"],
)
if close_col is None:
    close_col = find_contains(raw_enriched.columns, ["close"])

if close_col is None:
    raise RuntimeError("Could not find NIFTY close column.")

close = pd.to_numeric(raw_enriched[close_col], errors="coerce").reindex(features.index)
next_ret = close.shift(-1) / close - 1.0

# Standard next-day direction target for this research stage.
eligible = next_ret.notna()
y = (next_ret > 0).astype(int)

feature_cols = [
    c for c in features.columns
    if c not in {"Target", "Next_Return"}
    and features[c].notna().mean() >= 0.75
]

X = (
    features[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .loc[eligible]
)
y = y.loc[eligible]

print("=" * 72)
print("REGIME + CROSS-MARKET FEATURE STUDY")
print("=" * 72)
print(f"Rows:              {len(X)}")
print(f"Features:          {len(feature_cols)}")
print(f"New regime feats:  {len(added_features)}")
print(f"Range:             {X.index.min().date()} -> {X.index.max().date()}")
print("\nDetected source series:")
for k, v in found_sources.items():
    print(f"  {k:10s}: {v}")

print("\nAdded features:")
print(", ".join(added_features))


CONFIGS = [
    (
        "cat_base",
        dict(
            depth=4,
            learning_rate=0.025,
            iterations=350,
            l2_leaf_reg=12,
            random_strength=1.0,
            bagging_temperature=0.5,
        ),
    ),
    (
        "cat_conservative",
        dict(
            depth=4,
            learning_rate=0.02,
            iterations=500,
            l2_leaf_reg=20,
            random_strength=1.5,
            bagging_temperature=1.0,
        ),
    ),
]


def make_model(params):
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )


def impute(train, test):
    med = train.median(numeric_only=True)
    return train.fillna(med), test.fillna(med)


def score(y_true, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, p),
        "logloss": log_loss(y_true, p, labels=[0, 1]),
    }


# Compare:
# A) the original feature space from model_features_ultimate
# B) original + regime/cross-market additions
#
# The chronological outer test is untouched. The inner search only selects
# the CatBoost configuration. This is intentionally smaller than the earlier
# hyperparameter sweeps to reduce selection overfitting.

outer_initial = max(1200, int(len(X) * 0.68))
outer_size = 220
outer_starts = list(range(outer_initial, len(X), outer_size))

records = []

# Identify the original feature set by rebuilding features from the original raw data.
original_features = build_features(raw.copy())
original_cols = [
    c for c in original_features.columns
    if c not in {"Target", "Next_Return"}
    and original_features[c].notna().mean() >= 0.75
]

X_original = (
    original_features[original_cols]
    .replace([np.inf, -np.inf], np.nan)
    .reindex(X.index)
)

for start in outer_starts:
    end = min(start + outer_size, len(X))
    if end - start < 50:
        continue

    train_idx = X.index[:start]
    test_idx = X.index[start:end]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    for feature_space, matrix in [
        ("original", X_original),
        ("regime_cross_market", X),
    ]:
        Xt = matrix.loc[train_idx]
        Xv = matrix.loc[test_idx]

        for name, params in CONFIGS:
            m = make_model(params)
            A, B = impute(Xt, Xv)
            m.fit(A, y_train)
            p = m.predict_proba(B)[:, 1]
            s = score(y_test, p)

            records.append(
                {
                    "outer_start": str(test_idx[0].date()),
                    "outer_end": str(test_idx[-1].date()),
                    "feature_space": feature_space,
                    "config": name,
                    "features": matrix.shape[1],
                    **s,
                }
            )

            print(
                f"{test_idx[0].date()} -> {test_idx[-1].date()} | "
                f"{feature_space:20s} | {name:18s} | "
                f"acc={s['accuracy']:.4f} "
                f"bal={s['balanced_accuracy']:.4f} "
                f"auc={s['auc']:.4f}"
            )

results = pd.DataFrame(records)

summary = (
    results.groupby(["feature_space", "config", "features"])[
        ["accuracy", "balanced_accuracy", "f1", "auc", "logloss"]
    ]
    .mean()
    .sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=[False, False],
    )
)

print("\n" + "=" * 72)
print("OUTER WALK-FORWARD SUMMARY")
print("=" * 72)
print(summary.to_string())

# Stability matters too, so report the standard deviation of balanced accuracy.
stability = (
    results.groupby(["feature_space", "config"])[
        ["accuracy", "balanced_accuracy", "auc", "logloss"]
    ]
    .agg(["mean", "std"])
)

print("\nSTABILITY")
print(stability.to_string())

# Save research artifacts.
results.to_csv(DATA / "regime_cross_market_oos.csv", index=False)
summary.reset_index().to_csv(DATA / "regime_cross_market_summary.csv", index=False)
stability.reset_index().to_csv(DATA / "regime_cross_market_stability.csv", index=False)

meta = {
    "purpose": "Test whether backward-looking regime and cross-market transformations improve chronological out-of-sample NIFTY direction prediction.",
    "detected_sources": found_sources,
    "added_features": added_features,
    "feature_count_original": len(original_cols),
    "feature_count_regime_cross_market": len(feature_cols),
    "outer_initial": outer_initial,
    "outer_size": outer_size,
    "warning": "This is a research comparison. It does not establish a profitable trading strategy.",
}

(MODELS / "regime_cross_market_meta.json").write_text(
    json.dumps(meta, indent=2, default=str),
    encoding="utf-8",
)

print("\n" + "=" * 72)
print("SAVED")
print("=" * 72)
print(f"OOS results: {DATA / 'regime_cross_market_oos.csv'}")
print(f"Summary:     {DATA / 'regime_cross_market_summary.csv'}")
print(f"Stability:   {DATA / 'regime_cross_market_stability.csv'}")
print(f"Metadata:    {MODELS / 'regime_cross_market_meta.json'}")