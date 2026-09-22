import sys, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
sys.path.insert(0, str(BASE / "src"))

from model_features_ultimate import build_features

EXT_PATH = DATA / "nifty50_external_features.csv"
LIVE_PATH = DATA / "live_market_data.csv"

ext = pd.read_csv(EXT_PATH, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
live = pd.read_csv(LIVE_PATH, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
raw = pd.concat([ext, live[live.index > ext.index.max()]], axis=0, sort=False).sort_index()

feat = build_features(raw)

close_col = next(
    (c for c in raw.columns if str(c).lower() in {"close", "nifty_close", "adj close"}),
    None,
)
if close_col is None:
    raise RuntimeError("Could not find a NIFTY close column.")

close = raw[close_col].astype(float).reindex(feat.index)
next_ret = close.shift(-1) / close - 1.0

features = [
    c for c in feat.columns
    if c not in {"Target", "Next_Return"}
    and feat[c].notna().mean() >= 0.75
]

X_all = feat[features].replace([np.inf, -np.inf], np.nan)

# The target is deliberately fixed BEFORE any model selection:
# 1 = next-day move >= +0.5%
# 0 = next-day move <= -0.5%
# Small moves are FLAT and excluded from this directional task.
MOVE = 0.005
eligible = next_ret.abs() >= MOVE
y_all = (next_ret >= MOVE).astype(int)

valid = X_all.index[eligible & next_ret.notna()]
X = X_all.loc[valid]
y = y_all.loc[valid]

print("=" * 72)
print("NESTED WALK-FORWARD: ±0.5% NEXT-DAY TARGET")
print("=" * 72)
print(f"Eligible rows: {len(X)}")
print(f"Features:      {len(features)}")
print(f"Range:         {X.index.min().date()} -> {X.index.max().date()}")
print(f"Move target:   +/- {MOVE:.2%}")
print(f"UP labels:     {int(y.sum())}")
print(f"DOWN labels:   {int((1-y).sum())}")

CONFIGS = [
    ("cat_base", dict(depth=4, learning_rate=0.025, iterations=350, l2_leaf_reg=12, random_strength=1.0, bagging_temperature=0.5)),
    ("cat_conservative", dict(depth=4, learning_rate=0.02, iterations=500, l2_leaf_reg=20, random_strength=1.5, bagging_temperature=1.0)),
    ("cat_balanced", dict(depth=5, learning_rate=0.02, iterations=400, l2_leaf_reg=15, random_strength=1.0, bagging_temperature=0.8)),
]

FEATURE_COUNTS = [30, 80, len(features)]


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


def metrics(y_true, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan,
        "logloss": log_loss(y_true, p, labels=[0, 1]),
    }


# Outer folds are the only final evidence.
# Inner folds select feature count and CatBoost configuration.
n = len(X)
outer_initial = max(700, int(n * 0.60))
outer_size = max(100, int(n * 0.15))
outer_starts = list(range(outer_initial, n, outer_size))

outer_predictions = []
outer_selection = []

for os_ in outer_starts:
    oe = min(os_ + outer_size, n)
    if oe - os_ < 50:
        continue

    Xt = X.iloc[:os_]
    yt = y.iloc[:os_]
    Xtest = X.iloc[os_:oe]
    ytest = y.iloc[os_:oe]

    inner_n = len(Xt)
    inner_initial = max(450, int(inner_n * 0.62))
    inner_size = max(80, int(inner_n * 0.14))
    inner_starts = list(range(inner_initial, inner_n, inner_size))

    search_rows = []

    for s in inner_starts:
        e = min(s + inner_size, inner_n)
        if e - s < 40:
            continue

        Xi_t, yi_t = Xt.iloc[:s], yt.iloc[:s]
        Xi_v, yi_v = Xt.iloc[s:e], yt.iloc[s:e]

        # Use CatBoost's own feature importance from the inner training set.
        # The validation portion is never used to rank features.
        rank_model = make_model(CONFIGS[0][1])
        Xi_t2, Xi_v2 = impute(Xi_t, Xi_v)
        rank_model.fit(Xi_t2, yi_t)

        imp = pd.Series(rank_model.get_feature_importance(), index=features)
        ranking = imp.sort_values(ascending=False)

        for k in FEATURE_COUNTS:
            selected = ranking.head(k).index.tolist()

            for name, params in CONFIGS:
                m = make_model(params)
                A, B = impute(Xi_t[selected], Xi_v[selected])
                m.fit(A, yi_t)
                p = m.predict_proba(B)[:, 1]
                mm = metrics(yi_v, p)
                search_rows.append({
                    "config": name,
                    "features": k,
                    **mm,
                })

    search = pd.DataFrame(search_rows)
    if search.empty:
        continue

    summary = (
        search.groupby(["config", "features"])[
            ["balanced_accuracy", "accuracy", "f1", "auc", "logloss"]
        ].mean()
        .sort_values(
            ["balanced_accuracy", "accuracy", "f1"],
            ascending=[False, False, False],
        )
    )

    best_name, best_k = summary.index[0]
    params = dict(next(p for nme, p in CONFIGS if nme == best_name))

    rank_model = make_model(params)
    Xt_rank, _ = impute(Xt, Xtest)
    rank_model.fit(Xt_rank, yt)
    imp = pd.Series(rank_model.get_feature_importance(), index=features)
    selected = imp.sort_values(ascending=False).head(int(best_k)).index.tolist()

    final = make_model(params)
    A, B = impute(Xt[selected], Xtest[selected])
    final.fit(A, yt)
    p = final.predict_proba(B)[:, 1]

    mm = metrics(ytest, p)
    outer_selection.append({
        "outer_start": str(X.index[os_].date()),
        "outer_end": str(X.index[oe - 1].date()),
        "config": best_name,
        "features": int(best_k),
        **mm,
    })

    for dt, prob, actual in zip(Xtest.index, p, ytest):
        outer_predictions.append({
            "date": dt,
            "prob_up": float(prob),
            "prediction": int(prob >= 0.5),
            "actual": int(actual),
            "config": best_name,
            "features": int(best_k),
        })

    print(
        f"OUTER {X.index[os_].date()} -> {X.index[oe-1].date()} | "
        f"{best_name}/{best_k} | "
        f"acc={mm['accuracy']:.4f} bal={mm['balanced_accuracy']:.4f} "
        f"auc={mm['auc']:.4f}"
    )

oos = pd.DataFrame(outer_predictions)
if oos.empty:
    raise RuntimeError("No outer predictions produced.")

p = oos["prob_up"].to_numpy()
ytrue = oos["actual"].to_numpy()
overall = metrics(ytrue, p)

print("\n" + "=" * 72)
print("STRICT NESTED OOS RESULT")
print("=" * 72)
print(f"OOS signal samples: {len(oos)}")
for k, v in overall.items():
    print(f"{k:20s}: {v:.6f}")

# Coverage/accuracy by confidence threshold.
rows = []
conf = np.maximum(p, 1 - p)
for threshold in np.arange(0.50, 0.71, 0.02):
    mask = conf >= threshold
    if mask.sum() < 30:
        continue
    yy = ytrue[mask]
    pp = p[mask]
    mm = metrics(yy, pp)
    rows.append({
        "threshold": round(float(threshold), 2),
        "signals": int(mask.sum()),
        "coverage": float(mask.mean()),
        **mm,
    })

thresholds = pd.DataFrame(rows)

print("\nTHRESHOLD RESULTS")
print(thresholds.to_string(index=False) if not thresholds.empty else "No threshold with >=30 signals.")

selection = pd.DataFrame(outer_selection)
print("\nOUTER SELECTION FREQUENCY")
print(
    selection.groupby(["config", "features"]).size().sort_values(ascending=False).to_string()
)

# Final production model is fitted on all eligible observations using the
# most frequently selected configuration. This model is NOT used for OOS metrics.
if not selection.empty:
    freq = (
        selection.groupby(["config", "features"]).size()
        .sort_values(ascending=False)
    )
    prod_name, prod_k = freq.index[0]
else:
    prod_name, prod_k = "cat_base", len(features)

prod_params = dict(next(p for nme, p in CONFIGS if nme == prod_name))
rank_model = make_model(prod_params)
A, _ = impute(X, X)
rank_model.fit(A, y)
importance = pd.Series(rank_model.get_feature_importance(), index=features)
prod_features = importance.sort_values(ascending=False).head(int(prod_k)).index.tolist()

prod_model = make_model(prod_params)
A, _ = impute(X[prod_features], X[prod_features])
prod_model.fit(A, y)

joblib.dump(
    {
        "model": prod_model,
        "feature_cols": prod_features,
        "target": "next_day_abs_return_ge_0.5pct",
        "move_threshold": MOVE,
        "architecture": "NESTED_WALK_FORWARD_CATBOOST",
        "trained_through": str(X.index.max().date()),
    },
    MODELS / "nifty50_05pct_nested_model.pkl",
)

meta = {
    "target": "next_day_abs_return_ge_0.5pct",
    "move_threshold": MOVE,
    "outer_oos_metrics": overall,
    "outer_selection_frequency": selection.groupby(["config", "features"]).size().reset_index(name="count").to_dict("records"),
    "production_config": prod_name,
    "production_feature_count": int(prod_k),
    "production_features": prod_features,
    "warning": "Outer OOS metrics are the primary generalization estimate. Thresholds with low coverage are not reliable evidence.",
}

(MODELS / "nifty50_05pct_nested_meta.json").write_text(
    json.dumps(meta, indent=2, default=str),
    encoding="utf-8",
)

oos.to_csv(DATA / "nested_05pct_oos_predictions.csv", index=False)
thresholds.to_csv(DATA / "nested_05pct_thresholds.csv", index=False)
selection.to_csv(DATA / "nested_05pct_outer_selection.csv", index=False)

print("\n" + "=" * 72)
print("SAVED")
print("=" * 72)
print(f"Model:       {MODELS / 'nifty50_05pct_nested_model.pkl'}")
print(f"Metadata:    {MODELS / 'nifty50_05pct_nested_meta.json'}")
print(f"OOS:         {DATA / 'nested_05pct_oos_predictions.csv'}")
print(f"Thresholds:  {DATA / 'nested_05pct_thresholds.csv'}")
