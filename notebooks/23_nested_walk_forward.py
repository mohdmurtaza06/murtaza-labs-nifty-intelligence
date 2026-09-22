import sys, json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from model_features_ultimate import build_features

DATA = BASE / "data"
MODELS = BASE / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

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

feat = build_features(raw)
labeled = feat.dropna(subset=["Target"]).copy()

features = [
    c for c in feat.columns
    if c not in {"Target", "Next_Return"}
    and feat[c].notna().mean() >= 0.75
]

X = labeled[features].replace([np.inf, -np.inf], np.nan)
y = labeled["Target"].astype(int)

print("=" * 72)
print("NESTED WALK-FORWARD VALIDATION")
print("=" * 72)
print(
    f"Rows: {len(X)} | Features: {len(features)} | "
    f"Range: {X.index.min().date()} -> {X.index.max().date()}"
)

# ---------------------------------------------------------------------
# Model search space kept deliberately small enough to run locally.
# The OUTER folds are never used to choose the configuration.
# ---------------------------------------------------------------------

CONFIGS = [
    {
        "name": "cat_base",
        "depth": 4,
        "learning_rate": 0.025,
        "iterations": 350,
        "l2_leaf_reg": 12,
        "random_strength": 1.0,
        "bagging_temperature": 0.5,
    },
    {
        "name": "cat_balanced",
        "depth": 5,
        "learning_rate": 0.02,
        "iterations": 400,
        "l2_leaf_reg": 15,
        "random_strength": 1.0,
        "bagging_temperature": 0.8,
    },
    {
        "name": "cat_conservative",
        "depth": 4,
        "learning_rate": 0.02,
        "iterations": 500,
        "l2_leaf_reg": 20,
        "random_strength": 1.5,
        "bagging_temperature": 1.0,
    },
]

FEATURE_COUNTS = [30, 80, len(features)]


def make_model(cfg):
    return CatBoostClassifier(
        iterations=cfg["iterations"],
        depth=cfg["depth"],
        learning_rate=cfg["learning_rate"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        random_strength=cfg["random_strength"],
        bagging_temperature=cfg["bagging_temperature"],
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )


def rank_features(Xt, yt):
    imp = SimpleImputer(strategy="median")
    Xt_imp = pd.DataFrame(
        imp.fit_transform(Xt),
        index=Xt.index,
        columns=Xt.columns,
    )
    mi = mutual_info_classif(
        Xt_imp,
        yt,
        random_state=42,
    )
    return pd.Series(mi, index=Xt.columns).sort_values(ascending=False)


def evaluate(y_true, p):
    pred = (p >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "logloss": log_loss(y_true, p, labels=[0, 1]),
        "auc": roc_auc_score(y_true, p),
    }
    return out


def evaluate_baselines(y_true, previous_direction):
    rows = []

    # Majority class, learned only from the outer training population.
    majority = int(y_true.mean() >= 0.5)
    p_majority = np.full(len(y_true), 0.51 if majority else 0.49)
    m = evaluate(y_true, p_majority)
    m.update({"baseline": "majority_class"})
    rows.append(m)

    # Previous-session direction as a simple market baseline.
    prev = pd.Series(previous_direction).astype(float).to_numpy()
    prev = np.nan_to_num(prev, nan=0.5)
    m = evaluate(y_true, prev)
    m.update({"baseline": "previous_day_direction"})
    rows.append(m)

    # Always-up baseline.
    p_up = np.full(len(y_true), 0.51)
    m = evaluate(y_true, p_up)
    m.update({"baseline": "always_up"})
    rows.append(m)

    return rows


# ---------------------------------------------------------------------
# OUTER WALK-FORWARD
#
# Each outer test block is completely untouched by the inner search.
# ---------------------------------------------------------------------

n = len(X)
outer_initial = max(1200, int(n * 0.68))
outer_size = 220

outer_starts = list(range(outer_initial, n, outer_size))
outer_records = []
outer_baselines = []
selected_configs = []

for outer_start in outer_starts:
    outer_end = min(outer_start + outer_size, n)
    if outer_end - outer_start < 40:
        continue

    X_outer_train = X.iloc[:outer_start]
    y_outer_train = y.iloc[:outer_start]
    X_outer_test = X.iloc[outer_start:outer_end]
    y_outer_test = y.iloc[outer_start:outer_end]

    print(
        f"\nOUTER TEST: {X.index[outer_start].date()} -> "
        f"{X.index[outer_end - 1].date()} "
        f"(n={len(X_outer_test)})"
    )

    # -------------------------------------------------------------
    # INNER WALK-FORWARD SEARCH.
    # This is the only place where model/config selection happens.
    # -------------------------------------------------------------
    inner_n = len(X_outer_train)
    inner_initial = max(700, int(inner_n * 0.60))
    inner_size = max(120, int(inner_n * 0.14))
    inner_starts = list(range(inner_initial, inner_n, inner_size))

    inner_records = []

    for inner_start in inner_starts:
        inner_end = min(inner_start + inner_size, inner_n)
        if inner_end - inner_start < 60:
            continue

        Xi_t = X_outer_train.iloc[:inner_start]
        yi_t = y_outer_train.iloc[:inner_start]
        Xi_v = X_outer_train.iloc[inner_start:inner_end]
        yi_v = y_outer_train.iloc[inner_start:inner_end]

        rank = rank_features(Xi_t, yi_t)

        for k in FEATURE_COUNTS:
            selected = rank.head(k).index.tolist()

            for cfg in CONFIGS:
                model = make_model(cfg)
                model.fit(Xi_t[selected], yi_t)

                p = model.predict_proba(Xi_v[selected])[:, 1]
                metrics = evaluate(yi_v, p)

                inner_records.append(
                    {
                        "inner_fold": X_outer_train.index[inner_start].date(),
                        "config": cfg["name"],
                        "features": k,
                        **metrics,
                    }
                )

    inner_df = pd.DataFrame(inner_records)

    if inner_df.empty:
        print("No valid inner folds. Skipping outer fold.")
        continue

    inner_summary = (
        inner_df
        .groupby(["config", "features"])[
            ["accuracy", "balanced_accuracy", "logloss", "auc"]
        ]
        .mean()
        .sort_values(
            ["logloss", "balanced_accuracy"],
            ascending=[True, False],
        )
    )

    best_config, best_k = inner_summary.index[0]
    selected_configs.append(
        {
            "outer_test_start": str(X.index[outer_start].date()),
            "outer_test_end": str(X.index[outer_end - 1].date()),
            "config": best_config,
            "features": int(best_k),
        }
    )

    cfg = next(c for c in CONFIGS if c["name"] == best_config)

    # Feature ranking is recomputed using ONLY the outer training data.
    outer_rank = rank_features(X_outer_train, y_outer_train)
    selected = outer_rank.head(int(best_k)).index.tolist()

    model = make_model(cfg)
    model.fit(X_outer_train[selected], y_outer_train)

    p = model.predict_proba(X_outer_test[selected])[:, 1]
    metrics = evaluate(y_outer_test, p)

    for i, dt in enumerate(X_outer_test.index):
        outer_records.append(
            {
                "date": dt,
                "outer_fold": str(X.index[outer_start].date()),
                "config": best_config,
                "features": int(best_k),
                "prob_up": float(p[i]),
                "prediction": int(p[i] >= 0.5),
                "actual": int(y_outer_test.iloc[i]),
            }
        )

    # Previous-day direction baseline.
    # build_features is constructed from current-session information, so
    # we use the raw close series shifted one session as the baseline.
    close_col = next(
        (c for c in raw.columns if str(c).lower() in {"close", "nifty_close", "adj close"}),
        None,
    )

    if close_col is not None:
        close = raw[close_col].astype(float)
        previous_dir = close.pct_change().reindex(X_outer_test.index)
        previous_prob = (previous_dir > 0).astype(float).fillna(0.5).to_numpy()
        bm = evaluate(y_outer_test, previous_prob)
        bm.update(
            {
                "outer_fold": str(X.index[outer_start].date()),
                "baseline": "previous_day_direction",
            }
        )
        outer_baselines.append(bm)

    print(
        f"  Selected: {best_config} / {best_k} features | "
        f"accuracy={metrics['accuracy']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"logloss={metrics['logloss']:.4f} "
        f"auc={metrics['auc']:.4f}"
    )


# ---------------------------------------------------------------------
# OUTER RESULTS
# ---------------------------------------------------------------------

outer_df = pd.DataFrame(outer_records)

if outer_df.empty:
    raise RuntimeError("No outer-fold predictions were produced.")

oos_metrics = evaluate(
    outer_df["actual"].to_numpy(),
    outer_df["prob_up"].to_numpy(),
)

print("\n" + "=" * 72)
print("NESTED OUT-OF-SAMPLE RESULT")
print("=" * 72)
print(f"OOS samples:          {len(outer_df)}")
print(f"Accuracy:             {oos_metrics['accuracy']:.4f}")
print(f"Balanced accuracy:    {oos_metrics['balanced_accuracy']:.4f}")
print(f"Log loss:             {oos_metrics['logloss']:.4f}")
print(f"AUC:                  {oos_metrics['auc']:.4f}")

# Threshold analysis is descriptive only. It is NOT used to claim a
# production edge unless there is adequate coverage.
threshold_rows = []
for threshold in np.arange(0.50, 0.76, 0.02):
    confidence = np.maximum(
        outer_df["prob_up"],
        1.0 - outer_df["prob_up"],
    )
    mask = confidence >= threshold
    n_sig = int(mask.sum())

    if n_sig == 0:
        continue

    sig_y = outer_df.loc[mask, "actual"].to_numpy()
    sig_p = outer_df.loc[mask, "prob_up"].to_numpy()

    threshold_rows.append(
        {
            "threshold": round(float(threshold), 2),
            "predictions": n_sig,
            "coverage": n_sig / len(outer_df),
            "accuracy": accuracy_score(
                sig_y,
                (sig_p >= 0.5).astype(int),
            ),
            "balanced_accuracy": balanced_accuracy_score(
                sig_y,
                (sig_p >= 0.5).astype(int),
            ),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)

print("\nTHRESHOLD ANALYSIS")
if not threshold_df.empty:
    print(threshold_df.to_string(index=False))
else:
    print("No threshold had qualifying predictions.")

print("\nOUTER MODEL SELECTION FREQUENCY")
selection_df = pd.DataFrame(selected_configs)
print(
    selection_df
    .groupby(["config", "features"])
    .size()
    .sort_values(ascending=False)
    .to_string()
)

# ---------------------------------------------------------------------
# FINAL PRODUCTION SELECTION
#
# The outer test results remain untouched. For production, perform an
# INNER walk-forward over all available labeled data and select the
# configuration that has the best mean OOS log loss.
# ---------------------------------------------------------------------

print("\n" + "=" * 72)
print("FINAL PRODUCTION SELECTION")
print("=" * 72)

full_n = len(X)
prod_initial = max(900, int(full_n * 0.62))
prod_size = max(180, int(full_n * 0.12))
prod_starts = list(range(prod_initial, full_n, prod_size))

prod_records = []

for s in prod_starts:
    e = min(s + prod_size, full_n)
    if e - s < 60:
        continue

    Xt, yt = X.iloc[:s], y.iloc[:s]
    Xv, yv = X.iloc[s:e], y.iloc[s:e]

    rank = rank_features(Xt, yt)

    for k in FEATURE_COUNTS:
        selected = rank.head(k).index.tolist()

        for cfg in CONFIGS:
            model = make_model(cfg)
            model.fit(Xt[selected], yt)
            p = model.predict_proba(Xv[selected])[:, 1]
            metrics = evaluate(yv, p)

            prod_records.append(
                {
                    "fold": X.index[s].date(),
                    "config": cfg["name"],
                    "features": k,
                    **metrics,
                }
            )

prod_df = pd.DataFrame(prod_records)
prod_summary = (
    prod_df
    .groupby(["config", "features"])[
        ["accuracy", "balanced_accuracy", "logloss", "auc"]
    ]
    .mean()
    .sort_values(
        ["logloss", "balanced_accuracy"],
        ascending=[True, False],
    )
)

prod_config, prod_k = prod_summary.index[0]
prod_cfg = next(c for c in CONFIGS if c["name"] == prod_config)

final_rank = rank_features(X, y)
final_selected = final_rank.head(int(prod_k)).index.tolist()

final_model = make_model(prod_cfg)
final_model.fit(X[final_selected], y)

production_meta = {
    "architecture": "NESTED_WALK_FORWARD_CATBOOST",
    "production_config": prod_config,
    "production_feature_count": int(prod_k),
    "features": final_selected,
    "training_start": str(X.index.min().date()),
    "trained_through": str(X.index.max().date()),
    "samples": int(len(X)),
    "nested_outer_oos_metrics": oos_metrics,
    "outer_selection_frequency": (
        selection_df.groupby(["config", "features"]).size()
        .reset_index(name="count")
        .to_dict("records")
    ),
    "production_inner_summary": prod_summary.reset_index().to_dict("records"),
    "warning": (
        "Nested outer OOS metrics are the primary generalization estimate. "
        "Production model is refit after the untouched outer evaluation."
    ),
}

joblib.dump(
    {
        "model": final_model,
        "feature_cols": final_selected,
        "architecture": "NESTED_WALK_FORWARD_CATBOOST",
        "trained_through": str(X.index.max().date()),
    },
    MODELS / "nifty50_nested_model.pkl",
)

(MODELS / "nifty50_nested_meta.json").write_text(
    json.dumps(production_meta, indent=2, default=str),
    encoding="utf-8",
)

outer_df.to_csv(
    DATA / "nested_outer_oos_predictions.csv",
    index=False,
)

threshold_df.to_csv(
    DATA / "nested_outer_thresholds.csv",
    index=False,
)

prod_summary.reset_index().to_csv(
    DATA / "nested_production_search.csv",
    index=False,
)

if outer_baselines:
    pd.DataFrame(outer_baselines).to_csv(
        DATA / "nested_baseline_results.csv",
        index=False,
    )

print("\n" + "=" * 72)
print("SAVED")
print("=" * 72)
print(f"Production model: {MODELS / 'nifty50_nested_model.pkl'}")
print(f"Metadata:         {MODELS / 'nifty50_nested_meta.json'}")
print(f"Outer OOS:        {DATA / 'nested_outer_oos_predictions.csv'}")
print(f"Thresholds:       {DATA / 'nested_outer_thresholds.csv'}")
print(f"Production search:{DATA / 'nested_production_search.csv'}")
print(f"\nProduction configuration: {prod_config} / {prod_k} features")
print("Do NOT interpret threshold rows with tiny coverage as reliable.")
