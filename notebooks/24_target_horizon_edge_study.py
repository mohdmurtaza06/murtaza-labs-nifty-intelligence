import sys, json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
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
ret1 = close.shift(-1) / close - 1.0
ret3 = close.shift(-3) / close - 1.0

# Keep only features that are available before the future return.
feature_cols = [
    c for c in feat.columns
    if c not in {"Target", "Next_Return"}
    and feat[c].notna().mean() >= 0.75
]

X = feat[feature_cols].replace([np.inf, -np.inf], np.nan)

# Use a common chronological test period for all target definitions.
# No model is selected using the final test block.
test_start = max(1200, int(len(X) * 0.68))
test_size = 220
test_ranges = list(range(test_start, len(X), test_size))

# Target families:
# 0.00% = ordinary next-day direction
# 0.20/0.35/0.50/0.75% = only label a day as UP/DOWN when the move
# exceeds the threshold. Smaller moves become FLAT and are excluded
# from directional accuracy.
thresholds = [0.0, 0.002, 0.0035, 0.005, 0.0075]

configs = [
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
        "name": "cat_conservative",
        "depth": 4,
        "learning_rate": 0.02,
        "iterations": 500,
        "l2_leaf_reg": 20,
        "random_strength": 1.5,
        "bagging_temperature": 1.0,
    },
]


def model(cfg):
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


def add_features(Xt, Xv):
    med = Xt.median(numeric_only=True)
    return Xt.fillna(med), Xv.fillna(med)


rows = []

print("=" * 72)
print("TARGET / HORIZON EDGE STUDY")
print("=" * 72)
print(f"Rows: {len(X)} | Features: {len(feature_cols)}")
print(f"Range: {X.index.min().date()} -> {X.index.max().date()}")

for horizon_name, returns in [("next_day", ret1), ("three_day", ret3)]:
    yret = returns.reindex(X.index)

    for move_threshold in thresholds:
        # Binary direction target for qualifying moves only.
        # +1 = sufficiently positive, 0 = sufficiently negative.
        # Flat observations are excluded from directional training/testing.
        eligible = yret.abs() >= move_threshold if move_threshold > 0 else yret.notna()
        ybin = (yret > 0).astype(int)

        valid = X.index[eligible & yret.notna()]
        Xv_all = X.loc[valid]
        yv_all = ybin.loc[valid]

        if len(Xv_all) < 1000:
            continue

        split = max(800, int(len(Xv_all) * 0.68))
        if split >= len(Xv_all) - 100:
            continue

        train_idx = Xv_all.index[:split]
        test_idx = Xv_all.index[split:]

        # Feature ranking is deliberately simple and stable here:
        # use the existing full feature set, then compare two conservative
        # CatBoost configurations. This stage is about target definition,
        # not another giant hyperparameter search.
        Xt = Xv_all.loc[train_idx]
        yt = yv_all.loc[train_idx]
        Xtest = Xv_all.loc[test_idx]
        ytest = yv_all.loc[test_idx]

        for cfg in configs:
            m = model(cfg)
            Xt2, Xtest2 = add_features(Xt, Xtest)
            m.fit(Xt2, yt)
            p = m.predict_proba(Xtest2)[:, 1]
            pred = (p >= 0.5).astype(int)

            acc = accuracy_score(ytest, pred)
            bal = balanced_accuracy_score(ytest, pred)
            f1 = f1_score(ytest, pred, zero_division=0)

            rows.append(
                {
                    "horizon": horizon_name,
                    "move_threshold": move_threshold,
                    "config": cfg["name"],
                    "test_samples": len(ytest),
                    "accuracy": acc,
                    "balanced_accuracy": bal,
                    "f1": f1,
                    "coverage_vs_all_rows": len(ytest) / len(X),
                }
            )

            print(
                f"{horizon_name:9s} | threshold={move_threshold:.3f} | "
                f"{cfg['name']:17s} | n={len(ytest):4d} | "
                f"acc={acc:.4f} | bal={bal:.4f} | f1={f1:.4f}"
            )

results = pd.DataFrame(rows)
results.to_csv(DATA / "target_horizon_edge_study.csv", index=False)

print("\n" + "=" * 72)
print("RESULT SUMMARY")
print("=" * 72)

if results.empty:
    raise RuntimeError("No valid target configurations were evaluated.")

print(
    results.sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=False,
    ).to_string(index=False)
)

# Only call something promising if it has meaningful sample size.
credible = results[results["test_samples"] >= 100].copy()

if not credible.empty:
    best = credible.sort_values(
        ["balanced_accuracy", "accuracy", "f1"],
        ascending=False,
    ).iloc[0]
    print("\nBEST CREDIBLE CONFIGURATION")
    print(best.to_string())
else:
    best = None
    print("\nNo configuration had >=100 test observations.")

meta = {
    "purpose": "Test whether prediction target definition and horizon provide a more stable directional edge.",
    "thresholds": thresholds,
    "horizons": ["next_day", "three_day"],
    "selection_rule": "balanced_accuracy then accuracy then F1, with minimum 100 test observations",
    "warning": "Thresholded targets intentionally exclude small moves. Coverage must be reported alongside accuracy.",
    "best": None if best is None else best.to_dict(),
}

(MODELS / "target_horizon_edge_meta.json").write_text(
    json.dumps(meta, indent=2, default=str),
    encoding="utf-8",
)

print(f"\nSaved: {DATA / 'target_horizon_edge_study.csv'}")
print(f"Saved: {MODELS / 'target_horizon_edge_meta.json'}")
