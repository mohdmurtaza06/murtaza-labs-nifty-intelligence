"""One-shot chronological validation of the next-session open-to-close target.

This script does not alter the live model or turn on trading. It reserves the
last 252 labeled observations as a holdout, calibrates only on an earlier
chronological calibration segment, and refuses to overwrite a prior report
unless ALLOW_HOLDOUT_REUSE=true is explicitly set (which invalidates the
"untouched" claim and is intended only for exploratory debugging).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports" / "validation"
REPORT_FILE = REPORTS / "nifty50_time_split_validation.json"
PREDICTIONS_FILE = REPORTS / "nifty50_time_split_holdout_predictions.csv"
EXT_PATH = DATA / "nifty50_external_features.csv"
LIVE_PATH = DATA / "live_market_data.csv"
HOLDOUT_ROWS = int(os.environ.get("NIFTY_HOLDOUT_ROWS", "252"))
MIN_TRAIN_ROWS = int(os.environ.get("NIFTY_MIN_TRAIN_ROWS", "500"))
MIN_CALIBRATION_ROWS = int(os.environ.get("NIFTY_MIN_CALIBRATION_ROWS", "100"))
ROUND_TRIP_COST_BPS = float(os.environ.get("NIFTY_ROUND_TRIP_COST_BPS", "20"))
EXPECTED_TARGET_HORIZON = "next_session_open_to_close_direction"

sys.path.insert(0, str(BASE / "src"))
from model_features_ultimate import build_features
from readiness_checks import target_from_next_returns

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


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def _compound_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def _expected_calibration_error(y_true, probability, bins=10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p >= edges[idx]) & (p < edges[idx + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def main():
    if REPORT_FILE.exists() and os.environ.get("ALLOW_HOLDOUT_REUSE", "").lower() != "true":
        raise SystemExit(
            f"Refusing to overwrite prior holdout report: {REPORT_FILE}. "
            "Use a newly registered holdout for a genuine retest. "
            "ALLOW_HOLDOUT_REUSE=true is exploratory only and forfeits the untouched-holdout claim."
        )

    raw = load_data()
    if raw.empty:
        raise RuntimeError("No market observations found; validation aborted.")
    required_raw = {"Open", "High", "Low", "Close", "Volume"}
    missing_raw = sorted(required_raw - set(raw.columns))
    if missing_raw:
        raise RuntimeError(f"Required raw columns are missing: {missing_raw}")

    raw_enriched, detected_sources = add_safe_cross_market_features(raw)
    features = build_features(raw_enriched)

    production_columns = [c for c in raw_enriched.columns if c.startswith("PROD_")]
    if production_columns:
        extra = (
            raw_enriched[production_columns]
            .apply(pd.to_numeric, errors="coerce")
            .reindex(features.index)
        )
        features = pd.concat([features, extra], axis=1)
        features = features.loc[:, ~features.columns.duplicated()]

    opens = pd.to_numeric(raw_enriched["Open"], errors="coerce").reindex(features.index)
    closes = pd.to_numeric(raw_enriched["Close"], errors="coerce").reindex(features.index)

    # A post-close prediction can be entered at the next session's open, not
    # at today's already-known close. Score the return actually available to
    # an open-to-close paper trade on the next session.
    next_open = opens.shift(-1)
    next_close = closes.shift(-1)
    next_returns = next_close / next_open - 1.0
    target = target_from_next_returns(next_returns)

    feature_cols_all = [
        c for c in features.columns if c not in {"Target", "Next_Return"}
    ]
    X_all = features[feature_cols_all].replace([np.inf, -np.inf], np.nan)
    valid_mask = target.notna() & X_all.notna().any(axis=1)
    X_all = X_all.loc[valid_mask].sort_index()
    y_all = target.loc[X_all.index].astype(int)
    returns_all = next_returns.loc[X_all.index].astype(float)

    if len(X_all) <= HOLDOUT_ROWS + MIN_TRAIN_ROWS + MIN_CALIBRATION_ROWS:
        raise RuntimeError(
            f"Insufficient labeled history ({len(X_all)} rows) for "
            f"{MIN_TRAIN_ROWS} train + {MIN_CALIBRATION_ROWS} calibration + "
            f"{HOLDOUT_ROWS} holdout rows."
        )

    pre_holdout_count = len(X_all) - HOLDOUT_ROWS
    train_count = int(pre_holdout_count * 0.80)
    calibration_count = pre_holdout_count - train_count
    if train_count < MIN_TRAIN_ROWS or calibration_count < MIN_CALIBRATION_ROWS:
        raise RuntimeError(
            f"Temporal split is too small: train={train_count}, "
            f"calibration={calibration_count}, holdout={HOLDOUT_ROWS}."
        )

    train_index = X_all.index[:train_count]
    calibration_index = X_all.index[train_count:pre_holdout_count]
    holdout_index = X_all.index[pre_holdout_count:]

    # Feature inclusion and imputation are fitted only on the training slice.
    X_train_all = X_all.loc[train_index]
    feature_cols = [
        c for c in X_train_all.columns
        if X_train_all[c].notna().mean() >= 0.75
    ]
    if not feature_cols:
        raise RuntimeError("No features passed the training-only availability threshold.")

    X_train_raw = X_train_all[feature_cols].copy()
    medians = X_train_raw.median(numeric_only=True)
    X_train = X_train_raw.fillna(medians).fillna(0.0)
    X_calibration = X_all.loc[calibration_index, feature_cols].fillna(medians).fillna(0.0)
    X_holdout = X_all.loc[holdout_index, feature_cols].fillna(medians).fillna(0.0)

    y_train = y_all.loc[train_index]
    y_calibration = y_all.loc[calibration_index]
    y_holdout = y_all.loc[holdout_index]
    if y_train.nunique() < 2 or y_calibration.nunique() < 2 or y_holdout.nunique() < 2:
        raise RuntimeError("A temporal split lacks both target classes; validation is inconclusive.")

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
    model.fit(X_train, y_train)

    # Fit a simple sigmoid calibrator on its own later chronological slice.
    raw_calibration_probability = model.predict_proba(X_calibration)[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(raw_calibration_probability.reshape(-1, 1), y_calibration)

    raw_probability_up = model.predict_proba(X_holdout)[:, 1]
    probability_up = calibrator.predict_proba(raw_probability_up.reshape(-1, 1))[:, 1]
    probability_up = np.clip(probability_up, 0.0, 1.0)
    probability_down = 1.0 - probability_up
    predictions = (probability_up >= 0.5).astype(int)
    actual = y_holdout.to_numpy(dtype=int)
    actual_returns = returns_all.loc[holdout_index].to_numpy(dtype=float)

    accuracy = float(accuracy_score(actual, predictions))
    always_up_accuracy = float(np.mean(actual == 1))
    majority_accuracy = float(max(np.mean(actual == 1), np.mean(actual == 0)))
    brier = float(brier_score_loss(actual, probability_up))
    climatology_probability = float(np.mean(y_train))
    baseline_brier = float(brier_score_loss(
        actual, np.full(len(actual), climatology_probability, dtype=float)
    ))
    brier_skill = float(1.0 - brier / baseline_brier) if baseline_brier > 0 else None
    try:
        auc = float(roc_auc_score(actual, probability_up))
    except ValueError:
        auc = None

    # One long-only trade per positive signal: enter next session open, exit
    # that session close. Round-trip cost is a conservative configurable
    # assumption; it must be replaced with the instrument/broker-specific cost.
    positions = predictions.astype(bool)
    costs = ROUND_TRIP_COST_BPS / 10000.0
    strategy_net = np.where(
        positions,
        (1.0 + actual_returns) * (1.0 - costs) - 1.0,
        0.0,
    )
    always_long_net = (1.0 + actual_returns) * (1.0 - costs) - 1.0
    strategy_series = pd.Series(strategy_net, index=holdout_index)
    always_long_series = pd.Series(always_long_net, index=holdout_index)
    trades = int(positions.sum())

    # Pre-registered software-level screen. Passing this only authorizes a
    # paper-trading phase, never broker/live orders.
    gates = {
        "holdout_accuracy_beats_always_up": accuracy > always_up_accuracy,
        "holdout_accuracy_beats_majority_class": accuracy > majority_accuracy,
        "positive_brier_skill": brier_skill is not None and brier_skill > 0,
        "positive_net_open_to_close_strategy_return": _compound_return(strategy_series) > 0,
        "beats_always_long_same_horizon_after_costs": (
            _compound_return(strategy_series) > _compound_return(always_long_series)
        ),
        "minimum_30_holdout_trades": trades >= 30,
    }
    paper_gate_passed = all(gates.values())

    report = {
        "schema_version": 1,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "target_horizon": EXPECTED_TARGET_HORIZON,
        "execution_assumption": "prediction after prior close; enter at next open; exit same-session close",
        "market_data_latest_date": str(pd.Timestamp(raw.index.max()).date()),
        "data_rows_total": int(len(raw)),
        "labeled_rows": int(len(X_all)),
        "train": {
            "start": str(pd.Timestamp(train_index.min()).date()),
            "end": str(pd.Timestamp(train_index.max()).date()),
            "rows": int(len(train_index)),
        },
        "calibration": {
            "start": str(pd.Timestamp(calibration_index.min()).date()),
            "end": str(pd.Timestamp(calibration_index.max()).date()),
            "rows": int(len(calibration_index)),
            "method": "sigmoid_logistic_on_held-out_train-period_probabilities",
        },
        "holdout": {
            "start": str(pd.Timestamp(holdout_index.min()).date()),
            "end": str(pd.Timestamp(holdout_index.max()).date()),
            "rows": int(len(holdout_index)),
            "probabilities_calibrated_on_separate_earlier_slice": True,
        },
        "model": {
            "architecture": "CatBoostClassifier",
            "feature_count": int(len(feature_cols)),
            "features_selected_from_training_slice_only": True,
            "medians_fit_on_training_slice_only": True,
            "detected_cross_market_sources": detected_sources,
        },
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
            "precision_up": float(precision_score(actual, predictions, zero_division=0)),
            "recall_up": float(recall_score(actual, predictions, zero_division=0)),
            "f1_up": float(f1_score(actual, predictions, zero_division=0)),
            "roc_auc": auc,
            "always_up_accuracy": always_up_accuracy,
            "majority_class_accuracy": majority_accuracy,
            "accuracy_minus_always_up_percentage_points": (accuracy - always_up_accuracy) * 100,
            "brier_score_calibrated": brier,
            "brier_score_train_climatology_baseline": baseline_brier,
            "brier_skill_score": brier_skill,
            "log_loss_calibrated": float(log_loss(actual, probability_up, labels=[0, 1])),
            "expected_calibration_error_10_bins": _expected_calibration_error(actual, probability_up),
            "round_trip_cost_bps_assumption": ROUND_TRIP_COST_BPS,
            "long_signals": trades,
            "holdout_exposure_pct": float(positions.mean() * 100),
            "net_strategy_compounded_return_pct": _compound_return(strategy_series) * 100,
            "net_strategy_max_drawdown_pct": _max_drawdown(strategy_series) * 100,
            "net_always_long_same_horizon_return_pct": _compound_return(always_long_series) * 100,
            "strategy_minus_always_long_percentage_points": (
                _compound_return(strategy_series) - _compound_return(always_long_series)
            ) * 100,
            "strategy_win_rate_on_trades_pct": (
                float((strategy_net[positions] > 0).mean() * 100) if trades else None
            ),
        },
        "paper_trading_gate": "PASS_FOR_PAPER_TRADING_ONLY" if paper_gate_passed else "HOLD_NO_TRADE",
        "paper_gate_checks": gates,
        "prospective_paper_observations_closed": 0,
        "production_eligible": False,
        "live_orders_enabled": False,
        "real_money_approval": "NO",
        "note": (
            "A single chronological holdout is evidence for this candidate only. "
            "The result does not authorize live trading; 200 genuine prospective "
            "paper observations and independent instrument reconciliation remain required."
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "prediction_date": holdout_index.strftime("%Y-%m-%d"),
        "actual_open_to_close_return": actual_returns,
        "actual_label": actual,
        "raw_probability_up": raw_probability_up,
        "calibrated_probability_up": probability_up,
        "predicted_label": predictions,
        "long_position": positions.astype(int),
        "net_strategy_return": strategy_net,
    }).to_csv(PREDICTIONS_FILE, index=False)
    REPORT_FILE.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 76)
    print("MURTAZA LABS: CHRONOLOGICAL HOLDOUT VALIDATION")
    print("=" * 76)
    print(f"Training rows       : {len(train_index)}")
    print(f"Calibration rows    : {len(calibration_index)}")
    print(f"Holdout rows        : {len(holdout_index)}")
    print(f"Holdout date range  : {report['holdout']['start']} to {report['holdout']['end']}")
    print(f"Accuracy            : {accuracy:.3f}")
    print(f"Always-UP accuracy  : {always_up_accuracy:.3f}")
    print(f"Brier skill score   : {brier_skill}")
    print(f"Net strategy return : {report['metrics']['net_strategy_compounded_return_pct']:.2f}%")
    print(f"Same-horizon long return: {report['metrics']['net_always_long_same_horizon_return_pct']:.2f}%")
    print(f"Paper gate          : {report['paper_trading_gate']}")
    print("Live orders         : DISABLED")
    print(f"Report              : {REPORT_FILE}")
    print(f"Holdout predictions : {PREDICTIONS_FILE}")
    print("=" * 76)


if __name__ == "__main__":
    main()
