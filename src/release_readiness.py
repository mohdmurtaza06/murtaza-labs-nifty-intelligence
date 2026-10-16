"""Evidence-based, fail-closed release readiness inspection.

This module reports what exists on disk. File presence alone is not treated as
proof of a valid model, an untouched holdout, prospective paper evidence, or
authorization for real-money execution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .readiness_checks import validate_market_data_freshness
except ImportError:  # Support direct imports when the src directory is on sys.path.
    from readiness_checks import validate_market_data_freshness

EXPECTED_FEATURE_COUNT = 172
EXPECTED_TARGET_HORIZON = "next_session_open_to_close_direction"
REQUIRED_PAPER_OBSERVATIONS = 200


def _gate(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **extra}


def _safe_int(value: Any, fallback: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _schema_feature_names(schema: dict[str, Any]) -> list[str] | None:
    for key in ("ordered_features", "feature_names", "features", "columns"):
        values = schema.get(key)
        if not isinstance(values, list):
            continue
        if values and all(isinstance(item, str) and item for item in values):
            return values
        names = []
        for item in values:
            if not isinstance(item, dict):
                names = []
                break
            name = item.get("name") or item.get("feature_name") or item.get("feature")
            if not isinstance(name, str) or not name:
                names = []
                break
            names.append(name)
        if names:
            return names
    return None


def _fresh_data_gate(base: Path) -> dict[str, Any]:
    path = base / "data" / "live_market_data.csv"
    if not path.is_file():
        return _gate("NOT_VERIFIED", "Active-root data/live_market_data.csv is missing.")

    try:
        frame = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close"])
        if frame.empty:
            return _gate("FAIL", "Live market-data file is empty.")
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        if dates.empty:
            return _gate("FAIL", "No valid dates exist in live market data.")
        latest = dates.max()
        for column in ("Open", "High", "Low", "Close"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or (values <= 0).any():
                return _gate("FAIL", f"OHLC column {column} has missing or non-positive values.")
        open_ = pd.to_numeric(frame["Open"], errors="coerce")
        high = pd.to_numeric(frame["High"], errors="coerce")
        low = pd.to_numeric(frame["Low"], errors="coerce")
        close = pd.to_numeric(frame["Close"], errors="coerce")
        if (high < pd.concat([open_, close], axis=1).max(axis=1)).any():
            return _gate("FAIL", "OHLC integrity failed: a High is below Open/Close.")
        if (low > pd.concat([open_, close], axis=1).min(axis=1)).any():
            return _gate("FAIL", "OHLC integrity failed: a Low is above Open/Close.")
        if frame["Date"].astype(str).duplicated().any():
            return _gate("FAIL", "Duplicate market-data session dates were found.")
        age = validate_market_data_freshness(
            latest,
            max_age_days=4,
            require_latest_completed_session=True,
        )
        return _gate(
            "PASS",
            "Latest row matches the latest completed NSE session and basic OHLC checks pass.",
            latest_session=str(latest.date()),
            age_calendar_days=int(age),
            rows=int(len(frame)),
        )
    except Exception as exc:
        return _gate("FAIL", f"Market-data freshness/quality check failed: {exc}")


def _canonical_model_gate(base: Path) -> dict[str, Any]:
    schema_path = base / "data" / "production_model_schema.json"
    if not schema_path.is_file():
        return _gate("NOT_VERIFIED", "Canonical ordered feature manifest is missing.")
    schema = _read_json(schema_path)
    if schema is None:
        return _gate("FAIL", "Canonical feature manifest is not valid JSON.")
    names = _schema_feature_names(schema)
    declared_count = schema.get("feature_count", schema.get("n_features"))
    status = str(schema.get("status", "")).upper()
    if names is None or len(names) != EXPECTED_FEATURE_COUNT:
        return _gate("FAIL", "Manifest must contain exactly 172 ordered feature names.")
    if len(set(names)) != len(names):
        return _gate("FAIL", "Feature manifest contains duplicate names.")
    if declared_count is not None and _safe_int(declared_count) != EXPECTED_FEATURE_COUNT:
        return _gate("FAIL", "Manifest's declared feature count is not 172.")
    if status != "CANONICAL":
        return _gate("FAIL", f"Manifest status is {status}, not CANONICAL.")

    schema_hash = (
        schema.get("feature_hash")
        or schema.get("schema_hash")
        or schema.get("sha256")
    )
    meta_candidates = [
        (base / "models" / "nifty50_production_candidate_meta.json",
         base / "models" / "nifty50_production_candidate.pkl"),
        (base / "models" / "nifty50_research_candidate_meta.json",
         base / "models" / "nifty50_research_candidate.pkl"),
        (base / "models" / "nifty50_production_meta.json",
         base / "models" / "nifty50_production_model.pkl"),
    ]
    meta_path = None
    model_path = None
    meta = None
    for candidate_meta, candidate_model in meta_candidates:
        loaded = _read_json(candidate_meta) if candidate_meta.is_file() else None
        if loaded is not None:
            meta_path, model_path, meta = candidate_meta, candidate_model, loaded
            break
    if meta is None or model_path is None or not model_path.is_file():
        return _gate(
            "NOT_VERIFIED",
            "No active-root model binary with matching metadata is available.",
            manifest_feature_count=len(names),
            manifest_hash=schema_hash,
        )
    if _safe_int(meta.get("feature_count")) != EXPECTED_FEATURE_COUNT:
        return _gate("FAIL", "Model metadata feature count does not match the canonical 172-feature manifest.")
    meta_hash = meta.get("feature_hash") or meta.get("schema_hash") or meta.get("feature_manifest_hash")
    if not schema_hash or not meta_hash or str(meta_hash) != str(schema_hash):
        return _gate("FAIL", "Model metadata does not prove an exact feature-manifest hash match.")
    if str(meta.get("target", "")) != EXPECTED_TARGET_HORIZON:
        return _gate("FAIL", "Model target does not match the next-session open-to-close horizon.")
    return _gate(
        "PASS",
        "Model metadata and binary match the canonical ordered feature manifest and target.",
        feature_count=EXPECTED_FEATURE_COUNT,
        manifest_hash=schema_hash,
        model_metadata=str(meta_path.relative_to(base)),
    )


def _calibration_gate(base: Path) -> dict[str, Any]:
    path = base / "models" / "nifty50_calibration_meta.json"
    if not path.is_file():
        return _gate("NOT_VERIFIED", "Independent calibration metadata is missing.")
    meta = _read_json(path)
    if meta is None:
        return _gate("FAIL", "Calibration metadata is not valid JSON.")
    if _safe_int(meta.get("feature_count")) != EXPECTED_FEATURE_COUNT:
        return _gate("FAIL", "Calibration metadata feature count does not match the canonical 172-feature contract.")
    calibration = meta.get("calibration")
    samples = meta.get("oos_samples")
    if not isinstance(calibration, dict) or not isinstance(samples, int) or samples < 100:
        return _gate("FAIL", "Calibration metadata lacks a defined method or at least 100 out-of-sample observations.")
    metrics = meta.get("holdout_last_30pct_metrics")
    if not isinstance(metrics, dict) or not all(
        key in metrics for key in ("brier", "logloss", "accuracy")
    ):
        return _gate("FAIL", "Calibration report is missing required out-of-sample metrics.")
    return _gate(
        "PASS",
        "Calibration metadata is present for the 172-feature contract; inspect reliability and baseline metrics before approval.",
        out_of_sample_samples=samples,
        method=calibration.get("type", "unspecified"),
    )


def _holdout_gate(base: Path) -> dict[str, Any]:
    path = base / "reports" / "validation" / "nifty50_time_split_validation.json"
    if not path.is_file():
        return _gate("NOT_VERIFIED", "Preserved one-time chronological holdout report is missing.")
    report = _read_json(path)
    if report is None:
        return _gate("FAIL", "Chronological holdout report is not valid JSON.")
    holdout = report.get("holdout")
    if not isinstance(holdout, dict):
        return _gate("FAIL", "Holdout section is missing from the report.")
    if holdout.get("is_untouched") is not True or report.get("holdout_reuse_invalidates_gate") is True:
        return _gate("FAIL", "The holdout is reused, exposed, or not explicitly marked untouched.")
    rows = holdout.get("rows")
    if not isinstance(rows, int) or rows < 252:
        return _gate("FAIL", "At least 252 untouched chronological holdout rows are required.")
    if report.get("target_horizon") != EXPECTED_TARGET_HORIZON:
        return _gate("FAIL", "Holdout report uses an incompatible target horizon.")
    return _gate(
        "PASS",
        "Untouched chronological holdout exists; strategy gate outcome must still be reviewed separately.",
        rows=rows,
        start=holdout.get("start"),
        end=holdout.get("end"),
        paper_gate=report.get("paper_trading_gate", "not_recorded"),
    )


def _prospective_paper_gate(base: Path) -> dict[str, Any]:
    path = base / "data" / "paper_trading_ledger.csv"
    required = {
        "signal_id", "signal_time", "data_through_date", "model_hash",
        "feature_hash", "target_horizon", "is_prospective",
        "outcome_verified_at", "actual_open", "actual_close",
        "actual_return", "simulated_costs",
    }
    if not path.is_file():
        return _gate(
            "NOT_VERIFIED",
            "Append-only prospective ledger is missing; existing prediction history is not accepted as proof.",
            required_count=REQUIRED_PAPER_OBSERVATIONS,
            verified_count=0,
        )
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return _gate("FAIL", f"Paper ledger could not be read: {exc}", required_count=200, verified_count=0)
    missing = sorted(required - set(frame.columns))
    if missing:
        return _gate(
            "FAIL",
            f"Paper ledger lacks auditable fields: {', '.join(missing)}",
            required_count=REQUIRED_PAPER_OBSERVATIONS,
            verified_count=0,
        )
    prospective = frame["is_prospective"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    signal_time = pd.to_datetime(frame["signal_time"], errors="coerce", utc=True)
    data_through = pd.to_datetime(frame["data_through_date"], errors="coerce", utc=True)
    outcome_time = pd.to_datetime(frame["outcome_verified_at"], errors="coerce", utc=True)
    resolved = outcome_time.notna()
    valid_horizon = frame["target_horizon"].astype(str).eq(EXPECTED_TARGET_HORIZON)
    unique_ids = frame["signal_id"].notna() & ~frame["signal_id"].duplicated(keep=False)
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in ("actual_open", "actual_close", "actual_return", "simulated_costs")
    }
    numeric_ok = (
        numeric["actual_open"].gt(0)
        & numeric["actual_close"].gt(0)
        & numeric["actual_return"].notna()
        & numeric["simulated_costs"].ge(0)
    )
    time_ok = (
        signal_time.notna()
        & data_through.notna()
        & outcome_time.notna()
        & (data_through <= signal_time)
        & (outcome_time >= signal_time)
    )
    hashes_ok = (
        frame["model_hash"].astype(str).str.strip().ne("")
        & frame["feature_hash"].astype(str).str.strip().ne("")
        & frame["model_hash"].astype(str).str.lower().ne("nan")
        & frame["feature_hash"].astype(str).str.lower().ne("nan")
    )
    closed = frame[
        prospective & resolved & valid_horizon & unique_ids & numeric_ok & time_ok & hashes_ok
    ]
    count = int(len(closed))
    status = "PASS" if count >= REQUIRED_PAPER_OBSERVATIONS else "FAIL"
    return _gate(
        status,
        "At least 200 distinct, prospective, closed, horizon-compatible signals are required.",
        required_count=REQUIRED_PAPER_OBSERVATIONS,
        verified_count=count,
        total_ledger_rows=int(len(frame)),
        unique_signal_ids=bool(frame["signal_id"].is_unique),
    )


def _reconciliation_gate(base: Path) -> dict[str, Any]:
    candidates = [
        base / "reports" / "niftybees_reconciliation.json",
        base / "data" / "niftybees_reconciliation.json",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return _gate("NOT_VERIFIED", "Independent NIFTYBEES reconciliation artifact is missing.")
    report = _read_json(path)
    if report is None:
        return _gate("FAIL", "NIFTYBEES reconciliation artifact is not valid JSON.")
    status = str(report.get("status", "")).upper()
    mismatches = report.get("mismatch_count", report.get("mismatches"))
    if status != "PASS" or mismatches not in (0, [], None):
        return _gate("FAIL", "NIFTYBEES reconciliation is not explicitly PASS with no reported mismatches.")
    if mismatches is None:
        return _gate("FAIL", "NIFTYBEES reconciliation must explicitly record zero mismatches.")
    return _gate("PASS", "NIFTYBEES reconciliation records PASS with zero mismatches.")


def assess_readiness(base_dir: str | Path) -> dict[str, Any]:
    """Inspect available local artifacts and always keep real-money execution disabled."""
    base = Path(base_dir).resolve()
    gates = {
        "fresh_market_data": _fresh_data_gate(base),
        "canonical_172_feature_model": _canonical_model_gate(base),
        "probability_calibration": _calibration_gate(base),
        "untouched_chronological_holdout": _holdout_gate(base),
        "prospective_paper_signals": _prospective_paper_gate(base),
        "niftybees_reconciliation": _reconciliation_gate(base),
    }
    # These controls require explicit, separately reviewed operational setup;
    # this code release does not include an enabled broker adapter.
    gates["broker_integration"] = _gate(
        "NOT_CONFIGURED",
        "No validated broker adapter, account check, or exchange/broker compliance review is recorded.",
    )
    gates["live_orders"] = _gate(
        "DISABLED",
        "Real-money order execution is intentionally unavailable in this research-only release.",
    )
    gates["real_money_approval"] = _gate(
        "NOT_APPROVED",
        "No explicit, separately documented operational and compliance approval is recorded.",
    )
    return {
        "service_status": "ONLINE",
        "readiness_status": "NOT_READY",
        "trading_mode": "RESEARCH_ONLY",
        "trade_status": "NO_TRADE",
        "production_eligible": False,
        "live_orders_enabled": False,
        "release_blocked": True,
        "gates": gates,
    }
