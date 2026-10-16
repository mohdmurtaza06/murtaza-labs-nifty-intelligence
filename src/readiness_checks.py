"""Small, dependency-light safety checks shared by production scripts and tests.

These guards are operational safeguards, not evidence that a trading strategy is
profitable. A market-calendar-aware freshness check should replace the simple
calendar-age bound when an authoritative NSE session calendar is integrated.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


def target_from_next_returns(next_returns: pd.Series) -> pd.Series:
    """Convert next-session returns to labels without inventing a final label.

    Positive return -> 1 (UP); zero/negative return -> 0 (not UP).
    Missing/infinite returns remain NaN and must be excluded from training.
    """
    returns = pd.to_numeric(next_returns, errors="coerce")
    values = returns.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values)

    target = pd.Series(np.nan, index=returns.index, dtype=float)
    if valid.any():
        valid_index = returns.index[valid]
        target.loc[valid_index] = (returns.loc[valid_index] > 0).astype(float)
    return target


def _to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # ISO dates and ISO timestamps both begin with YYYY-MM-DD.
    return date.fromisoformat(str(value).strip()[:10])


def validate_market_data_freshness(
    latest_date,
    *,
    now=None,
    max_age_days: int = 4,
) -> int:
    """Reject missing/future/old observations; return calendar age in days.

    The four-day default tolerates a normal Friday-to-Monday gap. It is a
    conservative stale-data guard, not a substitute for the official NSE
    trading calendar. Extended market holidays can require a deliberate,
    documented override rather than silently accepting stale data.
    """
    if latest_date is None or str(latest_date).strip() == "":
        raise ValueError("Market data has no observation date.")
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative.")

    observed = _to_date(latest_date)

    if now is None:
        today = datetime.now(IST).date()
    elif isinstance(now, datetime):
        current = now.replace(tzinfo=IST) if now.tzinfo is None else now.astimezone(IST)
        today = current.date()
    else:
        today = _to_date(now)

    age_days = (today - observed).days
    if age_days < 0:
        raise ValueError(
            f"Market data date {observed.isoformat()} is in the future "
            f"relative to {today.isoformat()}."
        )
    if age_days > max_age_days:
        raise ValueError(
            f"Market data is stale: latest observation {observed.isoformat()} "
            f"is {age_days} calendar days old (limit {max_age_days})."
        )
    return age_days


def validate_probability_pair(
    probability_up,
    probability_down,
    *,
    tolerance: float = 0.002,
) -> tuple[float, float]:
    """Validate a binary classifier's probability output before publishing it."""
    try:
        up = float(probability_up)
        down = float(probability_down)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction probabilities must be numeric.") from exc

    if not np.isfinite(up) or not np.isfinite(down):
        raise ValueError("Prediction probabilities must be finite.")
    if not (0.0 <= up <= 1.0 and 0.0 <= down <= 1.0):
        raise ValueError("Prediction probabilities must be within [0, 1].")
    if abs((up + down) - 1.0) > tolerance:
        raise ValueError(
            "Binary prediction probabilities must sum to 1 "
            f"(received {up + down:.6f})."
        )
    return up, down
