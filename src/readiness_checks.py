"""Small, dependency-light safety checks shared by production scripts and tests.

These guards do not prove a strategy profitable. Production callers should
require NSE-session freshness; plain calendar-age checking is available for
unit tests and non-production tooling.
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


def target_from_next_returns(next_returns: pd.Series) -> pd.Series:
    """Convert next-session returns to labels without inventing missing labels.

    Positive return -> 1 (UP); negative return -> 0 (DOWN).
    Exact-flat, missing, and infinite returns remain NaN and are excluded from
    binary directional training, matching the prospective outcome verifier.
    """
    returns = pd.to_numeric(next_returns, errors="coerce")
    values = returns.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values) & (values != 0.0)

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
    return date.fromisoformat(str(value).strip()[:10])


def _last_completed_nse_session(current: datetime) -> date:
    """Resolve the latest fully completed NSE equity session from its calendar."""
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise ValueError(
            "NSE session-calendar dependency is unavailable; refusing to "
            "declare market data fresh."
        ) from exc

    today = current.astimezone(IST).date()
    try:
        calendar = mcal.get_calendar("NSE")
        schedule = calendar.schedule(
            start_date=(today - timedelta(days=45)).isoformat(),
            end_date=today.isoformat(),
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to resolve the NSE session calendar: {exc}"
        ) from exc

    if schedule.empty or "market_close" not in schedule.columns:
        raise ValueError("NSE session calendar returned no usable market schedule.")

    market_closes = pd.to_datetime(schedule["market_close"], utc=True).dt.tz_convert(IST)
    current_ts = pd.Timestamp(current)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize(IST)
    else:
        current_ts = current_ts.tz_convert(IST)

    completed = schedule.loc[market_closes <= current_ts]
    if completed.empty:
        raise ValueError("No completed NSE session could be resolved.")
    return pd.Timestamp(completed.index[-1]).date()


def validate_market_data_freshness(
    latest_date,
    *,
    now=None,
    max_age_days: int = 4,
    require_latest_completed_session: bool = False,
) -> int:
    """Reject missing/future/old observations.

    If require_latest_completed_session is true, the observed date must equal
    the last fully completed session according to pandas-market-calendars'
    NSE calendar. This avoids accepting Friday's bar as current on Monday after
    the exchange closes. The calendar is package-shipped, not a live NSE feed,
    so operators should keep it updated and verify exceptional/adhoc holidays.
    """
    if latest_date is None or str(latest_date).strip() == "":
        raise ValueError("Market data has no observation date.")
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative.")

    observed = _to_date(latest_date)

    if now is None:
        current = datetime.now(IST)
    elif isinstance(now, datetime):
        current = now.replace(tzinfo=IST) if now.tzinfo is None else now.astimezone(IST)
    else:
        current = datetime.combine(_to_date(now), datetime.min.time(), tzinfo=IST)

    today = current.date()
    age_days = (today - observed).days
    if age_days < 0:
        raise ValueError(
            f"Market data date {observed.isoformat()} is in the future "
            f"relative to {today.isoformat()}."
        )

    if require_latest_completed_session:
        expected = _last_completed_nse_session(current)
        if observed != expected:
            relation = "older than" if observed < expected else "newer than"
            raise ValueError(
                f"Market data is not current: observed {observed.isoformat()} is "
                f"{relation} the latest completed NSE session {expected.isoformat()}."
            )
        return age_days

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
