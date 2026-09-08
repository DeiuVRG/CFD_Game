"""Candle helpers for the live monitor."""
from datetime import datetime, timezone

import pandas as pd

from config.settings import parse_interval_hours


def split_incomplete_candle(df: pd.DataFrame, interval: str,
                            now: datetime = None):
    """(completed candles, forming candle as a dict or None).

    Data sources return the current (incomplete) candle as the last row,
    with its OPEN time as timestamp. The v3 model evaluates signals on the
    CLOSE of completed candles only, so a candle whose open + interval is
    still in the future must not be fed to the features or to the outcome
    replay - but its OPEN is exactly the backtest's entry price ("next
    candle open"), which is why it is returned separately.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return df, None
    now = now or datetime.now(timezone.utc)
    last_open = pd.Timestamp(df["timestamp"].iloc[-1])
    last_open = (last_open.tz_localize("UTC") if last_open.tzinfo is None
                 else last_open.tz_convert("UTC"))
    closes_at = last_open + pd.Timedelta(hours=parse_interval_hours(interval))
    if closes_at > pd.Timestamp(now):
        return df.iloc[:-1].reset_index(drop=True), df.iloc[-1].to_dict()
    return df, None


def drop_incomplete_candle(df: pd.DataFrame, interval: str,
                           now: datetime = None) -> pd.DataFrame:
    """Completed candles only (see split_incomplete_candle)."""
    return split_incomplete_candle(df, interval, now)[0]
