"""Candle helpers for the live monitor."""
from datetime import datetime, timezone

import pandas as pd

from config.settings import parse_interval_hours


def drop_incomplete_candle(df: pd.DataFrame, interval: str,
                           now: datetime = None) -> pd.DataFrame:
    """Drop the last row when it is the candle still forming.

    yfinance returns the current (incomplete) candle as the last row, with
    its OPEN time as timestamp. The v3 model evaluates signals on the CLOSE
    of completed candles only, so a candle whose open + interval is still in
    the future must not be fed to the features or to the outcome replay.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    now = now or datetime.now(timezone.utc)
    last_open = pd.Timestamp(df["timestamp"].iloc[-1])
    last_open = (last_open.tz_localize("UTC") if last_open.tzinfo is None
                 else last_open.tz_convert("UTC"))
    closes_at = last_open + pd.Timedelta(hours=parse_interval_hours(interval))
    if closes_at > pd.Timestamp(now):
        return df.iloc[:-1].reset_index(drop=True)
    return df
