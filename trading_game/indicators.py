"""Crupier-side indicator service.

Computes every indicator from the spec's catalog on daily OHLCV and converts
each one into a STANDARDIZED SIGNAL in [-1, +1] (positive = bullish). The
whole point of the game is to find which of these signals deserve weight, so
weak interpretations are fine - the competition prunes them.

Shared math (RSI/EMA/SMA/MACD/BB/ATR/ADX) comes from common.indicators - the
single implementation used across the repo.
"""
import numpy as np
import pandas as pd

from common.indicators import Indicators

# Canonical indicator catalog (what players may request)
CATALOG = [
    # trend
    "SMA_RATIO", "EMA_CROSS", "MACD_HIST", "ADX_TREND", "AROON",
    # momentum
    "RSI", "STOCH", "WILLR", "ROC", "CCI", "MFI",
    # volatility
    "BB_POSITION", "ATR_REGIME", "KELTNER_POSITION",
    # volume
    "OBV_TREND", "VOLUME_SURGE", "VWAP_VALUE", "CMF",
]


def _clip(s):
    return np.clip(s, -1.0, 1.0)


def _zscore(s: pd.Series, window: int = 60) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std.replace(0, np.nan)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """All catalog signals for one ticker. Index-aligned with df; NaN during
    warmup. Positive = bullish, negative = bearish, in [-1, +1]."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)

    # ---------------- TREND ----------------
    sma50 = Indicators.sma(close, 50)
    out["SMA_RATIO"] = _clip((close / sma50 - 1) * 10)

    ema12 = Indicators.ema(close, 12)
    ema26 = Indicators.ema(close, 26)
    out["EMA_CROSS"] = _clip((ema12 / ema26 - 1) * 25)

    _, _, hist = Indicators.macd(close, 12, 26, 9)
    out["MACD_HIST"] = _clip(_zscore(hist) / 2)

    # ADX with directional sign (+DI vs -DI)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ewm = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_ewm
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_ewm
    adx = Indicators.adx(high, low, close, 14)
    out["ADX_TREND"] = _clip(np.sign(plus_di - minus_di) * (adx / 50))

    # Aroon(25)
    period = 25
    days_since_high = high.rolling(period + 1).apply(
        lambda x: period - int(np.argmax(x)), raw=True)
    days_since_low = low.rolling(period + 1).apply(
        lambda x: period - int(np.argmin(x)), raw=True)
    aroon_up = 100 * (period - days_since_high) / period
    aroon_down = 100 * (period - days_since_low) / period
    out["AROON"] = _clip((aroon_up - aroon_down) / 100)

    # ---------------- MOMENTUM (canonical: contrarian oscillators) --------
    rsi = Indicators.rsi(close, 14)
    out["RSI"] = _clip((50 - rsi) / 50 * 1.5)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    out["STOCH"] = _clip((50 - stoch_k) / 50 * 1.2)

    willr = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)
    out["WILLR"] = _clip((-50 - willr) / 50 * 1.2)

    out["ROC"] = _clip(close.pct_change(12) * 8)  # momentum, trend-following

    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
    out["CCI"] = _clip(-cci / 200)

    money_flow = tp * volume
    pos_flow = money_flow.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg_flow = money_flow.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_flow / neg_flow.replace(0, np.nan))
    out["MFI"] = _clip((50 - mfi) / 50 * 1.2)

    # ---------------- VOLATILITY ----------------
    bb_up, bb_mid, bb_low = Indicators.bollinger_bands(close, 20, 2.0)
    bb_pos = (close - bb_low) / (bb_up - bb_low).replace(0, np.nan)
    out["BB_POSITION"] = _clip((0.5 - bb_pos) * 2.5)

    atr = Indicators.atr(high, low, close, 14)
    atr_avg = atr.rolling(50).mean()
    # Calm regime slightly bullish, volatility spike slightly bearish
    out["ATR_REGIME"] = _clip((atr_avg / atr.replace(0, np.nan) - 1))

    kel_mid = Indicators.ema(close, 20)
    kel_up = kel_mid + 2 * atr
    kel_low = kel_mid - 2 * atr
    kel_pos = (close - kel_low) / (kel_up - kel_low).replace(0, np.nan)
    out["KELTNER_POSITION"] = _clip((0.5 - kel_pos) * 2.5)

    # ---------------- VOLUME ----------------
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    out["OBV_TREND"] = _clip(_zscore(obv - obv.rolling(20).mean()) / 2)

    vol_sma = Indicators.sma(volume, 20)
    surge = (volume / vol_sma.replace(0, np.nan) - 1)
    ret1 = np.sign(close.pct_change())
    out["VOLUME_SURGE"] = _clip(ret1 * surge.clip(lower=0))

    pv = (tp * volume).rolling(20).sum()
    vv = volume.rolling(20).sum()
    vwap = pv / vv.replace(0, np.nan)
    out["VWAP_VALUE"] = _clip((vwap / close - 1) * 12)

    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    cmf = (mf_mult * volume).rolling(20).sum() / vv.replace(0, np.nan)
    out["CMF"] = _clip(cmf * 4)

    return out
