"""Data loading for the trading game.

Real mode: daily OHLCV for the universe + SPY + ^VIX via yfinance
(2015-2023), cached to CSV so the game is reproducible and re-runnable
offline. Synthetic mode: a regime-switching market-factor model that
produces the same structure (universe + market + VIX proxy) for fully
offline runs and tests.
"""
import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from trading_game.config import GameConfig, MARKET_SYMBOL, VIX_SYMBOL

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    elif "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    keep = ["timestamp"] + [c for c in REQUIRED_COLS if c in df.columns]
    df = df[keep].dropna(subset=["close"]).sort_values("timestamp")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df.reset_index(drop=True)


def _yf_session():
    """Reuse gold_monitor's proxy-aware session logic without importing the
    whole app: honor YF_IMPERSONATE / HTTPS_PROXY / CA env vars."""
    profile = os.getenv("YF_IMPERSONATE", "").strip()
    if not profile:
        return None
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None
    kwargs = {"impersonate": profile}
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if proxy:
        kwargs["proxies"] = {"https": proxy, "http": proxy}
    ca = os.getenv("CURL_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if ca and os.path.exists(ca):
        kwargs["verify"] = ca
    return curl_requests.Session(**kwargs)


def load_real_data(config: GameConfig) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Download (or read cached) daily data for universe + market + VIX."""
    import yfinance as yf

    os.makedirs(config.data_cache_dir, exist_ok=True)
    session = _yf_session()

    def fetch(symbol: str) -> pd.DataFrame:
        cache = os.path.join(config.data_cache_dir,
                             f"{symbol.replace('^', '_')}_1d.csv")
        if os.path.exists(cache):
            return pd.read_csv(cache, parse_dates=["timestamp"])
        ticker = yf.Ticker(symbol, session=session)
        raw = ticker.history(start=config.train_start, end=config.test_end,
                             interval="1d", auto_adjust=True)
        if raw is None or raw.empty:
            raise RuntimeError(f"No data for {symbol}")
        df = _normalize(raw)
        df.to_csv(cache, index=False)
        logger.info(f"Downloaded {symbol}: {len(df)} days")
        return df

    data = {}
    for symbol in config.universe:
        data[symbol] = fetch(symbol)
    market = fetch(MARKET_SYMBOL)
    vix = fetch(VIX_SYMBOL)
    return data, market, vix


def load_synthetic_data(config: GameConfig) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Regime-switching market factor + per-ticker beta/idiosyncratic noise.
    Produces plausible daily OHLCV with bull/bear/sideways phases and a
    VIX-like series so the whole game (regimes included) runs offline."""
    rng = np.random.default_rng(config.seed)
    dates = pd.bdate_range(config.train_start, config.test_end)
    n = len(dates)

    # Market regimes: annualized drift/vol; sticky transitions
    regimes = [(0.12, 0.12), (-0.20, 0.30), (0.02, 0.16)]  # bull, bear, side
    probs = np.array([
        [0.985, 0.005, 0.010],
        [0.015, 0.970, 0.015],
        [0.015, 0.010, 0.975],
    ])
    state = 0
    drift = np.empty(n)
    vol = np.empty(n)
    for i in range(n):
        state = rng.choice(3, p=probs[state])
        drift[i], vol[i] = regimes[state]

    daily_drift = drift / 252
    daily_vol = vol / np.sqrt(252)
    market_ret = daily_drift + daily_vol * rng.standard_normal(n)

    def build_ohlcv(returns: np.ndarray, start_price: float,
                    base_volume: float) -> pd.DataFrame:
        close = start_price * np.exp(np.cumsum(returns))
        open_ = np.empty_like(close)
        open_[0] = start_price
        open_[1:] = close[:-1] * (1 + 0.001 * rng.standard_normal(n - 1))
        intraday = np.abs(returns) + 0.004
        high = np.maximum(open_, close) * (1 + 0.5 * intraday * rng.random(n))
        low = np.minimum(open_, close) * (1 - 0.5 * intraday * rng.random(n))
        volume = base_volume * np.exp(0.3 * rng.standard_normal(n))
        return pd.DataFrame({
            "timestamp": dates, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        })

    market = build_ohlcv(market_ret, 200.0, 8e7)

    data = {}
    for symbol in config.universe:
        beta = 0.6 + 0.9 * rng.random()
        idio_vol = (0.10 + 0.25 * rng.random()) / np.sqrt(252)
        alpha = rng.normal(0, 0.03) / 252
        ret = alpha + beta * market_ret + idio_vol * rng.standard_normal(n)
        data[symbol] = build_ohlcv(ret, 20 + 200 * rng.random(), 5e6 * (0.5 + rng.random()))

    # VIX-like proxy: scaled realized vol of the market factor
    realized = pd.Series(market_ret).rolling(21).std().bfill() * np.sqrt(252) * 100
    vix_level = (realized * (1 + 0.15 * rng.standard_normal(n))).clip(lower=9)
    vix = pd.DataFrame({
        "timestamp": dates, "open": vix_level, "high": vix_level * 1.05,
        "low": vix_level * 0.95, "close": vix_level, "volume": 0.0,
    })

    return data, market, vix


def load_data(config: GameConfig):
    if config.synthetic:
        logger.info("Using SYNTHETIC data (offline mode)")
        return load_synthetic_data(config)
    try:
        return load_real_data(config)
    except Exception as e:
        logger.warning(f"Real data unavailable ({e}); falling back to synthetic")
        return load_synthetic_data(config)
