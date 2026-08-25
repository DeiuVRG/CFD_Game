import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config.settings import InstrumentConfig, TWELVEDATA_API_KEY
from data.fetch_utils import retry_call, get_yf_session

# Suppress yfinance "possibly delisted" spam when market is closed
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

TWELVEDATA_PRICE_URL = "https://api.twelvedata.com/price"
TRADINGVIEW_SCAN_URL = "https://scanner.tradingview.com/{exchange}/scan"
YAHOO_API_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Daily credit limit for Twelve Data free tier
TWELVEDATA_DAILY_LIMIT = 800


class CreditTracker:
    """Tracks data source and Twelve Data API credits."""

    def __init__(self, daily_limit: int = TWELVEDATA_DAILY_LIMIT):
        self.daily_limit = daily_limit
        self._credits_used: int = 0
        self._reset_date: str = ""
        self._exhausted = False
        self._source: str = "TradingView"

    @property
    def credits_remaining(self) -> int:
        self._check_reset()
        return max(0, self.daily_limit - self._credits_used)

    @property
    def is_exhausted(self) -> bool:
        self._check_reset()
        return self._exhausted

    @property
    def source_name(self) -> str:
        return self._source

    def set_source(self, name: str):
        self._source = name

    def use_credits(self, count: int):
        self._check_reset()
        self._credits_used += count
        if self._credits_used >= self.daily_limit:
            self._exhausted = True

    def mark_exhausted(self):
        self._exhausted = True
        self._credits_used = self.daily_limit

    def _check_reset(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._reset_date = today
            self._credits_used = 0
            self._exhausted = False
            logger.info("Twelve Data credits reset for new day.")


# Global credit tracker (shared across all fetchers)
credit_tracker = CreditTracker()


# ---------------------------------------------------------------------------
# TradingView Scanner (primary - free, no API key, spot prices)
# ---------------------------------------------------------------------------

def _fetch_tradingview_batch(instruments: list) -> dict:
    """Fetch real-time spot prices from TradingView scanner API.
    Free, no API key needed, no daily limit.
    Returns dict: { 'XAU/USD': price, 'EUR/USD': price, ... }
    """
    # Group instruments by TV exchange endpoint
    by_exchange: dict[str, list] = {}
    for inst in instruments:
        if inst.TV_SYMBOL:
            ex = inst.TV_EXCHANGE or "cfd"
            by_exchange.setdefault(ex, []).append(inst)

    prices = {}
    for exchange, insts in by_exchange.items():
        tickers = [i.TV_SYMBOL for i in insts]
        try:
            payload = {
                "symbols": {"tickers": tickers},
                "columns": ["close", "change", "change_abs"],
            }
            resp = requests.post(
                TRADINGVIEW_SCAN_URL.format(exchange=exchange),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            # Build a map from TV_SYMBOL -> TwelveData symbol for compatibility
            tv_to_td = {i.TV_SYMBOL: i.TWELVEDATA_SYMBOL for i in insts}

            for item in data.get("data", []):
                tv_sym = item["s"]
                close_price = item["d"][0]
                if close_price and tv_sym in tv_to_td:
                    prices[tv_to_td[tv_sym]] = float(close_price)

        except Exception as e:
            logger.error(f"TradingView scan error ({exchange}): {e}")

    return prices


# ---------------------------------------------------------------------------
# TwelveData (secondary - requires API key, 800 req/day)
# ---------------------------------------------------------------------------

def _fetch_twelvedata_batch(instruments: list) -> dict:
    """Batch fetch from Twelve Data API."""
    td_symbols = [i.TWELVEDATA_SYMBOL for i in instruments]

    try:
        resp = requests.get(
            TWELVEDATA_PRICE_URL,
            params={"symbol": ",".join(td_symbols), "apikey": TWELVEDATA_API_KEY},
            timeout=5,
        )

        if resp.status_code != 200:
            return {}

        data = resp.json()

        if "code" in data:
            code = data.get("code")
            if code == 429:
                credit_tracker.mark_exhausted()
                logger.warning("Twelve Data rate limit hit")
            return {}

        credit_tracker.use_credits(len(td_symbols))

        prices = {}
        if len(td_symbols) == 1:
            p = data.get("price")
            if p:
                prices[td_symbols[0]] = float(p)
        else:
            for sym in td_symbols:
                entry = data.get(sym, {})
                p = entry.get("price")
                if p:
                    prices[sym] = float(p)

        return prices

    except Exception as e:
        logger.error(f"Twelve Data batch error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Yahoo Finance (last resort fallback - futures prices, not spot)
# ---------------------------------------------------------------------------

def _fetch_yahoo_batch(instruments: list) -> dict:
    """Fetch prices from Yahoo Finance (fallback - futures, not spot)."""
    prices = {}
    for inst in instruments:
        try:
            url = YAHOO_API_URL.format(symbol=inst.SYMBOL)
            resp = requests.get(
                url,
                headers=YAHOO_HEADERS,
                params={"interval": "1m", "range": "1d"},
                timeout=5,
            )

            if resp.status_code != 200:
                continue

            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                continue

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            if price:
                prices[inst.TWELVEDATA_SYMBOL] = float(price)

        except Exception:
            continue

    return prices


# ---------------------------------------------------------------------------
# Main batch fetch: TradingView -> TwelveData -> Yahoo
# ---------------------------------------------------------------------------

def fetch_all_prices_batch(instruments: list) -> dict:
    """
    Fetch prices for ALL instruments.
    Priority: TradingView (spot, free) -> TwelveData -> Yahoo (futures).
    Fills missing instruments from lower-priority sources.
    Returns dict: { 'XAU/USD': price, 'EUR/USD': price, ... }
    """
    enabled = [i for i in instruments if i.ENABLED]
    if not enabled:
        return {}

    all_prices = {}
    source = "Yahoo"

    # 1. TradingView (primary - spot prices, free, no limits)
    tv_instruments = [i for i in enabled if i.TV_SYMBOL]
    if tv_instruments:
        tv_prices = _fetch_tradingview_batch(tv_instruments)
        if tv_prices:
            all_prices.update(tv_prices)
            source = "TradingView"

    # Find instruments still missing a price
    missing = [i for i in enabled if i.TWELVEDATA_SYMBOL not in all_prices]

    # 2. TwelveData (secondary - for missing instruments)
    if missing and TWELVEDATA_API_KEY and not credit_tracker.is_exhausted:
        td_instruments = [i for i in missing if i.TWELVEDATA_SYMBOL]
        if td_instruments:
            td_prices = _fetch_twelvedata_batch(td_instruments)
            if td_prices:
                all_prices.update(td_prices)
                if not source or source == "Yahoo":
                    source = "TwelveData"

    # Find instruments still missing
    missing = [i for i in enabled if i.TWELVEDATA_SYMBOL not in all_prices]

    # 3. Yahoo Finance (last resort - for remaining missing instruments)
    if missing:
        yahoo_prices = _fetch_yahoo_batch(missing)
        if yahoo_prices:
            all_prices.update(yahoo_prices)

    credit_tracker.set_source(source)
    return all_prices


class MarketFetcher:
    """Fetches live prices via TradingView/TwelveData/Yahoo, candles via yfinance."""

    def __init__(self, instrument: InstrumentConfig):
        self.instrument = instrument
        self._last_price: float = 0
        self._last_prev_close: float = 0
        self._last_change: float = 0
        self._last_change_pct: float = 0

    def update_from_batch(self, price: float):
        """Update price from batch fetch result."""
        if price <= 0:
            return

        prev_close = self._last_prev_close if self._last_prev_close > 0 else price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        self._last_price = round(price, 5)
        self._last_change = round(change, 5)
        self._last_change_pct = round(change_pct, 3)

    def get_live_price(self) -> dict:
        """Return cached price (updated via batch fetch from engine)."""
        if self._last_price > 0:
            return {
                "price": self._last_price,
                "prev_close": self._last_prev_close,
                "change": self._last_change,
                "change_pct": self._last_change_pct,
            }
        return self._fetch_initial()

    def _fetch_initial(self) -> dict:
        """Fetch price + prev_close at startup."""
        # Try TradingView first for current price
        tv_sym = self.instrument.TV_SYMBOL
        if tv_sym:
            try:
                exchange = self.instrument.TV_EXCHANGE or "cfd"
                payload = {
                    "symbols": {"tickers": [tv_sym]},
                    "columns": ["close", "prev_close_price"],
                }
                resp = requests.post(
                    TRADINGVIEW_SCAN_URL.format(exchange=exchange),
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    if items:
                        vals = items[0]["d"]
                        price = float(vals[0]) if vals[0] else 0
                        prev = float(vals[1]) if vals[1] else 0
                        if price:
                            return self._update_and_return(price, prev or price)
            except Exception:
                pass

        # Fallback: Yahoo Finance for prev_close
        try:
            def _download():
                ticker = yf.Ticker(self.instrument.SYMBOL, session=get_yf_session())
                return ticker.history(period="5d", interval="1d")

            df = retry_call(_download, what=f"initial price {self.instrument.SYMBOL}")
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else float(df["Open"].iloc[0])
                return self._update_and_return(price, prev_close)
        except Exception:
            pass

        return {"price": 0, "prev_close": 0, "change": 0, "change_pct": 0}

    def _update_and_return(self, price: float, prev_close: float) -> dict:
        change = price - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        self._last_price = round(price, 5)
        self._last_prev_close = round(prev_close, 5)
        self._last_change = round(change, 5)
        self._last_change_pct = round(change_pct, 3)

        return {
            "price": self._last_price,
            "prev_close": self._last_prev_close,
            "change": self._last_change,
            "change_pct": self._last_change_pct,
        }

    def get_candles(
        self,
        period: str = None,
        interval: str = None,
        count: int = None,
    ) -> pd.DataFrame:
        """Get OHLCV candles via yfinance (for AI analysis)."""
        period = period or self.instrument.HISTORY_PERIOD
        interval = interval or self.instrument.CANDLE_INTERVAL

        try:
            def _download():
                ticker = yf.Ticker(self.instrument.SYMBOL, session=get_yf_session())
                return ticker.history(period=period, interval=interval)

            # Retry with backoff; an instrument with no candles (e.g. gold on
            # a weekend) comes back as an empty frame, never as a crash.
            df = retry_call(
                _download,
                what=f"candles {self.instrument.SYMBOL} {interval}/{period}",
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "datetime" in df.columns:
                df.rename(columns={"datetime": "timestamp"}, inplace=True)
            elif "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)

            required = ["timestamp", "open", "high", "low", "close"]
            for col in required:
                if col not in df.columns:
                    return pd.DataFrame()

            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            if count and len(df) > count:
                df = df.tail(count).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Failed to get candles for {self.instrument.SYMBOL}: {e}")
            return pd.DataFrame()

    def get_training_data(self) -> pd.DataFrame:
        """Get historical data for AI training."""
        return self.get_candles(
            period=self.instrument.TRAIN_PERIOD,
            interval=self.instrument.TRAIN_INTERVAL,
        )


# Backwards compatibility alias
GoldFetcher = MarketFetcher
