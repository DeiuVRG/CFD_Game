"""Capital.com price-history client (candles), standalone.

Why: the broker's own candles are real-time, cover the exact CFD we trade
(spot gold, not the GC=F future), are 24/5 - 24/7 and have 2+ years of
history at 1000 bars per request with backward paging. Yahoo lagged hours
to days (weekends, US holidays).

No execution_capital import on purpose: this module is shared by
gold_monitor and the sentinel, whose processes cannot load
execution_capital's clashing package names. Read-only: it only ever calls
/session, /ping and /prices, always on the DEMO host (price data is the
same as live).
"""
import logging
import os
import threading
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEMO_URL = "https://demo-api-capital.backend-capital.com"
MAX_BARS_PER_REQUEST = 1000
SESSION_REFRESH_SEC = 480          # sessions expire after 10 min of inactivity
MIN_REQUEST_GAP_SEC = 0.12         # ~10 req/s API limit
READ_TIMEOUT_SEC = 30
TRANSIENT_RETRIES = 3              # read timeouts / connection resets / 5xx

RESOLUTIONS = {
    "1m": "MINUTE", "5m": "MINUTE_5", "15m": "MINUTE_15", "30m": "MINUTE_30",
    "1h": "HOUR", "4h": "HOUR_4", "1d": "DAY", "1wk": "WEEK",
}


class CapitalPricesError(RuntimeError):
    pass


def _mid(block) -> float:
    if isinstance(block, dict):
        bid, ask = block.get("bid"), block.get("ask")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2
        return float(bid if bid is not None else ask)
    return float(block)


class CapitalPrices:
    def __init__(self, api_key: str = None, identifier: str = None,
                 password: str = None, base_url: str = DEMO_URL, http=None):
        self.api_key = api_key if api_key is not None else os.getenv("CAPITAL_API_KEY", "")
        self.identifier = identifier if identifier is not None else os.getenv("CAPITAL_IDENTIFIER", "")
        self.password = password if password is not None else os.getenv("CAPITAL_PASSWORD", "")
        self.base_url = base_url.rstrip("/")
        self._http = http or requests.Session()
        self._tokens: dict = {}
        self._session_time: float = 0.0
        self._last_request: float = 0.0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.identifier and self.password)

    # ------------------------------------------------------------ session
    def _login(self):
        r = self._http.post(
            f"{self.base_url}/api/v1/session",
            json={"identifier": self.identifier, "password": self.password,
                  "encryptedPassword": False},
            headers={"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            raise CapitalPricesError(f"login failed: {r.status_code} {r.text[:120]}")
        self._tokens = {"CST": r.headers.get("CST", ""),
                        "X-SECURITY-TOKEN": r.headers.get("X-SECURITY-TOKEN", "")}
        self._session_time = time.time()
        logger.info("Capital.com price session created")

    def _ensure_session(self):
        if not self._tokens or time.time() - self._session_time > SESSION_REFRESH_SEC:
            self._login()

    def _get(self, path: str, params: dict) -> dict:
        with self._lock:
            self._ensure_session()
            last_error = None
            for attempt in range(1, TRANSIENT_RETRIES + 2):
                gap = MIN_REQUEST_GAP_SEC - (time.time() - self._last_request)
                if gap > 0:
                    time.sleep(gap)
                try:
                    r = self._http.get(f"{self.base_url}{path}", params=params,
                                       headers={"X-CAP-API-KEY": self.api_key, **self._tokens},
                                       timeout=READ_TIMEOUT_SEC)
                except requests.RequestException as e:      # timeout, reset, DNS...
                    last_error = e
                    self._last_request = time.time()
                    logger.warning(f"Capital.com {path}: {e} (attempt {attempt})")
                    time.sleep(min(2.0 * attempt, 6.0))
                    continue
                self._last_request = time.time()
                if r.status_code == 401:
                    self._login()
                    continue
                if r.status_code >= 500 or r.status_code == 429:
                    last_error = CapitalPricesError(f"{r.status_code} {r.text[:80]}")
                    time.sleep(min(2.0 * attempt, 6.0))
                    continue
                if r.status_code != 200:
                    raise CapitalPricesError(f"GET {path}: {r.status_code} {r.text[:120]}")
                self._session_time = time.time()
                return r.json()
            raise CapitalPricesError(f"GET {path}: giving up after retries ({last_error})")

    # ------------------------------------------------------------- candles
    def get_candles(self, epic: str, interval: str, count: int) -> pd.DataFrame:
        """Up to `count` most recent completed-or-forming bars, oldest first,
        columns timestamp (tz-aware UTC) / open / high / low / close / volume
        (mid of bid/ask). Pages backwards with `to=` in chunks of 1000."""
        if interval not in RESOLUTIONS:
            raise CapitalPricesError(f"unsupported interval {interval!r}")
        def ts_of(b):
            return b.get("snapshotTimeUTC") or b.get("snapshotTime")

        bars: list = []          # unique bars, oldest first
        seen: set = set()
        to: Optional[str] = None
        while len(bars) < count:
            # +1: a page ending at `to` repeats the `to` bar (deduplicated)
            params = {"resolution": RESOLUTIONS[interval],
                      "max": min(MAX_BARS_PER_REQUEST, count - len(bars) + 1)}
            if to:
                params["to"] = to
            page = self._get(f"/api/v1/prices/{epic}", params).get("prices", [])
            new = [b for b in page if ts_of(b) not in seen]
            if not new:
                break
            seen.update(ts_of(b) for b in new)
            bars = new + bars
            to = ts_of(page[0])
            if len(page) < params["max"]:
                break          # reached the start of the available history
        if not bars:
            return pd.DataFrame()
        rows = []
        for b in bars:
            ts = b.get("snapshotTimeUTC") or b.get("snapshotTime")
            rows.append({
                "timestamp": ts,
                "open": _mid(b["openPrice"]), "high": _mid(b["highPrice"]),
                "low": _mid(b["lowPrice"]), "close": _mid(b["closePrice"]),
                "volume": float(b.get("lastTradedVolume", 0) or 0),
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = (df.drop_duplicates("timestamp", keep="last")
                .sort_values("timestamp").tail(count).reset_index(drop=True))
        return df


def bars_for(period: str, interval: str, session_24_7: bool = False,
             cap: int = 15000) -> int:
    """How many bars of `interval` cover a yfinance-style `period`
    ('5d', '30d', '2y'), given the instrument's trading hours."""
    n, unit = int(period[:-1]), period[-1]
    days = {"d": n, "w": 7 * n, "y": 365 * n}[unit]
    hours_per_day = 24.0 if session_24_7 else 23 * 5 / 7   # gold/FX: ~23h x 5 days
    interval_hours = {"m": 1 / 60, "h": 1.0, "d": 24.0}[interval[-1]] * int(interval[:-1])
    return max(60, min(cap, int(days * hours_per_day / interval_hours)))
