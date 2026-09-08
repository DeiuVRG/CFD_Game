"""Capital.com candle source: client paging/mid prices (fake HTTP), bar-count
math, and the gold_monitor fetcher routing with Yahoo fallback."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from common.capital_prices import (CapitalPrices, CapitalPricesError, bars_for,
                                   RESOLUTIONS)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeHTTP:
    """Serves `n_bars` hourly bars ending at T0 + n_bars h, in pages of
    `max` ending at `to` (Capital.com semantics: `to` exclusive-ish, the
    page is the `max` bars before it)."""
    def __init__(self, n_bars=2500, login_ok=True, fail_401_once=False):
        self.n_bars, self.login_ok, self.fail_401_once = n_bars, login_ok, fail_401_once
        self.posts, self.gets = [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers))
        if not self.login_ok:
            return SimpleNamespace(status_code=401, text='{"errorCode":"error.invalid.details"}', headers={})
        return SimpleNamespace(status_code=200, text="", headers={"CST": "cst1", "X-SECURITY-TOKEN": "xst1"})

    def _bar(self, i):
        ts = (T0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
        p = 1000.0 + i
        return {"snapshotTimeUTC": ts,
                "openPrice": {"bid": p - 0.5, "ask": p + 0.5}, "highPrice": {"bid": p + 4.5, "ask": p + 5.5},
                "lowPrice": {"bid": p - 5.5, "ask": p - 4.5}, "closePrice": {"bid": p + 0.5, "ask": p + 1.5},
                "lastTradedVolume": 10}

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append((url, dict(params), dict(headers)))
        if self.fail_401_once:
            self.fail_401_once = False
            return SimpleNamespace(status_code=401, text="expired", headers={})
        end = self.n_bars
        if "to" in params:
            end = int((datetime.fromisoformat(params["to"]).replace(tzinfo=timezone.utc) - T0).total_seconds() // 3600) + 1
        start = max(0, end - params["max"])
        return SimpleNamespace(status_code=200, text="",
                               json=lambda: {"prices": [self._bar(i) for i in range(start, end)]})


def client(**kw):
    http = FakeHTTP(**kw)
    return CapitalPrices(api_key="k", identifier="me@x", password="pw", http=http), http


def test_login_then_get_candles_mid_prices_utc():
    c, http = client(n_bars=100)
    df = c.get_candles("GOLD", "1h", 50)
    assert len(df) == 50 and list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert http.posts[0][2]["X-CAP-API-KEY"] == "k"
    assert http.gets[0][2]["CST"] == "cst1" and http.gets[0][1]["resolution"] == "HOUR"
    last = df.iloc[-1]
    assert last["open"] == 1099.0 and last["close"] == 1100.0          # mid of bid/ask
    assert last["high"] == 1104.0 and last["low"] == 1094.0
    assert df["timestamp"].dt.tz is not None and df["timestamp"].iloc[-1] == T0 + timedelta(hours=99)
    assert df["timestamp"].is_monotonic_increasing


def test_paging_backwards_assembles_long_history_without_duplicates():
    c, http = client(n_bars=2500)
    df = c.get_candles("GOLD", "1h", 2300)
    assert len(df) == 2300 and df["timestamp"].is_unique
    assert df["timestamp"].iloc[0] == T0 + timedelta(hours=200)
    assert len(http.gets) == 3 and "to" in http.gets[1][1] and "to" not in http.gets[0][1]


def test_history_shorter_than_requested_stops_cleanly():
    c, _ = client(n_bars=120)
    assert len(c.get_candles("BTCUSD", "1h", 5000)) == 120


def test_relogin_on_401_and_login_failure():
    c, http = client(n_bars=10, fail_401_once=True)
    assert len(c.get_candles("GOLD", "1h", 5)) == 5 and len(http.posts) == 2
    bad, _ = client(login_ok=False)
    with pytest.raises(CapitalPricesError):
        bad.get_candles("GOLD", "1h", 5)
    with pytest.raises(CapitalPricesError):
        c.get_candles("GOLD", "7h", 5)


def test_available_needs_all_credentials():
    assert CapitalPrices(api_key="a", identifier="b", password="c").available
    assert not CapitalPrices(api_key="", identifier="b", password="c").available
    assert RESOLUTIONS["5m"] == "MINUTE_5" and RESOLUTIONS["1d"] == "DAY"


def test_bars_for():
    assert bars_for("30d", "1h", session_24_7=True) == 720
    assert bars_for("30d", "1h", session_24_7=False) == int(30 * 23 * 5 / 7)   # ~492
    assert bars_for("2y", "1h", session_24_7=False) == int(730 * 23 * 5 / 7)   # ~11992
    assert bars_for("2y", "1h", session_24_7=True) == 15000                    # capped
    assert bars_for("5d", "5m", session_24_7=True) == 1440
    assert bars_for("1d", "1h", session_24_7=True) == 60                       # floor


# ------------------------------------------------ gold_monitor fetcher --

def test_fetcher_uses_capital_then_falls_back_to_yahoo(monkeypatch):
    from config.settings import INSTRUMENTS, MONITOR
    from data import gold_fetcher
    gold = next(i for i in INSTRUMENTS if i.SYMBOL == "GC=F")
    monkeypatch.setattr(MONITOR, "CANDLE_SOURCE", "capital")
    calls = []

    class Stub:
        def __init__(self, fail=False): self.fail = fail
        def get_candles(self, epic, interval, count):
            calls.append((epic, interval, count))
            if self.fail:
                raise CapitalPricesError("boom")
            return pd.DataFrame({"timestamp": [T0], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})

    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: Stub())
    monkeypatch.setattr(gold_fetcher.MarketFetcher, "_yahoo_candles",
                        lambda self, period, interval, count=None: pd.DataFrame({"src": ["yahoo"]}))
    f = gold_fetcher.MarketFetcher(gold)
    df = f.get_candles(period="30d", interval="1h")
    assert "src" not in df.columns and calls == [("GOLD", "1h", bars_for("30d", "1h", False))]

    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: Stub(fail=True))
    assert list(f.get_candles(period="30d", interval="1h")["src"]) == ["yahoo"]

    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: None)   # no credentials
    assert list(f.get_candles(period="5d", interval="5m")["src"]) == ["yahoo"]

    monkeypatch.setattr(MONITOR, "CANDLE_SOURCE", "yahoo")
    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: Stub())
    assert list(f.get_candles(period="5d", interval="5m")["src"]) == ["yahoo"]


def test_training_data_never_falls_back_to_yahoo(monkeypatch):
    from config.settings import INSTRUMENTS, MONITOR
    from data import gold_fetcher
    gold = next(i for i in INSTRUMENTS if i.SYMBOL == "GC=F")
    monkeypatch.setattr(MONITOR, "CANDLE_SOURCE", "capital")
    yahoo_calls = []
    monkeypatch.setattr(gold_fetcher.MarketFetcher, "_yahoo_candles",
                        lambda self, period, interval, count=None: yahoo_calls.append(1) or pd.DataFrame({"src": ["yahoo"]}))

    class Failing:
        def get_candles(self, epic, interval, count):
            raise CapitalPricesError("timeout")
    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: Failing())
    assert gold_fetcher.MarketFetcher(gold).get_training_data().empty
    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: None)
    assert gold_fetcher.MarketFetcher(gold).get_training_data().empty
    assert yahoo_calls == []                                   # never mixed

    monkeypatch.setattr(MONITOR, "CANDLE_SOURCE", "yahoo")
    assert list(gold_fetcher.MarketFetcher(gold).get_training_data()["src"]) == ["yahoo"]


def test_client_retries_transient_errors(monkeypatch):
    import requests as rq
    import common.capital_prices as cp
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)

    class Flaky(FakeHTTP):
        def __init__(self):
            super().__init__(n_bars=10); self.fails = 2
        def get(self, url, params=None, headers=None, timeout=None):
            if self.fails:
                self.fails -= 1
                raise rq.exceptions.ReadTimeout("read timed out")
            return super().get(url, params=params, headers=headers, timeout=timeout)
    http = Flaky()
    c = CapitalPrices(api_key="k", identifier="i", password="p", http=http)
    assert len(c.get_candles("GOLD", "1h", 5)) == 5

    class Dead(FakeHTTP):
        def get(self, *a, **k):
            raise rq.exceptions.ConnectionError("down")
    dead = CapitalPrices(api_key="k", identifier="i", password="p", http=Dead())
    with pytest.raises(CapitalPricesError):
        dead.get_candles("GOLD", "1h", 5)


def test_snapshot_mid_price():
    class SnapHTTP(FakeHTTP):
        def get(self, url, params=None, headers=None, timeout=None):
            if "/markets/" in url:
                return SimpleNamespace(status_code=200, text="", json=lambda: {
                    "snapshot": {"bid": 100.0, "offer": 101.0, "marketStatus": "TRADEABLE",
                                 "updateTime": "2026-09-08T10:00:00"}})
            return super().get(url, params=params, headers=headers, timeout=timeout)
    c = CapitalPrices(api_key="k", identifier="i", password="p", http=SnapHTTP())
    snap = c.get_snapshot("GOLD")
    assert snap["mid"] == 100.5 and snap["status"] == "TRADEABLE" and snap["epic"] == "GOLD"


def test_live_price_batch_covers_demo_tier_and_prefers_capital(monkeypatch):
    """Regression: the batch used to filter on ENABLED and returned {} for
    demo-tier instruments -> the live price froze at start-up."""
    from config.settings import INSTRUMENTS, MONITOR
    from data import gold_fetcher
    demo = [i for i in INSTRUMENTS if i.active]
    assert demo and all(not i.ENABLED for i in demo)
    monkeypatch.setattr(MONITOR, "CANDLE_SOURCE", "capital")

    class Snap:
        def get_snapshot(self, epic):
            return {"epic": epic, "mid": {"GOLD": 4400.25, "BTCUSD": 78400.5}[epic], "status": "TRADEABLE"}
    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: Snap())
    tv_calls = []
    monkeypatch.setattr(gold_fetcher, "_fetch_tradingview_batch", lambda insts: tv_calls.append(insts) or {})
    prices = gold_fetcher.fetch_all_prices_batch(INSTRUMENTS)
    assert prices == {"XAU/USD": 4400.25, "BTC/USD": 78400.5}
    assert tv_calls == []                                   # Capital covered everything

    class Broken:
        def get_snapshot(self, epic):
            raise CapitalPricesError("down")
    monkeypatch.setattr(gold_fetcher, "_capital_client", lambda: Broken())
    monkeypatch.setattr(gold_fetcher, "_fetch_tradingview_batch",
                        lambda insts: {i.TWELVEDATA_SYMBOL: 1.0 for i in insts})
    prices = gold_fetcher.fetch_all_prices_batch(INSTRUMENTS)
    assert set(prices) == {"XAU/USD", "BTC/USD"}          # fell through to TradingView
