"""SIGNAL_MODE="ai_only": the live monitor mirrors the validated v3 backtest.

Everything runs offline: a scripted AI strategy, synthetic 1h candles, a
recording Discord stub, a temporary signals.db. No network, no model.
"""
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import InstrumentConfig
from data.candles import drop_incomplete_candle
from data.signal_store import SignalStore
from engine.monitor_engine import MonitorEngine
from engine.position_tracker import PositionTracker
from engine.signal import Signal

NAME = "TEST/USD"
T0 = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ stubs --

class FakeDiscord:
    def __init__(self):
        self.sent, self.closed, self.resets = [], [], []

    def send_signal(self, signal, confidence=0, probabilities=None):
        self.sent.append(signal)
        return True

    def send_close_signal(self, position):
        self.closed.append(position)
        return True

    def reset_last_direction(self, key):
        self.resets.append(key)


class ScriptedAI:
    """analyze() returns a fixed signal (or None) built on the last close
    with SL/TP distances 15 / 30 - like ATR=10, SL 1.5x, TP 3.0x."""
    def __init__(self, direction="BUY", sl_dist=15.0, tp_dist=30.0):
        self.direction, self.sl_dist, self.tp_dist = direction, sl_dist, tp_dist
        self.predictor = SimpleNamespace(is_ready=False)
        self.last_confidence, self.last_probs = 0.7, {"BUY": 0.7, "SELL": 0.1, "HOLD": 0.2}
        self.calls = 0

    def analyze(self, epic, df):
        self.calls += 1
        if self.direction is None:
            return None
        price = float(df["close"].iloc[-1])
        if self.direction == "BUY":
            sl, tp = price - self.sl_dist, price + self.tp_dist
        else:
            sl, tp = price + self.sl_dist, price - self.tp_dist
        return Signal(epic=epic, direction=self.direction, entry_price=price,
                      stop_loss=sl, take_profit=tp, strategy_name="AI (70%)",
                      strength=0.7)


def make_instrument(**over):
    params = dict(SYMBOL="TEST", SYMBOL_DISPLAY=NAME, MODEL_PATH="",
                  SPREAD_PIPS=0.0, PIP_VALUE=0.0, SL_ATR=1.5, TP_ATR=3.0,
                  CONFIDENCE=0.5, ADX_MIN=0.0, MIN_RR=1.0, SESSION_24_7=True)
    params.update(over)
    return InstrumentConfig(**params)


def trending_candles(n=70, start=T0, step=2.0, seed=1):
    """Hourly candles with a trend so that ADX(14) is defined (> 0)."""
    rng = np.random.default_rng(seed)
    rows, price = [], 1000.0
    for _ in range(n):
        o = price
        c = o + step + rng.normal(0, 0.5)
        rows.append((o, max(o, c) + 3, min(o, c) - 3, c))
        price = c
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = [start + timedelta(hours=i) for i in range(n)]
    return df


def append_candle(df, o, h, l, c):
    ts = pd.Timestamp(df["timestamp"].iloc[-1]) + timedelta(hours=1)
    row = pd.DataFrame([(o, h, l, c, ts)], columns=["open", "high", "low", "close", "timestamp"])
    return pd.concat([df, row], ignore_index=True)


def make_mon(ai=None, instrument=None, price=0.0):
    return SimpleNamespace(
        instrument=instrument or make_instrument(), display_name=NAME,
        ai_strategy=ai or ScriptedAI(), last_ai_candle_ts=None,
        last_ai_confidence=0.0, last_ai_probs={}, adx_1h=0.0, adx=0.0,
        rsi=50.0, price=price, session_active=True, last_signal=None,
    )


@pytest.fixture
def engine(tmp_path):
    store = SignalStore(db_path=str(tmp_path / "signals.db"))
    eng = MonitorEngine.__new__(MonitorEngine)
    eng.discord = FakeDiscord()
    eng.signal_store = store
    eng.position_tracker = PositionTracker(store=store)
    eng._model_versions = {}
    eng._signal_log_path = str(tmp_path / "signals.csv")
    eng._init_signal_log()
    yield eng
    store.close()


# ------------------------------------------------- drop_incomplete_candle --

def test_drop_incomplete_candle_drops_forming_candle():
    df = trending_candles(5)
    now = pd.Timestamp(df["timestamp"].iloc[-1]) + timedelta(minutes=20)
    out = drop_incomplete_candle(df, "1h", now=now)
    assert len(out) == 4
    assert out["timestamp"].iloc[-1] == df["timestamp"].iloc[3]


def test_drop_incomplete_candle_keeps_completed():
    df = trending_candles(5)
    now = pd.Timestamp(df["timestamp"].iloc[-1]) + timedelta(hours=1, seconds=5)
    assert len(drop_incomplete_candle(df, "1h", now=now)) == 5
    assert drop_incomplete_candle(pd.DataFrame(), "1h").empty


# ------------------------------------------------------------ signal path --

def test_signal_is_emitted_persisted_and_position_opened(engine):
    df = trending_candles()
    mon = make_mon(price=1234.5)
    res = engine._ai_only_step(mon, df, live_price=1234.5)

    assert res == {"discord_sent": True, "close_sent": False}
    sig = engine.discord.sent[0]
    # entry = live price (~ next open), distances kept from the signal candle
    assert sig.entry_price == 1234.5
    assert sig.stop_loss == pytest.approx(1234.5 - 15)
    assert sig.take_profit == pytest.approx(1234.5 + 30)

    pos = engine.position_tracker.get_position(NAME)
    assert pos is not None and pos.eod_close is False
    assert pos.signal_candle_ts == pd.Timestamp(df["timestamp"].iloc[-1])
    rows = engine.signal_store.fetch_all()
    assert len(rows) == 1 and rows[0]["direction"] == "BUY"
    assert rows[0]["entry_price"] == 1234.5 and rows[0]["prob_buy"] == 0.7
    assert pos.signal_id == rows[0]["id"]


def test_same_completed_candle_is_processed_once(engine):
    df = trending_candles()
    mon = make_mon(price=1000)
    engine._ai_only_step(mon, df, 1000)
    engine._ai_only_step(mon, df, 1000)
    engine._ai_only_step(mon, df, 1000)
    assert mon.ai_strategy.calls == 1
    assert len(engine.discord.sent) == 1


def test_no_new_signal_while_position_open(engine):
    df = trending_candles()
    mon = make_mon(price=1000)
    engine._ai_only_step(mon, df, 1000)
    df = append_candle(df, 1000, 1005, 995, 1002)   # nothing hit
    engine._ai_only_step(mon, df, 1002)
    assert mon.ai_strategy.calls == 1                 # AI not even asked
    assert len(engine.signal_store.fetch_all()) == 1


def test_sl_hit_on_next_candle_records_outcome_and_allows_new_signal(engine):
    df = trending_candles()
    mon = make_mon(price=1000)
    engine._ai_only_step(mon, df, 1000)              # BUY @1000, SL 985, TP 1030
    df = append_candle(df, 1000, 1004, 984, 995)     # wick through SL
    res = engine._ai_only_step(mon, df, 995)

    assert res["close_sent"] is True
    closed = engine.discord.closed[0]
    assert closed.close_reason == "SL_HIT" and closed.close_price == 985
    assert engine.discord.resets == [NAME]
    rows = engine.signal_store.fetch_all()
    assert rows[0]["outcome"] == "SL_HIT" and rows[0]["exit_price"] == 985
    # flat again -> the AI is asked on the same step and a new BUY is emitted
    assert res["discord_sent"] is True and len(rows) == 2
    assert rows[1]["outcome"] is None


def test_adx_gate_blocks_signal(engine):
    df = trending_candles()
    mon = make_mon(instrument=make_instrument(ADX_MIN=1000.0), price=1000)  # never satisfiable
    res = engine._ai_only_step(mon, df, 1000)
    assert res["discord_sent"] is False and mon.ai_strategy.calls == 0
    assert mon.adx_1h > 0                     # computed for the dashboard


def test_cost_and_rr_filter_like_the_backtester(engine):
    # reward 30/1000 = 3% must exceed the round-trip cost: SPREAD_PCT 5% -> filtered
    inst = make_instrument(SPREAD_PCT=0.05)
    mon = make_mon(instrument=inst, price=1000)
    assert engine._ai_only_step(mon, trending_candles(), 1000)["discord_sent"] is False
    # min R:R 3.0 > 30/15 = 2.0 -> filtered
    mon = make_mon(instrument=make_instrument(MIN_RR=3.0), price=1000)
    assert engine._ai_only_step(mon, trending_candles(), 1000)["discord_sent"] is False
    assert engine.signal_store.fetch_all() == []


def test_hold_prediction_emits_nothing(engine):
    mon = make_mon(ai=ScriptedAI(direction=None), price=1000)
    res = engine._ai_only_step(mon, trending_candles(), 1000)
    assert res == {"discord_sent": False, "close_sent": False}
    assert mon.ai_strategy.calls == 1


def test_sell_signal_anchors_sl_tp_above_and_below_entry(engine):
    mon = make_mon(ai=ScriptedAI(direction="SELL"), price=900)
    engine._ai_only_step(mon, trending_candles(), 900)
    sig = engine.discord.sent[0]
    assert sig.direction == "SELL"
    assert sig.stop_loss == pytest.approx(915) and sig.take_profit == pytest.approx(870)


def test_too_few_candles_is_a_noop(engine):
    mon = make_mon(price=1000)
    assert engine._ai_only_step(mon, trending_candles(30), 1000) == {
        "discord_sent": False, "close_sent": False}
