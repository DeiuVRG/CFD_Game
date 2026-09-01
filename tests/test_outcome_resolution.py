"""Candle-based outcome resolution for live signals (v3.1).

The hypothetical outcome of an emitted signal is decided by the SAME v3 rule
the backtester validated (engine/execution_rules.v3_exit): wicks count, SL
before TP in the same candle, gaps fill at the open. It is replayed from
candles, so it is deterministic and survives a restart
(PositionTracker.restore_from_store).
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data.signal_store import SignalStore
from engine.execution_rules import v3_exit
from engine.position_tracker import PositionTracker, floor_to_interval

INST = "XAU/USD (Gold)"
T0 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)   # signal candle open


def candles(rows, start=T0 + timedelta(hours=1)):
    """rows: list of (open, high, low, close); hourly from `start`."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = [start + timedelta(hours=i) for i in range(len(rows))]
    return df


@pytest.fixture
def store(tmp_path):
    s = SignalStore(db_path=str(tmp_path / "signals.db"))
    yield s
    s.close()


# ---------------------------------------------------------------- v3_exit --

def test_v3_exit_buy_rules():
    assert v3_exit("BUY", 985, 1030, 1000, 1010, 990) == (None, "")
    assert v3_exit("BUY", 985, 1030, 1000, 1010, 984) == (985, "SL")      # wick
    assert v3_exit("BUY", 985, 1030, 1000, 1031, 990) == (1030, "TP")
    assert v3_exit("BUY", 985, 1030, 1000, 1035, 980) == (985, "SL")      # both -> SL
    assert v3_exit("BUY", 985, 1030, 970, 1000, 960) == (970, "GAP_SL")    # gap at open


def test_v3_exit_sell_rules():
    assert v3_exit("SELL", 1015, 970, 1000, 1010, 990) == (None, "")
    assert v3_exit("SELL", 1015, 970, 1000, 1016, 990) == (1015, "SL")
    assert v3_exit("SELL", 1015, 970, 1000, 1010, 969) == (970, "TP")
    assert v3_exit("SELL", 1015, 970, 1000, 1020, 960) == (1015, "SL")
    assert v3_exit("SELL", 1015, 970, 1030, 1040, 1020) == (1030, "GAP_SL")


# ------------------------------------------------- resolve_with_candles --

def open_buy(tracker, signal_id=None, cost_pct=0.0):
    return tracker.open_position(INST, "BUY", 1000.0, 985.0, 1030.0, "AI",
                                 signal_id=signal_id, cost_pct=cost_pct,
                                 signal_candle_ts=T0)


def test_wick_through_sl_resolves_at_sl_with_candle_time(store):
    tracker = PositionTracker(store=store)
    sid = store.insert_signal(INST, "BUY", entry_price=1000.0, stop_loss=985.0,
                              take_profit=1030.0, strategy="AI")
    open_buy(tracker, signal_id=sid, cost_pct=0.05)
    df = candles([
        (1000, 1010, 995, 1005),   # entry candle: nothing
        (1005, 1012, 984, 1008),   # wick to 984, close back above SL
        (1008, 1040, 1000, 1035),  # would be TP - must never be reached
    ])
    closed = tracker.resolve_with_candles(INST, df)
    assert closed is not None
    assert closed.close_reason == "SL_HIT"
    assert closed.close_price == 985.0
    assert closed.closed_at == T0 + timedelta(hours=2)
    assert tracker.get_position(INST) is None

    row = store.fetch_all()[0]
    assert row["outcome"] == "SL_HIT"
    assert row["outcome_ts_utc"] == "2026-09-01T10:00:00Z"
    assert row["pnl_gross_pct"] == pytest.approx(-1.5)
    assert row["pnl_net_pct"] == pytest.approx(-1.55)


def test_candles_up_to_signal_candle_are_ignored():
    tracker = PositionTracker()
    open_buy(tracker)
    # a catastrophic candle AT the signal candle time and one before: ignored
    df = candles([(1000, 1001, 900, 1000), (1000, 1001, 900, 1000),
                  (1000, 1010, 995, 1005)], start=T0 - timedelta(hours=1))
    assert tracker.resolve_with_candles(INST, df) is None
    assert tracker.get_position(INST) is not None


def test_sl_and_tp_same_candle_resolves_as_sl():
    tracker = PositionTracker()
    open_buy(tracker)
    closed = tracker.resolve_with_candles(INST, candles([(1000, 1035, 980, 1020)]))
    assert closed.close_reason == "SL_HIT" and closed.close_price == 985.0


def test_gap_through_sl_resolves_at_open():
    tracker = PositionTracker()
    open_buy(tracker)
    closed = tracker.resolve_with_candles(
        INST, candles([(1000, 1010, 995, 1005), (970, 1000, 960, 990)]))
    assert closed.close_reason == "GAP_SL_HIT" and closed.close_price == 970.0


def test_tp_resolves_exactly_at_tp_level():
    tracker = PositionTracker()
    open_buy(tracker)
    closed = tracker.resolve_with_candles(INST, candles([(1000, 1050, 995, 1045)]))
    assert closed.close_reason == "TP_HIT" and closed.close_price == 1030.0


def test_still_open_when_nothing_hit():
    tracker = PositionTracker()
    open_buy(tracker)
    assert tracker.resolve_with_candles(INST, candles([(1000, 1010, 990, 1005)] * 5)) is None
    assert tracker.get_position(INST) is not None


def test_empty_frame_is_harmless():
    tracker = PositionTracker()
    open_buy(tracker)
    assert tracker.resolve_with_candles(INST, pd.DataFrame()) is None


# ----------------------------------------------------- restore_from_store --

def test_floor_to_interval():
    t = datetime(2026, 9, 1, 9, 3, 27, tzinfo=timezone.utc)
    assert floor_to_interval(t, 1.0) == datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    assert floor_to_interval(t, 4.0) == datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    assert floor_to_interval("2026-09-01T09:03:27Z", 1.0) == datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)


def test_restore_replays_old_signals_and_reopens_newest(store):
    # Signal A: BUY emitted 09:00:40 (candle 08:00 closed) -> nothing hits
    #   before signal B -> closed as SIGNAL_REVERSED at B's entry.
    # Signal B: SELL emitted 12:00:35 -> stays open, then its TP is hit at 14:00.
    a = store.insert_signal(INST, "BUY", entry_price=1000.0, stop_loss=985.0,
                            take_profit=1030.0, strategy="AI",
                            ts_utc="2026-09-01T09:00:40Z")
    b = store.insert_signal(INST, "SELL", entry_price=1008.0, stop_loss=1023.0,
                            take_profit=978.0, strategy="AI",
                            ts_utc="2026-09-01T12:00:35Z")
    df = candles([
        (1000, 1010, 995, 1005),   # 09:00 entry candle of A
        (1005, 1012, 998, 1008),   # 10:00
        (1008, 1015, 1000, 1008),  # 11:00 (signal B on this close)
        (1008, 1012, 990, 995),    # 12:00 entry candle of B
        (995, 1000, 985, 990),     # 13:00
        (990, 995, 975, 980),      # 14:00 -> B TP (978) hit
    ], start=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc))

    tracker = PositionTracker(store=store)
    info = tracker.restore_from_store(INST, df, cost_pct_fn=lambda p: 0.02,
                                      interval_hours=1.0)
    assert info == {"resolved": 2, "open": False}

    rows = {r["id"]: r for r in store.fetch_all()}
    assert rows[a]["outcome"] == "SIGNAL_REVERSED"
    assert rows[a]["exit_price"] == 1008.0
    assert rows[a]["outcome_ts_utc"] == "2026-09-01T12:00:35Z"
    assert rows[b]["outcome"] == "TP_HIT"
    assert rows[b]["exit_price"] == 978.0
    assert rows[b]["outcome_ts_utc"] == "2026-09-01T14:00:00Z"
    assert rows[b]["pnl_net_pct"] == pytest.approx((1008 - 978) / 1008 * 100 - 0.02)


def test_restore_keeps_newest_open_when_nothing_hit(store):
    sid = store.insert_signal(INST, "BUY", entry_price=1000.0, stop_loss=985.0,
                              take_profit=1030.0, strategy="AI",
                              ts_utc="2026-09-01T09:00:40Z")
    df = candles([(1000, 1010, 995, 1005), (1005, 1012, 998, 1008)],
                 start=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc))
    tracker = PositionTracker(store=store)
    info = tracker.restore_from_store(INST, df, cost_pct_fn=lambda p: 0.02)
    assert info == {"resolved": 0, "open": True}
    pos = tracker.get_position(INST)
    assert pos.signal_id == sid and pos.cost_pct == 0.02
    assert pos.signal_candle_ts == datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    assert store.fetch_open(INST)[0]["id"] == sid   # still unresolved in the DB

    # later candles resolve it through the normal path
    later = candles([(1008, 1035, 1000, 1030)],
                    start=datetime(2026, 9, 1, 11, 0, tzinfo=timezone.utc))
    closed = tracker.resolve_with_candles(INST, later)
    assert closed.close_reason == "TP_HIT"
    assert store.fetch_open(INST) == []


def test_restore_older_signal_hit_before_next_signal(store):
    a = store.insert_signal(INST, "BUY", entry_price=1000.0, stop_loss=985.0,
                            take_profit=1030.0, strategy="AI",
                            ts_utc="2026-09-01T09:00:40Z")
    b = store.insert_signal(INST, "BUY", entry_price=990.0, stop_loss=975.0,
                            take_profit=1020.0, strategy="AI",
                            ts_utc="2026-09-01T11:00:20Z")
    df = candles([
        (1000, 1005, 984, 990),    # 09:00 -> A SL hit
        (990, 995, 985, 990),      # 10:00 (B signal on this close)
        (990, 1000, 986, 995),     # 11:00 B entry candle, nothing
    ], start=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc))
    tracker = PositionTracker(store=store)
    info = tracker.restore_from_store(INST, df, cost_pct_fn=lambda p: 0.0)
    assert info == {"resolved": 1, "open": True}
    rows = {r["id"]: r for r in store.fetch_all()}
    assert rows[a]["outcome"] == "SL_HIT" and rows[a]["exit_price"] == 985.0
    assert rows[b]["outcome"] is None
