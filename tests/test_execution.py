"""
Tests for the v3 execution model (ai/backtester.py).

All tests use synthetic candles and a scripted predict_fn - no network, no
trained model. The warmup candles are identical (range 10) so that ATR(14) is
exactly 10 at signal time, which makes SL/TP levels exact:
    SL distance = 1.5 x ATR = 15, TP distance = 3.0 x ATR = 30.

Required scenarios (mission spec):
  1. Signal at idx N -> entry at the OPEN of idx N+1.
  2. Wick below SL with the close back above -> position closes at SL.
  3. SL + TP touched in the same candle -> exit at SL (conservative).
  4. Open gapping beyond SL -> exit at the open, not at the SL level.
Plus: end-of-period forced close updates equity; walk-forward state
(open position / pending signal / equity) carries across windows through the
same engine.
"""
import numpy as np
import pandas as pd
import pytest

from ai.backtester import Backtester, ExecutionState
from config.settings import InstrumentConfig


BASE = 1000.0
WARMUP = 21  # candles 0..20 are identical; signal fires on idx 20's close


def flat_candle():
    return (BASE, BASE + 5, BASE - 5, BASE)


def make_df(candles):
    df = pd.DataFrame(candles, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    return df


def make_instrument(**overrides) -> InstrumentConfig:
    params = dict(
        SYMBOL="TEST",
        SYMBOL_DISPLAY="TEST",
        MODEL_PATH="",
        SPREAD_PIPS=0.0,   # zero cost -> exact P&L assertions
        PIP_VALUE=0.0,
        SL_ATR=1.5,
        TP_ATR=3.0,
        CONFIDENCE=0.5,
        ADX_MIN=0.0,
        MIN_RR=1.0,
    )
    params.update(overrides)
    return InstrumentConfig(**params)


def scripted_predict(signals: dict):
    """predict_fn(idx) -> (signal, confidence); default HOLD."""
    def predict_fn(idx: int):
        return signals.get(idx, (0, 0.0))
    return predict_fn


def run_engine(candles, signals, instrument=None, **kwargs):
    inst = instrument or make_instrument()
    bt = Backtester(inst)
    df = make_df(candles)
    trades, curve, state = bt._simulate_trades(
        scripted_predict(signals), df, list(range(len(candles))), **kwargs
    )
    return trades, curve, state


def warmup_candles():
    return [flat_candle() for _ in range(WARMUP)]


# ---------------------------------------------------------------------------
# 1. Signal on close of idx N -> entry at the OPEN of idx N+1
# ---------------------------------------------------------------------------

def test_entry_at_next_candle_open():
    candles = warmup_candles()
    # idx 21: opens at 1002 (gap vs idx 20 close of 1000); stays inside SL/TP
    candles.append((1002.0, 1007.0, 997.0, 1002.0))
    candles.extend([(1002.0, 1007.0, 997.0, 1002.0)] * 3)  # idx 22..24 quiet

    trades, curve, state = run_engine(candles, {20: (1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    # Entry must be the NEXT candle's open, not the signal candle's close
    assert trade.entry_idx == 21
    assert trade.entry_price == pytest.approx(1002.0)
    assert trade.entry_price != 1000.0
    # SL/TP anchored to the actual entry price with ATR from signal time (10)
    assert trade.stop_loss == pytest.approx(1002.0 - 15.0)
    assert trade.take_profit == pytest.approx(1002.0 + 30.0)
    assert trade.exit_reason == "EOD"


# ---------------------------------------------------------------------------
# 2. Wick below SL with close back above -> closes at SL (wicks count)
# ---------------------------------------------------------------------------

def test_wick_through_sl_closes_at_sl():
    candles = warmup_candles()
    candles.append(flat_candle())                    # idx 21: entry at 1000
    candles.append(flat_candle())                    # idx 22: quiet
    # idx 23: low wicks to 980 (below SL=985) but closes back at 1002
    candles.append((1000.0, 1005.0, 980.0, 1002.0))
    candles.append(flat_candle())                    # idx 24

    trades, curve, state = run_engine(candles, {20: (1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_idx == 23
    assert trade.exit_reason == "SL"
    assert trade.exit_price == pytest.approx(985.0)   # exactly the SL level
    assert trade.pnl_pct == pytest.approx(-1.5)       # (985-1000)/1000
    # A close-only check would have missed this exit entirely (close=1002>SL)


# ---------------------------------------------------------------------------
# 3. SL and TP touched in the same candle -> conservative: SL first
# ---------------------------------------------------------------------------

def test_sl_and_tp_same_candle_exits_at_sl():
    candles = warmup_candles()
    candles.append(flat_candle())                    # idx 21: entry at 1000
    # idx 22: huge candle touching both TP (1030) and SL (985)
    candles.append((1000.0, 1035.0, 980.0, 1010.0))
    candles.append(flat_candle())                    # idx 23

    trades, curve, state = run_engine(candles, {20: (1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_idx == 22
    assert trade.exit_reason == "SL"
    assert trade.exit_price == pytest.approx(985.0)
    assert trade.pnl_net_pct < 0


def test_sl_and_tp_same_candle_exits_at_sl_for_sell():
    candles = warmup_candles()
    candles.append(flat_candle())                    # idx 21: entry SELL at 1000
    # SELL: SL = 1015, TP = 970; candle touches both
    candles.append((1000.0, 1020.0, 965.0, 990.0))
    candles.append(flat_candle())

    trades, curve, state = run_engine(candles, {20: (-1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == "SELL"
    assert trade.exit_reason == "SL"
    assert trade.exit_price == pytest.approx(1015.0)
    assert trade.pnl_net_pct < 0


# ---------------------------------------------------------------------------
# 4. Gap through SL -> exit at the open (worse), never at the SL level
# ---------------------------------------------------------------------------

def test_gap_through_sl_exits_at_open():
    candles = warmup_candles()
    candles.append(flat_candle())                    # idx 21: entry at 1000
    candles.append(flat_candle())                    # idx 22: quiet
    # idx 23: opens at 975, far below SL=985 (weekend-style gap)
    candles.append((975.0, 980.0, 970.0, 978.0))
    candles.append((978.0, 983.0, 973.0, 978.0))     # idx 24

    trades, curve, state = run_engine(candles, {20: (1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_idx == 23
    assert trade.exit_reason == "GAP_SL"
    assert trade.exit_price == pytest.approx(975.0)   # the (worse) open
    assert trade.exit_price != pytest.approx(985.0)   # NOT the SL level
    assert trade.pnl_pct == pytest.approx(-2.5)       # worse than SL's -1.5%


def test_tp_never_fills_better_than_tp_level():
    candles = warmup_candles()
    candles.append(flat_candle())                    # idx 21: entry at 1000
    # idx 22: gaps UP through TP (1030) - fill is exactly TP, never better
    candles.append((1040.0, 1045.0, 1035.0, 1042.0))
    candles.append(flat_candle())

    trades, curve, state = run_engine(candles, {20: (1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "TP"
    assert trade.exit_price == pytest.approx(1030.0)
    assert trade.pnl_pct == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# End-of-period forced close must update equity (historical bug)
# ---------------------------------------------------------------------------

def test_forced_close_updates_equity():
    candles = warmup_candles()
    candles.append(flat_candle())                    # idx 21: entry at 1000
    # idx 22 (last): closes at 1010, +1% unrealized -> forced close
    candles.append((1005.0, 1012.0, 1002.0, 1010.0))

    trades, curve, state = run_engine(candles, {20: (1, 0.99)})

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "EOD"
    assert trade.exit_price == pytest.approx(1010.0)
    # The +1% P&L must be reflected in equity (the pre-v3 bug dropped it)
    assert state.equity == pytest.approx(101000.0)
    assert curve[-1] == pytest.approx(101000.0)


# ---------------------------------------------------------------------------
# Walk-forward: position / pending signal / equity carry across windows
# through the SAME engine (no copies)
# ---------------------------------------------------------------------------

def test_walk_forward_carries_open_position_across_windows():
    candles = warmup_candles()                       # 0..20
    candles.append(flat_candle())                    # idx 21: entry at 1000
    candles.extend([flat_candle()] * 2)              # idx 22..23: window 1 end
    candles.extend([flat_candle()] * 2)              # idx 24..25: window 2
    candles.append((1000.0, 1005.0, 980.0, 1002.0))  # idx 26: SL wick
    candles.append(flat_candle())                    # idx 27

    inst = make_instrument()
    bt = Backtester(inst)
    df = make_df(candles)
    predict = scripted_predict({20: (1, 0.99)})

    state = ExecutionState()
    trades_w1, curve_w1, state = bt._simulate_trades(
        predict, df, list(range(0, 24)), state=state, close_at_end=False)
    # Window 1 ends with the position still open - no forced close
    assert trades_w1 == []
    assert state.position is not None
    assert state.equity == pytest.approx(100000.0)

    trades_w2, curve_w2, state = bt._simulate_trades(
        predict, df, list(range(24, 28)), state=state, close_at_end=True)
    # The carried position exits at SL inside window 2
    assert len(trades_w2) == 1
    trade = trades_w2[0]
    assert trade.entry_idx == 21          # entered in window 1
    assert trade.exit_idx == 26           # exited in window 2
    assert trade.exit_reason == "SL"
    assert state.equity == pytest.approx(100000.0 * (1 - 0.015))


def test_walk_forward_carries_pending_signal_across_windows():
    candles = warmup_candles()                       # 0..20, signal on idx 20
    candles.append((1003.0, 1008.0, 998.0, 1003.0))  # idx 21: first candle of w2
    candles.extend([(1003.0, 1008.0, 998.0, 1003.0)] * 2)  # idx 22..23

    inst = make_instrument()
    bt = Backtester(inst)
    df = make_df(candles)
    predict = scripted_predict({20: (1, 0.99)})

    state = ExecutionState()
    trades_w1, _, state = bt._simulate_trades(
        predict, df, list(range(0, 21)), state=state, close_at_end=False)
    # Signal fired on the last candle of window 1 -> still pending
    assert trades_w1 == []
    assert state.pending is not None
    assert state.position is None

    trades_w2, _, state = bt._simulate_trades(
        predict, df, list(range(21, 24)), state=state, close_at_end=True)
    # Entry happened at the open of the FIRST candle of window 2
    assert len(trades_w2) == 1
    assert trades_w2[0].entry_idx == 21
    assert trades_w2[0].entry_price == pytest.approx(1003.0)
