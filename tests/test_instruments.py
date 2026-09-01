"""Tests for per-instrument sessions (BTC 24/7 vs gold), the SPREAD_PCT cost
model priority, and the EOD-close exemption for 24/7 instruments."""
from datetime import datetime, timezone

import pytest

from config.settings import COSTS, INSTRUMENTS, InstrumentConfig
from engine.monitor_engine import MonitorEngine
from engine.position_tracker import PositionTracker


def get_instrument(symbol: str) -> InstrumentConfig:
    for inst in INSTRUMENTS:
        if inst.SYMBOL == symbol:
            return inst
    raise AssertionError(f"{symbol} not configured")


GOLD = get_instrument("GC=F")
BTC = get_instrument("BTC-USD")

SATURDAY_NOON = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
TUESDAY_NOON = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
TUESDAY_NIGHT = datetime(2026, 8, 25, 23, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sessions: gold keeps its window, BTC is always active
# ---------------------------------------------------------------------------

def test_gold_session_closed_on_weekend():
    assert MonitorEngine._is_session_active(GOLD, now=SATURDAY_NOON) is False


def test_gold_session_open_on_weekday_daytime():
    assert MonitorEngine._is_session_active(GOLD, now=TUESDAY_NOON) is True


def test_gold_session_closed_outside_hours():
    assert MonitorEngine._is_session_active(GOLD, now=TUESDAY_NIGHT) is False


def test_btc_session_always_active():
    assert MonitorEngine._is_session_active(BTC, now=SATURDAY_NOON) is True
    assert MonitorEngine._is_session_active(BTC, now=TUESDAY_NIGHT) is True
    assert MonitorEngine._is_session_active(BTC, now=TUESDAY_NOON) is True


def test_btc_config_sanity():
    assert BTC.ENABLED is False, "BTC must stay disabled until it passes the OOS gate"
    assert BTC.SESSION_24_7 is True
    assert BTC.PRICE_CHANGE_THRESHOLD >= 0.008
    assert BTC.THRESHOLD_GRID, "BTC needs a label-threshold grid for the optimizer"
    assert BTC.SPREAD_PCT >= 0.002, "BTC round-trip spread should be ~0.20-0.35%"


# ---------------------------------------------------------------------------
# Cost model: SPREAD_PCT takes priority over pips when set
# ---------------------------------------------------------------------------

def test_btc_cost_uses_spread_pct():
    price = 100000.0
    cost = COSTS.round_trip_cost_pct(BTC, price)
    expected = BTC.SPREAD_PCT * (1 + COSTS.SLIPPAGE_MULTIPLIER)
    assert cost == pytest.approx(expected)
    # And it must NOT depend on price (percentage model)
    assert COSTS.round_trip_cost_pct(BTC, 20000.0) == pytest.approx(expected)


def test_gold_cost_uses_pip_model():
    price = 3000.0
    cost = COSTS.round_trip_cost_pct(GOLD, price)
    one_way = GOLD.SPREAD_PIPS * GOLD.PIP_VALUE / price
    expected = one_way * 2 * (1 + COSTS.SLIPPAGE_MULTIPLIER)
    assert cost == pytest.approx(expected)
    # Pip model scales with price
    assert COSTS.round_trip_cost_pct(GOLD, 1500.0) == pytest.approx(expected * 2)


def test_spread_pct_priority_over_pips():
    inst = InstrumentConfig(
        SYMBOL="X", SYMBOL_DISPLAY="X", MODEL_PATH="",
        SPREAD_PIPS=100.0, PIP_VALUE=1.0,  # absurd pip cost...
        SPREAD_PCT=0.001,                   # ...must be ignored: pct wins
    )
    cost = COSTS.round_trip_cost_pct(inst, 100.0)
    assert cost == pytest.approx(0.001 * (1 + COSTS.SLIPPAGE_MULTIPLIER))


# ---------------------------------------------------------------------------
# EOD close: 24/7 positions are exempt
# ---------------------------------------------------------------------------

def test_eod_close_skips_24_7_positions(monkeypatch):
    tracker = PositionTracker()
    tracker.open_position("XAU/USD (Gold)", "BUY", 3000.0, 2980.0, 3040.0,
                          "test", eod_close=True)
    tracker.open_position("BTC/USD (Bitcoin)", "BUY", 100000.0, 98000.0,
                          104000.0, "test", eod_close=False)

    # Force the EOD hour check to pass
    monkeypatch.setattr(
        "engine.position_tracker.STRATEGY.EOD_CLOSE_UTC", -1, raising=False)

    closed = tracker.check_eod_close()
    closed_instruments = {p.instrument for p in closed}
    assert "XAU/USD (Gold)" in closed_instruments
    assert "BTC/USD (Bitcoin)" not in closed_instruments
    # BTC position is still open
    assert tracker.get_position("BTC/USD (Bitcoin)") is not None


# ---------------------------------------------------------------- v3.2 ----

def test_real_money_gate_stays_closed_and_demo_tier_is_explicit():
    """ENABLED is the real-money gate (Faza 4 OOS criteria) and must stay
    False for every instrument; DEMO_ENABLED is a separate, explicit flag."""
    for inst in INSTRUMENTS:
        assert inst.ENABLED is False, f"{inst.SYMBOL} must not pass the gate by hand"
    assert GOLD.DEMO_ENABLED is True and GOLD.active and GOLD.tier == "demo"
    assert BTC.DEMO_ENABLED is True and BTC.tier == "demo"
    assert get_instrument("EURUSD=X").active is False
    assert get_instrument("GBPUSD=X").active is False
