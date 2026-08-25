"""Tests for the quarantine-time risk fixes (bot stays dormant, logic stays
correct): position sizing with quote-currency conversion, MAX_TRADES_PER_DAY,
the FX anti-correlation rule, and the reduced risk defaults."""
import pytest

from broker.models import MarketInfo, OpenPosition
from config.settings import RISK, TRADING
from engine.signal import Signal
from risk.position_sizer import PositionSizer
from risk.risk_manager import RiskManager, is_fx_pair, quote_currency


def make_market_info(**overrides) -> MarketInfo:
    params = dict(
        epic="EURUSD", instrument_name="EUR/USD",
        min_deal_size=0.01, max_deal_size=1000000.0,
        pip_size=0.0001, margin_factor=0.033, market_open=True,
    )
    params.update(overrides)
    return MarketInfo(**params)


def make_signal(epic="EURUSD", direction="BUY",
                entry=1.1000, sl=1.0950, tp=1.1100) -> Signal:
    return Signal(
        epic=epic, direction=direction, entry_price=entry,
        stop_loss=sl, take_profit=tp,
        strategy_name="test", strength=0.8,
    )


def make_position(epic="EURUSD", direction="BUY") -> OpenPosition:
    return OpenPosition(
        deal_id="d1", epic=epic, direction=direction, size=1.0,
        open_level=1.1, stop_level=1.09, profit_level=1.12,
        profit_loss=0.0, created_date="2026-01-01",
    )


# ---------------------------------------------------------------------------
# Risk defaults reduced (3% -> 1% per trade, 10% -> 3% daily)
# ---------------------------------------------------------------------------

def test_risk_defaults_are_reduced():
    assert RISK.MAX_RISK_PER_TRADE == pytest.approx(0.01)
    assert RISK.MAX_DAILY_LOSS == pytest.approx(0.03)
    assert RISK.MAX_TRADES_PER_DAY >= 1
    assert RISK.MAX_FX_POSITIONS == 1


def test_default_symbols_are_account_currency_quoted():
    for symbol in TRADING.SYMBOLS:
        assert quote_currency(symbol) == "USD"


# ---------------------------------------------------------------------------
# PositionSizer: quote-currency conversion
# ---------------------------------------------------------------------------

def test_sizer_same_currency_risks_exact_amount():
    size = PositionSizer.calculate_size(
        equity=10000.0, risk_pct=0.01,
        entry_price=1.1000, stop_loss_price=1.0950,
        market_info=make_market_info(),
    )
    # risk 100 USD over 0.0050 distance -> 20000 units; loss at SL = 100 USD
    assert size == pytest.approx(20000.0)
    assert size * 0.0050 == pytest.approx(10000.0 * 0.01)


def test_sizer_converts_quote_currency():
    # USDJPY-style: quote JPY, 1 JPY = 0.0067 USD. The naive (unconverted)
    # sizer treats a 0.75 JPY move as if it were 0.75 USD per unit, so it
    # risks ~149x less USD than intended; the converted size fixes that.
    size = PositionSizer.calculate_size(
        equity=10000.0, risk_pct=0.01,
        entry_price=150.00, stop_loss_price=149.25,
        market_info=make_market_info(epic="USDJPY", pip_size=0.01),
        quote_to_account_rate=0.0067,
    )
    loss_jpy = size * 0.75
    loss_usd = loss_jpy * 0.0067
    assert loss_usd == pytest.approx(100.0, rel=0.01)
    # And it must differ hugely from the unconverted size
    unconverted = PositionSizer.calculate_size(
        equity=10000.0, risk_pct=0.01,
        entry_price=150.00, stop_loss_price=149.25,
        market_info=make_market_info(epic="USDJPY", pip_size=0.01),
    )
    assert size == pytest.approx(unconverted / 0.0067, rel=0.01)


def test_sizer_rejects_invalid_rate():
    size = PositionSizer.calculate_size(
        equity=10000.0, risk_pct=0.01,
        entry_price=150.0, stop_loss_price=149.0,
        market_info=make_market_info(),
        quote_to_account_rate=0.0,
    )
    assert size == 0.0


# ---------------------------------------------------------------------------
# RiskManager: non-account-currency quotes are rejected (no silent mis-size)
# ---------------------------------------------------------------------------

def test_non_usd_quoted_pair_is_rejected():
    rm = RiskManager(account_currency="USD")
    rm.update_day_start(10000.0)
    approved = rm.evaluate_signal(
        make_signal(epic="USDJPY", entry=150.0, sl=149.25, tp=151.5),
        equity=10000.0, open_positions=[],
        market_info=make_market_info(epic="USDJPY", pip_size=0.01),
    )
    assert approved is None


def test_usd_quoted_pair_is_accepted():
    rm = RiskManager(account_currency="USD")
    rm.update_day_start(10000.0)
    approved = rm.evaluate_signal(
        make_signal(), equity=10000.0, open_positions=[],
        market_info=make_market_info(),
    )
    assert approved is not None
    assert approved.size > 0


# ---------------------------------------------------------------------------
# MAX_TRADES_PER_DAY is enforced
# ---------------------------------------------------------------------------

def test_max_trades_per_day_enforced():
    rm = RiskManager(account_currency="USD")
    rm.update_day_start(10000.0)
    approvals = 0
    for i in range(RISK.MAX_TRADES_PER_DAY + 3):
        # Alternate direction so the duplicate epic+direction check never fires
        direction = "BUY" if i % 2 == 0 else "SELL"
        sig = make_signal(direction=direction,
                          sl=1.0950 if direction == "BUY" else 1.1050,
                          tp=1.1100 if direction == "BUY" else 1.0900)
        approved = rm.evaluate_signal(
            sig, equity=10000.0, open_positions=[],
            market_info=make_market_info(),
        )
        if approved:
            approvals += 1
    assert approvals == RISK.MAX_TRADES_PER_DAY


# ---------------------------------------------------------------------------
# Anti-correlation: max 1 simultaneous FX position
# ---------------------------------------------------------------------------

def test_second_fx_position_rejected():
    rm = RiskManager(account_currency="USD")
    rm.update_day_start(10000.0)
    open_positions = [make_position(epic="EURUSD", direction="BUY")]
    approved = rm.evaluate_signal(
        make_signal(epic="GBPUSD", entry=1.30, sl=1.295, tp=1.31),
        equity=10000.0, open_positions=open_positions,
        market_info=make_market_info(epic="GBPUSD"),
    )
    assert approved is None


def test_fx_detection():
    assert is_fx_pair("EURUSD")
    assert is_fx_pair("USDJPY")
    assert not is_fx_pair("GOLD")
    assert not is_fx_pair("BTCUSD")  # BTC is not an ISO currency here
    assert quote_currency("EURUSD") == "USD"
    assert quote_currency("USDJPY") == "JPY"
    assert quote_currency("GOLD") is None
