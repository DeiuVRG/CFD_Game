"""Annualization of candle-based metrics (v3.1).

Before: Sharpe/Sortino/Calmar and the trade-Sharpe were annualized with a
hardcoded 252 periods/year - correct for daily candles, wrong for the 1h
candles the evaluation runs on (~5980/year for gold, 8760 for BTC). The
trade-Sharpe used for the activation gate was shrunk ~5x on gold.
"""
import math

import numpy as np
import pytest

from ai.backtester import BacktestMetrics, BacktestTrade
from config.settings import INSTRUMENTS, InstrumentConfig, parse_interval_hours


def inst(symbol):
    return next(i for i in INSTRUMENTS if i.SYMBOL == symbol)


def test_parse_interval_hours():
    assert parse_interval_hours("5m") == pytest.approx(1 / 12)
    assert parse_interval_hours("1h") == 1.0
    assert parse_interval_hours("4h") == 4.0
    assert parse_interval_hours("1d") == 24.0
    with pytest.raises(ValueError):
        parse_interval_hours("1x")


def test_candles_per_year_gold_and_btc():
    assert inst("GC=F").candles_per_year() == pytest.approx(23 * 5 * 52)     # 5980
    assert inst("BTC-USD").candles_per_year() == pytest.approx(24 * 365)     # 8760
    assert inst("GC=F").candles_per_year("4h") == pytest.approx(5980 / 4)


def make_trades(pnls):
    return [BacktestTrade("BUY", 100, 100 + p, i, i + 1, 95, 110, "t",
                          pnl_pct=p, pnl_net_pct=p, cost_pct=0.0)
            for i, p in enumerate(pnls)]


def test_trade_sharpe_scales_with_periods_per_year():
    """Same trades and curve: the trade-Sharpe must scale with
    sqrt(periods_per_year), i.e. the old 252 understated it by
    sqrt(5980/252) ~ 4.9x on hourly gold data."""
    pnls = [1.0, -0.5, 2.0, -1.0, 1.5, 0.5, -0.7, 1.2]
    trades = make_trades(pnls)
    curve = np.array([100000.0] * (len(pnls) * 10))   # 80 candles
    old = BacktestMetrics.compute_all(trades, curve, periods_per_year=252)
    new = BacktestMetrics.compute_all(trades, curve, periods_per_year=5980)
    assert new["trade_sharpe"] / old["trade_sharpe"] == pytest.approx(
        math.sqrt(5980 / 252), rel=1e-9)
    assert new["periods_per_year"] == 5980


def test_trade_sharpe_sign_is_preserved():
    losing = make_trades([-1.0, -0.5, 0.2, -1.2, -0.3])
    curve = np.array([100000.0] * 50)
    for ppy in (252, 5980, 8760):
        assert BacktestMetrics.compute_all(losing, curve, ppy)["trade_sharpe"] < 0


def test_default_keeps_backwards_compatible_252():
    trades = make_trades([1.0, -0.5, 2.0])
    curve = np.array([100000.0] * 30)
    assert (BacktestMetrics.compute_all(trades, curve)["periods_per_year"]
            == 252)
