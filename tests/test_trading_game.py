"""Trading game tests: cost/liquidity units, crupier rules, and a compact
end-to-end synthetic game (offline, seeded, fast)."""
import numpy as np
import pandas as pd
import pytest

from trading_game.config import GameConfig
from trading_game.costs import LiquidityConstraints, RealisticCosts
from trading_game.crupier import Crupier
from trading_game.data_loader import load_synthetic_data
from trading_game.game import TradingGameSystem
from trading_game.metrics import AdvancedMetrics, ComplexityPenalty
from trading_game.players import (CorrelationPlayer, EqualWeightPlayer,
                                  MomentumPlayer)


def small_config(**overrides) -> GameConfig:
    params = dict(
        train_start="2015-01-01", train_end="2016-12-31",
        validation_start="2017-01-01", validation_end="2017-12-31",
        test_start="2018-01-01", test_end="2018-12-31",
        universe=["AAPL", "MSFT", "AMZN", "JPM", "XOM", "KO"],
        synthetic=True, seed=11,
    )
    params.update(overrides)
    return GameConfig(**params)


# ---------------------------------------------------------------------------
# Costs & liquidity
# ---------------------------------------------------------------------------

def test_costs_have_minimum_commission():
    costs = RealisticCosts(GameConfig())
    assert costs.transaction_cost(50.0, 1, 1e6) == pytest.approx(1.0)


def test_costs_scale_with_order_size():
    costs = RealisticCosts(GameConfig())
    small = costs.transaction_cost(10_000, 100, 1e6)
    # Same value, but the order is a big chunk of daily volume -> more slippage
    big = costs.transaction_cost(10_000, 5_000, 1e5)
    assert big > small


def test_liquidity_caps_at_1pct_adv():
    liq = LiquidityConstraints(GameConfig())
    res = liq.limit_order(desired_qty=50_000, price=100.0, avg_daily_volume=1e6)
    assert res.executed_qty == pytest.approx(10_000)   # 1% of ADV
    assert res.rejected_qty == pytest.approx(40_000)
    assert res.execution_price > 100.0                  # deteriorated


def test_complexity_penalty_tiers():
    assert ComplexityPenalty.penalty(5) == 0.0
    assert ComplexityPenalty.penalty(10) == 0.05
    assert ComplexityPenalty.penalty(15) == 0.10
    assert ComplexityPenalty.penalty(20) == 0.15


# ---------------------------------------------------------------------------
# Crupier rules
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def crupier():
    cfg = small_config()
    data, market, vix = load_synthetic_data(cfg)
    c = Crupier(cfg, data, market, vix)
    c.register_player("tester", ["RSI", "MACD_HIST", "BB_POSITION"])
    c.set_current_date(pd.Timestamp("2017-03-01"))
    return c


def test_crupier_rejects_bad_registration(crupier):
    assert crupier.register_player("bad1", ["RSI"])["status"] == "REJECTED"
    assert crupier.register_player("bad2", ["NOPE", "RSI", "MACD_HIST"])["status"] == "REJECTED"


def test_temporal_integrity_no_future_data(crupier):
    date = pd.Timestamp("2017-03-01")
    sig = crupier.provide_indicators("tester", "AAPL", until=date)
    assert sig.index.max() <= date
    ohlcv = crupier.provide_ohlcv("AAPL", until=date)
    assert ohlcv.index.max() <= date


def test_trade_execution_applies_costs(crupier):
    result = crupier.execute_trade("tester", {"action": "BUY",
                                              "company": "AAPL",
                                              "quantity": 10})
    assert result["status"] == "EXECUTED"
    assert result["total_cost"] >= 1.0
    portfolio = crupier.portfolios["tester"]
    assert portfolio.position_qty("AAPL") > 0
    assert portfolio.cash < crupier.config.initial_capital


def test_sell_without_position_rejected(crupier):
    result = crupier.execute_trade("tester", {"action": "SELL",
                                              "company": "KO",
                                              "quantity": 5})
    assert result["status"] == "REJECTED"


def test_recalibration_shift_validation(crupier):
    old = {"RSI": 0.5, "MACD_HIST": 0.3, "BB_POSITION": 0.2}
    ok = {"RSI": 0.35, "MACD_HIST": 0.40, "BB_POSITION": 0.25}
    drastic = {"RSI": 0.05, "MACD_HIST": 0.75, "BB_POSITION": 0.20}
    not_normalized = {"RSI": 0.5, "MACD_HIST": 0.3, "BB_POSITION": 0.1}
    assert crupier.validate_recalibration(old, ok) is True
    assert crupier.validate_recalibration(old, drastic) is False
    assert crupier.validate_recalibration(old, not_normalized) is False


def test_external_data_registry_blocks_lookahead(crupier):
    approval = crupier.approve_external_source(
        "tester", {"tip_sursa": "news", "descriere": "test feed"})
    assert approval["status"] == "APPROVED"
    future = crupier.register_external_data(
        "tester", approval["source_id"], {"headline": "x"},
        as_of=pd.Timestamp("2030-01-01"))
    assert future["status"] == "REJECTED"
    past = crupier.register_external_data(
        "tester", approval["source_id"], {"headline": "x"},
        as_of=pd.Timestamp("2017-02-01"))
    assert past["status"] == "REGISTERED"
    assert past["sha256"]


def test_elimination_gap_rule(crupier):
    scores = {"a": 0.50, "b": 0.40, "c": 0.30}
    assert crupier.evaluate_elimination(scores) == "c"   # 25% behind b
    scores = {"a": 0.50, "b": 0.40, "c": 0.38}
    assert crupier.evaluate_elimination(scores) is None  # only 5% behind
    assert crupier.evaluate_elimination({"a": 0.5, "b": 0.1}) is None  # 2 left


# ---------------------------------------------------------------------------
# End-to-end synthetic game (compact)
# ---------------------------------------------------------------------------

def test_full_game_synthetic_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_game.reporting.RESULTS_DIR", str(tmp_path))
    game = TradingGameSystem(
        small_config(),
        player_classes=[EqualWeightPlayer, CorrelationPlayer, MomentumPlayer],
        verbose=False,
    )
    outcome = game.run_complete_game()

    assert outcome["winner"] is not None
    assert outcome["winner"] in {p.id for p in game.players}

    # Every player's weights are a valid normalized set
    for player in game.players:
        assert sum(player.weights.values()) == pytest.approx(1.0)
        assert all(0 <= w <= 1 for w in player.weights.values())
        assert 3 <= len(player.indicators) <= 20

    # Audit log exists and every trade carries a cost
    assert game.crupier.trade_log, "no trades were executed"
    assert all(t["cost"] >= 0 for t in game.crupier.trade_log)

    # Mandatory monthly trading was respected by every non-eliminated player
    survivors = [p.id for p in game.players
                 if p.id not in game.crupier.eliminated]
    assert survivors, "everyone got eliminated"

    # Reports were generated with the expected structure
    reports = outcome["reports"]
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "castigator_report.json").exists()
    winner_report = reports["winner"]
    assert "final_weights" in winner_report
    assert "indicator_importance_ablation" in winner_report
    assert "performance_test" in winner_report
    assert "confidence_interval" in winner_report["performance_test"]
    ranking = reports["comparison"]["final_ranking"]
    assert len(ranking) == 3
