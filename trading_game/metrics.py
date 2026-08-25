"""Advanced metrics (spec Part IV): risk-adjusted composite score, complexity
penalty, market-regime evaluation, monthly final score."""
from typing import Dict, List

import numpy as np
import pandas as pd


class AdvancedMetrics:
    """Risk-adjusted metrics - NOT raw profit."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess = returns - (self.risk_free_rate / 252)
        return float(np.mean(excess) / np.std(returns) * np.sqrt(252))

    def sortino_ratio(self, returns: np.ndarray) -> float:
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        excess = returns - (self.risk_free_rate / 252)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(252))

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        equity_curve = np.asarray(equity_curve, dtype=float)
        if len(equity_curve) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return float(np.min(drawdown))

    def calmar_ratio(self, returns: np.ndarray, equity_curve: np.ndarray) -> float:
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        annual_return = float(np.mean(returns) * 252)
        max_dd = abs(self.max_drawdown(equity_curve))
        if max_dd == 0:
            return float("inf") if annual_return > 0 else 0.0
        return annual_return / max_dd

    @staticmethod
    def win_rate(trades: List[dict]) -> float:
        closed = [t for t in trades if "profit" in t]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t["profit"] > 0)
        return wins / len(closed)

    @staticmethod
    def profit_factor(trades: List[dict]) -> float:
        gross_profit = sum(t["profit"] for t in trades if t.get("profit", 0) > 0)
        gross_loss = abs(sum(t["profit"] for t in trades if t.get("profit", 0) < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def composite_score(self, returns: np.ndarray, equity_curve: np.ndarray,
                        trades: List[dict]) -> float:
        """Weighted combination in [0, 1] (spec 4.1)."""
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        calmar = self.calmar_ratio(returns, equity_curve)
        win_rate = self.win_rate(trades)
        pf = self.profit_factor(trades)

        def norm(x, cap):
            if x == float("inf"):
                return 1.0
            return max(0.0, min(x / cap, 1.0))

        return (
            0.25 * norm(sharpe, 2.0)
            + 0.25 * norm(sortino, 2.5)
            + 0.20 * norm(calmar, 1.0)
            + 0.15 * win_rate
            + 0.15 * norm(pf, 2.0)
        )

    def full_metrics(self, returns, equity_curve, trades) -> dict:
        return {
            "total_return_pct": float((equity_curve[-1] / equity_curve[0] - 1) * 100)
            if len(equity_curve) > 1 else 0.0,
            "sharpe": self.sharpe_ratio(returns),
            "sortino": self.sortino_ratio(returns),
            "calmar": self.calmar_ratio(returns, equity_curve),
            "max_drawdown": self.max_drawdown(equity_curve),
            "win_rate": self.win_rate(trades),
            "profit_factor": self.profit_factor(trades),
            "n_trades": len(trades),
            "composite_score": self.composite_score(returns, equity_curve, trades),
        }


class ComplexityPenalty:
    """Occam's razor: simpler strategies preferred (spec 4.2)."""

    @staticmethod
    def penalty(n_indicators: int) -> float:
        if n_indicators <= 5:
            return 0.0
        if n_indicators <= 10:
            return 0.05
        if n_indicators <= 15:
            return 0.10
        return 0.15

    @classmethod
    def adjusted_score(cls, composite: float, n_indicators: int) -> float:
        return composite * (1 - cls.penalty(n_indicators))


class RegimeEvaluation:
    """Market-regime classification and multi-regime scoring (spec 4.3)."""

    def __init__(self, market: pd.DataFrame, vix: pd.DataFrame):
        self.market = market.set_index("timestamp")["close"]
        self.vix = vix.set_index("timestamp")["close"]

    def classify_day(self, date: pd.Timestamp) -> List[str]:
        regimes = []
        vix_val = self.vix.asof(date)
        if not pd.isna(vix_val):
            if vix_val > 25:
                regimes.append("high_volatility")
            elif vix_val < 15:
                regimes.append("low_volatility")

        market_slice = self.market.loc[:date].tail(64)
        if len(market_slice) >= 40:
            trend_3m = market_slice.iloc[-1] / market_slice.iloc[0] - 1
            annualized = trend_3m * 4
            if annualized > 0.15:
                regimes.append("bull_market")
            elif annualized < -0.10:
                regimes.append("bear_market")
            else:
                regimes.append("sideways")
        return regimes

    def multi_regime_score(self, daily_returns: pd.Series,
                           metrics: AdvancedMetrics) -> dict:
        """Score per regime; final = 0.6*mean + 0.4*worst (must work
        everywhere, not just in bull markets)."""
        buckets: Dict[str, list] = {}
        for date, ret in daily_returns.items():
            for regime in self.classify_day(date):
                buckets.setdefault(regime, []).append(ret)

        per_regime = {}
        for regime, rets in buckets.items():
            if len(rets) < 10:
                continue
            arr = np.array(rets)
            equity = np.cumprod(1 + arr)
            per_regime[regime] = metrics.composite_score(arr, equity, [])

        if not per_regime:
            return {"score": 0.0, "per_regime": {}}
        values = list(per_regime.values())
        score = 0.6 * float(np.mean(values)) + 0.4 * float(np.min(values))
        return {"score": score, "per_regime": per_regime}


def monthly_final_score(metrics: AdvancedMetrics,
                        returns: np.ndarray,
                        equity_curve: np.ndarray,
                        trades: List[dict],
                        n_indicators: int,
                        month_index: int,
                        regime_eval: RegimeEvaluation = None,
                        daily_returns: pd.Series = None) -> float:
    """The score used for rankings and eliminations (spec 4.4)."""
    composite = metrics.composite_score(returns, equity_curve, trades)
    adjusted = ComplexityPenalty.adjusted_score(composite, n_indicators)

    if month_index >= 6 and regime_eval is not None and daily_returns is not None:
        regime_score = regime_eval.multi_regime_score(daily_returns, metrics)["score"]
        return 0.7 * adjusted + 0.3 * regime_score
    return adjusted
