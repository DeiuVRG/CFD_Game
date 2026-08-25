"""Statistical validation on the TEST set (spec Part VI): bootstrap CI on the
Sharpe ratio, permutation test vs. the market, and stress testing.

Stress-scenario note: the spec's canonical scenarios (2008, dot-com) predate
the game's 2015-2023 data, so resilience is evaluated on the crisis windows
that DO exist in the data (the 2020 COVID crash and the 2022 bear market)
plus the worst automatically-detected market drawdown window. Same intent:
does the strategy survive the ugliest regimes it could actually have faced.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from trading_game.metrics import AdvancedMetrics


class StatisticalValidation:
    def __init__(self, metrics: AdvancedMetrics, iterations: int = 1000,
                 seed: int = 42):
        self.metrics = metrics
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)

    def bootstrap_validation(self, returns: np.ndarray) -> dict:
        """95% bootstrap CI for the Sharpe ratio; significant if the CI
        excludes 0."""
        returns = np.asarray(returns, dtype=float)
        if len(returns) < 20:
            return {"sharpe_observed": 0.0, "confidence_interval": (0.0, 0.0),
                    "is_significant": False, "p_value": 1.0}
        sharpes = np.empty(self.iterations)
        for i in range(self.iterations):
            sample = self.rng.choice(returns, size=len(returns), replace=True)
            sharpes[i] = self.metrics.sharpe_ratio(sample)
        ci_lower = float(np.percentile(sharpes, 2.5))
        ci_upper = float(np.percentile(sharpes, 97.5))
        p_value = float(np.mean(sharpes <= 0))
        return {
            "sharpe_observed": self.metrics.sharpe_ratio(returns),
            "confidence_interval": (ci_lower, ci_upper),
            "is_significant": bool(ci_lower > 0),
            "p_value": p_value,
        }

    def permutation_test_vs_market(self, strategy_returns: np.ndarray,
                                   market_returns: np.ndarray) -> dict:
        """Does the strategy beat the market SIGNIFICANTLY (mean return)?"""
        s = np.asarray(strategy_returns, dtype=float)
        m = np.asarray(market_returns, dtype=float)
        if len(s) < 20 or len(m) < 20:
            return {"beats_market": False, "is_significant": False, "p_value": 1.0}
        observed = float(np.mean(s) - np.mean(m))
        combined = np.concatenate([s, m])
        n_s = len(s)
        diffs = np.empty(self.iterations)
        for i in range(self.iterations):
            self.rng.shuffle(combined)
            diffs[i] = np.mean(combined[:n_s]) - np.mean(combined[n_s:])
        p_value = float(np.mean(np.abs(diffs) >= abs(observed)))
        return {
            "observed_diff_daily": observed,
            "beats_market": bool(observed > 0),
            "is_significant": bool(p_value < 0.05),
            "p_value": p_value,
        }


class StressTesting:
    # Known crisis windows inside the game's data range
    KNOWN_SCENARIOS = {
        "covid_crash_2020": ("2020-02-14", "2020-03-23"),
        "bear_market_2022": ("2022-01-03", "2022-10-14"),
    }
    SURVIVAL_DD = -0.50

    def __init__(self, market: pd.DataFrame):
        self.market_close = market.set_index("timestamp")["close"]

    def worst_market_window(self, days: int = 60) -> tuple:
        """Auto-detected worst market drawdown window in the data."""
        close = self.market_close
        roll_ret = close.pct_change(days)
        if roll_ret.dropna().empty:
            return None
        end = roll_ret.idxmin()
        start = close.index[max(0, close.index.get_loc(end) - days)]
        return (str(start.date()), str(end.date()))

    def scenarios(self) -> Dict[str, tuple]:
        scen = dict(self.KNOWN_SCENARIOS)
        worst = self.worst_market_window()
        if worst:
            scen["worst_auto_window"] = worst
        return scen

    def evaluate_resilience(self, daily_equity: pd.Series) -> dict:
        """Max drawdown of the player's equity inside each scenario window.
        Survives when DD stays above -50% in every scenario."""
        results = {}
        for name, (start, end) in self.scenarios().items():
            window = daily_equity.loc[pd.Timestamp(start):pd.Timestamp(end)]
            if len(window) < 5:
                results[name] = {"max_drawdown": 0.0, "survives": True,
                                 "note": "window outside evaluated period"}
                continue
            dd = AdvancedMetrics.max_drawdown(window.to_numpy())
            results[name] = {"max_drawdown": dd,
                             "survives": bool(dd > self.SURVIVAL_DD)}
        return {
            "scenarios": results,
            "pass_stress_test": bool(all(r["survives"] for r in results.values())),
        }


def final_composite_score(test_score: float, is_significant: bool,
                          beats_market: bool, passes_stress: bool) -> float:
    """Spec 6.3: the REAL winner ranking."""
    return (
        0.40 * test_score
        + 0.25 * float(is_significant)
        + 0.20 * float(beats_market)
        + 0.15 * float(passes_stress)
    )
