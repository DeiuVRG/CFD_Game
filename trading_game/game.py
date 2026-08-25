"""Game orchestration: preparation on the training set, 24-month competition
on the validation set (eliminations + quarterly recalibrations), final
out-of-sample evaluation on the test set with statistical validation, and
reporting. Mirrors the flow in docs/trading_game_prompt.md Part VIII."""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_game.config import GameConfig
from trading_game.crupier import Crupier
from trading_game.data_loader import load_data
from trading_game.metrics import (AdvancedMetrics, ComplexityPenalty,
                                  RegimeEvaluation, monthly_final_score)
from trading_game.players import DEFAULT_PLAYER_CLASSES
from trading_game.validation import (StatisticalValidation, StressTesting,
                                     final_composite_score)

logger = logging.getLogger(__name__)


class TradingGameSystem:
    def __init__(self, config: GameConfig = None, player_classes=None,
                 verbose: bool = True):
        self.config = config or GameConfig()
        self.player_classes = player_classes or DEFAULT_PLAYER_CLASSES
        self.verbose = verbose

        self.data = None
        self.market = None
        self.vix = None
        self.crupier: Optional[Crupier] = None
        self.players = []
        self.metrics = AdvancedMetrics(self.config.risk_free_rate)
        self.regime_eval: Optional[RegimeEvaluation] = None
        self.monthly_scores: Dict[str, List[float]] = {}
        self.elimination_log: List[dict] = []
        self.test_results: Dict[str, dict] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Phase 1: preparation (training set)
    # ------------------------------------------------------------------

    def _bundle(self, until: pd.Timestamp = None, start: pd.Timestamp = None) -> dict:
        """Data bundles for player fitting, hard-limited to <= until."""
        bundle = {}
        for symbol in self.config.universe:
            signals = self.crupier._signals[symbol]
            close = self.crupier._close[symbol]
            if start is not None:
                signals = signals.loc[pd.Timestamp(start):]
                close = close.loc[pd.Timestamp(start):]
            if until is not None:
                signals = signals.loc[:pd.Timestamp(until)]
                close = close.loc[:pd.Timestamp(until)]
            bundle[symbol] = {"signals": signals, "close": close}
        return bundle

    def setup(self):
        self._log("=" * 78)
        self._log("  TRADING GAME - competitive indicator-weight discovery")
        self._log("=" * 78)
        self.data, self.market, self.vix = load_data(self.config)
        self.crupier = Crupier(self.config, self.data, self.market, self.vix)
        self.regime_eval = RegimeEvaluation(self.market, self.vix)

        self._log(f"\n[FAZA 1] Registration + initial weights on the training set "
                  f"({self.config.train_start} .. {self.config.train_end})")
        train_bundle = self._bundle(until=self.config.train_end)

        for i, cls in enumerate(self.player_classes):
            player = cls(f"P{i+1}_{cls.name}", self.config, seed=self.config.seed + i)
            registration = self.crupier.register_player(
                player.id, player.indicators, player.method_description)
            if registration["status"] != "APPROVED":
                raise RuntimeError(f"{player.id} rejected: {registration}")
            player.fit(train_bundle)
            self.players.append(player)
            self.monthly_scores[player.id] = []
            top = sorted(player.weights.items(), key=lambda kv: -kv[1])[:3]
            self._log(f"  + {player.id:26s} {len(player.indicators):>2} indicators"
                      f" | top: " + ", ".join(f"{k}={v:.2f}" for k, v in top))

    # ------------------------------------------------------------------
    # Phase 2: competition on the validation set
    # ------------------------------------------------------------------

    def _active_players(self) -> list:
        return [p for p in self.players if p.id not in self.crupier.eliminated]

    def _player_stats(self, player_id: str):
        equity = self.crupier.equity_series(player_id)
        if len(equity) < 2:
            return np.array([]), np.array([self.config.initial_capital]), []
        returns = equity.pct_change().dropna().to_numpy()
        return returns, equity.to_numpy(), self.crupier.closed_trades[player_id]

    def run_validation(self) -> list:
        cfg = self.config
        self._log(f"\n[FAZA 2] Competition on the validation set "
                  f"({cfg.validation_start} .. {cfg.validation_end})")
        days = self.crupier.trading_days(cfg.validation_start, cfg.validation_end)
        months = pd.PeriodIndex(days, freq="M").unique().sort_values()

        for month_index, month in enumerate(months, start=1):
            month_days = days[pd.PeriodIndex(days, freq="M") == month]
            first_day, last_day = month_days[0], month_days[-1]
            active = self._active_players()
            if len(active) <= 1:
                break

            # Quarterly recalibration BEFORE the new period (months 4, 7, ...)
            if month_index > 1 and (month_index - 1) % cfg.recalibration_every_months == 0:
                self._log(f"\n--- Luna {month_index}/{len(months)} ({month}) | "
                          f"quarterly recalibration point ---")
                until = first_day - pd.Timedelta(days=1)
                bundle = self._bundle(until=until)
                for player in active:
                    if player.recalibrations >= 8:
                        continue
                    proposal = player.propose_recalibration(bundle)
                    if self.crupier.validate_recalibration(player.weights, proposal):
                        player.apply_recalibration(proposal)
                    else:
                        self._log(f"  ! {player.id}: recalibration REJECTED "
                                  f"(shift > {cfg.max_weight_shift:.0%})")
            else:
                self._log(f"\n--- Luna {month_index}/{len(months)} ({month}) ---")

            # Decisions on the first trading day of the month
            self.crupier.set_current_date(first_day)
            for player in active:
                orders = player.decide_trades(self.crupier, first_day)
                for order in orders:
                    self.crupier.execute_trade(player.id, order)

            # Daily equity snapshots through the month
            for day in month_days:
                self.crupier.set_current_date(day)
                self.crupier.snapshot_equity([p.id for p in active], day)

            # End of month: violations, scores, elimination
            self.crupier.set_current_date(last_day)
            month_key = f"{first_day.year}-{first_day.month:02d}"
            for player in list(active):
                violation = self.crupier.check_violations(player.id, month_key)
                if violation:
                    self.crupier.eliminate(player.id, violation)
                    self.elimination_log.append({
                        "month": str(month), "player": player.id,
                        "reason": violation,
                    })
                    self._log(f"  x {player.id} ELIMINAT: {violation}")

            active = self._active_players()
            scores = {}
            for player in active:
                returns, equity_curve, trades = self._player_stats(player.id)
                daily_returns = self.crupier.equity_series(player.id).pct_change().dropna()
                score = monthly_final_score(
                    self.metrics, returns, equity_curve, trades,
                    len(player.indicators), month_index,
                    self.regime_eval, daily_returns,
                )
                scores[player.id] = score
                self.monthly_scores[player.id].append(score)

            eliminated = self.crupier.evaluate_elimination(scores)
            if eliminated:
                self.crupier.eliminate(eliminated,
                                       f"score gap >= {cfg.elimination_gap:.0%}")
                self.elimination_log.append({
                    "month": str(month), "player": eliminated,
                    "reason": "performance gap",
                })
                self._log(f"  x {eliminated} ELIMINAT: sub prag "
                          f"{cfg.elimination_gap:.0%}")

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            board = " | ".join(
                f"{pid.split('_', 1)[1]}={score:.3f}"
                + ("(x)" if pid in self.crupier.eliminated else "")
                for pid, score in ranked)
            self._log(f"  clasament: {board}")

        survivors = self._active_players()
        self._log(f"\n{'=' * 78}")
        self._log(f"SUPRAVIETUITORI: {[p.id for p in survivors]}")
        self._log(f"{'=' * 78}")
        return survivors

    # ------------------------------------------------------------------
    # Phase 3: out-of-sample evaluation on the test set
    # ------------------------------------------------------------------

    def simulate_period(self, player, start: str, end: str,
                        weights_override: dict = None):
        """Run one player's strategy with FIXED weights over [start, end] on
        a fresh portfolio; returns (daily equity series, closed trades).
        Used for the test set and for ablation studies."""
        sim = Crupier(self.config, self.data, self.market, self.vix)
        sim.register_player(player.id, player.indicators,
                            player.method_description)
        saved_weights = dict(player.weights)
        if weights_override is not None:
            player.weights = weights_override
        try:
            days = sim.trading_days(start, end)
            months = pd.PeriodIndex(days, freq="M").unique().sort_values()
            for month in months:
                month_days = days[pd.PeriodIndex(days, freq="M") == month]
                sim.set_current_date(month_days[0])
                for order in player.decide_trades(sim, month_days[0]):
                    sim.execute_trade(player.id, order)
                for day in month_days:
                    sim.set_current_date(day)
                    sim.snapshot_equity([player.id], day)
            return sim.equity_series(player.id), sim.closed_trades[player.id]
        finally:
            player.weights = saved_weights

    def run_test(self, survivors: list) -> Dict[str, dict]:
        cfg = self.config
        self._log(f"\n[FAZA 3] Final evaluation on the TEST set "
                  f"({cfg.test_start} .. {cfg.test_end}) - fixed weights, "
                  f"no recalibration")
        validator = StatisticalValidation(self.metrics,
                                          cfg.bootstrap_iterations, cfg.seed)
        stress = StressTesting(self.market)

        market_close = self.market.set_index("timestamp")["close"]
        market_test = market_close.loc[pd.Timestamp(cfg.test_start):
                                       pd.Timestamp(cfg.test_end)]
        market_returns = market_test.pct_change().dropna().to_numpy()

        results = {}
        for player in survivors:
            self._log(f"\n  Evaluare {player.id} pe Test Set...")
            equity, trades = self.simulate_period(player, cfg.test_start,
                                                  cfg.test_end)
            returns = equity.pct_change().dropna()
            test_metrics = self.metrics.full_metrics(
                returns.to_numpy(), equity.to_numpy(), trades)
            test_score = ComplexityPenalty.adjusted_score(
                test_metrics["composite_score"], len(player.indicators))

            bootstrap = validator.bootstrap_validation(returns.to_numpy())
            market_cmp = validator.permutation_test_vs_market(
                returns.to_numpy(), market_returns)

            # Stress: full equity history (validation + test)
            full_equity = pd.concat([
                self.crupier.equity_series(player.id), equity])
            full_equity = full_equity[~full_equity.index.duplicated(keep="first")]
            resilience = stress.evaluate_resilience(full_equity)

            final = final_composite_score(
                test_score, bootstrap["is_significant"],
                market_cmp["is_significant"] and market_cmp["beats_market"],
                resilience["pass_stress_test"])

            results[player.id] = {
                "test_metrics": test_metrics,
                "test_score_adjusted": test_score,
                "bootstrap": bootstrap,
                "vs_market": market_cmp,
                "stress": resilience,
                "final_score": final,
                "test_equity": equity,
            }
            self._log(f"    test return {test_metrics['total_return_pct']:+.2f}% | "
                      f"sharpe {test_metrics['sharpe']:.2f} | "
                      f"significant={bootstrap['is_significant']} | "
                      f"beats market={market_cmp['beats_market']} "
                      f"(p={market_cmp['p_value']:.3f}) | "
                      f"stress={'PASS' if resilience['pass_stress_test'] else 'FAIL'} | "
                      f"FINAL={final:.3f}")

        self.test_results = results
        return results

    # ------------------------------------------------------------------

    def determine_winner(self) -> Optional[str]:
        if not self.test_results:
            return None
        ranked = sorted(self.test_results.items(),
                        key=lambda kv: kv[1]["final_score"], reverse=True)
        return ranked[0][0]

    def run_complete_game(self) -> dict:
        self.setup()
        survivors = self.run_validation()
        self.run_test(survivors)
        winner = self.determine_winner()
        self._log(f"\n{'=' * 78}")
        self._log(f"  CASTIGATOR (test set + validare statistica): {winner}")
        self._log(f"{'=' * 78}")

        from trading_game.reporting import generate_all_reports
        reports = generate_all_reports(self)
        return {"winner": winner, "reports": reports}
