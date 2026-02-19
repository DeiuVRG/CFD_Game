"""
Backtest framework with realistic cost modeling and risk-adjusted metrics.

Two modes:
  1. Fixed split (50/25/25) - for comparison with optimizer results
  2. Walk-forward - re-trains model every N candles (realistic production sim)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import InstrumentConfig, COSTS, AI, STRATEGY
from data.indicators import Indicators


@dataclass
class BacktestTrade:
    direction: str
    entry_price: float
    exit_price: float
    entry_idx: int
    exit_idx: int
    stop_loss: float
    take_profit: float
    strategy: str
    pnl_pct: float = 0.0
    pnl_net_pct: float = 0.0   # After costs
    cost_pct: float = 0.0


class BacktestMetrics:
    """Risk-adjusted performance metrics."""

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess = returns - (risk_free_rate / 252)
        return float(np.mean(excess) / np.std(returns) * np.sqrt(252))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        excess = returns - (risk_free_rate / 252)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float('inf')
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(252))

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        if len(equity_curve) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return float(np.min(drawdown))

    @staticmethod
    def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        annual_return = float(np.mean(returns) * 252)
        max_dd = abs(BacktestMetrics.max_drawdown(equity_curve))
        if max_dd == 0:
            return float('inf')
        return annual_return / max_dd

    @staticmethod
    def profit_factor(trades: list) -> float:
        gross_profit = sum(t.pnl_net_pct for t in trades if t.pnl_net_pct > 0)
        gross_loss = abs(sum(t.pnl_net_pct for t in trades if t.pnl_net_pct < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def win_rate(trades: list) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl_net_pct > 0)
        return wins / len(trades)

    @staticmethod
    def compute_all(trades: list, equity_curve: np.ndarray) -> dict:
        if not trades or len(equity_curve) < 2:
            return {
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "sharpe": 0, "trade_sharpe": 0, "sortino": 0,
                "max_drawdown": 0, "calmar": 0,
                "total_return_pct": 0, "avg_trade_pnl": 0,
                "avg_win": 0, "avg_loss": 0, "total_costs_pct": 0,
            }

        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_costs = sum(t.cost_pct for t in trades)
        net_pnls = [t.pnl_net_pct for t in trades]
        trade_returns = np.array([t.pnl_net_pct / 100 for t in trades])

        # Trade-based Sharpe (more meaningful for infrequent trading)
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            # Assume ~2 trades per week on average
            trades_per_year = len(trade_returns) * 252 / max(len(equity_curve), 1)
            trade_sharpe = float(np.mean(trade_returns) / np.std(trade_returns)
                                 * np.sqrt(max(trades_per_year, 1)))
        else:
            trade_sharpe = 0.0

        return {
            "total_trades": len(trades),
            "win_rate": BacktestMetrics.win_rate(trades),
            "profit_factor": BacktestMetrics.profit_factor(trades),
            "sharpe": BacktestMetrics.sharpe_ratio(returns),
            "trade_sharpe": trade_sharpe,
            "sortino": BacktestMetrics.sortino_ratio(returns),
            "max_drawdown": BacktestMetrics.max_drawdown(equity_curve),
            "calmar": BacktestMetrics.calmar_ratio(returns, equity_curve),
            "total_return_pct": float((equity_curve[-1] / equity_curve[0] - 1) * 100),
            "avg_trade_pnl": float(np.mean(net_pnls)),
            "avg_win": float(np.mean([p for p in net_pnls if p > 0])) if any(p > 0 for p in net_pnls) else 0,
            "avg_loss": float(np.mean([p for p in net_pnls if p < 0])) if any(p < 0 for p in net_pnls) else 0,
            "total_costs_pct": total_costs,
        }


class Backtester:
    """Run backtest with same split logic as optimizer for fair comparison."""

    def __init__(self, instrument: InstrumentConfig):
        self.instrument = instrument

    def _simulate_trades(self, predictor, features, df, test_indices) -> tuple:
        """Core simulation loop - shared between run() modes."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_series = Indicators.atr(high, low, close, 14)
        adx_series = Indicators.adx(high, low, close, 14)

        trades = []
        equity = 100000.0
        equity_curve = [equity]
        position = None

        for idx in test_indices:
            price = close.iloc[idx]
            atr = atr_series.iloc[idx]

            if pd.isna(atr) or atr == 0:
                equity_curve.append(equity)
                continue

            spread_cost = (self.instrument.SPREAD_PIPS * self.instrument.PIP_VALUE) / price
            round_trip_cost = spread_cost * 2 * (1 + COSTS.SLIPPAGE_MULTIPLIER)

            # Check SL/TP for open position
            if position is not None:
                direction, entry, sl, tp, entry_i = position
                closed = False

                if direction == "BUY":
                    if price <= sl:
                        pnl = (sl - entry) / entry
                        closed = True
                    elif price >= tp:
                        pnl = (tp - entry) / entry
                        closed = True
                elif direction == "SELL":
                    if price >= sl:
                        pnl = (entry - sl) / entry
                        closed = True
                    elif price <= tp:
                        pnl = (entry - tp) / entry
                        closed = True

                if closed:
                    exit_price = sl if ((direction == "BUY" and price <= sl) or
                                        (direction == "SELL" and price >= sl)) else tp
                    pnl_net = pnl - round_trip_cost
                    trades.append(BacktestTrade(
                        direction=direction, entry_price=entry,
                        exit_price=exit_price,
                        entry_idx=entry_i, exit_idx=idx,
                        stop_loss=sl, take_profit=tp,
                        strategy="AI+ADX",
                        pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
                        cost_pct=round_trip_cost * 100,
                    ))
                    equity *= (1 + pnl_net)
                    position = None

            # Open new position
            if position is None:
                adx = adx_series.iloc[idx]
                if pd.isna(adx) or adx < STRATEGY.REGIME_ADX_THRESHOLD:
                    equity_curve.append(equity)
                    continue

                signal_val, confidence, _ = predictor.predict(features.iloc[:idx + 1])

                if signal_val != 0 and confidence >= AI.CONFIDENCE_THRESHOLD:
                    if signal_val == 1:
                        sl = price - (atr * STRATEGY.SCALP_ATR_SL)
                        tp = price + (atr * STRATEGY.SCALP_ATR_TP)
                    else:
                        sl = price + (atr * STRATEGY.SCALP_ATR_SL)
                        tp = price - (atr * STRATEGY.SCALP_ATR_TP)

                    risk = abs(price - sl) / price
                    reward = abs(tp - price) / price

                    if reward > round_trip_cost and (reward / risk) >= 1.5:
                        direction = "BUY" if signal_val == 1 else "SELL"
                        position = (direction, price, sl, tp, idx)

            equity_curve.append(equity)

        # Close remaining position
        if position is not None:
            direction, entry, sl, tp, entry_i = position
            last_price = close.iloc[test_indices[-1]]
            if direction == "BUY":
                pnl = (last_price - entry) / entry
            else:
                pnl = (entry - last_price) / entry
            spread_cost = (self.instrument.SPREAD_PIPS * self.instrument.PIP_VALUE) / last_price
            round_trip_cost = spread_cost * 2 * (1 + COSTS.SLIPPAGE_MULTIPLIER)
            pnl_net = pnl - round_trip_cost
            trades.append(BacktestTrade(
                direction=direction, entry_price=entry,
                exit_price=last_price,
                entry_idx=entry_i, exit_idx=test_indices[-1],
                stop_loss=sl, take_profit=tp,
                strategy="AI+ADX (EOD close)",
                pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
                cost_pct=round_trip_cost * 100,
            ))

        return trades, np.array(equity_curve)

    def run(self, df: pd.DataFrame, model_path: str = None) -> dict:
        """
        Run backtest using 50/25/25 split.
        Train on first 50%.
        Reports TWO test periods:
          - 'optim' = middle 25% (same data optimizer used → should match optimizer results)
          - 'oos'   = last 25%   (true out-of-sample → real performance estimate)
        """
        features = FeatureEngineer.create_features(df)
        labels = FeatureEngineer.create_labels(
            df, horizon=AI.PREDICTION_HORIZON,
            threshold=self.instrument.PRICE_CHANGE_THRESHOLD,
        )

        valid_mask = features.notna().all(axis=1) & labels.notna()
        valid_indices = valid_mask[valid_mask].index.tolist()

        if len(valid_indices) < 200:
            return {"error": "Not enough data", "trades": [], "metrics": {}}

        # 50/25/25 split
        n = len(valid_indices)
        train_end = int(n * 0.50)
        optim_end = int(n * 0.75)

        train_idx = valid_indices[:train_end]
        optim_idx = valid_indices[train_end:optim_end]   # Middle 25% (optimizer used this)
        oos_idx = valid_indices[optim_end:]               # Last 25% (true out-of-sample)

        # Train model on first 50%
        X_train = features.loc[train_idx]
        y_train = labels.loc[train_idx].astype(int)

        predictor = GoldPredictor(model_path=model_path or self.instrument.MODEL_PATH)
        metrics_train = predictor.train(X_train, y_train)

        print(f"  Split: train={len(train_idx)}, optim={len(optim_idx)}, oos={len(oos_idx)}")
        print(f"  Model CV accuracy: {metrics_train.get('cv_accuracy_mean', 0)*100:.1f}%")

        # Test 1: Optimizer period (middle 25%) - should match optimizer results
        print(f"\n  --- Optimizer period (middle 25%) ---")
        trades_opt, equity_opt = self._simulate_trades(predictor, features, df, optim_idx)
        metrics_opt = BacktestMetrics.compute_all(trades_opt, equity_opt)

        # Test 2: Out-of-sample (last 25%) - true performance
        print(f"  --- Out-of-sample period (last 25%) ---")
        trades_oos, equity_oos = self._simulate_trades(predictor, features, df, oos_idx)
        metrics_oos = BacktestMetrics.compute_all(trades_oos, equity_oos)

        return {
            "trades": trades_oos,           # Primary result = OOS
            "equity_curve": equity_oos,
            "metrics": metrics_oos,
            # Also include optimizer-period for comparison
            "optim_trades": trades_opt,
            "optim_equity_curve": equity_opt,
            "optim_metrics": metrics_opt,
        }

    def run_walk_forward(self, df: pd.DataFrame, retrain_every: int = 500,
                         min_train: int = 2000) -> dict:
        """
        Walk-forward backtest: re-train model every `retrain_every` candles.
        This simulates real production: model is always trained on past data only.

        Process:
          1. Start with min_train candles for initial training
          2. Test on next retrain_every candles
          3. Expand training window, retrain, repeat
        """
        features = FeatureEngineer.create_features(df)
        labels = FeatureEngineer.create_labels(
            df, horizon=AI.PREDICTION_HORIZON,
            threshold=self.instrument.PRICE_CHANGE_THRESHOLD,
        )

        valid_mask = features.notna().all(axis=1) & labels.notna()
        valid_indices = valid_mask[valid_mask].index.tolist()

        if len(valid_indices) < min_train + retrain_every:
            return {"error": "Not enough data for walk-forward", "trades": [], "metrics": {}}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_series = Indicators.atr(high, low, close, 14)
        adx_series = Indicators.adx(high, low, close, 14)

        all_trades = []
        equity = 100000.0
        equity_curve = [equity]
        position = None
        n_retrains = 0

        # Walk-forward windows
        window_start = 0
        test_start = min_train

        while test_start < len(valid_indices):
            test_end = min(test_start + retrain_every, len(valid_indices))

            train_idx = valid_indices[window_start:test_start]
            test_idx = valid_indices[test_start:test_end]

            # Train model on expanding window
            X_train = features.loc[train_idx]
            y_train = labels.loc[train_idx].astype(int)

            predictor = GoldPredictor(model_path=self.instrument.MODEL_PATH)
            predictor.train(X_train, y_train)
            n_retrains += 1

            train_price = close.iloc[train_idx[-1]]
            test_price_start = close.iloc[test_idx[0]]
            test_price_end = close.iloc[test_idx[-1]]
            print(f"  Window {n_retrains}: train={len(train_idx)} test={len(test_idx)} "
                  f"price={test_price_start:.0f}-{test_price_end:.0f}")

            # Simulate trades on test window
            for idx in test_idx:
                price = close.iloc[idx]
                atr = atr_series.iloc[idx]

                if pd.isna(atr) or atr == 0:
                    equity_curve.append(equity)
                    continue

                spread_cost = (self.instrument.SPREAD_PIPS * self.instrument.PIP_VALUE) / price
                round_trip_cost = spread_cost * 2 * (1 + COSTS.SLIPPAGE_MULTIPLIER)

                # Check SL/TP for open position
                if position is not None:
                    direction, entry, sl, tp, entry_i = position
                    closed = False

                    if direction == "BUY":
                        if price <= sl:
                            pnl = (sl - entry) / entry
                            closed = True
                        elif price >= tp:
                            pnl = (tp - entry) / entry
                            closed = True
                    elif direction == "SELL":
                        if price >= sl:
                            pnl = (entry - sl) / entry
                            closed = True
                        elif price <= tp:
                            pnl = (entry - tp) / entry
                            closed = True

                    if closed:
                        exit_price = sl if ((direction == "BUY" and price <= sl) or
                                            (direction == "SELL" and price >= sl)) else tp
                        pnl_net = pnl - round_trip_cost
                        all_trades.append(BacktestTrade(
                            direction=direction, entry_price=entry,
                            exit_price=exit_price,
                            entry_idx=entry_i, exit_idx=idx,
                            stop_loss=sl, take_profit=tp,
                            strategy=f"AI+ADX (wf{n_retrains})",
                            pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
                            cost_pct=round_trip_cost * 100,
                        ))
                        equity *= (1 + pnl_net)
                        position = None

                # Open new position
                if position is None:
                    adx = adx_series.iloc[idx]
                    if pd.isna(adx) or adx < STRATEGY.REGIME_ADX_THRESHOLD:
                        equity_curve.append(equity)
                        continue

                    signal_val, confidence, _ = predictor.predict(features.iloc[:idx + 1])

                    if signal_val != 0 and confidence >= AI.CONFIDENCE_THRESHOLD:
                        if signal_val == 1:
                            sl = price - (atr * STRATEGY.SCALP_ATR_SL)
                            tp = price + (atr * STRATEGY.SCALP_ATR_TP)
                        else:
                            sl = price + (atr * STRATEGY.SCALP_ATR_SL)
                            tp = price - (atr * STRATEGY.SCALP_ATR_TP)

                        risk = abs(price - sl) / price
                        reward = abs(tp - price) / price

                        if reward > round_trip_cost and (reward / risk) >= 1.5:
                            direction = "BUY" if signal_val == 1 else "SELL"
                            position = (direction, price, sl, tp, idx)

                equity_curve.append(equity)

            # Advance window
            test_start = test_end

        # Close remaining position
        if position is not None:
            direction, entry, sl, tp, entry_i = position
            last_price = close.iloc[valid_indices[-1]]
            if direction == "BUY":
                pnl = (last_price - entry) / entry
            else:
                pnl = (entry - last_price) / entry
            spread_cost = (self.instrument.SPREAD_PIPS * self.instrument.PIP_VALUE) / last_price
            round_trip_cost = spread_cost * 2 * (1 + COSTS.SLIPPAGE_MULTIPLIER)
            pnl_net = pnl - round_trip_cost
            all_trades.append(BacktestTrade(
                direction=direction, entry_price=entry,
                exit_price=last_price,
                entry_idx=entry_i, exit_idx=valid_indices[-1],
                stop_loss=sl, take_profit=tp,
                strategy="AI+ADX (final close)",
                pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
                cost_pct=round_trip_cost * 100,
            ))

        equity_arr = np.array(equity_curve)
        metrics = BacktestMetrics.compute_all(all_trades, equity_arr)

        print(f"\n  Walk-forward complete: {n_retrains} retrains, "
              f"{len(all_trades)} trades, "
              f"test period = {len(valid_indices) - min_train} candles")

        return {
            "trades": all_trades,
            "equity_curve": equity_arr,
            "metrics": metrics,
            "n_retrains": n_retrains,
        }

    @staticmethod
    def _print_section(metrics: dict, trades: list, label: str):
        """Print one section of metrics."""
        if not trades:
            print(f"    No trades in {label} period.")
            return

        print(f"\n  {label}:")
        print(f"    Total Return:    {metrics.get('total_return_pct', 0):+.2f}%")
        print(f"    Sharpe (candle): {metrics.get('sharpe', 0):.2f}")
        print(f"    Sharpe (trade):  {metrics.get('trade_sharpe', 0):.2f}")
        print(f"    Sortino Ratio:   {metrics.get('sortino', 0):.2f}")
        print(f"    Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}")
        print(f"    Calmar Ratio:    {metrics.get('calmar', 0):.2f}")
        print(f"    Trades:          {metrics.get('total_trades', 0)}")
        print(f"    Win Rate:        {metrics.get('win_rate', 0):.1%}")
        print(f"    Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
        print(f"    Avg Trade P&L:   {metrics.get('avg_trade_pnl', 0):+.3f}%")
        print(f"    Avg Win:         {metrics.get('avg_win', 0):+.3f}%")
        print(f"    Avg Loss:        {metrics.get('avg_loss', 0):+.3f}%")
        print(f"    Total Costs:     {metrics.get('total_costs_pct', 0):.2f}%")

    @staticmethod
    def print_report(result: dict, instrument_name: str = ""):
        metrics_oos = result.get("metrics", {})
        trades_oos = result.get("trades", [])
        metrics_opt = result.get("optim_metrics", {})
        trades_opt = result.get("optim_trades", [])

        print(f"\n{'=' * 60}")
        print(f"  BACKTEST REPORT: {instrument_name}")
        print(f"{'=' * 60}")

        # Optimizer-period results (should match optimizer output)
        Backtester._print_section(metrics_opt, trades_opt,
                                  "OPTIMIZER PERIOD (middle 25% - in-sample)")

        # Out-of-sample results (true performance)
        Backtester._print_section(metrics_oos, trades_oos,
                                  "OUT-OF-SAMPLE (last 25% - real test)")

        # Comparison
        if trades_opt and trades_oos:
            opt_ret = metrics_opt.get('total_return_pct', 0)
            oos_ret = metrics_oos.get('total_return_pct', 0)
            opt_sharpe = metrics_opt.get('sharpe', 0)
            oos_sharpe = metrics_oos.get('sharpe', 0)

            print(f"\n  COMPARISON:")
            print(f"    Return:  optim={opt_ret:+.2f}%  vs  OOS={oos_ret:+.2f}%")
            print(f"    Sharpe:  optim={opt_sharpe:.2f}  vs  OOS={oos_sharpe:.2f}")

            if opt_ret > 0 and oos_ret > 0:
                retention = (oos_ret / opt_ret) * 100
                print(f"    Performance retention: {retention:.0f}%")
                if retention >= 50:
                    print(f"    -> GOOD: Strategy generalizes well")
                else:
                    print(f"    -> WARNING: Significant performance decay OOS")
            elif opt_ret > 0 and oos_ret <= 0:
                print(f"    -> OVERFITTING: Positive in-sample, negative OOS")
            elif opt_ret <= 0:
                print(f"    -> NO EDGE: Even in-sample is not profitable")

        # Show last trades from OOS
        if trades_oos:
            print(f"\n  Last 10 OOS trades:")
            for t in trades_oos[-10:]:
                emoji = "W" if t.pnl_net_pct > 0 else "L"
                print(f"    [{emoji}] {t.direction:4s} @ {t.entry_price:.2f} -> "
                      f"{t.exit_price:.2f} | P&L: {t.pnl_net_pct:+.3f}% "
                      f"(cost: {t.cost_pct:.3f}%)")

        print(f"{'=' * 60}")
