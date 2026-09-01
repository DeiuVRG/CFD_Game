"""
Backtest framework with realistic cost modeling and risk-adjusted metrics.

v3 execution model (this is the single source of truth for trade simulation):
  1. The signal is computed on the CLOSE of candle N -> the entry is executed
     at the OPEN of candle N+1 (no same-candle close fills).
  2. SL/TP are checked against the HIGH/LOW of every candle (wicks count),
     never against the close alone.
  3. If both SL and TP are touched within the same candle, the conservative
     assumption is that SL was hit first.
  4. A gap through SL executes at the candle OPEN (worse price than SL);
     TP always executes exactly at the TP level (never better).
  5. The forced close at the end of the period updates equity (historical bug:
     the trade was recorded without an equity update).
  6. run_walk_forward() uses the SAME engine (not a copy): the open position,
     the pending signal and the equity are carried across retrain windows.

Two modes:
  1. Fixed split (50/25/25) - train on first 50%, report the optimizer window
     (middle 25%) and the true out-of-sample window (last 25%) separately.
  2. Walk-forward - re-trains the model every N candles (production sim).

A "legacy" execution mode reproduces the pre-v3 (close-based) simulation,
including its known bugs. It exists ONLY to quantify how much the old bugs
inflated results (see RESULTS.md); nothing in the decision path uses it.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional

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
    exit_reason: str = ""      # SL / GAP_SL / TP / EOD


@dataclass
class ExecutionState:
    """State carried across walk-forward windows (same engine, no copies)."""
    equity: float = 100000.0
    # Open position: (direction, entry_price, sl, tp, entry_idx)
    position: Optional[tuple] = None
    # Signal generated on the close of a candle, waiting for the next open:
    # (direction, atr_at_signal, signal_idx)
    pending: Optional[tuple] = None


class BacktestMetrics:
    """Risk-adjusted performance metrics."""

    # All ratios are annualized with `periods_per_year` = number of candles
    # in a year for the instrument's timeframe (InstrumentConfig.
    # candles_per_year()). The old hardcoded 252 assumed daily candles.
    DEFAULT_PERIODS_PER_YEAR = 252

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                     periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess = returns - (risk_free_rate / periods_per_year)
        return float(np.mean(excess) / np.std(returns) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                      periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> float:
        excess = returns - (risk_free_rate / periods_per_year)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float('inf')
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        if len(equity_curve) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return float(np.min(drawdown))

    @staticmethod
    def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray,
                     periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> float:
        annual_return = float(np.mean(returns) * periods_per_year)
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
    def compute_all(trades: list, equity_curve: np.ndarray,
                    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> dict:
        """`periods_per_year` must be the number of candles per year of the
        equity curve's timeframe (one point per candle)."""
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

        # Trade-based Sharpe (more meaningful for infrequent trading):
        # trades per year = trades / (years covered by the equity curve)
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            trades_per_year = (len(trade_returns) * periods_per_year
                               / max(len(equity_curve), 1))
            trade_sharpe = float(np.mean(trade_returns) / np.std(trade_returns)
                                 * np.sqrt(max(trades_per_year, 1)))
        else:
            trade_sharpe = 0.0

        return {
            "total_trades": len(trades),
            "win_rate": BacktestMetrics.win_rate(trades),
            "profit_factor": BacktestMetrics.profit_factor(trades),
            "sharpe": BacktestMetrics.sharpe_ratio(returns, periods_per_year=periods_per_year),
            "trade_sharpe": trade_sharpe,
            "sortino": BacktestMetrics.sortino_ratio(returns, periods_per_year=periods_per_year),
            "max_drawdown": BacktestMetrics.max_drawdown(equity_curve),
            "calmar": BacktestMetrics.calmar_ratio(returns, equity_curve, periods_per_year),
            "periods_per_year": float(periods_per_year),
            "total_return_pct": float((equity_curve[-1] / equity_curve[0] - 1) * 100),
            "avg_trade_pnl": float(np.mean(net_pnls)),
            "avg_win": float(np.mean([p for p in net_pnls if p > 0])) if any(p > 0 for p in net_pnls) else 0,
            "avg_loss": float(np.mean([p for p in net_pnls if p < 0])) if any(p < 0 for p in net_pnls) else 0,
            "total_costs_pct": total_costs,
        }


class Backtester:
    """Runs the v3 execution model over model predictions.

    The same `_simulate_trades` engine is used by:
      - run()               fixed 50/25/25 split (optim window + OOS window)
      - run_walk_forward()  expanding-window retraining with carried state
      - the optimizer       via a precomputed predict_fn
    """

    def __init__(self, instrument: InstrumentConfig):
        self.instrument = instrument

    # ------------------------------------------------------------------
    # Core v3 execution engine
    # ------------------------------------------------------------------

    def _round_trip_cost(self, price: float) -> float:
        return COSTS.round_trip_cost_pct(self.instrument, price)

    def _close_position(self, state: ExecutionState, exit_price: float,
                        exit_idx: int, reason: str, trades: list,
                        strategy_label: str):
        """Close the open position, book the trade and update equity."""
        direction, entry, sl, tp, entry_i = state.position
        if direction == "BUY":
            pnl = (exit_price - entry) / entry
        else:
            pnl = (entry - exit_price) / entry
        cost = self._round_trip_cost(entry)
        pnl_net = pnl - cost
        trades.append(BacktestTrade(
            direction=direction, entry_price=entry, exit_price=exit_price,
            entry_idx=entry_i, exit_idx=exit_idx,
            stop_loss=sl, take_profit=tp,
            strategy=strategy_label,
            pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
            cost_pct=cost * 100,
            exit_reason=reason,
        ))
        state.equity *= (1 + pnl_net)
        state.position = None

    def _simulate_trades(
        self,
        predict_fn: Callable[[int], tuple],
        df: pd.DataFrame,
        test_indices: list,
        state: ExecutionState = None,
        close_at_end: bool = True,
        strategy_label: str = "AI+ADX",
        sl_atr: float = None,
        tp_atr: float = None,
        conf_threshold: float = None,
        adx_min: float = None,
        min_rr: float = None,
        atr_series: pd.Series = None,
        adx_series: pd.Series = None,
    ) -> tuple:
        """v3 execution model core loop (see module docstring for the rules).

        predict_fn(idx) -> (signal_val, confidence) evaluated on the CLOSE of
        candle `idx`. Entries happen on the OPEN of the next processed candle.

        Returns (trades, equity_curve_segment, state). The equity curve
        segment has one point per processed candle (realized equity).
        """
        inst = self.instrument
        sl_atr = sl_atr if sl_atr is not None else inst.sl_atr()
        tp_atr = tp_atr if tp_atr is not None else inst.tp_atr()
        conf_threshold = (conf_threshold if conf_threshold is not None
                          else inst.confidence_threshold())
        adx_min = adx_min if adx_min is not None else inst.adx_min()
        min_rr = min_rr if min_rr is not None else inst.min_rr()

        if atr_series is None:
            atr_series = Indicators.atr(df["high"], df["low"], df["close"], 14)
        if adx_series is None:
            adx_series = Indicators.adx(df["high"], df["low"], df["close"], 14)

        # Positional numpy arrays for the hot loop (df has a RangeIndex)
        open_arr = df["open"].to_numpy()
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()
        close_arr = df["close"].to_numpy()
        atr_arr = atr_series.to_numpy()
        adx_arr = adx_series.to_numpy()

        if state is None:
            state = ExecutionState()

        trades: list = []
        equity_curve: list = []

        for idx in test_indices:
            o = open_arr[idx]
            h = high_arr[idx]
            l = low_arr[idx]
            c = close_arr[idx]

            # 1) Pending signal from the previous candle close -> enter at
            #    THIS candle's open. SL/TP anchored to the actual entry price
            #    using the ATR captured at signal time.
            if state.pending is not None:
                if state.position is None:
                    direction, atr_sig, sig_idx = state.pending
                    entry = o
                    if direction == "BUY":
                        sl = entry - atr_sig * sl_atr
                        tp = entry + atr_sig * tp_atr
                    else:
                        sl = entry + atr_sig * sl_atr
                        tp = entry - atr_sig * tp_atr
                    state.position = (direction, entry, sl, tp, idx)
                state.pending = None

            # 2) Exit checks on this candle: wicks count, SL has priority in
            #    the same candle, gaps through SL fill at the open, TP fills
            #    exactly at the TP level (never better).
            if state.position is not None:
                direction, entry, sl, tp, entry_i = state.position
                exit_price = None
                reason = ""
                if direction == "BUY":
                    if o <= sl:
                        exit_price, reason = o, "GAP_SL"
                    elif l <= sl:
                        exit_price, reason = sl, "SL"
                    elif h >= tp:
                        exit_price, reason = tp, "TP"
                else:  # SELL
                    if o >= sl:
                        exit_price, reason = o, "GAP_SL"
                    elif h >= sl:
                        exit_price, reason = sl, "SL"
                    elif l <= tp:
                        exit_price, reason = tp, "TP"

                if exit_price is not None:
                    self._close_position(state, exit_price, idx, reason,
                                         trades, strategy_label)

            # 3) Signal evaluation on the CLOSE of this candle (only when
            #    flat). The entry, if any, happens at the NEXT candle open.
            if state.position is None and state.pending is None:
                atr = atr_arr[idx]
                adx = adx_arr[idx]
                if not (np.isnan(atr) or atr == 0 or np.isnan(adx)) and adx >= adx_min:
                    signal_val, confidence = predict_fn(idx)
                    if signal_val != 0 and confidence >= conf_threshold:
                        risk = (atr * sl_atr) / c
                        reward = (atr * tp_atr) / c
                        cost = self._round_trip_cost(c)
                        if risk > 0 and reward > cost and (reward / risk) >= min_rr:
                            direction = "BUY" if signal_val == 1 else "SELL"
                            state.pending = (direction, atr, idx)

            equity_curve.append(state.equity)

        # 4) Forced close at the end of the period: exits at the last close
        #    and UPDATES EQUITY (the historical bug recorded the trade without
        #    the equity update). A pending-but-unfilled signal is dropped.
        if close_at_end:
            state.pending = None
            if state.position is not None:
                last_idx = test_indices[-1]
                self._close_position(
                    state, close_arr[last_idx], last_idx, "EOD",
                    trades, f"{strategy_label} (EOD close)",
                )
                if equity_curve:
                    equity_curve[-1] = state.equity

        return trades, equity_curve, state

    def _make_predict_fn(self, predictor, features: pd.DataFrame) -> Callable[[int], tuple]:
        def predict_fn(idx: int) -> tuple:
            signal_val, confidence, _ = predictor.predict(features.iloc[:idx + 1])
            return signal_val, confidence
        return predict_fn

    # ------------------------------------------------------------------
    # Legacy (pre-v3) execution - ONLY for the old-vs-new comparison
    # ------------------------------------------------------------------

    def _simulate_trades_legacy(self, predictor, features, df, test_indices) -> tuple:
        """Pre-v3 close-based simulation, bugs included, for RESULTS.md only:
        entry on the SIGNAL candle close, SL/TP checked on close (wicks
        ignored, gaps fill at the exact SL level), final forced close recorded
        WITHOUT an equity update. Do not use for decisions."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_series = Indicators.atr(high, low, close, 14)
        adx_series = Indicators.adx(high, low, close, 14)

        inst = self.instrument
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

            round_trip_cost = self._round_trip_cost(price)

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
                else:
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
                        strategy="AI+ADX (legacy)",
                        pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
                        cost_pct=round_trip_cost * 100,
                        exit_reason="SL" if exit_price == sl else "TP",
                    ))
                    equity *= (1 + pnl_net)
                    position = None

            if position is None:
                adx = adx_series.iloc[idx]
                if pd.isna(adx) or adx < inst.adx_min():
                    equity_curve.append(equity)
                    continue

                signal_val, confidence, _ = predictor.predict(features.iloc[:idx + 1])

                if signal_val != 0 and confidence >= inst.confidence_threshold():
                    if signal_val == 1:
                        sl = price - (atr * inst.sl_atr())
                        tp = price + (atr * inst.tp_atr())
                    else:
                        sl = price + (atr * inst.sl_atr())
                        tp = price - (atr * inst.tp_atr())

                    risk = abs(price - sl) / price
                    reward = abs(tp - price) / price

                    if reward > round_trip_cost and (reward / risk) >= inst.min_rr():
                        direction = "BUY" if signal_val == 1 else "SELL"
                        position = (direction, price, sl, tp, idx)

            equity_curve.append(equity)

        # Historical bug preserved on purpose: the final forced close is
        # recorded as a trade but equity is NOT updated.
        if position is not None:
            direction, entry, sl, tp, entry_i = position
            last_price = close.iloc[test_indices[-1]]
            if direction == "BUY":
                pnl = (last_price - entry) / entry
            else:
                pnl = (entry - last_price) / entry
            round_trip_cost = self._round_trip_cost(last_price)
            pnl_net = pnl - round_trip_cost
            trades.append(BacktestTrade(
                direction=direction, entry_price=entry,
                exit_price=last_price,
                entry_idx=entry_i, exit_idx=test_indices[-1],
                stop_loss=sl, take_profit=tp,
                strategy="AI+ADX (legacy EOD close)",
                pnl_pct=pnl * 100, pnl_net_pct=pnl_net * 100,
                cost_pct=round_trip_cost * 100,
                exit_reason="EOD",
            ))

        return trades, np.array(equity_curve)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, model_path: str = None,
            execution: str = "v3", threshold: float = None) -> dict:
        """
        Run backtest using a 50/25/25 split.
        Train on the first 50%.
        Reports TWO test periods:
          - 'optim' = middle 25% (same data the optimizer used)
          - 'oos'   = last 25%   (true out-of-sample -> the decision window)

        execution: "v3" (default) or "legacy" (bug-for-bug pre-v3 sim, used
        only for the old-vs-new comparison in RESULTS.md).
        """
        threshold = threshold or self.instrument.PRICE_CHANGE_THRESHOLD
        features = FeatureEngineer.create_features(df)
        labels = FeatureEngineer.create_labels(
            df, horizon=AI.PREDICTION_HORIZON,
            threshold=threshold,
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

        if execution == "legacy":
            print(f"\n  --- Optimizer period (middle 25%) [LEGACY EXECUTION] ---")
            trades_opt, equity_opt = self._simulate_trades_legacy(
                predictor, features, df, optim_idx)
            metrics_opt = BacktestMetrics.compute_all(trades_opt, np.array(equity_opt), self.instrument.candles_per_year())

            print(f"  --- Out-of-sample period (last 25%) [LEGACY EXECUTION] ---")
            trades_oos, equity_oos = self._simulate_trades_legacy(
                predictor, features, df, oos_idx)
            metrics_oos = BacktestMetrics.compute_all(trades_oos, np.array(equity_oos), self.instrument.candles_per_year())
        else:
            predict_fn = self._make_predict_fn(predictor, features)

            print(f"\n  --- Optimizer period (middle 25%) ---")
            trades_opt, curve_opt, _ = self._simulate_trades(
                predict_fn, df, optim_idx, state=ExecutionState())
            equity_opt = np.array([100000.0] + curve_opt)
            metrics_opt = BacktestMetrics.compute_all(trades_opt, equity_opt, self.instrument.candles_per_year())

            print(f"  --- Out-of-sample period (last 25%) ---")
            trades_oos, curve_oos, _ = self._simulate_trades(
                predict_fn, df, oos_idx, state=ExecutionState())
            equity_oos = np.array([100000.0] + curve_oos)
            metrics_oos = BacktestMetrics.compute_all(trades_oos, equity_oos, self.instrument.candles_per_year())

        return {
            "trades": trades_oos,           # Primary result = OOS
            "equity_curve": equity_oos,
            "metrics": metrics_oos,
            # Also include optimizer-period for comparison
            "optim_trades": trades_opt,
            "optim_equity_curve": equity_opt,
            "optim_metrics": metrics_opt,
            "execution": execution,
        }

    def run_walk_forward(self, df: pd.DataFrame, retrain_every: int = 500,
                         min_train: int = 2000, threshold: float = None) -> dict:
        """
        Walk-forward backtest: re-train model every `retrain_every` candles.
        This simulates real production: model is always trained on past data only.

        Uses the SAME v3 execution engine as run() - the open position, the
        pending signal and the equity are carried across retrain windows in a
        shared ExecutionState.
        """
        threshold = threshold or self.instrument.PRICE_CHANGE_THRESHOLD
        features = FeatureEngineer.create_features(df)
        labels = FeatureEngineer.create_labels(
            df, horizon=AI.PREDICTION_HORIZON,
            threshold=threshold,
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
        state = ExecutionState()
        equity_curve = [state.equity]
        n_retrains = 0

        window_start = 0
        test_start = min_train

        while test_start < len(valid_indices):
            test_end = min(test_start + retrain_every, len(valid_indices))

            train_idx = valid_indices[window_start:test_start]
            test_idx = valid_indices[test_start:test_end]

            X_train = features.loc[train_idx]
            y_train = labels.loc[train_idx].astype(int)

            predictor = GoldPredictor(model_path=self.instrument.MODEL_PATH)
            predictor.train(X_train, y_train)
            n_retrains += 1

            test_price_start = close.iloc[test_idx[0]]
            test_price_end = close.iloc[test_idx[-1]]
            print(f"  Window {n_retrains}: train={len(train_idx)} test={len(test_idx)} "
                  f"price={test_price_start:.0f}-{test_price_end:.0f}")

            predict_fn = self._make_predict_fn(predictor, features)

            is_last_window = test_end >= len(valid_indices)
            trades, curve, state = self._simulate_trades(
                predict_fn, df, test_idx,
                state=state,
                close_at_end=is_last_window,
                strategy_label=f"AI+ADX (wf{n_retrains})",
                atr_series=atr_series,
                adx_series=adx_series,
            )
            all_trades.extend(trades)
            equity_curve.extend(curve)

            test_start = test_end

        equity_arr = np.array(equity_curve)
        metrics = BacktestMetrics.compute_all(all_trades, equity_arr, self.instrument.candles_per_year())

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
        if result.get("execution") == "legacy":
            print(f"  (LEGACY execution - comparison only, not for decisions)")
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
                      f"{t.exit_price:.2f} ({t.exit_reason}) | P&L: {t.pnl_net_pct:+.3f}% "
                      f"(cost: {t.cost_pct:.3f}%)")

        print(f"{'=' * 60}")
