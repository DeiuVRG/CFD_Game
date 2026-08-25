"""
Parameter optimizer - grid search over SL/TP/confidence/ADX/label-threshold
combinations, executed through the SAME v3 execution engine as the backtester
(next-open entries, wick-based SL/TP, SL-first, gap handling).

Split discipline: the model is trained on the first 50% of the data and every
parameter combo is scored on the middle 25% (optimization window). The last
25% is NEVER touched here - it is reserved for the one-shot out-of-sample run
in the backtester.
"""
import numpy as np
import pandas as pd
import logging
from itertools import product
from dataclasses import dataclass

from ai.backtester import Backtester, BacktestMetrics, ExecutionState
from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import InstrumentConfig, AI

logger = logging.getLogger(__name__)


@dataclass
class OptimResult:
    threshold: float   # Label threshold (PRICE_CHANGE_THRESHOLD) used
    sl_atr: float
    tp_atr: float
    confidence_threshold: float
    min_rr: float
    adx_filter: float  # Minimum ADX to trade
    total_return: float
    sharpe: float
    trade_sharpe: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    avg_pnl: float

    @property
    def score(self) -> float:
        """Composite ranking score (consistency x profitability)."""
        pf = min(self.profit_factor, 10.0)  # Cap inf/huge PF from tiny samples
        return self.sharpe * pf


# Parameter grid (shared by all instruments)
SL_ATR_RANGE = [1.0, 1.5, 2.0, 2.5]
TP_ATR_RANGE = [1.5, 2.0, 2.5, 3.0, 4.0]
CONF_RANGE = [0.45, 0.50, 0.55, 0.60]
MIN_RR_RANGE = [1.0, 1.5, 2.0]
ADX_FILTER_RANGE = [0, 15, 20, 25]

# Minimum trades in the optimization window for a combo to be considered a
# reliable candidate (defined BEFORE looking at any OOS data).
MIN_TRADES_FOR_SELECTION = 15


def optimize_instrument(instrument: InstrumentConfig, df: pd.DataFrame) -> list:
    """
    Grid search over label threshold + execution parameters.
    Returns sorted list of OptimResult (best first).
    """
    print(f"\n{'='*60}")
    print(f"  OPTIMIZER: {instrument.SYMBOL_DISPLAY}")
    print(f"{'='*60}")

    features = FeatureEngineer.create_features(df)
    bt = Backtester(instrument)

    thresholds = instrument.threshold_grid()
    print(f"  Label threshold grid: {thresholds}")

    all_results: list = []

    for threshold in thresholds:
        labels = FeatureEngineer.create_labels(
            df, horizon=AI.PREDICTION_HORIZON, threshold=threshold,
        )

        valid_mask = features.notna().all(axis=1) & labels.notna()
        valid_indices = valid_mask[valid_mask].index.tolist()

        if len(valid_indices) < 300:
            print(f"  [thr={threshold}] Not enough data for optimization")
            continue

        # Split: train on first 50%, optimize on next 25%; last 25% untouched.
        n = len(valid_indices)
        train_end = int(n * 0.50)
        optim_end = int(n * 0.75)

        train_idx = valid_indices[:train_end]
        optim_idx = valid_indices[train_end:optim_end]

        X_train = features.loc[train_idx]
        y_train = labels.loc[train_idx].astype(int)

        label_counts = y_train.value_counts().to_dict()
        print(f"\n  [thr={threshold}] Training on {len(train_idx)} samples "
              f"(labels: {label_counts})...")
        predictor = GoldPredictor(model_path=instrument.MODEL_PATH)
        metrics = predictor.train(X_train, y_train)
        print(f"  [thr={threshold}] Model CV accuracy: "
              f"{metrics['cv_accuracy_mean']*100:.1f}%")

        # Pre-compute predictions once per threshold (model is fixed)
        print(f"  [thr={threshold}] Pre-computing predictions for "
              f"{len(optim_idx)} optimization samples...")
        predictions = {}
        for idx in optim_idx:
            signal_val, confidence, _ = predictor.predict(features.iloc[:idx + 1])
            predictions[idx] = (signal_val, confidence)

        def predict_fn(idx: int) -> tuple:
            return predictions.get(idx, (0, 0.0))

        combos = [
            (sl, tp, conf, rr, adx)
            for sl, tp, conf, rr, adx in product(
                SL_ATR_RANGE, TP_ATR_RANGE, CONF_RANGE,
                MIN_RR_RANGE, ADX_FILTER_RANGE)
            if tp > sl and (tp / sl) >= rr
        ]
        print(f"  [thr={threshold}] Testing {len(combos)} parameter combinations "
              f"through the v3 execution engine...")

        for sl_atr, tp_atr, conf_thresh, min_rr, adx_min in combos:
            trades, curve, _ = bt._simulate_trades(
                predict_fn, df, optim_idx,
                state=ExecutionState(),
                sl_atr=sl_atr, tp_atr=tp_atr,
                conf_threshold=conf_thresh,
                adx_min=adx_min, min_rr=min_rr,
            )

            if len(trades) < 5:
                continue

            equity_curve = np.array([100000.0] + curve)
            m = BacktestMetrics.compute_all(trades, equity_curve)

            all_results.append(OptimResult(
                threshold=threshold,
                sl_atr=sl_atr, tp_atr=tp_atr,
                confidence_threshold=conf_thresh, min_rr=min_rr,
                adx_filter=adx_min,
                total_return=m["total_return_pct"],
                sharpe=m["sharpe"],
                trade_sharpe=m["trade_sharpe"],
                win_rate=m["win_rate"],
                profit_factor=m["profit_factor"],
                max_drawdown=m["max_drawdown"],
                total_trades=m["total_trades"],
                avg_pnl=m["avg_trade_pnl"],
            ))

    all_results.sort(key=lambda r: r.score, reverse=True)

    print(f"\n  Tested {len(all_results)} valid combinations")
    print(f"\n  TOP 10 PARAMETER COMBINATIONS:")
    print(f"  {'Thr':>6} {'SL':>5} {'TP':>5} {'Conf':>5} {'RR':>4} {'ADX':>4} | "
          f"{'Return':>8} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'MaxDD':>7} {'#':>4}")
    print(f"  {'-'*82}")

    for r in all_results[:10]:
        print(
            f"  {r.threshold:>6.3f} {r.sl_atr:>5.1f} {r.tp_atr:>5.1f} "
            f"{r.confidence_threshold:>5.2f} "
            f"{r.min_rr:>4.1f} {r.adx_filter:>4.0f} | "
            f"{r.total_return:>+7.2f}% {r.sharpe:>7.2f} "
            f"{r.win_rate:>5.1%} {r.profit_factor:>5.2f} "
            f"{r.max_drawdown:>7.2%} {r.total_trades:>4}"
        )

    best = select_best(all_results)
    if best:
        print(f"\n  BEST (>= {MIN_TRADES_FOR_SELECTION} trades): "
              f"thr={best.threshold} SL={best.sl_atr}xATR TP={best.tp_atr}xATR "
              f"Conf={best.confidence_threshold} ADX>{best.adx_filter} RR>={best.min_rr}")
        print(f"  Return={best.total_return:+.2f}% Sharpe={best.sharpe:.2f} "
              f"WR={best.win_rate:.1%} PF={best.profit_factor:.2f}")

    print(f"{'='*60}")
    return all_results


def select_best(results: list, min_trades: int = MIN_TRADES_FOR_SELECTION):
    """Best combo with a minimum sample size (avoids tiny-sample flukes).
    Falls back to the overall best if nothing reaches min_trades."""
    for r in results:
        if r.total_trades >= min_trades:
            return r
    return results[0] if results else None
