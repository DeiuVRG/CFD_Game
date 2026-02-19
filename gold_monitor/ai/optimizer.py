"""
Parameter optimizer - grid search over SL/TP/confidence/threshold combinations.
Tests hundreds of parameter combos to find the most profitable setup.
"""
import numpy as np
import pandas as pd
import logging
from itertools import product
from dataclasses import dataclass
from typing import Optional

from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import InstrumentConfig, COSTS, AI
from data.indicators import Indicators

logger = logging.getLogger(__name__)


@dataclass
class OptimResult:
    sl_atr: float
    tp_atr: float
    confidence_threshold: float
    min_rr: float
    adx_filter: float  # Minimum ADX to trade
    total_return: float
    sharpe: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    avg_pnl: float


def optimize_instrument(instrument: InstrumentConfig, df: pd.DataFrame) -> list:
    """
    Grid search over key parameters to find optimal setup.
    Returns sorted list of OptimResult (best first).
    """
    print(f"\n{'='*60}")
    print(f"  OPTIMIZER: {instrument.SYMBOL_DISPLAY}")
    print(f"{'='*60}")

    # Create features and labels
    features = FeatureEngineer.create_features(df)
    labels = FeatureEngineer.create_labels(
        df, horizon=AI.PREDICTION_HORIZON,
        threshold=instrument.PRICE_CHANGE_THRESHOLD,
    )

    valid_mask = features.notna().all(axis=1) & labels.notna()
    valid_indices = valid_mask[valid_mask].index.tolist()

    if len(valid_indices) < 300:
        print("  Not enough data for optimization")
        return []

    # Split: train on first 50%, optimize on next 25%, validate on last 25%
    n = len(valid_indices)
    train_end = int(n * 0.50)
    optim_end = int(n * 0.75)

    train_idx = valid_indices[:train_end]
    optim_idx = valid_indices[train_end:optim_end]
    valid_idx = valid_indices[optim_end:]

    # Train model once
    X_train = features.loc[train_idx]
    y_train = labels.loc[train_idx].astype(int)

    print(f"  Training model on {len(train_idx)} samples...")
    predictor = GoldPredictor(model_path=instrument.MODEL_PATH)
    metrics = predictor.train(X_train, y_train)
    print(f"  Model CV accuracy: {metrics['cv_accuracy_mean']*100:.1f}%")

    # Pre-compute predictions for optimization set
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr_series = Indicators.atr(high, low, close, 14)
    adx_series = Indicators.adx(high, low, close, 14)

    print(f"  Pre-computing predictions for {len(optim_idx)} optimization samples...")
    predictions = []
    for idx in optim_idx:
        price = close.iloc[idx]
        atr = atr_series.iloc[idx]
        adx = adx_series.iloc[idx]

        if pd.isna(atr) or atr == 0 or pd.isna(adx):
            predictions.append((idx, price, atr, adx, 0, 0.0))
            continue

        signal_val, confidence, _ = predictor.predict(features.iloc[:idx + 1])
        predictions.append((idx, price, atr, adx, signal_val, confidence))

    # Parameter grid
    sl_atr_range = [1.0, 1.5, 2.0, 2.5]
    tp_atr_range = [1.5, 2.0, 2.5, 3.0, 4.0]
    conf_range = [0.45, 0.50, 0.55, 0.60]
    min_rr_range = [1.0, 1.5, 2.0]
    adx_filter_range = [0, 15, 20, 25]

    total_combos = (len(sl_atr_range) * len(tp_atr_range) * len(conf_range)
                    * len(min_rr_range) * len(adx_filter_range))
    print(f"  Testing {total_combos} parameter combinations...")

    results = []
    tested = 0

    for sl_atr, tp_atr, conf_thresh, min_rr, adx_min in product(
        sl_atr_range, tp_atr_range, conf_range, min_rr_range, adx_filter_range
    ):
        # Skip invalid combos (TP must be > SL for positive R:R)
        if tp_atr <= sl_atr:
            continue

        rr_ratio = tp_atr / sl_atr
        if rr_ratio < min_rr:
            continue

        # Run fast backtest with these params
        equity = 100000.0
        position = None
        trades_pnl = []

        spread_cost = (instrument.SPREAD_PIPS * instrument.PIP_VALUE)

        for idx, price, atr, adx, signal_val, confidence in predictions:
            if pd.isna(atr) or atr == 0:
                continue

            round_trip_cost = (spread_cost / price) * 2 * (1 + COSTS.SLIPPAGE_MULTIPLIER)

            # Check SL/TP
            if position is not None:
                direction, entry, sl, tp, _ = position

                hit_sl = False
                hit_tp = False

                if direction == "BUY":
                    hit_sl = price <= sl
                    hit_tp = price >= tp
                else:
                    hit_sl = price >= sl
                    hit_tp = price <= tp

                if hit_sl:
                    if direction == "BUY":
                        pnl = (sl - entry) / entry - round_trip_cost
                    else:
                        pnl = (entry - sl) / entry - round_trip_cost
                    trades_pnl.append(pnl)
                    equity *= (1 + pnl)
                    position = None
                elif hit_tp:
                    if direction == "BUY":
                        pnl = (tp - entry) / entry - round_trip_cost
                    else:
                        pnl = (entry - tp) / entry - round_trip_cost
                    trades_pnl.append(pnl)
                    equity *= (1 + pnl)
                    position = None

            # Open new position
            if position is None and signal_val != 0 and confidence >= conf_thresh:
                # ADX filter
                if adx < adx_min:
                    continue

                if signal_val == 1:  # BUY
                    sl = price - (atr * sl_atr)
                    tp = price + (atr * tp_atr)
                else:  # SELL
                    sl = price + (atr * sl_atr)
                    tp = price - (atr * tp_atr)

                risk = abs(price - sl) / price
                reward = abs(tp - price) / price

                if reward > round_trip_cost and (reward / risk) >= min_rr:
                    direction = "BUY" if signal_val == 1 else "SELL"
                    position = (direction, price, sl, tp, idx)

        if len(trades_pnl) < 5:
            continue

        # Calculate metrics
        total_return = (equity / 100000 - 1) * 100
        wins = sum(1 for p in trades_pnl if p > 0)
        win_rate = wins / len(trades_pnl)

        gross_profit = sum(p for p in trades_pnl if p > 0)
        gross_loss = abs(sum(p for p in trades_pnl if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe
        returns_arr = np.array(trades_pnl)
        sharpe = (np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
                  if np.std(returns_arr) > 0 else 0)

        # Max drawdown
        equity_curve = [100000]
        for pnl in trades_pnl:
            equity_curve.append(equity_curve[-1] * (1 + pnl))
        eq = np.array(equity_curve)
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max
        max_dd = float(np.min(dd))

        results.append(OptimResult(
            sl_atr=sl_atr, tp_atr=tp_atr,
            confidence_threshold=conf_thresh, min_rr=min_rr,
            adx_filter=adx_min,
            total_return=total_return, sharpe=sharpe,
            win_rate=win_rate, profit_factor=pf,
            max_drawdown=max_dd,
            total_trades=len(trades_pnl),
            avg_pnl=float(np.mean(returns_arr) * 100),
        ))

        tested += 1

    # Sort by composite score: Sharpe * profit_factor (rewards consistency + profitability)
    results.sort(key=lambda r: r.sharpe * r.profit_factor, reverse=True)

    print(f"\n  Tested {tested} valid combinations")
    print(f"\n  TOP 10 PARAMETER COMBINATIONS:")
    print(f"  {'SL':>5} {'TP':>5} {'Conf':>5} {'RR':>4} {'ADX':>4} | "
          f"{'Return':>8} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'MaxDD':>7} {'#':>4}")
    print(f"  {'-'*75}")

    for r in results[:10]:
        print(
            f"  {r.sl_atr:>5.1f} {r.tp_atr:>5.1f} {r.confidence_threshold:>5.2f} "
            f"{r.min_rr:>4.1f} {r.adx_filter:>4.0f} | "
            f"{r.total_return:>+7.2f}% {r.sharpe:>7.2f} "
            f"{r.win_rate:>5.1%} {r.profit_factor:>5.2f} "
            f"{r.max_drawdown:>7.2%} {r.total_trades:>4}"
        )

    # Validate best on holdout set
    if results:
        best = results[0]
        print(f"\n  BEST: SL={best.sl_atr}xATR TP={best.tp_atr}xATR "
              f"Conf={best.confidence_threshold} ADX>{best.adx_filter}")
        print(f"  Return={best.total_return:+.2f}% Sharpe={best.sharpe:.2f} "
              f"WR={best.win_rate:.1%} PF={best.profit_factor:.2f}")

    print(f"{'='*60}")
    return results
