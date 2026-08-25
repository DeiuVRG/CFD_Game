#!/usr/bin/env python3
"""
Faza 4 evaluation runner: optimizer -> one-shot v3 backtest -> activation
verdict, per instrument. Produces the numbers for RESULTS.md.

Protocol (anti-overfitting discipline):
  1. Data: 2 years of TRAIN_INTERVAL (1h) candles, downloaded once and cached
     to CSV so every step sees identical data.
  2. Optimizer: trains on the first 50%, grid-searches parameters (and the
     label threshold) on the middle 25%. The last 25% is never touched.
  3. Selection rule (declared before any OOS run): best sharpe*PF score among
     combos with >=30 trades, falling back to >=15, then overall best.
  4. One v3 backtest run() per instrument configuration: reports the
     optimization window and the OOS window separately. The OOS window is
     the DECISION window and is run exactly once per configuration.
  5. Legacy-execution comparison (same model, same params) to quantify how
     much the pre-v3 bugs inflated results.
  6. Final production model trained on the full 2y with the chosen threshold.

Usage:
    cd gold_monitor
    python tools/run_evaluation.py gold
    python tools/run_evaluation.py btc
    python tools/run_evaluation.py all
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_MONITOR = os.path.dirname(HERE)
sys.path.insert(0, GOLD_MONITOR)

import pandas as pd  # noqa: E402

from ai.backtester import Backtester  # noqa: E402
from ai.feature_engineer import FeatureEngineer  # noqa: E402
from ai.model import GoldPredictor  # noqa: E402
from ai.optimizer import optimize_instrument, select_best  # noqa: E402
from config.settings import AI, INSTRUMENTS  # noqa: E402
from data.gold_fetcher import MarketFetcher  # noqa: E402

CACHE_DIR = os.path.join(GOLD_MONITOR, "data_cache")
OUTPUT_DIR = os.path.join(GOLD_MONITOR, "output")

# OOS activation criteria (all must hold, net of costs)
CRITERIA = {
    "min_trades": 30,
    "min_profit_factor": 1.15,
    "min_expectancy_pct": 0.0,     # avg net P&L per trade must be > 0
    "max_drawdown_limit": -0.15,   # max DD must be shallower than -15%
    "min_trade_sharpe": 0.5,
}


def sanitize(obj):
    """Make metrics JSON-safe (inf/nan -> None)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


def load_data(inst, use_cache=True) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        CACHE_DIR,
        f"{inst.SYMBOL.replace('=', '_').replace('-', '_')}"
        f"_{inst.TRAIN_INTERVAL}_{inst.TRAIN_PERIOD}.csv",
    )
    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        print(f"  Loaded {len(df)} cached candles from {cache_file}")
        return df

    fetcher = MarketFetcher(inst)
    df = fetcher.get_training_data()
    if df.empty:
        raise RuntimeError(f"No data for {inst.SYMBOL}")
    df.to_csv(cache_file, index=False)
    print(f"  Downloaded {len(df)} candles -> cached to {cache_file}")
    return df


def check_criteria(oos_metrics: dict) -> dict:
    checks = {
        "trades>=30": bool(oos_metrics.get("total_trades", 0) >= CRITERIA["min_trades"]),
        "profit_factor>=1.15": bool(oos_metrics.get("profit_factor", 0) >= CRITERIA["min_profit_factor"]),
        "expectancy>0": bool(oos_metrics.get("avg_trade_pnl", 0) > CRITERIA["min_expectancy_pct"]),
        "max_dd<15%": bool(oos_metrics.get("max_drawdown", -1) > CRITERIA["max_drawdown_limit"]),
        "trade_sharpe>0.5": bool(oos_metrics.get("trade_sharpe", 0) > CRITERIA["min_trade_sharpe"]),
    }
    return {"checks": checks, "enabled": all(checks.values())}


def evaluate(inst, skip_train_final=False, with_legacy=True) -> dict:
    print(f"\n{'#' * 70}")
    print(f"#  EVALUATION: {inst.SYMBOL_DISPLAY}")
    print(f"{'#' * 70}")

    df = load_data(inst)
    data_info = {
        "candles": len(df),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "interval": inst.TRAIN_INTERVAL,
    }

    # ------------------------------------------------------------------
    # 1) Optimizer (train 50% / optimize middle 25%; OOS untouched)
    # ------------------------------------------------------------------
    results = optimize_instrument(inst, df)
    if not results:
        return {"error": "optimizer produced no results", "data": data_info}
    best = select_best(results)

    chosen = {
        "threshold": best.threshold,
        "sl_atr": best.sl_atr,
        "tp_atr": best.tp_atr,
        "confidence": best.confidence_threshold,
        "adx_min": best.adx_filter,
        "min_rr": best.min_rr,
        "optim_window_stats": {
            "total_return_pct": best.total_return,
            "sharpe": best.sharpe,
            "trade_sharpe": best.trade_sharpe,
            "win_rate": best.win_rate,
            "profit_factor": best.profit_factor,
            "max_drawdown": best.max_drawdown,
            "total_trades": best.total_trades,
            "avg_pnl_pct": best.avg_pnl,
        },
    }

    # Apply the chosen parameters to the instrument (in-memory)
    inst.SL_ATR = best.sl_atr
    inst.TP_ATR = best.tp_atr
    inst.CONFIDENCE = best.confidence_threshold
    inst.ADX_MIN = float(best.adx_filter)
    inst.MIN_RR = best.min_rr

    # ------------------------------------------------------------------
    # 2) One-shot v3 backtest: optimization window + OOS window
    # ------------------------------------------------------------------
    print(f"\n  >>> v3 backtest with chosen params "
          f"(thr={best.threshold}, SL={best.sl_atr}, TP={best.tp_atr}, "
          f"conf={best.confidence_threshold}, ADX>{best.adx_filter}, "
          f"RR>={best.min_rr})")
    bt = Backtester(inst)
    v3 = bt.run(df, execution="v3", threshold=best.threshold)
    Backtester.print_report(v3, f"{inst.SYMBOL_DISPLAY} [v3]")

    verdict = check_criteria(v3.get("metrics", {}))

    # ------------------------------------------------------------------
    # 3) Legacy execution comparison (same params/model; RESULTS.md only)
    # ------------------------------------------------------------------
    legacy = None
    if with_legacy:
        print(f"\n  >>> LEGACY execution comparison (bug-for-bug pre-v3)")
        legacy = bt.run(df, execution="legacy", threshold=best.threshold)
        Backtester.print_report(legacy, f"{inst.SYMBOL_DISPLAY} [LEGACY]")

    # ------------------------------------------------------------------
    # 4) Final production model on the full 2y (chosen threshold)
    # ------------------------------------------------------------------
    final_model = None
    if not skip_train_final:
        print(f"\n  >>> Training final production model on full data...")
        features = FeatureEngineer.create_features(df)
        labels = FeatureEngineer.create_labels(
            df, horizon=AI.PREDICTION_HORIZON, threshold=best.threshold)
        valid = features.notna().all(axis=1) & labels.notna()
        predictor = GoldPredictor(model_path=inst.MODEL_PATH)
        metrics = predictor.train(features[valid], labels[valid].astype(int))
        predictor.save()
        final_model = {
            "path": inst.MODEL_PATH,
            "samples": int(valid.sum()),
            "test_accuracy": metrics["accuracy"],
            "cv_accuracy_mean": metrics["cv_accuracy_mean"],
            "cv_accuracy_std": metrics["cv_accuracy_std"],
        }
        print(f"  Model saved: acc={metrics['accuracy']:.3f} "
              f"cv={metrics['cv_accuracy_mean']:.3f}")

    report = {
        "instrument": inst.SYMBOL_DISPLAY,
        "symbol": inst.SYMBOL,
        "data": data_info,
        "cost_model": {
            "spread_pct": inst.SPREAD_PCT,
            "spread_pips": inst.SPREAD_PIPS,
            "pip_value": inst.PIP_VALUE,
        },
        "chosen_params": chosen,
        "optimizer_top10": [vars(r) for r in results[:10]],
        "v3": {
            "optim_metrics": v3.get("optim_metrics", {}),
            "oos_metrics": v3.get("metrics", {}),
        },
        "legacy": None if legacy is None else {
            "optim_metrics": legacy.get("optim_metrics", {}),
            "oos_metrics": legacy.get("metrics", {}),
        },
        "criteria": CRITERIA,
        "verdict": verdict,
        "final_model": final_model,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    key = inst.SYMBOL.replace("=", "_").replace("-", "_").lower()
    out_path = os.path.join(OUTPUT_DIR, f"evaluation_{key}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize(report), f, indent=2, default=str)
    print(f"\n  Saved evaluation to {out_path}")

    print(f"\n  VERDICT for {inst.SYMBOL_DISPLAY}: "
          f"{'ENABLED' if verdict['enabled'] else 'STAYS DISABLED'}")
    for name, ok in verdict["checks"].items():
        print(f"    [{'x' if ok else ' '}] {name}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Faza 4 evaluation runner")
    parser.add_argument("target", nargs="?", default="all",
                        help="gold | btc | all | <symbol substring>")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training the final production model")
    parser.add_argument("--no-legacy", action="store_true",
                        help="Skip the legacy-execution comparison")
    args = parser.parse_args()

    targets = []
    for inst in INSTRUMENTS:
        name = inst.SYMBOL_DISPLAY.lower() + " " + inst.SYMBOL.lower()
        if args.target == "all":
            if inst.SYMBOL in ("GC=F", "BTC-USD"):
                targets.append(inst)
        elif args.target.lower() in name:
            targets.append(inst)

    if not targets:
        print(f"No instrument matches '{args.target}'")
        sys.exit(1)

    for inst in targets:
        evaluate(inst, skip_train_final=args.skip_train,
                 with_legacy=not args.no_legacy)


if __name__ == "__main__":
    main()
