import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineer import FeatureEngineer
from ai.model import GoldPredictor
from config.settings import AI, INSTRUMENTS, InstrumentConfig
from data.gold_fetcher import MarketFetcher

logger = logging.getLogger(__name__)


def train_instrument(instrument: InstrumentConfig) -> bool:
    """Train XGBoost model for a single instrument."""
    print(f"\n{'=' * 60}")
    print(f"  Training: {instrument.SYMBOL_DISPLAY} ({instrument.SYMBOL})")
    print(f"{'=' * 60}")

    # Step 1: Download data
    print(f"\n[1/5] Downloading {instrument.TRAIN_PERIOD} of price data...")
    fetcher = MarketFetcher(instrument)
    df = fetcher.get_training_data()

    if df.empty:
        print(f"ERROR: Could not download data for {instrument.SYMBOL}!")
        return False

    print(f"  Downloaded {len(df)} candles")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

    # Step 2: Create features
    print("\n[2/5] Engineering features (v2: ADX, StochRSI, ATR ratio)...")
    features = FeatureEngineer.create_features(df)
    labels = FeatureEngineer.create_labels(
        df,
        horizon=AI.PREDICTION_HORIZON,
        threshold=instrument.PRICE_CHANGE_THRESHOLD,
    )

    valid_mask = features.notna().all(axis=1) & labels.notna()
    features = features[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask].reset_index(drop=True).astype(int)

    print(f"  Features: {features.shape[1]} columns, {len(features)} rows")
    print(f"  Label distribution:")
    print(f"    BUY:  {(labels == 1).sum():>5} ({(labels == 1).mean()*100:.1f}%)")
    print(f"    HOLD: {(labels == 0).sum():>5} ({(labels == 0).mean()*100:.1f}%)")
    print(f"    SELL: {(labels == -1).sum():>5} ({(labels == -1).mean()*100:.1f}%)")

    if len(features) < 200:
        print(f"ERROR: Not enough data ({len(features)} rows, need 200+)!")
        return False

    # Step 3: Train with walk-forward CV + class weights
    print("\n[3/5] Training XGBoost (walk-forward CV, class-weighted)...")
    predictor = GoldPredictor(model_path=instrument.MODEL_PATH)
    metrics = predictor.train(features, labels)

    # Step 4: Results
    print("\n[4/5] Results:")
    print(f"  Holdout Test Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Walk-Forward CV:       {metrics['cv_accuracy_mean']*100:.1f}% "
          f"+/- {metrics['cv_accuracy_std']*100:.1f}%")

    cv_accs = metrics.get("cv_accuracies", [])
    if cv_accs:
        fold_str = " | ".join(f"{a*100:.1f}%" for a in cv_accs)
        print(f"  Per-fold accuracies:   [{fold_str}]")

    print(f"  Train: {metrics['train_size']} | Test: {metrics['test_size']}")

    print(f"\n  Per-class performance:")
    for cls in ["SELL", "HOLD", "BUY"]:
        r = metrics["report"].get(cls, {})
        print(
            f"    {cls:4s}: P={r.get('precision', 0):.2f} "
            f"R={r.get('recall', 0):.2f} F1={r.get('f1-score', 0):.2f}"
        )

    print(f"\n  Top features:")
    for name, imp in metrics["top_features"]:
        bar = "#" * int(imp * 100)
        print(f"    {name:25s} {imp:.4f} {bar}")

    # Overfitting check
    cv_mean = metrics["cv_accuracy_mean"]
    test_acc = metrics["accuracy"]
    diff = abs(test_acc - cv_mean)
    if diff > 0.10:
        print(f"\n  WARNING: Overfitting! Test ({test_acc:.1%}) vs CV ({cv_mean:.1%})")
    else:
        print(f"\n  Overfitting check: OK (diff={diff:.1%})")

    # Step 5: Save
    print(f"\n[5/5] Saving model to {instrument.MODEL_PATH}...")
    predictor.save()
    print(f"  Done! Test={test_acc*100:.1f}% CV={cv_mean*100:.1f}%")
    return True


def train_model(instrument_filter: str = None) -> bool:
    """Train AI models for all enabled instruments."""
    print("=" * 60)
    print("  AI Trainer v2 - Walk-Forward + Class Weights")
    print("=" * 60)

    targets = [i for i in INSTRUMENTS if i.ENABLED]
    if instrument_filter:
        targets = [i for i in targets if instrument_filter.lower() in i.SYMBOL.lower()
                    or instrument_filter.lower() in i.SYMBOL_DISPLAY.lower()]
        if not targets:
            print(f"ERROR: No instrument matching '{instrument_filter}'")
            return False

    print(f"\n  Training {len(targets)} instrument(s):")
    for t in targets:
        print(f"    - {t.SYMBOL_DISPLAY} ({t.SYMBOL})")

    results = {}
    for inst in targets:
        success = train_instrument(inst)
        results[inst.SYMBOL_DISPLAY] = success

    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print(f"{'=' * 60}")

    return all(results.values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
