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
    print(f"\n{'=' * 55}")
    print(f"  Training: {instrument.SYMBOL_DISPLAY} ({instrument.SYMBOL})")
    print(f"{'=' * 55}")

    # Step 1: Download data
    print(f"\n[1/5] Downloading {instrument.TRAIN_PERIOD} of price data...")
    fetcher = MarketFetcher(instrument)
    df = fetcher.get_training_data()

    if df.empty:
        print(f"ERROR: Could not download training data for {instrument.SYMBOL}!")
        return False

    print(f"  Downloaded {len(df)} candles")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

    # Step 2: Create features
    print("\n[2/5] Engineering features...")
    features = FeatureEngineer.create_features(df)
    labels = FeatureEngineer.create_labels(
        df,
        horizon=AI.PREDICTION_HORIZON,
        threshold=instrument.PRICE_CHANGE_THRESHOLD,
    )

    valid_mask = features.notna().all(axis=1) & labels.notna()
    features = features[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask].reset_index(drop=True)

    print(f"  Features: {features.shape[1]} columns, {len(features)} rows")
    print(f"  Label distribution:")
    print(f"    BUY:  {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    print(f"    HOLD: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"    SELL: {(labels == -1).sum()} ({(labels == -1).mean()*100:.1f}%)")

    if len(features) < 100:
        print(f"ERROR: Not enough training data ({len(features)} rows)!")
        return False

    # Step 3: Train model
    print("\n[3/5] Training XGBoost model...")
    predictor = GoldPredictor(model_path=instrument.MODEL_PATH)
    metrics = predictor.train(features, labels)

    # Step 4: Print results
    print("\n[4/5] Results:")
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Train size: {metrics['train_size']} | Test size: {metrics['test_size']}")
    print(f"\n  Test set distribution:")
    for cls, count in metrics["class_distribution"].items():
        print(f"    {cls}: {count}")

    print(f"\n  Per-class performance:")
    for cls in ["SELL", "HOLD", "BUY"]:
        r = metrics["report"].get(cls, {})
        print(
            f"    {cls:4s}: precision={r.get('precision', 0):.2f} "
            f"recall={r.get('recall', 0):.2f} f1={r.get('f1-score', 0):.2f}"
        )

    print(f"\n  Top 10 features:")
    for name, imp in metrics["top_features"]:
        bar = "#" * int(imp * 100)
        print(f"    {name:25s} {imp:.4f} {bar}")

    # Step 5: Save model
    print(f"\n[5/5] Saving model to {instrument.MODEL_PATH}...")
    predictor.save()
    print("  Model saved successfully!")

    print(f"\n  {instrument.SYMBOL_DISPLAY} training complete! Accuracy: {metrics['accuracy']*100:.1f}%")
    return True


def train_model(instrument_filter: str = None) -> bool:
    """Train AI models for all enabled instruments (or a specific one)."""
    print("=" * 55)
    print("  AI Trainer - XGBoost (Multi-Instrument)")
    print("=" * 55)

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

    # Summary
    print(f"\n{'=' * 55}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 55}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print(f"{'=' * 55}")

    return all(results.values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
