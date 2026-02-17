import logging
import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from config.settings import AI

logger = logging.getLogger(__name__)


class GoldPredictor:
    """XGBoost classifier for BUY/SELL/HOLD signals."""

    def __init__(self, model_path: str = None):
        self.model: XGBClassifier = None
        self._is_loaded = False
        self._model_path = model_path or AI.MODEL_PATH

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self.model is not None

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train XGBoost on features X and labels y. Returns metrics."""
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = y_train.map(label_map)
        y_test_mapped = y_test.map(label_map)

        self.model = XGBClassifier(
            n_estimators=AI.N_ESTIMATORS,
            max_depth=AI.MAX_DEPTH,
            learning_rate=AI.LEARNING_RATE,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )

        self.model.fit(
            X_train, y_train_mapped,
            eval_set=[(X_test, y_test_mapped)],
            verbose=False,
        )

        y_pred_mapped = self.model.predict(X_test)
        reverse_map = {0: -1, 1: 0, 2: 1}
        y_pred = pd.Series(y_pred_mapped).map(reverse_map)
        y_test_orig = y_test.reset_index(drop=True)

        accuracy = (y_pred.values == y_test_orig.values).mean()

        from sklearn.metrics import classification_report
        report = classification_report(
            y_test_orig, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            output_dict=True,
            zero_division=0,
        )

        importance = dict(zip(AI.FEATURE_NAMES, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        self._is_loaded = True
        metrics = {
            "accuracy": accuracy,
            "report": report,
            "top_features": top_features,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "class_distribution": {
                "SELL": int((y_test_orig == -1).sum()),
                "HOLD": int((y_test_orig == 0).sum()),
                "BUY": int((y_test_orig == 1).sum()),
            },
        }

        logger.info(f"Model trained: accuracy={accuracy:.3f}")
        return metrics

    def predict(self, features: pd.DataFrame) -> tuple:
        """
        Predict signal for the latest candle.
        Returns (signal: int, confidence: float, probabilities: dict)
        """
        if not self.is_ready:
            logger.warning("Model not loaded")
            return 0, 0.0, {}

        X = features.iloc[[-1]]

        probs = self.model.predict_proba(X)[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

        class_map = {0: -1, 1: 0, 2: 1}
        signal = class_map[pred_class]

        prob_dict = {
            "SELL": round(probs[0], 3),
            "HOLD": round(probs[1], 3),
            "BUY": round(probs[2], 3),
        }

        return signal, confidence, prob_dict

    def save(self, path: str = None):
        """Save model to disk."""
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = None) -> bool:
        """Load model from disk."""
        path = path or self._model_path
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        self.model = joblib.load(path)
        self._is_loaded = True
        logger.info(f"Model loaded from {path}")
        return True
