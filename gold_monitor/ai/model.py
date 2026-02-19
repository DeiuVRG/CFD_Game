import logging
import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from ai.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class GoldPredictor:
    """XGBoost classifier with walk-forward validation and class weighting."""

    def __init__(self, model_path: str = None):
        self.model: XGBClassifier = None
        self._is_loaded = False
        self._model_path = model_path or "models/default_xgb.pkl"

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self.model is not None

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train XGBoost with:
        - Walk-forward (TimeSeriesSplit) cross-validation
        - Class weights to handle BUY/SELL/HOLD imbalance
        - Regularization to prevent overfitting
        """
        label_map = {-1: 0, 0: 1, 1: 2}
        reverse_map = {0: -1, 1: 0, 2: 1}
        y_mapped = y.map(label_map).astype(int)

        # --- Walk-Forward Cross-Validation ---
        tscv = TimeSeriesSplit(n_splits=5)
        cv_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]

            sample_weights = compute_sample_weight("balanced", y_tr)

            model = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
            )

            model.fit(
                X_tr, y_tr,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            acc = (y_pred == y_val.values).mean()
            cv_accuracies.append(acc)

        # --- Final model: train on first 80%, test on last 20% ---
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_mapped.iloc[:split_idx], y_mapped.iloc[split_idx:]

        sample_weights_final = compute_sample_weight("balanced", y_train)

        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights_final,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # --- Evaluate ---
        y_pred_mapped = self.model.predict(X_test)
        y_pred = pd.Series(y_pred_mapped).map(reverse_map)
        y_test_orig = y_test.map(reverse_map).reset_index(drop=True)

        accuracy = (y_pred.values == y_test_orig.values).mean()

        from sklearn.metrics import classification_report
        report = classification_report(
            y_test_orig, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            output_dict=True,
            zero_division=0,
        )

        feature_names = FeatureEngineer.FEATURE_NAMES
        if len(feature_names) == len(self.model.feature_importances_):
            importance = dict(zip(feature_names, self.model.feature_importances_))
        else:
            importance = {f"f{i}": v for i, v in enumerate(self.model.feature_importances_)}
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        self._is_loaded = True

        metrics = {
            "accuracy": accuracy,
            "cv_accuracy_mean": np.mean(cv_accuracies),
            "cv_accuracy_std": np.std(cv_accuracies),
            "cv_accuracies": cv_accuracies,
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

        logger.info(
            f"Model trained: test_acc={accuracy:.3f}, "
            f"cv_acc={np.mean(cv_accuracies):.3f}+/-{np.std(cv_accuracies):.3f}"
        )
        return metrics

    def predict(self, features: pd.DataFrame) -> tuple:
        if not self.is_ready:
            return 0, 0.0, {}

        X = features.iloc[[-1]]
        probs = self.model.predict_proba(X)[0]
        pred_class = np.argmax(probs)
        confidence = float(probs[pred_class])

        class_map = {0: -1, 1: 0, 2: 1}
        signal = class_map[pred_class]

        prob_dict = {
            "SELL": round(float(probs[0]), 3),
            "HOLD": round(float(probs[1]), 3),
            "BUY": round(float(probs[2]), 3),
        }

        return signal, confidence, prob_dict

    def save(self, path: str = None):
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = None) -> bool:
        path = path or self._model_path
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        self.model = joblib.load(path)
        self._is_loaded = True
        logger.info(f"Model loaded from {path}")
        return True
