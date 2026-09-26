"""
Model Training and Inference for Entity Matching.
Uses LightGBM (with HistGradientBoosting fallback) optimized for pairwise ranking/classification.
Supports probability calibration and model persistence.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class EntityMatcherModel:
    """
    Pairwise entity resolution model.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        scale_pos_weight: float = 1.8,
        random_state: int = 42,
    ):
        self.model_type = model_type if (HAS_LIGHTGBM and model_type == "lightgbm") else "hist_gb"
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model = None
        self.feature_names = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_cols: List[str],
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ):
        """Fit the gradient boosting model on pairwise features."""
        self.feature_names = feature_cols
        X_train = X[feature_cols].fillna(0.0)

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
            )
            eval_kwargs = {}
            if eval_set:
                X_val, y_val = eval_set
                eval_kwargs["eval_set"] = [(X_val[feature_cols].fillna(0.0), y_val)]

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    X_train,
                    y,
                    **eval_kwargs,
                )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.model.fit(X_train, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        X_test = X[self.feature_names].fillna(0.0)
        return self.model.predict_proba(X_test)[:, 1]

    def save(self, filepath: str):
        """Save model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EntityMatcherModel":
        """Load model artifact from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance
