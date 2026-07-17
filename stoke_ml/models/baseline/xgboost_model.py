"""XGBoost baseline model for stock direction prediction (3-class)."""
import numpy as np
import xgboost as xgb


class XGBoostBaseline:
    """XGBoost classifier for next-day price direction.

    Labels: 0=DOWN, 1=FLAT, 2=UP (consistent with TFT 3-class output).
    Positions with label=-100 (limit-up/down masked) are filtered in fit().
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "multi:softmax",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "verbosity": 0,
        }
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        mask = y != -100
        X_clean = X[mask]
        y_clean = y[mask].astype(int)
        self._model = xgb.XGBClassifier(**self._params)
        self._model.fit(X_clean, y_clean, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict_proba(X)

    def save(self, path: str):
        if self._model is None:
            raise RuntimeError("Model not trained. Nothing to save.")
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "XGBoostBaseline":
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(path)
        return instance
