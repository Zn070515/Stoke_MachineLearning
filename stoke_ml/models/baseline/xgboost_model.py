"""XGBoost baseline model for stock direction prediction."""
import numpy as np
import xgboost as xgb


class XGBoostBaseline:
    """XGBoost classifier for next-day price direction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
    ):
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        if scale_pos_weight is not None:
            self._params["scale_pos_weight"] = scale_pos_weight
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self._params.get("scale_pos_weight") is None:
            neg = np.sum(y == 0)
            pos = np.sum(y == 1)
            self._params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
        self._model = xgb.XGBClassifier(**self._params)
        self._model.fit(X, y, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]

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
