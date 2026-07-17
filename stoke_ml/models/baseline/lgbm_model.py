"""LightGBM baseline with EFB for high-dimensional financial features (3-class)."""
import numpy as np
import lightgbm as lgb


class LGBMBaseline:
    """LightGBM classifier with Exclusive Feature Bundling for wide data.

    EFB automatically bundles mutually exclusive sparse features,
    reducing effective dimensionality without information loss.
    feature_fraction=0.7 provides additional regularization for
    wide financial datasets (>1000 features).

    Labels: 0=DOWN, 1=FLAT, 2=UP (consistent with TFT 3-class output).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        feature_fraction: float = 0.7,
        subsample: float = 0.8,
    ):
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": subsample,
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "random_state": 42,
            "verbosity": -1,
            "enable_bundle": True,  # EFB — key for 24,300-dim data
        }
        self._model: lgb.Booster | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        mask = y != -100
        X_clean = X[mask]
        y_clean = y[mask].astype(int)
        train_data = lgb.Dataset(X_clean, label=y_clean)
        self._model = lgb.train(
            {k: v for k, v in self._params.items() if k != "n_estimators"},
            train_data,
            num_boost_round=self._params["n_estimators"],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        proba = self._model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(X)

    def save(self, path: str):
        if self._model is None:
            raise RuntimeError("Model not trained. Nothing to save.")
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "LGBMBaseline":
        instance = cls()
        instance._model = lgb.Booster(model_file=path)
        return instance
