"""LightGBM baseline with EFB for high-dimensional financial features."""
import numpy as np
import lightgbm as lgb


class LGBMBaseline:
    """LightGBM classifier with Exclusive Feature Bundling for wide data.

    EFB automatically bundles mutually exclusive sparse features,
    reducing effective dimensionality without information loss.
    feature_fraction=0.7 provides additional regularization for
    wide financial datasets (>1000 features).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        feature_fraction: float = 0.7,
        subsample: float = 0.8,
        scale_pos_weight: float | None = None,
    ):
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": subsample,
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 42,
            "verbosity": -1,
            "enable_bundle": True,  # EFB — key for 24,300-dim data
        }
        if scale_pos_weight is not None:
            self._params["scale_pos_weight"] = scale_pos_weight
        self._model: lgb.Booster | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self._params.get("scale_pos_weight") is None:
            neg = np.sum(y == 0)
            pos = np.sum(y == 1)
            self._params["scale_pos_weight"] = neg / pos if (pos > 0 and neg > 0) else 1.0
        train_data = lgb.Dataset(X, label=y)
        self._model = lgb.train(
            {k: v for k, v in self._params.items() if k != "n_estimators"},
            train_data,
            num_boost_round=self._params["n_estimators"],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        proba = self._model.predict(X)
        return (proba > 0.5).astype(int)

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
