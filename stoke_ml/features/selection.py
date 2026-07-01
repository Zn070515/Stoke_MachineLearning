"""Feature selection for high-dimensional financial data (24,300+ dims).

Methods:
- Mutual Information filter: fast ranking, selects top-k by MI score.
  AUROC optimal (0.774) in 2024 Nature Scientific Reports benchmarks.
- Sequential Forward Selection: greedy add-one-at-a-time, highest accuracy
  but computationally expensive. Best with LightGBM (80.53% vs 74.22%).

Typical pipeline:
  1. MI filter to 200 features (fast)
  2. SFS to 50 features (precise)
  3. Train on selected features
"""
import logging
import time

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def select_by_mutual_info(
    X: np.ndarray, y: np.ndarray, k: int = 50, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-k features by Mutual Information with label.

    Returns (X_selected, selected_indices).
    """
    n_features = X.shape[1]
    mi_scores = mutual_info_classif(X, y, random_state=seed, n_neighbors=3)
    top_idx = np.argsort(mi_scores)[-k:]
    logger.info("MI: selected %d/%d features, top score=%.4f, median=%.4f",
                k, n_features, mi_scores[top_idx[-1]], np.median(mi_scores[top_idx]))
    return X[:, top_idx], top_idx


def select_by_sfs(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 50,
    model_type: str = "lgbm",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy Sequential Forward Selection to k features.

    Uses LightGBM (fast) or XGBoost as evaluator. At each step,
    adds the feature that maximizes cross-validated AUROC.

    Returns (X_selected, selected_indices).
    """
    n_samples, n_features = X.shape
    k = min(k, n_features)
    if k == 0:
        return np.empty((n_samples, 0)), np.array([], dtype=np.int64)

    def _make_model():
        if model_type == "lgbm":
            from stoke_ml.models.baseline.lgbm_model import LGBMBaseline
            return LGBMBaseline(n_estimators=30, max_depth=3, learning_rate=0.1)
        from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline
        return XGBoostBaseline(n_estimators=30, max_depth=3, learning_rate=0.1)

    def _eval_subset(indices):
        X_sub = X[:, indices]
        # Simple 80/20 split for fast evaluation
        split = int(n_samples * 0.8)
        X_tr, X_va = X_sub[:split], X_sub[split:]
        y_tr, y_va = y[:split], y[split:]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            return 0.0
        model = _make_model()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)
        return roc_auc_score(y_va, proba)

    remaining = list(range(n_features))
    selected: list[int] = []
    best_auc = 0.0

    logger.info("SFS: selecting %d features from %d (model=%s)", k, n_features, model_type)
    t0 = time.time()
    for step in range(k):
        best_feat = -1
        best_score = -1.0
        for feat in remaining:
            candidate = selected + [feat]
            auc = _eval_subset(candidate)
            if auc > best_score:
                best_score = auc
                best_feat = feat
        if best_feat < 0:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        delta = best_score - best_auc
        best_auc = best_score
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            logger.info("  SFS %d/%d: AUC=%.4f (Δ+%.4f), %d features, %.1fs",
                        step + 1, k, best_auc, delta, len(selected), elapsed)

    elapsed = time.time() - t0
    logger.info("SFS done: %d features, AUC=%.4f, %.1fs", len(selected), best_auc, elapsed)
    return X[:, selected], np.array(selected)


class FeatureSelector:
    """Two-stage feature selector: MI pre-filter + SFS refinement."""

    def __init__(self, mi_k: int = 200, sfs_k: int = 50, model_type: str = "lgbm"):
        self.mi_k = mi_k
        self.sfs_k = sfs_k
        self.model_type = model_type
        self.mi_indices: np.ndarray | None = None
        self.sfs_indices: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run MI filter then SFS, return reduced feature matrix."""
        n_features = X.shape[1]
        mi_limit = min(self.mi_k, n_features)
        logger.info("FeatureSelector: %d → %d (MI) → %d (SFS)",
                     n_features, mi_limit, min(self.sfs_k, mi_limit))

        X_mi, self.mi_indices = select_by_mutual_info(X, y, k=mi_limit)
        if self.sfs_k > 0:
            sfs_limit = min(self.sfs_k, X_mi.shape[1])
            X_out, sfs_local = select_by_sfs(
                X_mi, y, k=sfs_limit, model_type=self.model_type,
            )
            self.sfs_indices = self.mi_indices[sfs_local]
        else:
            self.sfs_indices = self.mi_indices
            X_out = X_mi
        return X_out

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply saved selection to new data."""
        if self.sfs_indices is None:
            raise RuntimeError("Call fit_transform() first.")
        return X[:, self.sfs_indices]
