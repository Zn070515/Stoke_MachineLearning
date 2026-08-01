import numpy as np


class PanelFeatureSelector:
    """3-stage feature selection for panel (N_stocks x T x D) data.

    Stages:
    1. IC filter: |Spearman RankIC| > ic_threshold (default 0.01)
    2. Blocked correlation dedup: Spearman rho < corr_threshold (default 0.85)
       within predefined feature blocks
    3. LightGBM importance: discard bottom 1% by gain

    Runs once per fold on training data only.  Returns boolean masks
    for PK and PO columns.
    """

    IC_THRESHOLD = 0.01
    CORR_THRESHOLD = 0.85
    IMPORTANCE_PCT = 1.0  # discard bottom 1%

    # Feature blocks for blocked correlation dedup.
    # Features within the same block are correlated; cross-block
    # features are assumed independent.
    BLOCKS: list[tuple[str, list[str]]] = [
        ("capital_flow_nets", ["main_net", "mid_net", "small_net", "large_net", "super_net"]),
        ("capital_flow_ratios", ["main_ratio", "mid_ratio", "small_ratio", "large_ratio", "super_ratio"]),
        ("sector_momentum", ["momentum_5d", "momentum_20d", "momentum_60d", "momentum_252d"]),
        ("valuation", ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]),
        ("sentiment_news", []),   # filled dynamically: columns starting with "news_"
        ("sentiment_guba", []),   # filled dynamically: columns starting with "guba_"
        ("temporal_ma", []),      # filled dynamically: columns ending with _ma5/_ma10/_ma20
        ("temporal_std", []),     # filled dynamically: columns ending with _std20
    ]

    def __init__(
        self,
        ic_threshold: float = 0.01,
        corr_threshold: float = 0.85,
        importance_pct: float = 1.0,
    ):
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold
        self.importance_pct = importance_pct
        self._pk_mask: np.ndarray | None = None
        self._po_mask: np.ndarray | None = None

    def select(
        self,
        pk_arr: np.ndarray,      # (N, T, D_pk) or (samples, D_pk)
        po_arr: np.ndarray,      # (N, T, D_po) or (samples, D_po)
        y: np.ndarray,           # (N, T) or (samples,)
        pk_cols: list[str],
        po_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        """Run 3-stage selection, return (selected_pk_cols, selected_po_cols)."""
        import logging
        from scipy.stats import spearmanr

        logger = logging.getLogger(__name__)

        # Flatten panel -> (samples, D) for correlation computation
        y_flat = y.reshape(-1)
        valid = (y_flat != -100) & np.isfinite(y_flat)
        if valid.sum() < 100:
            logger.warning("PanelFeatureSelector: <100 valid samples, skipping")
            return pk_cols, po_cols

        y_valid = y_flat[valid]

        def _flatten(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3:
                return arr.reshape(-1, arr.shape[-1])[valid]
            return arr[valid]

        all_cols = pk_cols + po_cols
        all_data = np.concatenate([_flatten(pk_arr), _flatten(po_arr)], axis=1)

        n_total = len(all_cols)
        logger.info(
            "PanelFeatureSelector: %d features, %d valid samples",
            n_total, valid.sum(),
        )

        # ---- Stage 1: IC filter ----
        ic_scores = np.zeros(n_total, dtype=np.float64)
        for j in range(n_total):
            col_data = all_data[:, j]
            finite = np.isfinite(col_data)
            if finite.sum() < 30:
                ic_scores[j] = 0.0
                continue
            rho, _ = spearmanr(col_data[finite], y_valid[finite])
            ic_scores[j] = abs(rho)

        ic_mask = ic_scores > self.ic_threshold
        n_ic = ic_mask.sum()
        logger.info("  Stage 1 (IC>%.3f): %d -> %d features", self.ic_threshold, n_total, n_ic)
        if n_ic == 0:
            logger.warning("  IC filter removed ALL features, keeping top 10 by IC")
            top10 = np.argsort(ic_scores)[-10:]
            ic_mask[top10] = True
            n_ic = ic_mask.sum()

        # ---- Stage 2: Blocked correlation dedup ----
        surviving_indices = np.where(ic_mask)[0]
        surviving_scores = ic_scores[ic_mask]
        order = np.argsort(-surviving_scores)
        sorted_indices = surviving_indices[order]

        blocks = self._build_blocks(all_cols)

        keep = np.zeros(n_total, dtype=bool)
        for block_name, block_indices in blocks:
            # Sort by descending IC so higher-IC features survive correlation dedup
            block_sorted = sorted(
                [i for i in block_indices if i in set(sorted_indices)],
                key=lambda idx: ic_scores[idx], reverse=True,
            )
            block_kept = []
            for idx in block_sorted:
                col_vec = all_data[:, idx]
                reject = False
                for kept_idx in block_kept:
                    rho, _ = spearmanr(
                        np.nan_to_num(col_vec, 0.0),
                        np.nan_to_num(all_data[:, kept_idx], 0.0),
                    )
                    if abs(rho) >= self.corr_threshold:
                        reject = True
                        break
                if not reject:
                    block_kept.append(idx)
                    keep[idx] = True

        n_corr = keep.sum()
        logger.info("  Stage 2 (corr<%.2f, %d blocks): %d -> %d features",
                     self.corr_threshold, len(blocks), n_ic, n_corr)

        # ---- Stage 3: LightGBM importance ----
        try:
            import lightgbm as lgb
            keep_indices = np.where(keep)[0]
            X_sub = all_data[:, keep_indices]
            X_sub = np.nan_to_num(X_sub, 0.0)
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, num_leaves=31,
                verbose=-1, random_state=42, n_jobs=-1,
            )
            model.fit(X_sub, y_valid)
            gains = model.booster_.feature_importance(importance_type="gain")
            threshold = np.percentile(gains, self.importance_pct)
            gain_mask = gains > threshold
            final_indices = keep_indices[gain_mask]
            final_mask = np.zeros(n_total, dtype=bool)
            final_mask[final_indices] = True
            n_final = final_mask.sum()
            logger.info("  Stage 3 (LGBM gain>p%.1f): %d -> %d features",
                         self.importance_pct, n_corr, n_final)
        except ImportError:
            logger.info("  Stage 3 skipped (lightgbm not available): keeping %d features", n_corr)
            final_mask = keep
            n_final = n_corr

        # Split result back into PK / PO
        n_pk = len(pk_cols)
        self._pk_mask = final_mask[:n_pk]
        self._po_mask = final_mask[n_pk:]

        selected_pk = [c for c, m in zip(pk_cols, self._pk_mask) if m]
        selected_po = [c for c, m in zip(po_cols, self._po_mask) if m]

        logger.info("  Final: %d PK + %d PO = %d features (from %d)",
                     len(selected_pk), len(selected_po), n_final, n_total)

        return selected_pk, selected_po

    @property
    def pk_mask(self) -> np.ndarray | None:
        return self._pk_mask

    @property
    def po_mask(self) -> np.ndarray | None:
        return self._po_mask

    # ------------------------------------------------------------------
    # Block builder
    # ------------------------------------------------------------------

    def _build_blocks(self, all_cols: list[str]) -> list[tuple[str, list[int]]]:
        """Build feature blocks for correlation dedup."""
        col_to_idx = {c: i for i, c in enumerate(all_cols)}
        blocks: list[tuple[str, list[int]]] = []

        for block_name, patterns in self.BLOCKS:
            indices = []
            if patterns:
                # Static block: patterns are exact column names
                for pat in patterns:
                    if pat in col_to_idx:
                        indices.append(col_to_idx[pat])
            else:
                # Dynamic block: fill by naming convention
                if block_name == "sentiment_news":
                    indices = [i for c, i in col_to_idx.items() if c.startswith("news_")]
                elif block_name == "sentiment_guba":
                    indices = [i for c, i in col_to_idx.items() if c.startswith("guba_")]
                elif block_name == "temporal_ma":
                    indices = [i for c, i in col_to_idx.items()
                               if c.endswith(("_ma5", "_ma10", "_ma20"))]
                elif block_name == "temporal_std":
                    indices = [i for c, i in col_to_idx.items() if c.endswith("_std20")]
            if indices:
                blocks.append((block_name, indices))

        # Any column not in a block gets its own singleton block
        assigned = set()
        for _, idxs in blocks:
            assigned.update(idxs)
        for c, i in col_to_idx.items():
            if i not in assigned:
                blocks.append((f"_singleton_{c}", [i]))

        return blocks
