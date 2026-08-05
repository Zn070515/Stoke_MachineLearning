"""Feature pipeline orchestrating all feature engineering steps.

Integrates K-line, sentiment, market-wide (margin/northbound/dragon-tiger),
ETF sector flow, and fundamental data into a unified feature set.
"""
import json
import logging
import os
import re

import pandas as pd
import numpy as np
from stoke_ml.features import cache_manifest
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
from stoke_ml.features.interaction import InteractionFeatures
from stoke_ml.features.temporal import (
    add_lag_features, add_calendar_features,
)
from stoke_ml.features.transform import TemporalTransformer
from stoke_ml.features.emotion import EmotionRefiner
from stoke_ml.features.fundamental import FundamentalRefiner
from stoke_ml.features.market_env import MarketEnvRefiner
from stoke_ml.features.aux_aligner import (
    AuxAligner,
    SENTIMENT_COLS,
    MARGIN_COLS,
    NORTHBOUND_COLS,
    DRAGON_TIGER_COLS,
    FUNDAMENTAL_COLS,
    EARNINGS_COLS,
    VALUATION_COLS,
    ETF_FLOW_COLS,
    GUBA_COLS,
    COMMENT_COLS,
    MACRO_COLS,
    MARKET_ENV_COLS,
    INDUSTRY_COLS,
    _batch_fill_shift,
    _merge_daily_aux,
)

logger = logging.getLogger(__name__)

# The official A-share trading calendar used to validate every
# stock's date axis before it joins the panel's UNION date axis.  Lazy-loaded
# (module-level singleton) so the panel path pays for it only when used.
_panel_calendar = None


def _get_panel_calendar():
    global _panel_calendar
    if _panel_calendar is None:
        from stoke_ml.data.calendar import TradingCalendar
        _panel_calendar = TradingCalendar("a_shares")
    return _panel_calendar

# Sparse feature policy: drop structurally-dead columns — constant on every
# observed day of a stock's series (flat path, _prep_feature_df) or of a fold's
# training window (panel path, fold_dead_feature_columns) — from training.  The
# SPARSE_KEEP_PREFIXES families are exempt: they are genuinely-rare events that
# stay constant for most stocks but carry real signal on their activation days
# (market-state regimes, dragon-tiger presence, dividend growth, pledge plan
# stage, transfer ratio).  The judgment is per-stock / per-fold and uses only
# data already available at the decision point, so a future period never picks
# an earlier period's feature set (research-selection leakage).
DEAD_FEATURE_RATIO = 0.9
SPARSE_KEEP_PREFIXES = (
    "market_state_", "is_dt", "dividend_growth", "plan_stage_", "transfer_ratio",
)


def _constant_col_indices(arr: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-stock dead-feature mask: (N, D) const[stock, feature].

    const[i, f] is True when feature f's value is identical on every day of
    stock i whose observation_mask is True, or stock i has no observed days in
    the slice.  Zero-padded rows carry no evidence and are excluded from the
    scan.
    """
    n_stocks, _n_dates, n_feat = arr.shape
    listed = obs.sum(axis=1) > 0
    hi = np.float32(1e30)
    const = np.zeros((n_stocks, n_feat), dtype=bool)
    # Chunk over stocks so the two where() temporaries stay bounded.
    for s0 in range(0, n_stocks, 512):
        s1 = min(s0 + 512, n_stocks)
        m = obs[s0:s1, :, None]
        vmin = np.where(m, arr[s0:s1], hi).min(axis=1)   # (chunk, D)
        vmax = np.where(m, arr[s0:s1], -hi).max(axis=1)  # (chunk, D)
        const[s0:s1] = vmin == vmax
    const[~listed] = True
    return const


def fold_dead_feature_columns(
    train_data: dict,
    pk_cols: list[str],
    po_cols: list[str],
    ratio: float = DEAD_FEATURE_RATIO,
) -> tuple[list[int], list[int]]:
    """Axis-2 indices of dead columns in a fold's training window.

    A column is dead when >= `ratio` of the fold's eligible stocks show it
    time-constant over their observed days within the training period (the
    same per-stock constant_stock_ratio measure the old report used), EXCEPT
    the SPARSE_KEEP_PREFIXES rare-event families, which are kept — their
    signal lives on sparse activation days.  The judgment uses ONLY the
    training slice — never validation/test — so a future period can't decide
    an earlier fold's feature set.  Returned index lists are safe to np.delete
    from the past_known / past_observed grids of ALL fold slices (train/val/
    test share the column layout).
    """
    obs = train_data["observation_mask"]
    pk_const = _constant_col_indices(train_data["past_known"], obs)
    po_const = _constant_col_indices(train_data["past_observed"], obs)
    pk_dead = (pk_const.mean(axis=0) >= ratio) & ~_sparse_kept(pk_cols)
    po_dead = (po_const.mean(axis=0) >= ratio) & ~_sparse_kept(po_cols)
    pk_idx = [int(i) for i in np.where(pk_dead)[0]]
    po_idx = [int(i) for i in np.where(po_dead)[0]]
    return pk_idx, po_idx


def _sparse_kept(cols: list[str]) -> np.ndarray:
    """Per-column boolean: True for the exempt SPARSE_KEEP_PREFIXES families."""
    return np.array(
        [c.startswith(SPARSE_KEEP_PREFIXES) for c in cols], dtype=bool,
    )

ANNOUNCEMENT_COLS = [
    "ann_sentiment_mean", "ann_sentiment_std", "ann_count",
    "ann_positive_ratio", "ann_negative_ratio", "has_announce",
]

TEMPORAL_BASE_COLS = [
    "open", "high", "low", "close", "volume",
    "volume_ratio", "atr_14", "rsi_12",
]

# Rich text features from DailyAggregator (new preprocessing text chain).
# Per-source prefixes are applied by the benchmark/data-loading layer.
_AGGREGATOR_BASE_COLS = [
    "bipolar_sent", "agreement", "attention", "weighted_sent",
]

# ── New multi-shape preprocessing (spec §6) ──

FLOW_COLS = [
    "flow_intensity", "flow_z", "flow_momentum",
    "flow_persistence_5d", "flow_persistence_10d", "flow_persistence_20d",
    "flow_divergence", "flow_residual", "flow_spread_large_small",
]

BLOCK_TRADE_COLS = [
    "bt_count", "bt_total_amount", "bt_vwap_premium",
    "bt_deep_discount_count", "bt_permanent_impact", "bt_temporary_impact",
    "bt_volatility_6d",
]

SHAREHOLDER_COLS = [
    "sh_hnum_change_pct", "sh_hnum_zscore", "sh_pcrc",
    "sh_consecutive_neg", "sh_dual_concentration_signal", "sh_avg_shares_held",
]

LOCKUP_COLS = [
    "lu_pressure", "lu_ratio", "lu_days_until",
    "lu_event_count", "lu_is_vc_backed",
]

DIVIDEND_COLS = [
    "dv_yield", "dv_effective_yield", "dv_months_since_last",
]

BOARD_COLS = [
    "is_zt", "is_zb", "is_dt", "is_yzt",
    "consecutive_zt", "board_height_20d", "seal_strength", "seal_success",
    "net_zt_proportion", "break_rate", "advance_rate", "max_height",
]

SECTOR_COLS = [
    "sector_relative_strength", "sector_breadth_z",
    "sector_rrg_y", "sector_rrg_x", "sector_rrg_quadrant",
]

CONCEPT_COLS = [
    "board_count", "has_hot_board", "avg_concept_heat",
    "is_concept_leader", "board_overlap_score",
]

LIMIT_UP_COLS = [
    "zt_first_seal_hour", "zt_last_seal_hour", "zt_seal_fund_ratio",
    "zt_break_times", "zt_limit_days", "zt_pct",
    "zb_first_seal_hour", "zb_break_times", "zb_amplitude", "zb_speed",
    "dt_seal_fund_ratio", "dt_open_times", "dt_days", "dt_pe",
    "yzt_first_seal_hour", "yzt_limit_days",
    "has_zt", "has_zb", "has_dt", "has_yzt",
]  # DEFERRED (limit-up ecology family, top scope note) — defined for future re-enable, NOT wired

PLEDGE_COLS = [
    "pledge_ratio", "pledge_margin_dist", "pledge_risk",
    "pledge_count_20d", "has_pledge",
]

INDEX_MEMBER_COLS = [
    "is_index_member", "n_indexes", "idx_change_30d",
]  # no index_weight — Baostock has no historical weights (A4a scope note)

DRAGON_TIGER_SEAT_COLS = [
    "lhb_is_wave", "lhb_is_sustained", "lhb_is_drop", "lhb_count_5d",
]


class FeaturePipeline:
    """End-to-end feature engineering for stock prediction."""

    TARGET_COLS = ["open", "high", "low", "close", "volume"]
    LAGS = [1, 2, 3, 5, 10, 20]
    ROLLING_WINDOWS = [5, 10, 20, 60]

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        flat_mode: bool = False,
        use_technical: bool = True,
        use_scoring: bool = True,
        use_temporal: bool = True,
        use_sentiment: bool = True,
        use_announcements: bool = True,
        use_guba: bool = True,
        use_comment: bool = True,
        use_margin: bool = True,
        use_northbound: bool = True,
        use_dragon_tiger: bool = True,
        use_fundamental: bool = True,
        use_earnings: bool = True,
        use_valuation: bool = True,
        use_etf_flow: bool = True,
        use_interaction: bool = True,
        use_feature_selection: bool = False,
        use_capital_flow: bool = True,
        use_block_trade: bool = True,
        use_shareholder: bool = True,
        use_lockup: bool = True,
        use_dividend: bool = True,
        use_board: bool = True,
        use_sector: bool = True,
        use_concept: bool = True,
        use_macro: bool = True,
        use_industry: bool = True,
        use_limit_up: bool = False,  # DEFERRED (limit-up ecology, top scope note)
        use_pledge: bool = True,
        use_market_env: bool = True,
        use_index_membership: bool = True,
        use_market_env_refine: bool = True,
        minute_mode: bool = False,
        feature_selection_k: int = 500,
        use_new_preprocessing: bool = False,
        preprocessing_config: dict | str | None = None,
        # Feature engineering v2
        use_emotion_refine: bool = True,
        use_fundamental_refine: bool = True,
        use_temporal_stats: bool = True,
        drop_dead_features: bool = True,
        min_history: int = 50,
    ):
        self.seq_len = seq_len
        self.min_history = min_history
        self.horizon = horizon
        self.flat_mode = flat_mode
        self.use_technical = use_technical
        self.use_scoring = use_scoring
        self.use_temporal = use_temporal
        self.use_sentiment = use_sentiment
        self.use_announcements = use_announcements
        self.use_guba = use_guba
        self.use_comment = use_comment
        self.use_margin = use_margin
        self.use_northbound = use_northbound
        self.use_dragon_tiger = use_dragon_tiger
        self.use_fundamental = use_fundamental
        self.use_earnings = use_earnings
        self.use_valuation = use_valuation
        self.use_etf_flow = use_etf_flow
        self.use_interaction = use_interaction
        self.use_feature_selection = use_feature_selection
        self.use_capital_flow = use_capital_flow
        self.use_block_trade = use_block_trade
        self.use_shareholder = use_shareholder
        self.use_lockup = use_lockup
        self.use_dividend = use_dividend
        self.use_board = use_board
        self.use_sector = use_sector
        self.use_concept = use_concept
        self.use_macro = use_macro
        self.use_industry = use_industry
        self.use_limit_up = use_limit_up  # inert while deferred (not wired in _engineer_features)
        self.use_pledge = use_pledge
        self.use_market_env = use_market_env
        self.use_index_membership = use_index_membership
        self.use_market_env_refine = use_market_env_refine
        self._aux = AuxAligner(
            {k: getattr(self, f"use_{k}") for k in AuxAligner.AUX_KEYS}
        )
        self.minute_mode = minute_mode
        self.feature_selection_k = feature_selection_k
        self.use_new_preprocessing = use_new_preprocessing
        self._preprocessing_config = preprocessing_config
        self._preprocessing = None
        if use_new_preprocessing and preprocessing_config:
            self._preprocessing = self._build_preprocessing()
        self._intraday = None
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()
        self._interaction = InteractionFeatures()
        self.use_emotion_refine = use_emotion_refine
        self.use_fundamental_refine = use_fundamental_refine
        self.use_temporal_stats = use_temporal_stats
        self._temporal_transformer = TemporalTransformer() if use_temporal_stats else None
        self._emotion_refiner = EmotionRefiner() if use_emotion_refine else None
        self._fundamental_refiner = FundamentalRefiner() if use_fundamental_refine else None
        self._market_env_refiner = MarketEnvRefiner() if use_market_env_refine else None
        self.drop_dead_features = drop_dead_features

    # ------------------------------------------------------------------
    # Preprocessing integration
    # ------------------------------------------------------------------

    def _build_preprocessing(self):
        """Lazily build PreprocessingPipeline from stored config."""
        from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
        cfg = self._preprocessing_config
        if isinstance(cfg, str):
            from stoke_ml.config import load_config as _load_cfg
            cfg = _load_cfg(cfg)
        if cfg is not None and not isinstance(cfg, dict):
            try:
                from omegaconf import OmegaConf
                cfg = OmegaConf.to_container(cfg, resolve=True)
            except Exception as exc:
                from stoke_ml.utils.error_summary import classify_error
                logger.warning(
                    "OmegaConf conversion failed (category=%s), using raw cfg",
                    classify_error(exc).value,
                )
                cfg = {}
        if isinstance(cfg, dict):
            return PreprocessingPipeline.from_config(cfg.get("preprocessing", cfg))
        return None

    @property
    def preprocessing(self):
        """Return the PreprocessingPipeline, building it lazily if needed."""
        if self._preprocessing is None and self._preprocessing_config:
            self._preprocessing = self._build_preprocessing()
        return self._preprocessing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        sentiment_df: pd.DataFrame | None = None,
        margin_df: pd.DataFrame | None = None,
        northbound_df: pd.DataFrame | None = None,
        dragon_tiger_df: pd.DataFrame | None = None,
        fundamental_df: pd.DataFrame | None = None,
        earnings_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
        etf_flow_df: pd.DataFrame | None = None,
        announcement_df: pd.DataFrame | None = None,
        guba_df: pd.DataFrame | None = None,
        comment_df: pd.DataFrame | None = None,
        capital_flow_df: pd.DataFrame | None = None,
        block_trade_df: pd.DataFrame | None = None,
        shareholder_df: pd.DataFrame | None = None,
        lockup_df: pd.DataFrame | None = None,
        dividend_df: pd.DataFrame | None = None,
        board_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
        concept_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
        industry_df: pd.DataFrame | None = None,
        limit_up_df: pd.DataFrame | None = None,  # unused hook while deferred (top scope note)
        pledge_df: pd.DataFrame | None = None,
        market_env_df: pd.DataFrame | None = None,
        index_membership_df: pd.DataFrame | None = None,
        return_dates: bool = False,
    ) -> tuple:
        """Build features for a single stock. Returns (X, y, aligned_close).

        If *return_dates* is True, also returns (sample_dates) as a 4-tuple.
        Dates track the prediction date for each sample after dropna + sequencing.
        """
        feats = self._engineer_features(
            df, sentiment_df, margin_df, northbound_df,
            dragon_tiger_df, fundamental_df, earnings_df, valuation_df, etf_flow_df,
            announcement_df, guba_df, comment_df,
            capital_flow_df, block_trade_df, shareholder_df,
            lockup_df, dividend_df, board_df, sector_df, concept_df,
            macro_df=macro_df, industry_df=industry_df,
            limit_up_df=limit_up_df, pledge_df=pledge_df,
            market_env_df=market_env_df, index_membership_df=index_membership_df,
        )
        X, y, aligned_close = self._create_sequences(feats, target_col)

        if return_dates:
            dates = self._get_sample_dates(feats)
            return X, y, aligned_close, dates

        if self.use_feature_selection and self.flat_mode and len(X) > 0:
            from stoke_ml.features.selection import FeatureSelector
            selector = FeatureSelector(mi_k=self.feature_selection_k, sfs_k=0)
            X = selector.fit_transform(X, y)

        return X, y, aligned_close

    def engineer_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
        margin_df: pd.DataFrame | None = None,
        northbound_df: pd.DataFrame | None = None,
        dragon_tiger_df: pd.DataFrame | None = None,
        fundamental_df: pd.DataFrame | None = None,
        earnings_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
        etf_flow_df: pd.DataFrame | None = None,
        announcement_df: pd.DataFrame | None = None,
        guba_df: pd.DataFrame | None = None,
        comment_df: pd.DataFrame | None = None,
        capital_flow_df: pd.DataFrame | None = None,
        block_trade_df: pd.DataFrame | None = None,
        shareholder_df: pd.DataFrame | None = None,
        lockup_df: pd.DataFrame | None = None,
        dividend_df: pd.DataFrame | None = None,
        board_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
        concept_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
        industry_df: pd.DataFrame | None = None,
        limit_up_df: pd.DataFrame | None = None,  # unused hook while deferred (top scope note)
        pledge_df: pd.DataFrame | None = None,
        market_env_df: pd.DataFrame | None = None,
        index_membership_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Engineer features for a single stock, returning the full DataFrame.

        Unlike ``build_features``, this does NOT slice into (X, y) sequences.
        It returns the raw engineered daily DataFrame suitable for saving to
        parquet and later fast loading.
        """
        return self._engineer_features(
            df, sentiment_df, margin_df, northbound_df,
            dragon_tiger_df, fundamental_df, earnings_df, valuation_df, etf_flow_df,
            announcement_df, guba_df, comment_df,
            capital_flow_df, block_trade_df, shareholder_df,
            lockup_df, dividend_df, board_df, sector_df, concept_df,
            macro_df=macro_df, industry_df=industry_df,
            limit_up_df=limit_up_df, pledge_df=pledge_df,
            market_env_df=market_env_df, index_membership_df=index_membership_df,
        )

    def save_features(
        self,
        output_path: str,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
        margin_df: pd.DataFrame | None = None,
        northbound_df: pd.DataFrame | None = None,
        dragon_tiger_df: pd.DataFrame | None = None,
        fundamental_df: pd.DataFrame | None = None,
        earnings_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
        etf_flow_df: pd.DataFrame | None = None,
        announcement_df: pd.DataFrame | None = None,
        guba_df: pd.DataFrame | None = None,
        comment_df: pd.DataFrame | None = None,
        capital_flow_df: pd.DataFrame | None = None,
        block_trade_df: pd.DataFrame | None = None,
        shareholder_df: pd.DataFrame | None = None,
        lockup_df: pd.DataFrame | None = None,
        dividend_df: pd.DataFrame | None = None,
        board_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
        concept_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
        industry_df: pd.DataFrame | None = None,
        limit_up_df: pd.DataFrame | None = None,  # unused hook while deferred (top scope note)
        pledge_df: pd.DataFrame | None = None,
        market_env_df: pd.DataFrame | None = None,
        index_membership_df: pd.DataFrame | None = None,
        panel_mode: bool = False,
    ) -> str:
        """Engineer features and save to parquet. Returns output_path.

        panel_mode=True emits panel-format features (skip_temporal + calendar)
        matching what ``build_panel_features`` consumes, so a prebuilt parquet
        can be replayed through it with identical model input semantics.
        """
        feats = self._engineer_features(
            df, sentiment_df, margin_df, northbound_df,
            dragon_tiger_df, fundamental_df, earnings_df, valuation_df, etf_flow_df,
            announcement_df, guba_df, comment_df,
            capital_flow_df, block_trade_df, shareholder_df,
            lockup_df, dividend_df, board_df, sector_df, concept_df,
            macro_df=macro_df, industry_df=industry_df,
            limit_up_df=limit_up_df, pledge_df=pledge_df,
            market_env_df=market_env_df, index_membership_df=index_membership_df,
            skip_temporal=panel_mode,
        )
        if panel_mode:
            feats = add_calendar_features(feats)
        tmp_path = f"{output_path}.tmp"
        feats.to_parquet(tmp_path, index=False, compression="lz4")
        os.replace(tmp_path, output_path)
        return output_path

    @staticmethod
    def load_features(path: str) -> pd.DataFrame:
        """Load pre-built engineered features from parquet."""
        import pandas as _pd
        return _pd.read_parquet(path)

    def build_features_from_panel(
        self,
        panel: pd.DataFrame,
        target_col: str = "close",
        *,
        cross_sectional: bool = True,
        cs_stages: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build features from a multi-stock panel with cross-sectional normalization.

        Parameters
        ----------
        panel : DataFrame from PanelBuilder
            Must have columns: date, stock_code, open, high, low, close,
            volume, sector, size_proxy.
        target_col : str
            Column to use as prediction target (default: "close").
        cross_sectional : bool
            If True, apply CrossSectionNormalizer after feature engineering.
        cs_stages : list[str] or None
            Stages for CrossSectionNormalizer. Default: ["sector", "size", "rank"].

        Returns
        -------
        X : ndarray (n_total_samples, seq_len, n_features) or (n_total_samples, n_features*seq_len)
        y : ndarray (n_total_samples,)
        aligned_close : ndarray (n_total_samples+1,)
        stock_indices : ndarray (n_total_samples,) int
            Maps each sample back to its stock index in panel["stock_code"].unique().
        """
        if panel.empty:
            empty = np.array([], dtype=np.float32)
            return empty, np.array([], dtype=np.int64), empty, np.array([], dtype=np.int64)

        codes = sorted(panel["stock_code"].unique())

        # 1. Engineer features per stock
        engineered_frames: list[pd.DataFrame] = []
        for code in codes:
            mask = panel["stock_code"] == code
            df_stock = panel[mask].copy()
            feats = self._engineer_features(df_stock)
            engineered_frames.append(feats)

        # 2. Recombine into panel
        feats_panel = pd.concat(engineered_frames, ignore_index=True)
        feats_panel = feats_panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

        # 3. Cross-sectional normalization on the feature panel
        if cross_sectional:
            from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer
            csn = CrossSectionNormalizer(
                enabled=True,
                stages=cs_stages or ["sector", "size", "rank"],
            )
            feats_panel = csn.fit_transform(feats_panel)
            logger.info(
                "CrossSectionNormalizer fit range: %s → %s "
                "(per-date stats, PIT-safe)",
                csn.fit_start, csn.fit_end,
            )

        # 4. Create sequences per stock, track stock origin
        X_parts, y_parts, close_parts, idx_parts = [], [], [], []
        for i, code in enumerate(codes):
            mask = feats_panel["stock_code"] == code
            df_stock = feats_panel[mask].sort_values("date").reset_index(drop=True)
            X_s, y_s, close_s = self._create_sequences(df_stock, target_col)
            if len(X_s) > 0:
                X_parts.append(X_s)
                y_parts.append(y_s)
                close_parts.append(close_s)
                idx_parts.append(np.full(len(X_s), i, dtype=np.int64))

        if not X_parts:
            empty = np.array([], dtype=np.float32)
            return empty, np.array([], dtype=np.int64), empty, np.array([], dtype=np.int64)

        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)
        close_all = np.concatenate(close_parts, axis=0)
        stock_idx = np.concatenate(idx_parts, axis=0)

        # 5. Optional: feature selection on the combined dataset
        if self.use_feature_selection and self.flat_mode and len(X_all) > 0:
            from stoke_ml.features.selection import FeatureSelector
            selector = FeatureSelector(mi_k=self.feature_selection_k, sfs_k=0)
            X_all = selector.fit_transform(X_all, y_all)

        return X_all, y_all, close_all, stock_idx

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
        margin_df: pd.DataFrame | None = None,
        northbound_df: pd.DataFrame | None = None,
        dragon_tiger_df: pd.DataFrame | None = None,
        fundamental_df: pd.DataFrame | None = None,
        earnings_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
        etf_flow_df: pd.DataFrame | None = None,
        announcement_df: pd.DataFrame | None = None,
        guba_df: pd.DataFrame | None = None,
        comment_df: pd.DataFrame | None = None,
        capital_flow_df: pd.DataFrame | None = None,
        block_trade_df: pd.DataFrame | None = None,
        shareholder_df: pd.DataFrame | None = None,
        lockup_df: pd.DataFrame | None = None,
        dividend_df: pd.DataFrame | None = None,
        board_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
        concept_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
        industry_df: pd.DataFrame | None = None,
        limit_up_df: pd.DataFrame | None = None,  # unused hook while deferred (top scope note)
        pledge_df: pd.DataFrame | None = None,
        market_env_df: pd.DataFrame | None = None,
        index_membership_df: pd.DataFrame | None = None,
        skip_temporal: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Preserve the K-line pct_change column across technical computation.
        # compute_all drops it as an intermediate column; restoring the daily
        # value here keeps pct_change as a same-day PK feature (no lag) and
        # prevents aux merges from injecting their own (possibly stale) one.
        _pct_change = df["pct_change"].copy() if "pct_change" in df.columns else None

        if self.use_new_preprocessing and self.preprocessing:
            df = self.preprocessing.run("numeric", df)

        # 1. Technical indicators + scoring + microstructure (no aux dependency)
        if self.use_technical:
            df = self._ti.compute_all(df)
        if _pct_change is not None:
            df["pct_change"] = _pct_change.fillna(0.0).astype(np.float32)
        if self.use_scoring:
            df = self._scorer.score(df)

        df = self._add_microstructure(df)

        if self.minute_mode:
            if self._intraday is None:
                from stoke_ml.features.minute_technical import MinuteIntradayFeatures
                self._intraday = MinuteIntradayFeatures()
            df = self._intraday.compute_all(df)

        # 2. Merge aux DataFrames (expanded PO columns).  The per-dimension
        # merge family lives in AuxAligner (aux_aligner.py, §十七-1).
        df = self._aux.merge_all(
            df,
            sentiment=sentiment_df,
            announcements=announcement_df,
            margin=margin_df,
            northbound=northbound_df,
            dragon_tiger=dragon_tiger_df,
            fundamental=fundamental_df,
            earnings=earnings_df,
            valuation=valuation_df,
            etf_flow=etf_flow_df,
            guba=guba_df,
            comment=comment_df,
            capital_flow=capital_flow_df,
            block_trade=block_trade_df,
            shareholder=shareholder_df,
            lockup=lockup_df,
            dividend=dividend_df,
            board=board_df,
            sector=sector_df,
            concept=concept_df,
            macro=macro_df,
            industry=industry_df,
            # limit_up is DEFERRED (limit-up ecology family, top scope note) —
            # the merge method exists but is intentionally NOT wired.
            pledge=pledge_df,
            market_env=market_env_df,
            index_membership=index_membership_df,
        )

        # Interaction features require merged sentiment columns — must run
        # after the aux merges (was previously a silent no-op).
        if self.use_interaction:
            df = self._interaction.compute_all(df)

        # Defragment after merge calls
        df = df.copy()

        # 3. Emotion refinement (news + guba sentiment features)
        if self._emotion_refiner is not None:
            df = self._emotion_refiner.refine(df)

        # 4. Per-stock fundamental refinement (quality, stability, valuation, trends)
        if self._fundamental_refiner is not None:
            df = self._fundamental_refiner.refine(df)

        # 4b. Market-environment factors (macro composite + regime score)
        if self._market_env_refiner is not None:
            df = self._market_env_refiner.refine(df)

        # 5. Temporal statistics on PO columns (replaces add_rolling_features)
        if self._temporal_transformer is not None:
            df = self._temporal_transformer.transform(df)

        # 6. Lag features + calendar (PO rolling handled by TemporalTransformer)
        if self.use_temporal and not skip_temporal:
            temporal_cols = list(TEMPORAL_BASE_COLS)
            temporal_cols += _active_cols(df, [
                "sentiment_mean", "sentiment_std",
                "positive_ratio", "negative_ratio",
            ])
            temporal_cols += _active_cols(df, [
                "ann_sentiment_mean", "ann_sentiment_std",
                "ann_positive_ratio", "ann_negative_ratio",
            ])
            temporal_cols += _active_cols(df, (
                MARGIN_COLS + NORTHBOUND_COLS + DRAGON_TIGER_COLS
            ))
            temporal_cols += _active_cols(df, FUNDAMENTAL_COLS)
            temporal_cols += _active_cols(df, VALUATION_COLS)
            temporal_cols += _active_cols(df, ETF_FLOW_COLS)
            temporal_cols += _active_cols(df, GUBA_COLS)
            temporal_cols += _active_cols(df, COMMENT_COLS)
            temporal_cols += _active_cols(df, FLOW_COLS)
            temporal_cols += _active_cols(df, BLOCK_TRADE_COLS)
            temporal_cols += _active_cols(df, SHAREHOLDER_COLS)
            temporal_cols += _active_cols(df, LOCKUP_COLS)
            temporal_cols += _active_cols(df, DIVIDEND_COLS)
            temporal_cols += _active_cols(df, BOARD_COLS)
            temporal_cols += _active_cols(df, SECTOR_COLS)
            temporal_cols += _active_cols(df, CONCEPT_COLS)
            temporal_cols += _active_cols(df, MACRO_COLS)
            temporal_cols += _active_cols(df, INDUSTRY_COLS)
            temporal_cols += _active_cols(df, LIMIT_UP_COLS)
            temporal_cols += _active_cols(df, PLEDGE_COLS)
            temporal_cols += _active_cols(df, INDEX_MEMBER_COLS)
            temporal_cols += _active_cols(df, MARKET_ENV_COLS)
            temporal_cols += _active_cols(df, DRAGON_TIGER_SEAT_COLS)
            temporal_cols += _active_cols(df, [
                c for c in df.columns if c.startswith("menv_")
            ])
            # Dynamic PO columns
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith("momentum_") or c.startswith("concept_momentum_")
                or c.startswith("board_momentum_") or c.startswith("sector_rrg_")
                or c.startswith("seal_type_") or c.startswith("market_state_")
                or c.startswith("cb_")
            ])
            # Text features from preprocessing chains
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.endswith("_bipolar_sent") or c.endswith("_agreement")
                or c.endswith("_attention") or c.endswith("_weighted_sent")
                or c in ("bipolar_sent", "agreement", "attention", "weighted_sent")
            ])
            # Emotion refinement outputs
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith("news_") or c.startswith("guba_")
                or c in (
                    "news_guba_divergence", "news_guba_ratio",
                    "total_attention", "cross_source_agreement", "retail_panic",
                )
            ])
            # Fundamental refinement outputs
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith(("f_score", "quality_", "earnings_", "profitability_",
                                 "margin_stability", "growth_quality", "pe_", "pb_",
                                 "deep_value", "roe_", "revenue_", "margin_trend",
                                 "earnings_surprise"))
                or c in ("pe_pb_divergence",)
            ])
            df = add_lag_features(df, temporal_cols, self.LAGS)
            df = add_calendar_features(df)

        return df

    # ------------------------------------------------------------------
    # Microstructure features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features from OHLCV data.

        Computes limit-up/down signals and seal quality proxies from K-line
        alone — no dependency on limit-up pool data (which only covers ~2
        weeks via EastMoney push2ex).
        """
        df = df.copy()
        close = df.get("close")
        _open = df.get("open")
        high = df.get("high")
        low = df.get("low")
        volume = df.get("volume")
        if close is None:
            return df

        prev_close = close.shift(1)

        # Limit up/down (A-share: ±10% daily limit; STAR/GEM: ±20%)
        pct = (close - prev_close) / prev_close.replace(0, np.nan)
        df["is_limit_up"] = (pct >= 0.098).astype(np.float32)
        df["is_limit_down"] = (pct <= -0.098).astype(np.float32)

        # Gap open
        if _open is not None:
            gap = (_open - prev_close) / prev_close.replace(0, np.nan)
            df["gap_up_pct"] = gap.clip(lower=0).fillna(0).astype(np.float32)
            df["gap_down_pct"] = (-gap).clip(lower=0).fillna(0).astype(np.float32)

        # Seal quality proxies (no pool data needed)
        if high is not None and low is not None:
            is_up = df["is_limit_up"] > 0
            # One-word board (一字板): limit-up with open==high==low==close
            df["is_one_word_board"] = (
                is_up & (_open == high) & (high == low) & (low == close)
            ).astype(np.float32)
            # Seal quality on limit-up days: close/high ratio
            # 1.0 = sealed at day high (strong), < 1.0 = retreated from high
            seal_q = np.where(
                is_up & (high > 0),
                (close / high.replace(0, np.nan)).clip(0, 1),
                0.0,
            )
            df["seal_quality"] = seal_q.astype(np.float32)

        # Volume anomaly: ratio of current volume to 20-day median
        if volume is not None:
            vol_med = volume.shift(1).rolling(20, min_periods=5).median()
            df["volume_ratio_20"] = (volume / vol_med.replace(0, np.nan)).clip(0, 20)
            df["volume_ratio_20"] = df["volume_ratio_20"].fillna(1.0).astype(np.float32)

            # Turnover anomaly flag: volume > 3x 20-day median
            df["volume_anomaly"] = (df["volume_ratio_20"] > 3.0).astype(np.float32)

        # Consecutive limit-up streak
        df["limit_up_streak"] = (
            df["is_limit_up"]
            .groupby((df["is_limit_up"] == 0).cumsum())
            .cumsum()
            .astype(np.float32)
        )

        # Rolling limit-up count (short-term momentum proxy)
        df["limit_up_count_5d"] = (
            df["is_limit_up"].rolling(5, min_periods=1).sum().astype(np.float32)
        )
        df["limit_up_count_20d"] = (
            df["is_limit_up"].rolling(20, min_periods=5).sum().astype(np.float32)
        )

        return df

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    def _prep_feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop metadata/dead columns and rows with inf/NaN — shared by sequencing methods."""
        drop_cols = ["stock_code", "sector", "sector_code", "size_proxy"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if self.drop_dead_features:
            # Per-stock constancy: a column constant across the stock's whole
            # series is constant in every walk-forward prefix, so dropping it
            # here is leak-free — the full series never decides a column that
            # varies (those are always kept).
            nuniq = feat_df.nunique()
            dead = [
                c for c, u in nuniq.items()
                if u <= 1 and not c.startswith(SPARSE_KEEP_PREFIXES)
            ]
            if dead:
                feat_df = feat_df.drop(columns=dead)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        return feat_df.dropna()

    def _create_sequences(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat_df = self._prep_feature_df(df)

        close = feat_df[target_col].values
        ret = (close[self.horizon:] - close[: -self.horizon]) / (close[: -self.horizon] + 1e-8)
        target = np.where(ret > 0.003, 2, np.where(ret < -0.003, 0, 1))

        price_cols = ["open", "high", "low", "close", "date"]
        X_cols = [c for c in feat_df.columns if c not in price_cols]
        X_data = feat_df[X_cols].values.astype(np.float32)

        n_samples = len(X_data) - self.seq_len - self.horizon + 1
        if n_samples <= 0:
            empty = np.array([], dtype=np.float32)
            return empty, np.array([], dtype=np.int64), empty

        if self.flat_mode:
            X = np.array([
                X_data[i: i + self.seq_len].flatten()
                for i in range(n_samples)
            ], dtype=np.float32)
        else:
            X = np.array([
                X_data[i: i + self.seq_len]
                for i in range(n_samples)
            ], dtype=np.float32)

        y = target[self.seq_len - 1: self.seq_len - 1 + n_samples]
        # aligned_close steps by horizon so diff(aligned_close)[k] equals the
        # horizon-day return of the NON-OVERLAPPING sample k*horizon (y[k*horizon]),
        # not y[k]. For horizon=1 this reduces to consecutive closes and pairs
        # with every prediction. For horizon>1 the overlapping windows mean
        # n_samples horizon returns cannot be derived from n_samples+1 prices,
        # so compute_financial_metrics(prices, predictions) is only correct for
        # horizon=1; multi-horizon financial metrics are unsupported
        # (see audit-doc follow-up 2026-08-02).
        aligned_close = close[self.seq_len - 1 :: self.horizon][: n_samples + 1]
        if len(aligned_close) < n_samples + 1:
            aligned_close = np.concatenate([
                aligned_close,
                np.full(n_samples + 1 - len(aligned_close), close[-1]),
            ])

        return X, y, aligned_close.astype(np.float32)

    def _get_sample_dates(self, feats: pd.DataFrame) -> np.ndarray:
        """Return the prediction date for each sample after dropna + sequencing.

        Uses the same row filtering as _create_sequences via _prep_feature_df.
        """
        feat_df = self._prep_feature_df(feats)
        valid_dates = feat_df["date"].values
        if len(valid_dates) < self.seq_len + self.horizon:
            return np.array([], dtype="datetime64[ns]")
        n_samples = len(valid_dates) - self.seq_len - self.horizon + 1
        # Sample i predicts return ending at valid_dates[seq_len-1+i+horizon]
        return valid_dates[self.seq_len - 1 + self.horizon:
                           self.seq_len - 1 + self.horizon + n_samples]

    # ------------------------------------------------------------------
    # Dynamic column discovery (replaces hardcoded PK/PO lists)
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_pk_columns(df: pd.DataFrame) -> list[str]:
        """Auto-discover past-known columns from a reference DataFrame.

        PK columns come from: OHLCV, technical indicators, scoring,
        microstructure, calendar, intraday, and fundamental refinements.
        Everything else is PO.
        """
        pk_prefixes = [
            "open", "high", "low", "close", "volume", "amount",
            "ma_", "ema_", "macd_", "rsi_", "kdj_", "boll_", "atr_",
            "roc_", "wr_", "cci_", "vol_", "volume_", "amount_",
            "kmid", "klen", "kup", "klow", "ksft",
            "open0", "high0", "low0",
            "adx", "adxr", "pdi", "mdi",
            "mfi_", "cmo_", "trix",
            "obv", "turnover",
            "is_limit_", "gap_", "volume_anomaly", "limit_up_",
            "is_one_word", "seal_quality",
            "day_of_", "month", "quarter",
            "minutes_", "is_am_", "is_pm_", "session_", "bar_of_", "opening_",
            "session_high",
            "trend_level", "bias_", "buy_signal",
            "pct_change",
            "max_", "min_", "qtlu_", "qtld_", "rank_", "rsv_",
            "corr_", "cord_", "beta_", "rsqr_", "resi_",
            "vma_", "vstd_",
            "cntp_", "cntn_", "cntd_",
            "sump_", "sumn_", "sumd_",
            "imax_", "imin_", "imxd_",
            "wvma_", "vsump_", "vsumn_", "vsumd_",
            "interaction_",
            "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
            "debt_ratio", "gross_margin", "net_margin",
            "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm",
            "f_score", "quality_composite", "earnings_quality",
            "profitability_stability", "margin_stability", "growth_quality",
            "pe_percentile_", "pb_percentile_", "pe_pb_divergence", "deep_value",
            "roe_trend_", "revenue_trend_", "margin_trend_", "roe_accel",
            "earnings_surprise",
            "pe_sector_ratio", "pb_sector_ratio", "ps_sector_ratio",
            "leverage_warning", "valuation_composite_z",
        ]
        pk = []
        for col in df.columns:
            if col in ("date", "stock_code", "sector", "size_proxy", "sector_code"):
                continue
            for prefix in pk_prefixes:
                if col == prefix or col.startswith(prefix):
                    pk.append(col)
                    break
        return pk

    @staticmethod
    def _discover_po_columns(df: pd.DataFrame, pk_set: set[str] | None = None) -> list[str]:
        """Auto-discover PO columns — everything not PK and not metadata."""
        if pk_set is None:
            pk_set = set(FeaturePipeline._discover_pk_columns(df))
        skip = {"date", "stock_code", "sector", "size_proxy", "sector_code"}
        return [c for c in df.columns if c not in pk_set and c not in skip]

    def _clean_calendar_dates(self, df: pd.DataFrame, stock_code: str):
        """Keep a stock's date axis calendar-clean before the panel
        UNION date axis is built from it.

        One wrong weekend/closed-day bar, or a duplicated/out-of-order date,
        would add a phantom column to the global panel calendar and corrupt
        cross-sectional alignment for every stock.  Repair in place: drop
        unparsable dates, de-dup (keep the last row, matching storage
        overwrite semantics), drop dates that are not official trading days,
        then re-sort so the surviving dates are unique and strictly increasing.
        Returns the cleaned frame, or None when nothing valid remains.
        """
        if df is None or len(df) == 0:
            return None
        d = pd.to_datetime(df["date"])
        unparsable = d.isna()
        if unparsable.any():
            logger.warning(
                "Panel %s: dropping %d row(s) with unparsable date",
                stock_code, int(unparsable.sum()),
            )
            df = df[~unparsable]
            d = d[~unparsable]
            if len(df) == 0:
                return None
        dup = d.duplicated(keep="last")
        trading = set(_get_panel_calendar().get_trading_days(
            d.min().date(), d.max().date()))
        off_cal = ~np.array([x.date() in trading for x in d])
        bad = dup | off_cal
        n_bad = int(bad.sum())
        if n_bad:
            examples = [
                pd.Timestamp(x).date().isoformat() for x in d[bad].head(8)
            ]
            logger.warning(
                "Panel %s: dropping %d date-invalid row(s) "
                "(duplicate or not an official trading day; e.g. %s)",
                stock_code, n_bad, examples,
            )
            df = df[~bad]
            if len(df) == 0:
                return None
        return df.sort_values("date").reset_index(drop=True)

    def build_panel_features(
        self,
        panel: pd.DataFrame,
        target_col: str = "close",
        aux_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        horizon: int = 1,
        prebuilt_dir: str | None = None,
        min_history: int | None = None,
        require_feature_manifest: bool = False,
    ) -> dict:
        """Build panel-format features for VSN+xLSTM training from a multi-stock panel.

        The input panel must have columns: date, stock_code, open, high, low,
        close, volume (plus any auxiliary feature columns already merged).

        Args:
            panel: multi-stock DataFrame with columns date, stock_code, OHLCV.
            target_col: column name for close price.
            aux_data: optional dict stock_code → {aux_type: DataFrame}.
                      aux_type keys: "sentiment", "guba", "comment",
                      "announcement", "margin", "northbound", "dragon_tiger",
                      "fundamental", "etf_flow", "capital_flow", "block_trade",
                      "shareholder", "lockup", "dividend", "board", "sector", "concept".
            horizon: forward return horizon in days (1/5/20). Direction
                     threshold scales as 0.003 * sqrt(horizon).
            prebuilt_dir: optional dir of panel-mode feature parquets
                      (``save_features(panel_mode=True)``).  When set, per-stock
                      features are loaded from ``{prebuilt_dir}/{code}.parquet``
                      instead of being engineered live; ``aux_data`` is ignored.
                      Parquets must be built with the SAME ``--use-*`` flags, but
                      column SETS may legitimately differ per stock (merge
                      methods skip columns a stock has no data for) — those gaps
                      are reconciled by the all_cols ZI-alignment block below.
            require_feature_manifest: when True and prebuilt_dir is set, FAIL
                      (raise) instead of warning if any sidecar manifest is
                      missing/stale/schema-drifted or built by a different git
                      commit (or no ``.manifests/`` exists at all).  Formal
                      training passes the CLI's ``--require-feature-manifest``
                      (default on); legacy prebuilt dirs / unit tests pass False
                      to keep warn-only behavior.

        Returns:
            dict with numpy arrays: static_features, past_known, past_observed,
            y_direction, y_return, y_volatility.
        """
        codes = sorted(panel["stock_code"].unique())
        aux_data = aux_data or {}
        if min_history is None:
            min_history = self.min_history

        if prebuilt_dir:
            # A missing prebuilt parquet would otherwise surface as a bare
            # FileNotFoundError mid-loop (or an empty frame corrupting the
            # panel).  Drop missing stocks up front; fail loudly if the dir
            # holds nothing usable at all.
            prebuilt_paths = {
                c: os.path.join(prebuilt_dir, f"{c}.parquet") for c in codes
            }
            missing = [c for c, p in prebuilt_paths.items() if not os.path.isfile(p)]
            if len(missing) == len(codes):
                raise FileNotFoundError(
                    f"No prebuilt feature parquets found in {prebuilt_dir}"
                )
            if missing:
                if require_feature_manifest:
                    raise FileNotFoundError(
                        f"prebuilt_dir {prebuilt_dir}: {len(missing)}/{len(codes)} "
                        f"feature parquets missing (first 20: {missing[:20]}). "
                        f"Regenerate with build_features.py before a formal run "
                        f"(--no-require-feature-manifest to override)."
                    )
                logger.warning(
                    "prebuilt_dir missing %d/%d parquets (first 20: %s); "
                    "dropping those stocks from the panel",
                    len(missing), len(codes), missing[:20],
                )
                codes = [c for c in codes if c not in set(missing)]

            # Lineage guard: surface prebuilt features that lack a
            # sidecar manifest, or whose manifest no longer matches the file
            # (schema drift) or the current code (built by another git commit).
            # require_feature_manifest makes these FAIL — silently reusing
            # unverified/stale features would corrupt a formal experiment —
            # while warn-only keeps legacy un-manifested dirs trainable.
            manifest_dir = os.path.join(prebuilt_dir, ".manifests")
            missing_manifest: list[str] = []
            stale_manifest: list[str] = []
            if os.path.isdir(manifest_dir) and os.listdir(manifest_dir):
                commit = cache_manifest.git_head()
                # §十一-3: config.yaml can change under the SAME git commit
                # (or outside git entirely) — compare the recorded config_hash
                # against the current config snapshot too.  None when config
                # cannot load → comparison skipped.
                cfg_hash = cache_manifest.current_config_hash()
                for code in codes:
                    mp = os.path.join(manifest_dir, f"{code}.json")
                    if not os.path.isfile(mp):
                        missing_manifest.append(code)
                        continue
                    try:
                        with open(mp, encoding="utf-8") as f:
                            m = json.load(f)
                    except Exception as exc:
                        from stoke_ml.utils.error_summary import classify_error
                        logger.warning(
                            "manifest %s unreadable (category=%s), marking stale",
                            mp, classify_error(exc).value,
                        )
                        stale_manifest.append(code)
                        continue
                    if (
                        m.get("feature_schema_hash")
                        != cache_manifest.schema_hash(
                            os.path.join(prebuilt_dir, f"{code}.parquet")
                        )
                        or m.get("git_commit") != commit
                        or (cfg_hash is not None and m.get("config_hash") != cfg_hash)
                    ):
                        stale_manifest.append(code)
            else:
                # No .manifests/ at all: every stock is unverifiable, so both
                # the warn path and the require path speak the same language.
                missing_manifest = list(codes)

            if require_feature_manifest and (missing_manifest or stale_manifest):
                raise RuntimeError(
                    f"prebuilt_dir {prebuilt_dir}: feature-manifest check FAILED "
                    f"({len(missing_manifest)} missing, {len(stale_manifest)} "
                    f"stale — schema drift, built by a different git commit, or "
                    f"built with a different config; first missing: "
                    f"{missing_manifest[:5]}, first stale: {stale_manifest[:5]}). "
                    f"Regenerate with build_features.py --panel-mode before a "
                    f"formal run (--no-require-feature-manifest to override)."
                )
            if missing_manifest:
                logger.warning(
                    "prebuilt_dir %s: %d/%d stocks lack sidecar manifests "
                    "(first 10: %s) — regenerate with build_features.py for "
                    "verifiable lineage",
                    prebuilt_dir, len(missing_manifest), len(codes),
                    missing_manifest[:10],
                )
            if stale_manifest:
                logger.warning(
                    "prebuilt_dir %s: %d/%d stocks have STALE manifests "
                    "(schema drift, different git commit, or config drift; "
                    "first 10: %s) — rebuild features before trusting "
                    "training output",
                    prebuilt_dir, len(stale_manifest), len(codes),
                    stale_manifest[:10],
                )

        N = len(codes)

        # Engineer features per stock (reuses existing pipeline)
        all_feat_dfs = []
        for code in codes:
            if prebuilt_dir:
                path = os.path.join(prebuilt_dir, f"{code}.parquet")
                feats = self.load_features(path)
                feats["date"] = pd.to_datetime(feats["date"])
                # Flat prebuilt (data/features/) carries temporal lag columns
                # (skip_temporal=False).  Panel training uses skip_temporal=True
                # (xLSTM learns the time structure itself), so drop *_lag{N}
                # columns — the remainder matches a --panel-mode build.
                lag_cols = [c for c in feats.columns if re.search(r"_lag\d+$", c)]
                if lag_cols:
                    feats = feats.drop(columns=lag_cols)
                # A stale/hand-built parquet may carry a
                # weekend/duplicate bar that would pollute the UNION date axis.
                feats = self._clean_calendar_dates(feats, code)
                if feats is None:
                    continue
                # Calendar features are idempotent (overwrite in place); safe
                # to re-apply even though save_features(panel_mode=True) already
                # added them — guards against hand-built parquets.
                feats = add_calendar_features(feats)
                # Schema note: parquets must be built with the SAME --use-*
                # flags (build_features.py).  Column SETS still legitimately
                # differ per stock — merge methods skip columns when a stock
                # has no data for a sparse aux type (block_trade, dividend,
                # valuation, ...).  No strict equality check here: those gaps
                # are reconciled by the all_cols ZI-alignment block after the
                # loop, and PK/PO columns are discovered BY NAME (never by
                # position), so a missing column simply becomes all-zero.
            else:
                mask = panel["stock_code"] == code
                df_stock = panel[mask].sort_values("date").reset_index(drop=True)
                # Drop phantom/duplicate/out-of-calendar rows before
                # feature engineering so a bad bar neither pollutes the UNION
                # date axis nor corrupts the rolling indicators around it.
                df_stock = self._clean_calendar_dates(df_stock, code)
                if df_stock is None:
                    continue
                stock_aux = aux_data.get(code, {})
                feats = self._engineer_features(
                    df_stock,
                    sentiment_df=stock_aux.get("sentiment"),
                    guba_df=stock_aux.get("guba"),
                    comment_df=stock_aux.get("comment"),
                    announcement_df=stock_aux.get("announcement"),
                    margin_df=stock_aux.get("margin"),
                    northbound_df=stock_aux.get("northbound"),
                    dragon_tiger_df=stock_aux.get("dragon_tiger"),
                    fundamental_df=stock_aux.get("fundamental"),
                    valuation_df=stock_aux.get("valuation"),
                    etf_flow_df=stock_aux.get("etf_flow"),
                    capital_flow_df=stock_aux.get("capital_flow"),
                    block_trade_df=stock_aux.get("block_trade"),
                    shareholder_df=stock_aux.get("shareholder"),
                    lockup_df=stock_aux.get("lockup"),
                    dividend_df=stock_aux.get("dividend"),
                    board_df=stock_aux.get("board"),
                    sector_df=stock_aux.get("sector"),
                    concept_df=stock_aux.get("concept"),
                    skip_temporal=True,  # xLSTM learns temporal patterns natively
                )
                # Calendar features are normally added by the temporal path;
                # we still want them when skip_temporal=True (panel model benefits
                # from day-of-week/month/quarter signals for seasonality).
                feats = add_calendar_features(feats)
            # Defragment after many df["col"] = ... assignments in merge methods.
            # Without this, pandas emits PerformanceWarning and slows down
            # subsequent operations.
            feats = feats.copy()
            all_feat_dfs.append(feats)

        # ── Compute targets from RAW close BEFORE cross-sectional normalization ──
        # Cross-sectional z-score normalization mutates close (and all PK/PO
        # columns) to relative-value space.  Targets MUST be computed from raw
        # prices — using z-score changes as returns distorts the signal.
        # ── Global trading-calendar alignment ──
        # Every stock is aligned to the UNION of all stock dates (sorted), so
        # array column t is the SAME calendar date for every stock.  Without
        # this, a short-history stock would start at position 0 and its column
        # t would be a different date than a long-history stock's column t —
        # corrupting cross-sectional IC / Top-K / long-short evaluation (which
        # index by column) and walk-forward fold boundaries.
        all_dates = sorted({d for df in all_feat_dfs for d in pd.to_datetime(df["date"])})
        # §九-1 defensive invariant: the panel time axis MUST be a subset of the
        # official A-share trading calendar.  `_clean_calendar_dates` enforces
        # this per stock on both entry paths and the merge methods only left-join
        # aux data onto the K-line axis, so an off-calendar date surviving to the
        # UNION signals an upstream regression — fail loudly instead of silently
        # widening column t (the global calendar column) for every stock.
        if all_dates:
            _cal = _get_panel_calendar()
            _official = set(_cal.get_trading_days(
                all_dates[0].date(), all_dates[-1].date()))
            _off = [d.strftime("%Y-%m-%d") for d in all_dates
                    if d.date() not in _official]
            if _off:
                raise ValueError(
                    "panel union axis contains dates that are not in the "
                    f"official a_shares trading calendar: "
                    f"{_off[:10]}{' ...' if len(_off) > 10 else ''}")
        max_T = len(all_dates)
        global_dates = np.array(all_dates, dtype="datetime64[ns]")
        # `all_dates` holds pandas Timestamps (which have .date()); iterating
        # the numpy global_dates array would yield datetime64 scalars instead.
        date_to_pos = {str(d.date()): i for i, d in enumerate(all_dates)}

        N_stocks = len(all_feat_dfs)
        y_dir_arr = np.full((N_stocks, max_T), -100, dtype=np.int64)  # CE ignore_index
        y_ret_arr = np.zeros((N_stocks, max_T), dtype=np.float32)
        y_vol_arr = np.zeros((N_stocks, max_T), dtype=np.float32)
        stock_T = np.zeros(N_stocks, dtype=np.int32)

        # ── Per-task target masks ──
        # One `y_direction != -100` cannot carry four distinct
        # jobs — "tradable today", "clean label exists", "loss applies here",
        # "portfolio P&L computable".  Split them:
        #   obs_arr        — real close at t (base observation / history count)
        #   entry_arr      — real open at t → can enter a position at open[t]
        #   ret_tgt_arr    — clean forward return open[t+h]/open[t]-1 available
        #   vol_tgt_arr    — vol window (t, t+h] has >=2 valid daily returns
        #   realized_arr   — evaluation P&L: clean open return, else carry to the
        #                    last real close in (t, t+h], else 0 (flat) — so a
        #                    stock that suspends/delists AFTER entry still counts
        #                    and the Top-K pool never conditions on the future.
        obs_arr = np.zeros((N_stocks, max_T), dtype=bool)
        entry_arr = np.zeros((N_stocks, max_T), dtype=bool)
        ret_tgt_arr = np.zeros((N_stocks, max_T), dtype=bool)
        vol_tgt_arr = np.zeros((N_stocks, max_T), dtype=bool)
        realized_arr = np.zeros((N_stocks, max_T), dtype=np.float32)
        # Daily price paths for sleeve-account evaluation: the
        # realized_return array is per-ENTRY-day open-to-exit, which cannot
        # reconstruct a true daily mark-to-market series.  Expose the raw
        # close/open paths (NaN outside a stock's trading days) so evaluate.py
        # can build chronological sleeve daily returns + exit_status.
        close_price_arr = np.full((N_stocks, max_T), np.nan, dtype=np.float32)
        open_price_arr = np.full((N_stocks, max_T), np.nan, dtype=np.float32)

        # Row i of a stock's feature df → its column on the global calendar.
        # Computed once here and reused by the feature-array scatter below.
        stock_pos: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(N_stocks)]

        # Raw PIT-static inputs: trailing 60d means of close and
        # turnover (canonical `amount`), plus first-listed global
        # column — captured HERE because the per-date z-score normalization
        # later mutates the feature dfs.
        price60_raw = np.zeros((N_stocks, max_T), dtype=np.float32)
        amt60_raw = np.zeros((N_stocks, max_T), dtype=np.float32)
        first_col = np.full(N_stocks, -1, dtype=np.int32)
        # The canonical `amount` (real CNY turnover) is REQUIRED by the formal
        # daily contract — a panel without it raises above (§十一-5).  Every
        # stock here has it, so every stock qualifies for the liquidity floor.
        has_amount_arr = np.ones(N_stocks, dtype=bool)

        # Direction noise threshold — scale by sqrt(horizon)
        # (0.003 per day, 1.0% / 5-day, 1.3% / 20-day)
        dir_threshold = 0.003 * (horizon ** 0.5)

        for i, df in enumerate(all_feat_dfs):
            if len(df) == 0:
                continue
            df_sorted = df.sort_values("date").reset_index(drop=True)
            dates = pd.to_datetime(df_sorted["date"])
            pos = np.array([date_to_pos[str(d.date())] for d in dates], dtype=np.int32)
            stock_pos[i] = pos
            T_i = len(pos)
            stock_T[i] = T_i
            # Trading-time convention: features up to close[t]
            # (window's last column end-1) → signal after close[t] → ENTER at
            # open[end]=open[t+1] → hold h days → EXIT at open[end+h].  Labels
            # are therefore open-to-open; entry eligibility needs a real open.
            close_full = np.full(max_T, np.nan, dtype=np.float64)
            close_full[pos] = df_sorted[target_col].to_numpy(dtype=np.float64)
            open_col = "open" if "open" in df_sorted.columns else target_col
            open_full = np.full(max_T, np.nan, dtype=np.float64)
            open_full[pos] = df_sorted[open_col].to_numpy(dtype=np.float64)
            # Row-level quality = REPAIR/MASK, not stock
            # ejection.  A non-positive price is DATA-MISSING (a dead data row,
            # indistinguishable from a delisting) — mask it
            # like a suspension so it never becomes a training observation or an
            # entry, instead of ejecting the whole stock because of one bad row.
            close_valid = ~np.isnan(close_full) & (close_full > 0)
            open_valid = ~np.isnan(open_full) & (open_full > 0)
            obs_arr[i] = close_valid
            entry_arr[i] = open_valid
            close_price_arr[i] = close_full.astype(np.float32)
            open_price_arr[i] = open_full.astype(np.float32)

            # PIT static raw inputs — trailing 60d means over the trading days
            # in each global-calendar window (NaNs from pre-listing/suspension
            # are skipped).  Computed here on the RAW df before z-scoring.
            price60_raw[i] = _trailing_mean(close_full, 60).astype(np.float32)
            # The formal daily contract requires canonical CNY turnover
            # (`amount`, real 成交额).  volume×qfq-close misstates historical
            # nominal turnover because qfq prices are rescaled while volume is
            # not.  Fail loudly rather than silently substituting a proxy that
            # is not a real turnover measure (§十一-5).
            if "amount" not in df_sorted.columns:
                raise ValueError(
                    f"Stock {codes[i]}: daily K-line lacks canonical `amount` — "
                    "the formal daily contract requires it (§十一-5); no "
                    "volume×close / price fallback."
                )
            has_amount_arr[i] = True
            amt_full = np.full(max_T, np.nan, dtype=np.float64)
            amt_full[pos] = df_sorted["amount"].to_numpy(dtype=np.float64)
            amt60_raw[i] = _trailing_mean(amt_full, 60).astype(np.float32)
            first_col[i] = int(pos[0]) if len(pos) else -1

            # Clean forward return (training label): open[t] and open[t+h] both
            # real.  Positions without one stay NaN → direction -100 / return 0
            # with ret_tgt_arr False so training ignores them.
            ret_fwd = np.full(max_T, np.nan, dtype=np.float32)
            if max_T > horizon:
                both = open_valid[:-horizon] & open_valid[horizon:]
                num = open_full[horizon:][both] - open_full[:-horizon][both]
                ret_fwd[:max_T - horizon][both] = (num / (open_full[:-horizon][both] + 1e-8)).astype(np.float32)
            ret_tgt_arr[i] = np.isfinite(ret_fwd)
            valid = ret_tgt_arr[i]
            y_dir_arr[i, valid] = np.where(
                ret_fwd[valid] > dir_threshold, 2,
                np.where(ret_fwd[valid] < -dir_threshold, 0, 1),
            )
            y_ret_arr[i] = np.nan_to_num(ret_fwd, nan=0.0)

            # Realized return for portfolio evaluation — defined for EVERY
            # entry-eligible (open-valid) day so the candidate pool never
            # depends on whether a future label exists:
            #   clean open[t]->open[t+h] where available; else carry to the last
            #   real close in (t, t+h] / open[t] - 1; else 0 (no exit → flat).
            realized = np.zeros(max_T, dtype=np.float32)
            for t in range(max_T):
                if not (open_valid[t] and open_full[t] > 0):
                    continue
                if t < max_T - horizon and np.isfinite(ret_fwd[t]):
                    realized[t] = ret_fwd[t]
                    continue
                hi = min(t + horizon, max_T - 1)
                later = np.nonzero(close_valid[t + 1:hi + 1])[0]
                if later.size:
                    k = t + 1 + int(later[-1])
                    if close_full[k] > 0:
                        realized[t] = float(close_full[k] / open_full[t] - 1.0)
            realized_arr[i] = realized

            # FORWARD-looking realized volatility: std of the daily returns
            # realized over the NEXT `horizon` days (return[t+1 : t+horizon+1]),
            # spanning the same forward window as y_return.  The target is
            # strictly positive, matching VolatilityHead's softplus — train_panel
            # must NOT z-score it.  Suspended days get a 0 return and the
            # resumption day records the accumulated close gap, so a "5-day vol"
            # label uses all 5 days instead of silently collapsing to however
            # many days actually traded.  A window with <2
            # valid closes sets vol_tgt_arr False so the vol loss never sees a
            # degenerate single-price window.
            ret_daily = np.zeros(max_T, dtype=np.float32)
            last_valid = np.maximum.accumulate(
                np.where(close_valid, np.arange(max_T), -1))
            prev_close = np.full(max_T, -1)
            prev_close[1:] = last_valid[:-1]
            ok = close_valid & (prev_close >= 0)
            ret_daily[ok] = (
                close_full[ok] / close_full[prev_close[ok]] - 1.0
            )
            for t in range(max_T - horizon):
                win = ret_daily[t + 1:t + 1 + horizon]
                if close_valid[t + 1:t + 1 + horizon].sum() < 2:
                    continue
                y_vol_arr[i, t] = float(np.std(win))
                vol_tgt_arr[i, t] = True

        # ── Decision / history eligibility ──
        # decision_arr[t] = close[t-1] is real, so a signal computed after
        # close[t-1] (features through column t-1) can rank this stock and
        # ENTER at open[t].  Aligned to the ENTRY column t so the candidate
        # pool is decision & entry & history on one grid.
        decision_arr = np.zeros((N_stocks, max_T), dtype=bool)
        if max_T > 1:
            decision_arr[:, 1:] = obs_arr[:, :-1]
        # history_arr[t] = the seq_len input window ending at t-1 (columns
        # [t-seq_len, t-1]) holds >= min_history real observations — excludes
        # freshly-listed stocks whose window is mostly zero padding.
        if min_history <= 0:
            history_arr = np.ones((N_stocks, max_T), dtype=bool)
        else:
            obs_i = obs_arr.astype(np.int32)
            cum = np.concatenate(
                [np.zeros((N_stocks, 1), dtype=np.int32), np.cumsum(obs_i, axis=1)],
                axis=1,
            )
            t_idx = np.arange(max_T)
            lo = np.maximum(t_idx - self.seq_len, 0)
            history_arr = (cum[:, t_idx] - cum[:, lo]) >= min_history

        # ── Research-universe eligibility (§七-3) ──
        # Data-derived PIT gates merged into the decision pool:
        #   已上市  (first_col) + 当日未长期停牌 + 符合研究流动性规则.
        # The 未退市 delist gate and the per-day index-membership gate need the
        # EXTERNAL universe status / membership records, so they are applied
        # per-fold in train_panel and ANDed into this same decision mask there.
        from stoke_ml.config import load_config
        uni_cfg = dict(load_config().get("universe", {}) or {})
        long_susp_thr = int(uni_cfg.get("long_suspension_days", 60))
        susp_lookback = int(uni_cfg.get("suspension_lookback", 60))
        min_amount_60d = float(uni_cfg.get("min_amount_60d", 5_000_000))
        universe_eligible_arr = _not_long_suspended(
            obs_arr, first_col, max_T, long_susp_thr, susp_lookback,
        )
        if min_amount_60d > 0 and has_amount_arr.any():
            # Causal trailing-60d turnover known at close[t-1] → entry day t:
            # shift amt60_raw (mean over [t-59, t]) right by one column.  Only
            # stocks with a canonical `amount` get the floor; the volume×close
            # / price proxies are not a real turnover measure.
            amt_causal = np.zeros_like(amt60_raw, dtype=np.float32)
            if max_T > 1:
                amt_causal[:, 1:] = amt60_raw[:, :-1]
            liquid = np.ones((N_stocks, max_T), dtype=bool)
            liquid[has_amount_arr] = amt_causal[has_amount_arr] >= min_amount_60d
            universe_eligible_arr &= liquid
        decision_arr &= universe_eligible_arr

        # Align columns across all stocks — sparse data types (dragon_tiger,
        # block_trade, lockup, etc.) may have data for some stocks but not
        # others, producing different column sets. Missing columns get ZI fill.
        if all_feat_dfs:
            all_cols = set()
            for df in all_feat_dfs:
                all_cols.update(df.columns)
            for i, df in enumerate(all_feat_dfs):
                missing = all_cols - set(df.columns)
                if not missing:
                    continue
                fill_data: dict[str, np.ndarray] = {}
                n = len(df)
                for col in missing:
                    if col == "date":
                        continue
                    elif col.startswith("has_"):
                        fill_data[col] = np.full(n, False)
                    elif col.endswith("_count") or col.endswith("_streak"):
                        fill_data[col] = np.zeros(n, dtype=np.int16)
                    else:
                        fill_data[col] = np.zeros(n, dtype=np.float32)
                if fill_data:
                    fill_df = pd.DataFrame(fill_data, index=df.index)
                    all_feat_dfs[i] = pd.concat([df, fill_df], axis=1)

        # ── Cross-sectional fundamental features ──
        # Sector-relative valuation, leverage warning, composite cheapness.
        # Computed on the full multi-stock panel so sector medians are meaningful.
        cs_fund_cols = ["date", "stock_code", "sector_code",
                        "pe_ttm", "pb_mrq", "ps_ttm", "debt_ratio",
                        "pe_percentile_252d", "pb_percentile_252d"]
        if (self._fundamental_refiner is not None
                and any("sector_code" in df.columns for df in all_feat_dfs)):
            panel_parts: list[pd.DataFrame] = []
            for i, df in enumerate(all_feat_dfs):
                if len(df) == 0:
                    continue
                avail = [c for c in cs_fund_cols if c in df.columns]
                if "sector_code" not in avail:
                    continue
                part = df[avail].copy()
                panel_parts.append(part)
            if panel_parts:
                cs_panel = pd.concat(panel_parts, ignore_index=True)
                cs_panel = FundamentalRefiner.add_cross_sectional(cs_panel)
                new_cs_cols = [c for c in cs_panel.columns
                               if c not in set(cs_fund_cols)]
                if new_cs_cols:
                    for i, df in enumerate(all_feat_dfs):
                        if len(df) == 0 or "sector_code" not in df.columns:
                            continue
                        stock_code = codes[i]
                        stock_cs = cs_panel[cs_panel["stock_code"] == stock_code]
                        if stock_cs.empty:
                            continue
                        merge_df = stock_cs[["date"] + new_cs_cols].copy()
                        df = df.merge(merge_df, on="date", how="left")
                        for col in new_cs_cols:
                            if col not in df.columns:
                                df[col] = np.float32(0.0)
                            else:
                                df[col] = df[col].fillna(0.0).astype(np.float32)
                        all_feat_dfs[i] = df

        if max_T < self.seq_len + 5:
            raise ValueError(
                f"Max timesteps ({max_T}) must be > seq_len+5 ({self.seq_len + 5})"
            )

        # ── Dynamic column discovery (replaces hardcoded _PAST_KNOWN/OBSERVED_COLS) ──
        first_df = all_feat_dfs[0]
        # PIT static columns: time-varying per-window
        # context derived from data available at each decision day.  Replaces the
        # leaky first-20-days permanent quantiles.  All are derivable from OHLCV
        # + date + stock code — see _PIT_STATIC_COLS.
        static_cols_available = list(_PIT_STATIC_COLS)
        pk_cols_available = self._discover_pk_columns(first_df)
        pk_set = set(pk_cols_available)
        po_cols_available = self._discover_po_columns(first_df, pk_set)
        # Dead-feature drop is deliberately NOT applied here: a column's
        # constancy over the FULL history (including future periods) must never
        # decide an earlier fold's feature set.  All engineered columns stay in
        # the grids; train_panel drops dead columns per-fold using only its own
        # training window (fold_dead_feature_columns).

        # ── Per-date cross-sectional z-score normalization ──
        # Normalize each feature across stocks within each date, so that
        # a feature's value is expressed relative to the cross-section that
        # day.  This avoids pooling future dates' statistics into today's
        # normalized value and is the standard panel-finance treatment.
        norm_cols = [c for c in pk_cols_available + po_cols_available
                     if c not in _CS_NORM_SKIP_COLS]
        all_feat = pd.concat([
            df[["date"] + norm_cols]
            for df in all_feat_dfs
            if len(df) > 0
        ], ignore_index=True)
        # Strip non-finite BEFORE any cross-sectional statistic.
        # A single inf (e.g. a near-zero divisor in a factor) pollutes the
        # groupby mean/std, corrupting the whole date's z-score before the
        # final nan_to_num silently zeroes it out.
        finite_cols = [c for c in norm_cols if c in all_feat.columns]
        for c in finite_cols:
            vals = all_feat[c]
            if not np.isfinite(vals.to_numpy()).all():
                all_feat[c] = vals.replace([np.inf, -np.inf], np.nan)

        date_stats: dict[str, pd.DataFrame] = {}
        for col in norm_cols:
            if col not in all_feat.columns:
                continue
            stats = all_feat.groupby("date")[col].agg(["mean", "std", "count"])
            stats["std"] = stats["std"].fillna(1.0).clip(lower=1e-8)
            # Dates with very few listed stocks (the 2000-2015 backfill has
            # 1-5 stocks/day) give a degenerate cross-section: std→0 inflates
            # z-scores to ±hundreds, which dominates the loss.  Fall back to
            # the pooled global mean/std for those sparse dates — the global
            # moments are stable even when the daily cross-section is tiny.
            sparse = stats["count"] < 5
            if sparse.any():
                # Full-panel pooled moments would leak future dates' statistics
                # into early dates' z-scores (the exact bias the per-date
                # cross-section avoids).  Use expanding moments over dates <=
                # the sparse date — strictly point-in-time.  Cumulative sums
                # give O(dates) per column instead of O(dates^2).
                sdf = all_feat[["date", col]].sort_values("date")
                col_vals = sdf[col].to_numpy(dtype=np.float64)
                # Treat inf as invalid too (np.nanmean/np.nanstd choke on it
                # and would leak NaN through the z-score).
                valid_vals = np.isfinite(col_vals)
                x = np.where(valid_vals, col_vals, 0.0)
                ccount = np.cumsum(valid_vals.astype(np.float64))
                csum = np.cumsum(x)
                csq = np.cumsum(x * x)
                sdates = pd.to_datetime(sdf["date"]).to_numpy(dtype="datetime64[ns]")
                sparse_dates = pd.to_datetime(stats.index[sparse]).to_numpy(dtype="datetime64[ns]")
                pos = np.clip(
                    np.searchsorted(sdates, sparse_dates, side="right") - 1,
                    0, len(sdates) - 1,
                )
                cnt = np.maximum(ccount[pos], 1.0)
                mean = csum[pos] / cnt
                var = np.maximum(csq[pos] / cnt - mean * mean, 0.0)
                std = np.maximum(np.sqrt(var), 1e-8)
                # groupby agg returns float32 columns; the float64 arrays must
                # be cast back or pandas raises LossySetitemError.
                stats.loc[sparse, "mean"] = mean.astype(stats["mean"].dtype)
                stats.loc[sparse, "std"] = std.astype(stats["std"].dtype)
            date_stats[col] = stats

        for df in all_feat_dfs:
            for col in norm_cols:
                if col not in df.columns or col not in date_stats:
                    continue
                aligned_mean = df["date"].map(date_stats[col]["mean"])
                aligned_std = df["date"].map(date_stats[col]["std"]).clip(lower=1e-8)
                df[col] = (df[col] - aligned_mean) / aligned_std

        static_dim = len(static_cols_available)
        pk_dim = len(pk_cols_available)
        po_dim = len(po_cols_available)

        # Pre-allocate feature arrays aligned to the GLOBAL calendar (column t
        # is the same date for every stock).  Days before listing or during a
        # suspension keep zero features; observation_mask / entry_eligible_mask
        # (returned below) tell training & evaluation which positions are real.
        static_arr = np.zeros((N_stocks, max_T, static_dim), dtype=np.float32)
        pk_arr = np.zeros((N_stocks, max_T, pk_dim), dtype=np.float32)
        po_arr = np.zeros((N_stocks, max_T, po_dim), dtype=np.float32)

        for i, df in enumerate(all_feat_dfs):
            if len(df) == 0:
                continue

            df_sorted = df.sort_values("date").reset_index(drop=True)
            pos = stock_pos[i]
            if len(pos) == 0:
                continue

            # PIT static — per-row series scattered onto global-calendar columns.
            # price_60d_q / amt_60d_q hold RAW trailing 60d means (captured in
            # the mask loop before z-score); their cross-sectional per-date
            # quantiles are computed over the whole (N, T) grid after the loop.
            if static_dim > 0:
                s = np.zeros((len(pos), static_dim), dtype=np.float32)
                sidx = {c: k for k, c in enumerate(static_cols_available)}
                if "price_60d_q" in sidx:
                    s[:, sidx["price_60d_q"]] = price60_raw[i][pos]
                if "amt_60d_q" in sidx:
                    s[:, sidx["amt_60d_q"]] = amt60_raw[i][pos]
                if "listing_days" in sidx:
                    glob_col = pos.astype(np.float32)
                    if first_col[i] >= 0:
                        glob_col = np.maximum(glob_col - first_col[i], 0.0)
                    s[:, sidx["listing_days"]] = glob_col / 250.0
                bid = _board_index(codes[i])
                bcol = _BOARD_ONEHOT_COLS[bid]
                if bcol in sidx:
                    s[:, sidx[bcol]] = 1.0
                static_arr[i, pos] = s

            # Past known / observed — scattered onto global-calendar columns.
            pk_arr[i, pos] = df_sorted[pk_cols_available].fillna(0.0).values.astype(np.float32)
            po_arr[i, pos] = df_sorted[po_cols_available].fillna(0.0).values.astype(np.float32)

        # Cross-sectional per-date quantile for the trailing-mean size/price
        # features.  Rank within each column's cross-section of stocks that are
        # genuinely listed there (obs True) with a nonzero trailing mean.
        # PIT-safe: every value in column t uses only data through close t, and
        # the within-column rank is itself known at t.
        for qname in ("price_60d_q", "amt_60d_q"):
            if qname not in static_cols_available:
                continue
            qk = static_cols_available.index(qname)
            qcol = static_arr[:, :, qk]
            qlisted = obs_arr & (qcol > 0)
            for qt in range(max_T):
                qidxs = np.nonzero(qlisted[:, qt])[0]
                if qidxs.size < 2:
                    if qidxs.size == 1:
                        qcol[qidxs, qt] = 0.5  # singleton cross-section → neutral rank
                    continue
                qvals = qcol[qidxs, qt]
                # Average-rank ties — pandas rank(method="average",
                # pct=True).  argsort ordinal ranks would give equal values
                # different quantiles purely from stock array order.
                qorder = np.argsort(qvals, kind="mergesort")
                q0 = qvals[qorder]
                qsz = qidxs.size
                grp_start = np.concatenate([[0], np.nonzero(np.diff(q0))[0] + 1])
                grp_end = np.concatenate([grp_start[1:], [qsz]])
                grp_rank1 = (grp_start + grp_end + 1) / 2.0  # 1-based avg rank/group
                qranks = np.empty(qsz, dtype=np.float64)
                qranks[qorder] = np.repeat(grp_rank1, grp_end - grp_start)
                qcol[qidxs, qt] = (qranks / qsz).astype(np.float32)

        # ── Sanitize: replace NaN/Inf with zeros and clip extreme values ──
        # Alpha158 factors can produce Inf from near-zero divisors (e.g.
        # open0 = open/close with close≈0 for suspended stocks).  The z-score
        # normalization also amplifies tiny variance features.
        pk_arr = np.nan_to_num(pk_arr, nan=0.0, posinf=0.0, neginf=0.0)
        pk_arr = np.clip(pk_arr, -10.0, 10.0)
        po_arr = np.nan_to_num(po_arr, nan=0.0, posinf=0.0, neginf=0.0)
        po_arr = np.clip(po_arr, -10.0, 10.0)
        static_arr = np.nan_to_num(static_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_ret_arr = np.nan_to_num(y_ret_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_vol_arr = np.nan_to_num(y_vol_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # NOTE: Targets are NOT scaled here.  Per-stock z-score normalization
        # is applied in the training script (train_panel.py) using only training
        # statistics, which avoids look-ahead bias and gives each stock equal
        # weight regardless of its native return volatility.
        # After per-stock z-scoring (μ=0, σ=1), the MSE baseline ≈ 1.0,
        # which is naturally balanced with CE loss (~1.0).

        # Per-stock date indices — maps each time step to its absolute
        # position in the global trading calendar.  With global alignment the
        # mapping is IDENTICAL for every stock (column t == global_dates[t]),
        # so PairwiseRankingLoss groups true same-day cross-sections.
        date_idx_arr = np.tile(np.arange(max_T, dtype=np.int32), (N_stocks, 1))

        # Union trading calendar (datetime64[ns]) — lets callers convert a
        # panel column index back to the real trading date.
        return {
            "static_features": static_arr,
            "past_known": pk_arr,
            "past_observed": po_arr,
            "y_direction": y_dir_arr,
            "y_return": y_ret_arr,
            "y_volatility": y_vol_arr,
            "date_indices": date_idx_arr,
            "global_dates": global_dates,
            "observation_mask": obs_arr,
            "entry_eligible_mask": entry_arr,
            "return_target_mask": ret_tgt_arr,
            "vol_target_mask": vol_tgt_arr,
            "realized_return": realized_arr,
            "decision_eligible_mask": decision_arr,
            "history_eligible_mask": history_arr,
            # Data-derived research-universe gate (已上市 + 未长期停牌 +
            # 流动性); the delist / index-membership halves are ANDed in per-fold
            # by train_panel (§七-3).
            "universe_eligible_mask": universe_eligible_arr,
            "close_price": close_price_arr,
            "open_price": open_price_arr,
            # Column order of the feature grids (axis 2) — lets consumers probe
            # per-channel presence via the has_* flags.  Full engineered order:
            # per-fold dead-column removal happens in train_panel after slicing.
            "past_known_cols": list(pk_cols_available),
            "past_observed_cols": list(po_cols_available),
        }


# ── Panel model feature column definitions ──────────────────────────────────


# PIT-static features:
#   price_60d_q / amt_60d_q  trailing 60d means → per-date cross-sectional rank
#   listing_days             (global col − first listed col) / 250
#   board_*                  exchange-board one-hot derived from the stock code
# All nine are computed purely from data known at the decision day.
# NOTE on size: a genuine PIT float market cap (real 流通市值) is NOT currently
# derivable from canonical on-disk data — valuation data begins 2015 and is
# PE/PB/PS/PCF only, the daily contract has no share counts, and fundamentals
# are quarterly without share counts.  `amt_60d_q` (trailing 60d turnover rank)
# is the size/liquidity axis; `price_60d_q` is a price tier, NOT a size proxy.
# Replacing these with real PIT float market cap requires new data acquisition
# (e.g. Sina backup `float_mcap_yi` or Baostock `turn`), not a derivation (§十一-5).
# `industry_code` is deliberately EXCLUDED: the only available stock→industry
# sources (sector_map.json / stock_sector_cache.csv) are current-snapshot maps
# with no point-in-time membership history, so a static industry_code would
# backfill today's classification onto historical rows — a present-backfill
# that must never happen.  Re-add only behind a genuine PIT membership source.
# The per-stock industry-relative features (ind_matched_return /
# stock_vs_industry) are likewise removed from _merge_industry for the same
# reason.
_BOARD_NAMES = ("unknown", "sh_main", "star", "sz_main", "chinext", "bse")
_BOARD_ONEHOT_COLS = [f"board_{n}" for n in _BOARD_NAMES]

_PIT_STATIC_COLS = [
    "price_60d_q",     # trailing 60d mean close → cross-sectional price tier
    "amt_60d_q",       # trailing 60d mean turnover (canonical amount) → size/liquidity
    "listing_days",    # days since first bar (scaled by 250 → years)
    *_BOARD_ONEHOT_COLS,  # exchange-board one-hot derived from the stock code
]


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Point-in-time trailing mean over [t-window+1, t].

    Early rows use the partial window (mean of the days available so far);
    NaNs inside the window are skipped.  Rows with no valid value → 0.
    Vectorized via cumulative sums — O(n) per call.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    valid = np.isfinite(values)
    vals = np.where(valid, values, 0.0)
    scs = np.concatenate([[0.0], np.cumsum(vals)])
    vcs = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])
    t_idx = np.arange(n)
    lo = np.maximum(t_idx - window + 1, 0)
    cnt = vcs[1:] - vcs[lo]
    s = scs[1:] - scs[lo]
    return np.where(cnt >= 1.0, s / np.maximum(cnt, 1.0), 0.0)


def _not_long_suspended(
    obs_arr: np.ndarray,
    first_col: np.ndarray,
    max_T: int,
    threshold: int,
    lookback: int,
) -> np.ndarray:
    """Point-in-time long-suspension eligibility ``(n_stocks, T)`` (§七-3).

    A stock is disqualified from the first column its consecutive missing-close
    run reaches ``threshold``, through ``lookback`` trading columns after the
    run resumes.  Pre-listing columns are never "missing" (the stock is not yet
    listed, not suspended).  Fully vectorized: cumulative run lengths + a
    difference array for the active-window intervals.
    """
    n = obs_arr.shape[0]
    if n == 0 or max_T == 0:
        return np.zeros((n, max_T), dtype=bool)
    cols = np.arange(max_T)[None, :]
    listed = (first_col[:, None] >= 0) & (cols >= first_col[:, None])
    missing = (~obs_arr) & listed
    # Consecutive missing-run length ending at t: last non-missing column up to
    # t, so run[t] = t - last_ok[t] when missing, else 0.
    last_ok = np.maximum.accumulate(np.where(missing, -1, cols), axis=1)
    run = np.where(missing, cols - last_ok, 0)
    trig = run >= threshold
    if not trig.any():
        return np.ones((n, max_T), dtype=bool)
    trig_start = trig.copy()
    trig_start[:, 1:] &= ~trig[:, :-1]
    trig_end = trig.copy()
    trig_end[:, :-1] &= ~trig[:, 1:]
    rows_s, cols_s = np.where(trig_start)
    rows_e, cols_e = np.where(trig_end)
    diff = np.zeros((n, max_T + 1), dtype=np.int64)
    np.add.at(diff, (rows_s, cols_s), 1)
    end = np.minimum(cols_e + lookback + 1, max_T)
    np.add.at(diff, (rows_e, end), -1)
    active = np.cumsum(diff[:, :max_T], axis=1)
    return active == 0


def _board_index(code) -> int:
    """Map a 6-digit A-share code to an index into _BOARD_ONEHOT_COLS."""
    s = str(code).zfill(6)
    if s.startswith("60"):
        return 1   # SH main board
    if s.startswith("68"):
        return 2   # STAR market
    if s.startswith("00"):
        return 3   # SZ main board (incl. 002 SME)
    if s.startswith("30"):
        return 4   # ChiNext
    if s[0] in ("8", "4"):
        return 5   # Beijing Stock Exchange
    return 0



# Alpha158 rolling-window factor name generator.
# Must stay in sync with _WINDOWS in stoke_ml/features/technical.py.
_ALPHA158_WINDOWS = [5, 10, 20, 30, 60]


def _alpha158_factor_names() -> list[str]:
    """Return Alpha158 rolling-window factor column names for all windows."""
    names: list[str] = []
    for d in _ALPHA158_WINDOWS:
        names.extend([
            f"max_{d}d", f"min_{d}d", f"qtlu_{d}d", f"qtld_{d}d",
            f"rank_{d}d", f"rsv_{d}d",
            f"corr_{d}d", f"cord_{d}d",
            f"beta_{d}d", f"rsqr_{d}d", f"resi_{d}d",
            f"vma_{d}d", f"vstd_{d}d",
            f"cntp_{d}d", f"cntn_{d}d", f"cntd_{d}d",
            f"sump_{d}d", f"sumn_{d}d", f"sumd_{d}d",
            f"imax_{d}d", f"imin_{d}d", f"imxd_{d}d",
            f"wvma_{d}d", f"vsump_{d}d", f"vsumn_{d}d", f"vsumd_{d}d",
        ])
    return names


# Features that are stock-invariant (same value for every stock on a given date).
# Cross-sectional z-score normalization would divide by near-zero std, producing
# saturated ±10.0 values with no signal. Skip them.
_CS_NORM_SKIP_COLS = frozenset({
    "minutes_from_open", "minutes_to_close", "is_am_session", "is_pm_session",
    "session_progress", "bar_of_day",
    "day_of_week", "day_of_month", "month", "quarter",
    # Macro (market-wide, identical for every stock on a date)
    "shibor_O_N", "shibor_1W", "shibor_2W", "shibor_1M",
    "shibor_3M", "shibor_6M", "shibor_9M", "shibor_1Y",
    "fx_usd_cny", "fx_eur_cny", "fx_jpy_cny", "fx_hkd_cny", "fx_gbp_cny",
    "bond_cn_2y", "bond_cn_5y", "bond_cn_10y", "bond_cn_30y",
    "bond_cn_10y2y_spread",
    "bond_us_2y", "bond_us_5y", "bond_us_10y", "bond_us_30y",
    "bond_us_10y2y_spread",
    "gdp_cn_yoy", "m2_yoy", "m1_yoy", "sf_total", "cpi_yoy",
    # Industry cross-sectional stats (row-wise over ALL industries — market-wide).
    # NOTE: ind_matched_return / stock_vs_industry are PER-STOCK and intentionally
    # excluded from this set (they carry cross-sectional signal).
    "ind_pct_up", "ind_return_mean", "ind_return_std",
    "ind_return_max", "ind_return_min", "ind_return_skew",
    "ind_dispersion_20d",
})


def _active_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the subset of *candidates* that exist in *df*."""
    return [c for c in candidates if c in df.columns]

