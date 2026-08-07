"""Feature pipeline orchestrating all feature engineering steps.

Integrates K-line, sentiment, market-wide (margin/northbound/dragon-tiger),
ETF sector flow, and fundamental data into a unified feature set.
"""
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

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

from stoke_ml.features.panel_helpers import (
    ANNOUNCEMENT_COLS,
    TEMPORAL_BASE_COLS,
    _AGGREGATOR_BASE_COLS,
    FLOW_COLS,
    BLOCK_TRADE_COLS,
    SHAREHOLDER_COLS,
    LOCKUP_COLS,
    DIVIDEND_COLS,
    BOARD_COLS,
    SECTOR_COLS,
    CONCEPT_COLS,
    LIMIT_UP_COLS,
    PLEDGE_COLS,
    INDEX_MEMBER_COLS,
    DRAGON_TIGER_SEAT_COLS,
    DEAD_FEATURE_RATIO,
    SPARSE_KEEP_PREFIXES,
    _constant_col_indices,
    fold_dead_feature_columns,
    _sparse_kept,
    _panel_calendar,  # frozen None snapshot — use _get_panel_calendar() (lives in panel_helpers)
    _get_panel_calendar,
    _BOARD_NAMES,
    _BOARD_ONEHOT_COLS,
    _ABSOLUTE_PRICE_COLS,
    _PIT_STATIC_COLS,
    _trailing_mean,
    _not_long_suspended,
    _board_index,
    _alpha158_factor_names,
    _ALPHA158_WINDOWS,
    _CS_NORM_SKIP_COLS,
    _active_cols,
    _manifest_check_config,
    _min_vol_nobs,
)
from stoke_ml.features.panel_builder import build_panel_features as _build_panel_features

logger = logging.getLogger(__name__)


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
        # §T5: the ACCOUNT sub-part of the market_env file (monthly
        # investor/mkt-cap stats) is PROXY-PIT while the raw source records no
        # real publish date, so it is OFF by default (ablation-only, mirroring
        # use_topic).  The PRICE sub-part is always consumed via use_market_env.
        use_market_env_account: bool = False,
        use_index_membership: bool = True,
        use_market_env_refine: bool = True,
        minute_mode: bool = False,
        feature_selection_k: int = 500,
        use_new_preprocessing: bool = False,
        preprocessing_config: dict | str | None = None,
        # Feature engineering v2
        use_emotion_refine: bool = True,
        # Coupled to use_fundamental: without the fundamental channel there is
        # nothing to refine, so use_fundamental=False forces use_fundamental_refine
        # off as well (§T7) — see the __init__ body.
        use_fundamental_refine: bool = True,
        use_temporal_stats: bool = True,
        drop_dead_features: bool = True,
        min_history: int = 50,
        # §七: topic-model features (topic_* from the global_frozen topic model)
        # are OFF by default.  The model is fit once on a pinned reference corpus
        # and then TRANSFORMS ALL historical headlines, so a headline that enters
        # the reference after an earlier day's decision leaks future vocabulary
        # into that day's representation — an ablation-only dimension.  Explicitly
        # enabling use_topic restores the columns for a controlled study of their
        # marginal value; keeping the default OFF means the headline/Lockbox
        # feature set never silently consumes them.
        use_topic: bool = False,
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
        # §T7: fundamental_refine is COUPLED to the fundamental channel — without
        # the channel there is nothing to refine, so turning fundamental off
        # silently turns the refiner off too.  This is what makes the
        # revision-safe vintage switch set (which turns use_fundamental off) also
        # drop the fundamental_refine columns on the prebuilt scrub path, instead
        # of the refiner running anyway on columns that were never requested
        # (§T3 leak).
        if not use_fundamental:
            use_fundamental_refine = False
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
        self.use_market_env_account = use_market_env_account
        self.use_index_membership = use_index_membership
        self.use_market_env_refine = use_market_env_refine
        self._aux = AuxAligner(
            {k: getattr(self, f"use_{k}") for k in AuxAligner.AUX_KEYS}
            | {"market_env_account": use_market_env_account}
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
        # §七: topic_* features OFF by default (global_frozen topic-model
        # representation leakage) — see the __init__ docstring.
        self.use_topic = use_topic

    @staticmethod
    def _drop_topic_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop topic_* columns (topic_entropy / topic_dominant / topic_*_sent /
        topic_*_ratio / topic_sent_dispersion) from a feature frame.

        The global_frozen topic model (preprocessing.text.topics) is fit on a
        pinned reference corpus and then transforms every historical headline,
        so its outputs are not point-in-time and must not feed the default
        feature set (§七).  ``use_topic=True`` (ablation) skips this drop.
        """
        cols = [c for c in df.columns if c.startswith("topic_")]
        if cols:
            df = df.drop(columns=cols)
        return df

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
                from omegaconf.errors import OmegaConfBaseException
                cfg = OmegaConf.to_container(cfg, resolve=True)
            except (ValueError, TypeError, OmegaConfBaseException) as exc:
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

        # §七: topic_* columns ride in as "extra" merge columns on the
        # sentiment/guba/announcement channels.  OFF by default — drop them
        # immediately after the merge so no downstream step (interaction,
        # emotion refinement, temporal/lag assembly) consumes the
        # future-corpus-contaminated representation.
        if not self.use_topic:
            df = self._drop_topic_columns(df)

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

        §五 P0: the raw absolute qfq ``open/high/low/close`` are EXCLUDED (see
        ``_ABSOLUTE_PRICE_COLS``) — qfq levels re-anchor with future corporate
        actions and would leak future behaviour into historical decisions.
        Scale-invariant close-derived relatives (open0/high0/low0, kmid/klen/
        ... ratios) remain PK.
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
        # §五 P0: drop the absolute qfq price level (exact name only) — see
        # _ABSOLUTE_PRICE_COLS docstring.  volume/amount stay: amount is real
        # CNY (not re-anchored), volume is unchanged under qfq adjustment.
        pk = [c for c in pk if c not in _ABSOLUTE_PRICE_COLS]
        return pk

    @staticmethod
    def _discover_po_columns(df: pd.DataFrame, pk_set: set[str] | None = None) -> list[str]:
        """Auto-discover PO columns — everything not PK and not metadata."""
        if pk_set is None:
            pk_set = set(FeaturePipeline._discover_pk_columns(df))
        skip = {"date", "stock_code", "sector", "size_proxy", "sector_code"}
        # §五 P0: absolute qfq OHLC must not leak into the observed grid either —
        # excluding them from PK alone would just reclassify them as PO.
        return [
            c for c in df.columns
            if c not in pk_set and c not in skip and c not in _ABSOLUTE_PRICE_COLS
        ]

    def _clean_calendar_dates(self, df: pd.DataFrame, stock_code: str,
                              data_dir: str | None = None):
        """Keep a stock's date axis calendar-clean before the panel
        UNION date axis is built from it.

        One wrong weekend/closed-day bar, or a duplicated/out-of-order date,
        would add a phantom column to the global panel calendar and corrupt
        cross-sectional alignment for every stock.  Repair in place: drop
        unparsable dates, de-dup (keep the last row, matching storage
        overwrite semantics), drop dates that are not official trading days,
        then re-sort so the surviving dates are unique and strictly increasing.
        Returns the cleaned frame, or None when nothing valid remains.

        ``data_dir`` is forwarded to :func:`_get_panel_calendar` so the strict
        calendar follows the frozen ``exchange_calendar`` artifact at the data
        root the caller actually reads (None → the config default).
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
        trading = set(_get_panel_calendar(data_dir).get_trading_days(
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
        data_dir: str | None = None,
        daily_membership: pd.DataFrame | None = None,
        memmap_dir: str | None = None,
    ) -> dict:
        """Build panel-format features for VSN+xLSTM training.

        Thin delegate to ``panel_builder.build_panel_features`` (extracted
        §二十一); keeps the public signature so all callers are unchanged.
        See the panel_builder function for the full parameter documentation.
        """
        return _build_panel_features(
            self,
            panel,
            target_col=target_col,
            aux_data=aux_data,
            horizon=horizon,
            prebuilt_dir=prebuilt_dir,
            min_history=min_history,
            require_feature_manifest=require_feature_manifest,
            data_dir=data_dir,
            daily_membership=daily_membership,
            memmap_dir=memmap_dir,
        )
