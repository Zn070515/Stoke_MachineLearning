"""Feature pipeline orchestrating all feature engineering steps.

Integrates K-line, sentiment, market-wide (margin/northbound/dragon-tiger),
ETF sector flow, and fundamental data into a unified feature set.
"""
import logging

import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
from stoke_ml.features.interaction import InteractionFeatures
from stoke_ml.features.temporal import (
    add_lag_features, add_rolling_features, add_calendar_features,
)

logger = logging.getLogger(__name__)

SENTIMENT_COLS = [
    "sentiment_mean", "sentiment_std", "news_count",
    "positive_ratio", "negative_ratio", "has_news",
]

ANNOUNCEMENT_COLS = [
    "ann_sentiment_mean", "ann_sentiment_std", "ann_count",
    "ann_positive_ratio", "ann_negative_ratio", "has_announce",
]

MARGIN_COLS = [
    "margin_balance", "margin_buy", "short_balance", "margin_net",
]

NORTHBOUND_COLS = [
    "north_hold_pct", "north_net_buy",
]

DRAGON_TIGER_COLS = [
    "lhb_net_amount", "lhb_buy_ratio", "lhb_present",
]

ETF_FLOW_COLS = [
    "sector_etf_flow", "sector_etf_amount",
]

GUBA_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio", "has_guba_post",
]

COMMENT_COLS = [
    "comment_score", "comment_attention", "comment_institution",
    "comment_trend", "has_comment",
]

XUEQIU_COLS = [
    "xueqiu_sentiment_mean", "xueqiu_sentiment_std", "xueqiu_post_count",
    "xueqiu_positive_ratio", "xueqiu_negative_ratio", "has_xueqiu_post",
]

FUNDAMENTAL_COLS = [
    "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
    "debt_ratio", "gross_margin", "net_margin",
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
    "net_zt_proportion", "break_rate", "advance_rate", "max_board_height",
]

SECTOR_COLS = [
    "sector_relative_strength", "sector_breadth_z",
    "sector_rrg_y", "sector_rrg_x", "sector_rrg_quadrant",
]

CONCEPT_COLS = [
    "board_count", "has_hot_board", "avg_concept_heat",
    "is_concept_leader", "board_overlap_score",
]

INDUSTRY_COLS = [
    "ind_pct_up", "ind_return_mean", "ind_return_std",
    "ind_return_max", "ind_return_min", "ind_return_skew",
    "ind_dispersion_20d",
]

MACRO_COLS = [
    "shibor_O_N", "shibor_1W", "shibor_2W", "shibor_1M",
    "shibor_3M", "shibor_6M", "shibor_9M", "shibor_1Y",
    "fx_usd_cny", "fx_eur_cny", "fx_jpy_cny", "fx_hkd_cny", "fx_gbp_cny",
    "bond_cn_2y", "bond_cn_5y", "bond_cn_10y", "bond_cn_30y",
    "bond_cn_10y2y_spread",
    "bond_us_2y", "bond_us_5y", "bond_us_10y", "bond_us_30y",
    "bond_us_10y2y_spread",
    "gdp_cn_yoy", "m2_yoy", "m1_yoy", "sf_total", "cpi_yoy",
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
        use_etf_flow: bool = True,
        use_xueqiu: bool = True,
        use_interaction: bool = True,
        use_feature_selection: bool = False,
        use_capital_flow: bool = False,
        use_block_trade: bool = False,
        use_shareholder: bool = False,
        use_lockup: bool = False,
        use_dividend: bool = False,
        use_board: bool = False,
        use_sector: bool = False,
        use_concept: bool = False,
        use_macro: bool = True,
        use_industry: bool = True,
        feature_selection_k: int = 500,
        use_new_preprocessing: bool = False,
        preprocessing_config: dict | str | None = None,
    ):
        self.seq_len = seq_len
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
        self.use_etf_flow = use_etf_flow
        self.use_xueqiu = use_xueqiu
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
        self.feature_selection_k = feature_selection_k
        self.use_new_preprocessing = use_new_preprocessing
        self._preprocessing_config = preprocessing_config
        self._preprocessing = None
        if use_new_preprocessing and preprocessing_config:
            self._preprocessing = self._build_preprocessing()
        self._warned_missing: set[str] = set()
        self._macro_cache: pd.DataFrame | None = None
        self._industry_cache: pd.DataFrame | None = None
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()
        self._interaction = InteractionFeatures()

    def _warn_if_missing(self, key: str) -> None:
        """Emit one-time warning when use_*=True but no data was passed."""
        if key not in self._warned_missing:
            logger.warning("use_%s=True but no %s data provided", key, key)
            self._warned_missing.add(key)

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
            except Exception:
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
        etf_flow_df: pd.DataFrame | None = None,
        announcement_df: pd.DataFrame | None = None,
        guba_df: pd.DataFrame | None = None,
        comment_df: pd.DataFrame | None = None,
        xueqiu_df: pd.DataFrame | None = None,
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
        return_dates: bool = False,
    ) -> tuple:
        """Build features for a single stock. Returns (X, y, aligned_close).

        If *return_dates* is True, also returns (sample_dates) as a 4-tuple.
        Dates track the prediction date for each sample after dropna + sequencing.
        """
        feats = self._engineer_features(
            df, sentiment_df, margin_df, northbound_df,
            dragon_tiger_df, fundamental_df, etf_flow_df,
            announcement_df, guba_df, comment_df, xueqiu_df,
            capital_flow_df, block_trade_df, shareholder_df,
            lockup_df, dividend_df, board_df, sector_df, concept_df,
            macro_df=macro_df, industry_df=industry_df,
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
        etf_flow_df: pd.DataFrame | None = None,
        announcement_df: pd.DataFrame | None = None,
        guba_df: pd.DataFrame | None = None,
        comment_df: pd.DataFrame | None = None,
        xueqiu_df: pd.DataFrame | None = None,
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
        skip_temporal: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if self.use_new_preprocessing and self.preprocessing:
            df = self.preprocessing.run("numeric", df)

        df = self._merge_sentiment(df, sentiment_df)
        df = self._merge_announcements(df, announcement_df)
        df = self._merge_margin(df, margin_df)
        df = self._merge_northbound(df, northbound_df)
        df = self._merge_dragon_tiger(df, dragon_tiger_df)
        df = self._merge_fundamental(df, fundamental_df)
        df = self._merge_etf_flow(df, etf_flow_df)
        df = self._merge_guba(df, guba_df)
        df = self._merge_comment(df, comment_df)
        df = self._merge_xueqiu(df, xueqiu_df)

        # New multi-shape preprocessing
        df = self._merge_capital_flow(df, capital_flow_df)
        df = self._merge_block_trade(df, block_trade_df)
        df = self._merge_shareholder(df, shareholder_df)
        df = self._merge_lockup(df, lockup_df)
        df = self._merge_dividend(df, dividend_df)
        df = self._merge_board(df, board_df)
        df = self._merge_sector(df, sector_df)
        df = self._merge_concept(df, concept_df)
        df = self._merge_macro(df, macro_df)
        df = self._merge_industry(df, industry_df)

        if self.use_technical:
            df = self._ti.compute_all(df)
        if self.use_scoring:
            df = self._scorer.score(df)

        df = self._add_microstructure(df)

        if self.use_interaction:
            df = self._interaction.compute_all(df)

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
            temporal_cols += _active_cols(df, ETF_FLOW_COLS)
            temporal_cols += _active_cols(df, GUBA_COLS)
            temporal_cols += _active_cols(df, COMMENT_COLS)
            temporal_cols += _active_cols(df, XUEQIU_COLS)
            # New multi-shape columns (dynamic — pick up whatever was merged)
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
            # Dynamic columns: concept momentum, board momentum, sector momentum
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith("momentum_") or c.startswith("concept_momentum_")
                or c.startswith("board_momentum_") or c.startswith("sector_rrg_")
                or c.startswith("seal_type_") or c.startswith("market_state_")
                or c.startswith("cb_")
            ])
            # New text features from DailyAggregator (any source prefix)
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.endswith("_bipolar_sent") or c.endswith("_agreement")
                or c.endswith("_attention") or c.endswith("_weighted_sent")
                or c in ("bipolar_sent", "agreement", "attention", "weighted_sent")
            ])
            df = add_lag_features(df, temporal_cols, self.LAGS)
            df = add_rolling_features(df, temporal_cols, self.ROLLING_WINDOWS)
            df = add_calendar_features(df)

        return df

    # ------------------------------------------------------------------
    # Merge helpers — each returns a (possibly enriched) DataFrame
    # ------------------------------------------------------------------

    def _merge_sentiment(self, df: pd.DataFrame,
                         sentiment_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_sentiment:
            return df
        if sentiment_df is None or sentiment_df.empty:
            self._warn_if_missing("sentiment")
            return df
        s = sentiment_df.copy()
        s["date"] = pd.to_datetime(s["date"])
        available = [c for c in SENTIMENT_COLS if c in s.columns]
        extra = [c for c in s.columns
                 if c not in SENTIMENT_COLS and c not in ("date", "stock_code")
                 and not c.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(s[["date"] + available + extra], on="date", how="left")
        for col in available + extra:
            if col == "has_news":
                df[col] = df[col].fillna(False).astype(bool)
            elif col == "news_count":
                df[col] = df[col].fillna(0).astype("int16")
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)

        # Lag sentiment by 1 trading day to prevent look-ahead bias.
        for col in available + extra:
            df[col] = df[col].shift(1)
        df["has_news"] = df["has_news"].fillna(False).astype(bool)
        df["news_count"] = df["news_count"].fillna(0).astype("int16")
        for col in available + extra:
            if col not in ("has_news", "news_count") and col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)

        return df

    def _merge_announcements(self, df: pd.DataFrame,
                             announcement_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_announcements:
            return df
        if announcement_df is None or announcement_df.empty:
            self._warn_if_missing("announcements")
            return df
        a = announcement_df.copy()
        a["date"] = pd.to_datetime(a["date"])
        # Map storage column names to prefixed feature column names
        col_map = {
            "sentiment_mean": "ann_sentiment_mean",
            "sentiment_std": "ann_sentiment_std",
            "announce_count": "ann_count",
            "positive_ratio": "ann_positive_ratio",
            "negative_ratio": "ann_negative_ratio",
            "has_announce": "has_announce",
        }
        mapped_cols = {k: v for k, v in col_map.items() if k in a.columns}
        # Extra columns (e.g. ann_bipolar_sent from DailyAggregator) — merge directly
        extra = [c for c in a.columns
                 if c not in col_map and c not in ("date", "stock_code")
                 and not c.startswith("has_")]
        if not mapped_cols and not extra:
            return df
        rename = {k: v for k, v in mapped_cols.items()}
        merged_cols = list(rename.values()) + extra
        source_cols = list(rename.keys()) + extra
        a_renamed = a[["date"] + source_cols].rename(columns=rename)
        df = df.merge(a_renamed, on="date", how="left")
        for target_col in merged_cols:
            if target_col == "has_announce":
                df[target_col] = df[target_col].fillna(False).astype(bool)
            elif "count" in target_col:
                df[target_col] = df[target_col].fillna(0).astype("int16")
            else:
                df[target_col] = df[target_col].fillna(0.0).astype(np.float32)
        # Same PIT lag as news sentiment
        for target_col in merged_cols:
            df[target_col] = df[target_col].shift(1)
        df["has_announce"] = df["has_announce"].fillna(False).astype(bool)
        df["ann_count"] = df["ann_count"].fillna(0).astype("int16")
        for col in ["ann_sentiment_mean", "ann_sentiment_std",
                     "ann_positive_ratio", "ann_negative_ratio"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_margin(self, df: pd.DataFrame,
                      margin_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_margin:
            return df
        if margin_df is None or margin_df.empty:
            self._warn_if_missing("margin")
            return df
        m = margin_df.copy()
        m["date"] = pd.to_datetime(m["date"])
        m = m.drop(columns=["stock_code"], errors="ignore")
        m = m.drop_duplicates(subset="date", keep="last")
        available = [c for c in MARGIN_COLS if c in m.columns]
        if not available:
            return df
        df = df.merge(m[["date"] + available], on="date", how="left")
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: market-wide data available after close → shift to t+1
        for col in available:
            df[col] = df[col].shift(1)
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_northbound(self, df: pd.DataFrame,
                          northbound_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_northbound:
            return df
        if northbound_df is None or northbound_df.empty:
            self._warn_if_missing("northbound")
            return df
        nb = northbound_df.copy()
        nb["date"] = pd.to_datetime(nb["date"])
        nb = nb.drop(columns=["stock_code"], errors="ignore")
        nb = nb.drop_duplicates(subset="date", keep="last")
        available = [c for c in NORTHBOUND_COLS if c in nb.columns]
        if not available:
            return df
        df = df.merge(nb[["date"] + available], on="date", how="left")
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: market-wide data available after close → shift to t+1
        for col in available:
            df[col] = df[col].shift(1)
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_dragon_tiger(self, df: pd.DataFrame,
                            dt_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_dragon_tiger:
            return df
        if dt_df is None or dt_df.empty:
            self._warn_if_missing("dragon_tiger")
            return df
        dt = dt_df.copy()
        dt["date"] = pd.to_datetime(dt["date"])
        dt = dt.drop(columns=["stock_code", "stock_name", "lhb_reason"],
                      errors="ignore")
        # Aggregate multiple entries per date
        agg = dt.groupby("date").agg(
            lhb_net_amount=("net_amount", "sum"),
            lhb_buy_ratio=(
                "buy_amount",
                lambda x: x.sum() / (x.sum()
                                     + dt.loc[x.index, "sell_amount"].sum()
                                     + 1),
            ),
            lhb_present=("net_amount", "count"),
        ).reset_index()
        agg["lhb_present"] = (agg["lhb_present"] > 0).astype(np.float32)
        agg["lhb_buy_ratio"] = agg["lhb_buy_ratio"].fillna(0.5).astype(np.float32)
        agg["lhb_net_amount"] = agg["lhb_net_amount"].fillna(0.0).astype(np.float32)
        df = df.merge(agg, on="date", how="left")
        for col in DRAGON_TIGER_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: market-wide data available after close → shift to t+1
        for col in DRAGON_TIGER_COLS:
            if col in df.columns:
                df[col] = df[col].shift(1)
        for col in DRAGON_TIGER_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_fundamental(self, df: pd.DataFrame,
                           fundamental_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_fundamental:
            return df
        if fundamental_df is None or fundamental_df.empty:
            self._warn_if_missing("fundamental")
            return df
        fd = fundamental_df.copy()
        # Drop metadata columns
        fd = fd.drop(columns=["stock_code", "report_date"], errors="ignore")
        available = [c for c in FUNDAMENTAL_COLS if c in fd.columns]
        if not available:
            return df

        if "disclose_date" in fd.columns:
            # Raw quarterly data — forward-fill to daily
            fd["disclose_date"] = pd.to_datetime(fd["disclose_date"])
            fd = fd.drop_duplicates(subset="disclose_date", keep="last")
            fd = fd.sort_values("disclose_date").set_index("disclose_date")
            full_idx = pd.date_range(fd.index.min(), df["date"].max(), freq="D")
            fd = fd[available].reindex(full_idx).ffill().reset_index(names="date")
        else:
            # Already daily data — just ensure date column
            fd["date"] = pd.to_datetime(fd["date"])
            fd = fd.drop_duplicates(subset="date", keep="last")

        df = df.merge(fd[["date"] + available], on="date", how="left")
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_etf_flow(self, df: pd.DataFrame,
                        etf_flow_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_etf_flow:
            return df
        if etf_flow_df is None or etf_flow_df.empty:
            self._warn_if_missing("etf_flow")
            return df
        ef = etf_flow_df.copy()
        ef["date"] = pd.to_datetime(ef["date"])
        ef = ef.drop(columns=["sector_name", "etf_count"], errors="ignore")
        ef = ef.drop_duplicates(subset="date", keep="last")
        available = [c for c in ETF_FLOW_COLS if c in ef.columns]
        if not available:
            return df
        df = df.merge(ef[["date"] + available], on="date", how="left")
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_guba(self, df: pd.DataFrame,
                    guba_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_guba:
            return df
        if guba_df is None or guba_df.empty:
            self._warn_if_missing("guba")
            return df
        g = guba_df.copy()
        g["date"] = pd.to_datetime(g["date"])
        available = [c for c in GUBA_COLS if c in g.columns]
        extra = [c for c in g.columns
                 if c not in GUBA_COLS and c not in ("date", "stock_code")
                 and not c.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(g[["date"] + available + extra], on="date", how="left")
        for col in available + extra:
            if col == "has_guba_post":
                df[col] = df[col].fillna(False).astype(bool)
            elif col == "guba_post_count":
                df[col] = df[col].fillna(0).astype("int16")
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: sentiment[t-1] paired with price[t]
        for col in available + extra:
            df[col] = df[col].shift(1)
        df["has_guba_post"] = df["has_guba_post"].fillna(False).astype(bool)
        df["guba_post_count"] = df["guba_post_count"].fillna(0).astype("int16")
        for col in available + extra:
            if col not in ("has_guba_post", "guba_post_count") and col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_comment(self, df: pd.DataFrame,
                       comment_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_comment:
            return df
        if comment_df is None or comment_df.empty:
            self._warn_if_missing("comment")
            return df
        c = comment_df.copy()
        c["date"] = pd.to_datetime(c["date"])
        available = [col for col in COMMENT_COLS if col in c.columns]
        extra = [col for col in c.columns
                 if col not in COMMENT_COLS and col not in ("date", "stock_code")
                 and not col.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(c[["date"] + available + extra], on="date", how="left")
        for col in available + extra:
            if col == "has_comment":
                df[col] = df[col].fillna(False).astype(bool)
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: comment data[t-1] paired with price[t]
        for col in available + extra:
            df[col] = df[col].shift(1)
        if "has_comment" in df.columns:
            df["has_comment"] = df["has_comment"].fillna(False).astype(bool)
        else:
            df["has_comment"] = df.get("comment_score", pd.Series(dtype=float)).notna()
        for col in COMMENT_COLS + extra:
            if col != "has_comment" and col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_xueqiu(self, df: pd.DataFrame,
                      xueqiu_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_xueqiu:
            return df
        if xueqiu_df is None or xueqiu_df.empty:
            self._warn_if_missing("xueqiu")
            return df
        x = xueqiu_df.copy()
        x["date"] = pd.to_datetime(x["date"])
        available = [c for c in XUEQIU_COLS if c in x.columns]
        extra = [c for c in x.columns
                 if c not in XUEQIU_COLS and c not in ("date", "stock_code")
                 and not c.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(x[["date"] + available + extra], on="date", how="left")
        for col in available + extra:
            if col == "has_xueqiu_post":
                df[col] = df[col].fillna(False).astype(bool)
            elif col == "xueqiu_post_count":
                df[col] = df[col].fillna(0).astype("int16")
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: sentiment[t-1] paired with price[t]
        for col in available + extra:
            df[col] = df[col].shift(1)
        df["has_xueqiu_post"] = df["has_xueqiu_post"].fillna(False).astype(bool)
        df["xueqiu_post_count"] = df["xueqiu_post_count"].fillna(0).astype("int16")
        for col in available + extra:
            if col not in ("has_xueqiu_post", "xueqiu_post_count") and col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    # ── Multi-shape preprocessing merge methods ──────────────────────

    def _merge_capital_flow(self, df: pd.DataFrame,
                            flow_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_capital_flow:
            return df
        if flow_df is None or flow_df.empty:
            self._warn_if_missing("capital_flow")
            return df
        return _merge_daily_aux(df, flow_df)

    def _merge_block_trade(self, df: pd.DataFrame,
                           bt_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_block_trade:
            return df
        if bt_df is None or bt_df.empty:
            self._warn_if_missing("block_trade")
            return df
        return _merge_daily_aux(df, bt_df)

    def _merge_shareholder(self, df: pd.DataFrame,
                           sh_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_shareholder:
            return df
        if sh_df is None or sh_df.empty:
            self._warn_if_missing("shareholder")
            return df
        return _merge_daily_aux(df, sh_df)

    def _merge_lockup(self, df: pd.DataFrame,
                      lu_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_lockup:
            return df
        if lu_df is None or lu_df.empty:
            self._warn_if_missing("lockup")
            return df
        return _merge_daily_aux(df, lu_df)

    def _merge_dividend(self, df: pd.DataFrame,
                        dv_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_dividend:
            return df
        if dv_df is None or dv_df.empty:
            self._warn_if_missing("dividend")
            return df
        return _merge_daily_aux(df, dv_df)

    def _merge_board(self, df: pd.DataFrame,
                     board_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_board:
            return df
        if board_df is None or board_df.empty:
            self._warn_if_missing("board")
            return df
        return _merge_daily_aux(df, board_df)

    def _merge_sector(self, df: pd.DataFrame,
                      sector_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_sector:
            return df
        if sector_df is None or sector_df.empty:
            self._warn_if_missing("sector")
            return df
        return _merge_daily_aux(df, sector_df)

    def _merge_concept(self, df: pd.DataFrame,
                       concept_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_concept:
            return df
        if concept_df is None or concept_df.empty:
            self._warn_if_missing("concept")
            return df
        # Aggregate from long format (one row per stock-board-date) to wide
        # (one row per stock-date) before merging.
        if "board_name" in concept_df.columns:
            concept_df = _aggregate_concept_long(concept_df)
        return _merge_daily_aux(df, concept_df)

    def _merge_macro(self, df: pd.DataFrame,
                     macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.use_macro:
            return df
        if macro_df is None:
            macro_df = getattr(self, '_macro_cache', None)
            if macro_df is None:
                import os
                from stoke_ml.config import load_config
                cfg = load_config()
                path = os.path.join(cfg.project.data_dir, "a_shares", "macro", "macro_daily.parquet")
                if not os.path.exists(path):
                    self._warn_if_missing("macro")
                    return df
                macro_df = pd.read_parquet(path)
                self._macro_cache = macro_df
        if macro_df.empty:
            return df
        macro = macro_df.reset_index() if macro_df.index.name == "date" else macro_df.copy()
        if "date" not in macro.columns:
            if isinstance(macro.index, pd.DatetimeIndex):
                macro = macro.reset_index()
                macro = macro.rename(columns={"index": "date"})
            else:
                return df
        macro["date"] = pd.to_datetime(macro["date"])
        available = [c for c in MACRO_COLS if c in macro.columns]
        if not available:
            return df
        df = df.merge(macro[["date"] + available], on="date", how="left")
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: macro data available after close → shift to t+1
        for col in available:
            df[col] = df[col].shift(1)
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    def _merge_industry(self, df: pd.DataFrame,
                        industry_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Merge cross-sectional industry stats (no per-stock mapping needed)."""
        if not self.use_industry:
            return df
        if industry_df is None:
            industry_df = self._industry_cache
            if industry_df is None:
                import os
                from stoke_ml.config import load_config
                cfg = load_config()
                path = os.path.join(cfg.project.data_dir, "a_shares", "industry", "industry_returns.parquet")
                if not os.path.exists(path):
                    self._warn_if_missing("industry")
                    return df
                raw = pd.read_parquet(path)
                # Compute cross-sectional stats from 90 industry returns
                industry_df = pd.DataFrame({
                    "date": pd.to_datetime(raw.index),
                    "ind_pct_up": (raw > 0).sum(axis=1).values / raw.notna().sum(axis=1).values,
                    "ind_return_mean": raw.mean(axis=1).values,
                    "ind_return_std": raw.std(axis=1).values,
                    "ind_return_max": raw.max(axis=1).values,
                    "ind_return_min": raw.min(axis=1).values,
                    "ind_return_skew": raw.skew(axis=1).values,
                })
                # Rolling dispersion: 20-day std of cross-sectional std
                industry_df["ind_dispersion_20d"] = (
                    industry_df["ind_return_std"].rolling(20).std().fillna(0.0)
                )
                for col in INDUSTRY_COLS:
                    industry_df[col] = industry_df[col].astype(np.float32)
                self._industry_cache = industry_df
        if industry_df.empty:
            return df
        ind = industry_df.copy()
        ind["date"] = pd.to_datetime(ind["date"])
        available = [c for c in INDUSTRY_COLS if c in ind.columns]
        if not available:
            return df
        df = df.merge(ind[["date"] + available], on="date", how="left")
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        for col in available:
            df[col] = df[col].shift(1)
        for col in available:
            df[col] = df[col].fillna(0.0).astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Microstructure features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features from OHLCV data.

        Adds: limit_up, limit_down, gap_up_pct, gap_down_pct,
        volume_ratio_20, turnover_anomaly.
        """
        df = df.copy()
        close = df.get("close")
        _open = df.get("open")
        volume = df.get("volume")
        if close is None:
            return df

        prev_close = close.shift(1)

        # Limit up/down (A-share: 10% daily limit)
        pct = (close - prev_close) / prev_close.replace(0, np.nan)
        df["is_limit_up"] = (pct >= 0.098).astype(np.float32)
        df["is_limit_down"] = (pct <= -0.098).astype(np.float32)

        # Gap open
        if _open is not None:
            gap = (_open - prev_close) / prev_close.replace(0, np.nan)
            df["gap_up_pct"] = gap.clip(lower=0).fillna(0).astype(np.float32)
            df["gap_down_pct"] = (-gap).clip(lower=0).fillna(0).astype(np.float32)

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

        return df

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    @staticmethod
    def _prep_feature_df(df: pd.DataFrame) -> pd.DataFrame:
        """Drop metadata columns and rows with inf/NaN — shared by sequencing methods."""
        drop_cols = ["stock_code", "sector", "size_proxy"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
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
        aligned_close = close[self.seq_len - 1: self.seq_len - 1 + n_samples + 1]
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

    def build_panel_features(
        self,
        panel: pd.DataFrame,
        target_col: str = "close",
        aux_data: dict[str, dict[str, pd.DataFrame]] | None = None,
    ) -> dict:
        """Build panel-format features for TFT training from a multi-stock panel.

        The input panel must have columns: date, stock_code, open, high, low,
        close, volume (plus any auxiliary feature columns already merged).

        Args:
            panel: multi-stock DataFrame with columns date, stock_code, OHLCV.
            target_col: column name for close price.
            aux_data: optional dict stock_code → {aux_type: DataFrame}.
                      aux_type keys: "sentiment", "guba", "xueqiu", "comment",
                      "announcement", "margin", "northbound", "dragon_tiger",
                      "fundamental", "etf_flow", "capital_flow", "block_trade",
                      "shareholder", "lockup", "dividend", "board", "sector", "concept".

        Returns:
            dict with numpy arrays: static_features, past_known, past_observed,
            y_direction, y_return, y_volatility.
        """
        codes = sorted(panel["stock_code"].unique())
        N = len(codes)
        aux_data = aux_data or {}

        # Engineer features per stock (reuses existing pipeline)
        all_feat_dfs = []
        for code in codes:
            mask = panel["stock_code"] == code
            df_stock = panel[mask].sort_values("date").reset_index(drop=True)
            stock_aux = aux_data.get(code, {})
            feats = self._engineer_features(
                df_stock,
                sentiment_df=stock_aux.get("sentiment"),
                guba_df=stock_aux.get("guba"),
                xueqiu_df=stock_aux.get("xueqiu"),
                comment_df=stock_aux.get("comment"),
                announcement_df=stock_aux.get("announcement"),
                margin_df=stock_aux.get("margin"),
                northbound_df=stock_aux.get("northbound"),
                dragon_tiger_df=stock_aux.get("dragon_tiger"),
                fundamental_df=stock_aux.get("fundamental"),
                etf_flow_df=stock_aux.get("etf_flow"),
                capital_flow_df=stock_aux.get("capital_flow"),
                block_trade_df=stock_aux.get("block_trade"),
                shareholder_df=stock_aux.get("shareholder"),
                lockup_df=stock_aux.get("lockup"),
                dividend_df=stock_aux.get("dividend"),
                board_df=stock_aux.get("board"),
                sector_df=stock_aux.get("sector"),
                concept_df=stock_aux.get("concept"),
                skip_temporal=True,  # TFT LSTM learns temporal patterns natively
            )
            # Calendar features are normally added by the temporal path;
            # we still want them when skip_temporal=True (TFT benefits from
            # day-of-week/month/quarter signals for seasonality).
            feats = add_calendar_features(feats)
            all_feat_dfs.append(feats)

        # Find common max length
        lengths = [len(df) for df in all_feat_dfs]
        max_T = min(max(lengths), 3000)  # cap at ~12 years

        if max_T < self.seq_len + 5:
            raise ValueError(
                f"Max timesteps ({max_T}) must be > seq_len+5 ({self.seq_len + 5})"
            )

        # Determine feature dimensions from first stock
        first_df = all_feat_dfs[0]
        static_cols_available = [c for c in _STATIC_FEATURE_COLS if c in first_df.columns]
        pk_cols_available = [c for c in _PAST_KNOWN_COLS if c in first_df.columns]
        po_cols_available = [c for c in _PAST_OBSERVED_COLS if c in first_df.columns]

        # Compute market_cap_quantile if not present (size proxy from avg close)
        if "market_cap_quantile" not in first_df.columns:
            stock_avg_closes = {}
            for i, df in enumerate(all_feat_dfs):
                if len(df) > 0 and "close" in df.columns:
                    # Use first 20 days average as size proxy — zero look-ahead
                    stock_avg_closes[codes[i]] = df["close"].iloc[:min(20, len(df))].mean()
            if stock_avg_closes:
                all_avgs = np.array(list(stock_avg_closes.values()))
                quantiles = np.searchsorted(
                    np.sort(all_avgs), all_avgs
                ).astype(np.float32) / len(all_avgs)
                cap_map = {c: q for c, q in zip(stock_avg_closes.keys(), quantiles)}
                for i, df in enumerate(all_feat_dfs):
                    df["market_cap_quantile"] = cap_map.get(codes[i], 0.5)
                static_cols_available = [c for c in _STATIC_FEATURE_COLS
                                         if c in first_df.columns or c in all_feat_dfs[0].columns]

        # ── Per-date cross-sectional z-score normalization ──
        # Normalize each feature across stocks within each date, so that
        # a feature's value is expressed relative to the cross-section that
        # day.  This avoids pooling future dates' statistics into today's
        # normalized value and is the standard panel-finance treatment.
        norm_cols = pk_cols_available + po_cols_available
        all_feat = pd.concat([
            df[["date"] + norm_cols]
            for df in all_feat_dfs
            if len(df) > 0
        ], ignore_index=True)

        date_stats: dict[str, pd.DataFrame] = {}
        for col in norm_cols:
            if col not in all_feat.columns:
                continue
            stats = all_feat.groupby("date")[col].agg(["mean", "std"])
            stats["std"] = stats["std"].fillna(1.0).clip(lower=1e-8)
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

        # Pre-allocate
        static_arr = np.zeros((N, static_dim), dtype=np.float32)
        pk_arr = np.zeros((N, max_T, pk_dim), dtype=np.float32)
        po_arr = np.zeros((N, max_T, po_dim), dtype=np.float32)
        y_dir_arr = np.zeros((N, max_T), dtype=np.int64)
        y_ret_arr = np.zeros((N, max_T), dtype=np.float32)
        y_vol_arr = np.zeros((N, max_T), dtype=np.float32)

        for i, df in enumerate(all_feat_dfs):
            if len(df) == 0:
                continue

            df = df.sort_values("date").reset_index(drop=True)
            T_i = min(len(df), max_T)

            # Static: take first row values
            if static_dim > 0:
                static_arr[i] = df[static_cols_available].iloc[0].fillna(0.0).values.astype(np.float32)

            # Past known
            pk_arr[i, :T_i] = df[pk_cols_available].fillna(0.0).values[:T_i].astype(np.float32)

            # Past observed
            po_arr[i, :T_i] = df[po_cols_available].fillna(0.0).values[:T_i].astype(np.float32)

            # Targets
            close = df[target_col].values[:T_i]
            ret_1d = np.full(T_i, np.nan, dtype=np.float32)
            ret_1d[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
            # Direction label with 0.3% noise threshold (3-class: up/flat/down)
            y_dir_arr[i, :T_i] = np.where(
                ret_1d > 0.003, 2, np.where(ret_1d < -0.003, 0, 1),
            ).astype(np.int64)
            y_ret_arr[i, :T_i] = ret_1d

            # 5-day realized volatility (lagged: excludes current return)
            for t in range(6, T_i):
                y_vol_arr[i, t] = float(np.std(ret_1d[t - 5:t]))  # ret_1d[t-5:t] excludes ret_1d[t]

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

        return {
            "static_features": static_arr,
            "past_known": pk_arr,
            "past_observed": po_arr,
            "y_direction": y_dir_arr,
            "y_return": y_ret_arr,
            "y_volatility": y_vol_arr,
        }


# ── TFT feature column definitions ──────────────────────────────────────

_STATIC_FEATURE_COLS = [
    "market_cap_quantile",
]

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


_PAST_KNOWN_COLS = [
    # OHLCV
    "open", "high", "low", "close", "volume",
    # Moving averages
    "ma_5", "ma_10", "ma_20", "ma_60", "ma_120",
    "ema_12", "ema_26",
    # MACD
    "macd_dif", "macd_dea", "macd_hist",
    # RSI
    "rsi_6", "rsi_12", "rsi_24",
    # KDJ (9-day and 14-day)
    "kdj_k_9", "kdj_d_9", "kdj_j_9",
    "kdj_k_14", "kdj_d_14", "kdj_j_14",
    # Bollinger Bands
    "boll_mid", "boll_upper", "boll_lower", "boll_pct",
    # ATR
    "atr_14",
    # ROC
    "roc_6", "roc_12", "roc_20",
    # Williams %R
    "wr_10", "wr_20",
    # CCI
    "cci_14", "cci_20",
    # Historical volatility
    "vol_5", "vol_20",
    # Volume
    "volume_ma5", "volume_ratio", "vol_up_ratio_20", "obv",
    # Amount (conditional on availability)
    "amount_ma5", "amount_ratio", "turnover_proxy",
    # K-bar microstructural (Alpha158 K-series, 9 factors)
    "kmid", "klen", "kmid2", "kup", "kup2",
    "klow", "klow2", "ksft", "ksft2",
    # Price standardization (Alpha158, 3 factors)
    "open0", "high0", "low0",
    # ADX family (trend strength)
    "adx", "adxr", "pdi", "mdi",
    # MFI / CMO / TRIX
    "mfi_14", "cmo_14", "trix",
    # Microstructure
    "is_limit_up", "is_limit_down", "gap_up_pct", "gap_down_pct",
    "volume_ratio_20", "volume_anomaly", "limit_up_streak",
    # Calendar
    "day_of_week", "day_of_month", "month", "quarter",
    # Fundamental (forward-filled quarterly)
    "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
    "debt_ratio", "gross_margin", "net_margin",
] + _alpha158_factor_names()

_PAST_OBSERVED_COLS = [
    # Sentiment (news)
    "sentiment_mean", "sentiment_std", "news_count",
    "positive_ratio", "negative_ratio",
    # Guba
    "guba_sentiment_mean", "guba_sentiment_std",
    "guba_positive_ratio", "guba_negative_ratio", "guba_post_count",
    # Xueqiu
    "xueqiu_sentiment_mean", "xueqiu_sentiment_std",
    "xueqiu_positive_ratio", "xueqiu_negative_ratio", "xueqiu_post_count",
    # Comment
    "comment_score", "comment_attention", "comment_institution", "comment_trend",
    # Announcement
    "ann_sentiment_mean", "ann_sentiment_std",
    # Margin
    "margin_balance", "margin_buy", "short_balance", "margin_net",
    # Northbound
    "north_hold_pct", "north_net_buy",
    # Dragon Tiger
    "lhb_net_amount", "lhb_buy_ratio",
    # ETF flow
    "sector_etf_flow", "sector_etf_amount",
    # Capital flow
    "flow_intensity", "flow_z", "flow_momentum",
    # Board
    "is_zt", "is_zb", "board_height_20d", "seal_strength",
    # Sector
    "sector_relative_strength", "sector_breadth_z",
    # Concept
    "avg_concept_heat", "is_concept_leader",
]


def _active_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the subset of *candidates* that exist in *df*."""
    return [c for c in candidates if c in df.columns]


def _merge_daily_aux(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """Merge a preprocessed auxiliary DataFrame on date with ZI fill + PIT lag.

    Any column that exists in *aux* (except date, stock_code, has_* flags)
    is merged and lagged by 1 trading day.
    """
    a = aux.copy()
    a["date"] = pd.to_datetime(a["date"])
    # Drop stock-level columns — we merge on date only
    a = a.drop(columns=["stock_code"], errors="ignore")
    a = a.drop_duplicates(subset="date", keep="last")

    skip = {"date", "stock_code"}
    available = [c for c in a.columns if c not in skip]
    if not available:
        return df

    df = df.merge(a[["date"] + available], on="date", how="left")
    for col in available:
        if col.startswith("has_"):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0).astype(np.int8)
            else:
                df[col] = df[col].fillna(False).astype(bool)
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0).astype("int16")
        elif col.endswith("_streak") or col.endswith("_quadrant"):
            df[col] = df[col].fillna(0).astype(np.int16)
        else:
            df[col] = df[col].fillna(0.0).astype(np.float32)

    # PIT lag: feature[t-1] paired with price[t]
    for col in available:
        df[col] = df[col].shift(1)

    # Re-fill after shift (first row becomes NaN)
    for col in available:
        if col.startswith("has_"):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0).astype(np.int8)
            else:
                df[col] = df[col].fillna(False).astype(bool)
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0).astype("int16")
        elif col.endswith("_streak") or col.endswith("_quadrant"):
            df[col] = df[col].fillna(0).astype(np.int16)
        else:
            df[col] = df[col].fillna(0.0).astype(np.float32)

    return df


def _aggregate_concept_long(concept_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate concept data from long format to per-stock-per-date.

    ConceptBlockEncoder outputs one row per (date, stock_code, board_name).
    Multi-hot columns (cb_*) and per-board momentum columns need to be
    collapsed to a single row per (date, stock_code) before merging with
    the main feature DataFrame.
    """
    agg_spec = {}
    # Multi-hot: max works as OR (1 if any board has the flag)
    cb_cols = [c for c in concept_df.columns if c.startswith("cb_")]
    agg_spec.update({c: (c, "max") for c in cb_cols})
    # Per-board momentum: average across boards
    mom_cols = [c for c in concept_df.columns if c.startswith("concept_momentum_")]
    agg_spec.update({c: (c, "mean") for c in mom_cols})
    bmom_cols = [c for c in concept_df.columns if c.startswith("board_momentum_")]
    agg_spec.update({c: (c, "mean") for c in bmom_cols})
    # Per-stock columns: same value across rows (take first)
    static_cols = [
        c for c in concept_df.columns
        if c not in {"date", "stock_code", "board_name"}
        and c not in cb_cols
        and c not in mom_cols
        and c not in bmom_cols
    ]
    agg_spec.update({c: (c, "first") for c in static_cols})

    key_cols = ["date", "stock_code"]
    available = [c for c in key_cols if c in concept_df.columns]
    return (
        concept_df.groupby(available, as_index=False)
        .agg(**agg_spec)
        .reset_index(drop=True)
    )
