"""Feature pipeline orchestrating all feature engineering steps.

Integrates K-line, sentiment, market-wide (margin/northbound/dragon-tiger),
ETF sector flow, and fundamental data into a unified feature set.
"""
import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
from stoke_ml.features.interaction import InteractionFeatures
from stoke_ml.features.temporal import (
    add_lag_features, add_rolling_features, add_calendar_features,
)

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
        self.feature_selection_k = feature_selection_k
        self.use_new_preprocessing = use_new_preprocessing
        self._preprocessing_config = preprocessing_config
        self._preprocessing = None
        if use_new_preprocessing and preprocessing_config:
            self._preprocessing = self._build_preprocessing()
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()
        self._interaction = InteractionFeatures()

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

        if self.use_technical:
            df = self._ti.compute_all(df)
        if self.use_scoring:
            df = self._scorer.score(df)

        df = self._add_microstructure(df)

        if self.use_interaction:
            df = self._interaction.compute_all(df)

        if self.use_temporal:
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
        if not (self.use_sentiment and sentiment_df is not None
                and not sentiment_df.empty):
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
        if not (self.use_announcements and announcement_df is not None
                and not announcement_df.empty):
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
        if not (self.use_margin and margin_df is not None
                and not margin_df.empty):
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
        return df

    def _merge_northbound(self, df: pd.DataFrame,
                          northbound_df: pd.DataFrame | None) -> pd.DataFrame:
        if not (self.use_northbound and northbound_df is not None
                and not northbound_df.empty):
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
        return df

    def _merge_dragon_tiger(self, df: pd.DataFrame,
                            dt_df: pd.DataFrame | None) -> pd.DataFrame:
        if not (self.use_dragon_tiger and dt_df is not None
                and not dt_df.empty):
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
        return df

    def _merge_fundamental(self, df: pd.DataFrame,
                           fundamental_df: pd.DataFrame | None) -> pd.DataFrame:
        if not (self.use_fundamental and fundamental_df is not None
                and not fundamental_df.empty):
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
        if not (self.use_etf_flow and etf_flow_df is not None
                and not etf_flow_df.empty):
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
        if not (self.use_guba and guba_df is not None
                and not guba_df.empty):
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
        if not (self.use_comment and comment_df is not None
                and not comment_df.empty):
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
        if not (self.use_xueqiu and xueqiu_df is not None
                and not xueqiu_df.empty):
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
            vol_med = volume.rolling(20, min_periods=5).median()
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

    def _create_sequences(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        drop_cols = ["date", "stock_code", "sector", "size_proxy"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # Replace inf with NaN so dropna handles both
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = feat_df.dropna()

        close = feat_df[target_col].values
        target = (close[self.horizon:] > close[: -self.horizon]).astype(int)

        price_cols = ["open", "high", "low", "close", "amount"]
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
        aligned_close = close[self.seq_len - 1: self.seq_len + n_samples]
        return X, y, aligned_close.astype(np.float32)

    def _get_sample_dates(self, feats: pd.DataFrame) -> np.ndarray:
        """Return the prediction date for each sample after dropna + sequencing.

        Must match _create_sequences exactly in the rows it keeps.
        """
        drop_cols = ["date", "stock_code", "sector", "size_proxy"]
        # Reconstruct the dropna mask (same as _create_sequences)
        feat_df = feats.drop(columns=[c for c in drop_cols if c in feats.columns])
        valid_mask = feat_df.notna().all(axis=1)
        # Get dates for valid rows
        valid_dates = feats.loc[valid_mask.values, "date"].values
        if len(valid_dates) < self.seq_len + self.horizon:
            return np.array([], dtype="datetime64[ns]")
        n_samples = len(valid_dates) - self.seq_len - self.horizon + 1
        # Sample i predicts return ending at valid_dates[seq_len-1+i+horizon]
        return valid_dates[self.seq_len - 1 + self.horizon:
                           self.seq_len - 1 + self.horizon + n_samples]


def _active_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the subset of *candidates* that exist in *df*."""
    return [c for c in candidates if c in df.columns]
