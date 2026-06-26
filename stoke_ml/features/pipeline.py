"""Feature pipeline orchestrating all feature engineering steps.

Integrates K-line, sentiment, market-wide (margin/northbound/dragon-tiger),
ETF sector flow, and fundamental data into a unified feature set.
"""
import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
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

FUNDAMENTAL_COLS = [
    "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
    "debt_ratio", "gross_margin", "net_margin",
]

TEMPORAL_BASE_COLS = [
    "open", "high", "low", "close", "volume",
    "volume_ratio", "atr_14", "rsi_12",
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
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build features and return (X, y, aligned_close).

        All auxiliary DataFrames must have a 'date' column and be
        pre-filtered to the stock's date range.  Market data are merged
        by date; fundamentals should already be forward-filled to daily;
        ETF flow is merged by date after mapping stock to sector;
        guba_df provides Guba forum sentiment aggregated to daily.
        """
        feats = self._engineer_features(
            df, sentiment_df, margin_df, northbound_df,
            dragon_tiger_df, fundamental_df, etf_flow_df,
            announcement_df, guba_df,
        )
        X, y, aligned_close = self._create_sequences(feats, target_col)
        return X, y, aligned_close

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
    ) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        df = self._merge_sentiment(df, sentiment_df)
        df = self._merge_announcements(df, announcement_df)
        df = self._merge_margin(df, margin_df)
        df = self._merge_northbound(df, northbound_df)
        df = self._merge_dragon_tiger(df, dragon_tiger_df)
        df = self._merge_fundamental(df, fundamental_df)
        df = self._merge_etf_flow(df, etf_flow_df)
        df = self._merge_guba(df, guba_df)

        if self.use_technical:
            df = self._ti.compute_all(df)
        if self.use_scoring:
            df = self._scorer.score(df)

        df = self._add_microstructure(df)

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
        df = df.merge(s[["date"] + SENTIMENT_COLS], on="date", how="left")
        for col in SENTIMENT_COLS:
            if col == "has_news":
                df[col] = df[col].fillna(False).astype(bool)
            elif col == "news_count":
                df[col] = df[col].fillna(0).astype("int16")
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)

        # Lag sentiment by 1 trading day to prevent look-ahead bias.
        # Without timestamps, day-t news may include post-15:00 releases
        # the market hasn't absorbed yet. Using sentiment[t-1] with
        # price[t] ensures all news was fully priced before prediction.
        for col in SENTIMENT_COLS:
            df[col] = df[col].shift(1)
        df["has_news"] = df["has_news"].fillna(False).astype(bool)
        df["news_count"] = df["news_count"].fillna(0).astype("int16")
        for col in ["sentiment_mean", "sentiment_std",
                     "positive_ratio", "negative_ratio"]:
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
        available = {k: v for k, v in col_map.items() if k in a.columns}
        if not available:
            return df
        rename = {k: v for k, v in available.items()}
        a_renamed = a[["date"] + list(available.keys())].rename(columns=rename)
        df = df.merge(a_renamed, on="date", how="left")
        for _, target_col in available.items():
            if target_col == "has_announce":
                df[target_col] = df[target_col].fillna(False).astype(bool)
            elif "count" in target_col:
                df[target_col] = df[target_col].fillna(0).astype("int16")
            else:
                df[target_col] = df[target_col].fillna(0.0).astype(np.float32)
        # Same PIT lag as news sentiment
        for _, target_col in available.items():
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
        if margin_df is None or margin_df.empty:
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
        if northbound_df is None or northbound_df.empty:
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
        if dt_df is None or dt_df.empty:
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
        if fundamental_df is None or fundamental_df.empty:
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
        if etf_flow_df is None or etf_flow_df.empty:
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
        if not available:
            return df
        df = df.merge(g[["date"] + available], on="date", how="left")
        for col in available:
            if col == "has_guba_post":
                df[col] = df[col].fillna(False).astype(bool)
            elif col == "guba_post_count":
                df[col] = df[col].fillna(0).astype("int16")
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: guba sentiment[t-1] paired with price[t]
        for col in available:
            df[col] = df[col].shift(1)
        df["has_guba_post"] = df["has_guba_post"].fillna(False).astype(bool)
        df["guba_post_count"] = df["guba_post_count"].fillna(0).astype("int16")
        for col in ["guba_sentiment_mean", "guba_sentiment_std",
                     "guba_positive_ratio", "guba_negative_ratio"]:
            if col in df.columns:
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
        drop_cols = ["date", "stock_code"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
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


def _active_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the subset of *candidates* that exist in *df*."""
    return [c for c in candidates if c in df.columns]
