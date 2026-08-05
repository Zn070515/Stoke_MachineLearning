"""Auxiliary-data alignment for the feature pipeline (§十七-1).

Extracted from ``stoke_ml.features.pipeline``: every per-stock aux merge
(sentiment / margin / guba / ...) lives in :class:`AuxAligner`, which owns the
per-dimension ``use_*`` switches and the lazy-loaded market-wide caches (macro /
market-env / industry), and delegates the ZI fill → PIT lag → ZI fill
choreography to the helpers in ``stoke_ml.features.aux_helpers``.

``FeaturePipeline`` delegates to ``AuxAligner.merge_all`` during
``_engineer_features``; the column constants are re-imported there for the
temporal-feature assembly.
"""
import logging

import numpy as np
import pandas as pd

from stoke_ml.features.aux_cols import (
    SENTIMENT_COLS,
    MARGIN_COLS,
    NORTHBOUND_COLS,
    DRAGON_TIGER_COLS,
    ETF_FLOW_COLS,
    GUBA_COLS,
    COMMENT_COLS,
    FUNDAMENTAL_COLS,
    EARNINGS_COLS,
    VALUATION_COLS,
    INDUSTRY_COLS,
    MACRO_COLS,
    MARKET_ENV_COLS,
    STATE_MAX_STALENESS,
)
from stoke_ml.features.aux_helpers import (
    _append_state_staleness,  # noqa: F401  re-exported for import-compat (used only inside aux_helpers)
    _batch_fill_shift,
    _merge_daily_aux,
    _aggregate_concept_long,
    _load_macro_features,
)

logger = logging.getLogger(__name__)


class AuxAligner:
    """Align auxiliary per-stock data onto a daily K-line frame.

    One ``_merge_<source>`` method per data dimension; each returns the frame
    with that dimension's columns merged on ``date`` and ZI-filled / PIT-lagged
    as documented per method.  ``merge_all`` wires them in the same order the
    original ``FeaturePipeline._engineer_features`` dispatch used.
    """

    AUX_KEYS = (
        "sentiment", "announcements", "margin", "northbound", "dragon_tiger",
        "fundamental", "earnings", "valuation", "etf_flow", "guba", "comment",
        "capital_flow", "block_trade", "shareholder", "lockup", "dividend",
        "board", "sector", "concept", "macro", "industry", "limit_up",
        "pledge", "market_env", "index_membership",
    )

    def __init__(self, flags: dict[str, bool] | None = None):
        flags = dict(flags or {})
        for key in self.AUX_KEYS:
            setattr(self, f"use_{key}", bool(flags.get(key, True)))
        self._warned_missing: set[str] = set()
        self._macro_cache: pd.DataFrame | None = None
        self._industry_cache: pd.DataFrame | None = None
        self._market_env_cache: pd.DataFrame | None = None

    def _warn_if_missing(self, key: str) -> None:
        """Emit one-time debug log when use_*=True but no data was passed.

        Many data types (lockup, shareholder, block_trade, etc.) are sparse
        by nature — only a subset of stocks or dates have records.  This is
        expected, not a problem, so we log at DEBUG instead of WARNING to
        avoid noise during training runs.
        """
        if key not in self._warned_missing:
            logger.debug("use_%s=True but no %s data for this stock", key, key)
            self._warned_missing.add(key)

    def merge_all(
        self,
        df: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
        announcements: pd.DataFrame | None = None,
        margin: pd.DataFrame | None = None,
        northbound: pd.DataFrame | None = None,
        dragon_tiger: pd.DataFrame | None = None,
        fundamental: pd.DataFrame | None = None,
        earnings: pd.DataFrame | None = None,
        valuation: pd.DataFrame | None = None,
        etf_flow: pd.DataFrame | None = None,
        guba: pd.DataFrame | None = None,
        comment: pd.DataFrame | None = None,
        capital_flow: pd.DataFrame | None = None,
        block_trade: pd.DataFrame | None = None,
        shareholder: pd.DataFrame | None = None,
        lockup: pd.DataFrame | None = None,
        dividend: pd.DataFrame | None = None,
        board: pd.DataFrame | None = None,
        sector: pd.DataFrame | None = None,
        concept: pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
        industry: pd.DataFrame | None = None,
        limit_up: pd.DataFrame | None = None,
        pledge: pd.DataFrame | None = None,
        market_env: pd.DataFrame | None = None,
        index_membership: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = self._merge_sentiment(df, sentiment)
        df = self._merge_announcements(df, announcements)
        df = self._merge_margin(df, margin)
        df = self._merge_northbound(df, northbound)
        df = self._merge_dragon_tiger(df, dragon_tiger)
        df = self._merge_fundamental(df, fundamental)
        df = self._merge_earnings(df, earnings)
        df = self._merge_valuation(df, valuation)
        df = self._merge_etf_flow(df, etf_flow)
        df = self._merge_guba(df, guba)
        df = self._merge_comment(df, comment)
        df = self._merge_capital_flow(df, capital_flow)
        df = self._merge_block_trade(df, block_trade)
        df = self._merge_shareholder(df, shareholder)
        df = self._merge_lockup(df, lockup)
        df = self._merge_dividend(df, dividend)
        df = self._merge_board(df, board)
        df = self._merge_sector(df, sector)
        df = self._merge_concept(df, concept)
        df = self._merge_macro(df, macro)
        df = self._merge_industry(df, industry)

        # _merge_limit_up is DEFERRED (limit-up ecology family, top scope note):
        # the method exists but is intentionally NOT wired.
        df = self._merge_pledge(df, pledge)
        df = self._merge_market_env(df, market_env)
        df = self._merge_index_membership(df, index_membership)
        return df

    def _merge_sentiment(self, df: pd.DataFrame,
                         sentiment_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_sentiment:
            return df
        if sentiment_df is None or sentiment_df.empty:
            self._warn_if_missing("sentiment")
            return df
        s = sentiment_df.copy()
        s["date"] = pd.to_datetime(s["date"])
        s = s.drop_duplicates(subset="date", keep="last")
        available = [c for c in SENTIMENT_COLS if c in s.columns]
        extra = [c for c in s.columns
                 if c not in SENTIMENT_COLS and c not in ("date", "stock_code")
                 and not c.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(s[["date"] + available + extra], on="date", how="left")
        # NewsStorage maps post-close → next trading day; date is already the
        # effective_trade_date, so no extra shift.
        _batch_fill_shift(df, available + extra, lag=False)
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
        _batch_fill_shift(df, merged_cols)
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
        # Margin balance is a state: a missing day means "unchanged", never 0.
        _batch_fill_shift(df, available, policy="ffill", prefix="margin",
                          max_staleness=STATE_MAX_STALENESS["margin"])
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
        # Holdings are a state snapshot — forward-fill gaps, never zero.
        _batch_fill_shift(df, available, policy="ffill", prefix="northbound",
                          max_staleness=STATE_MAX_STALENESS["northbound"])
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
        reason = dt.get("lhb_reason",
                        pd.Series(index=dt.index, dtype=str)).fillna("").astype(str)
        dt["lhb_is_wave"] = reason.str.contains("振幅|换手", regex=True)
        dt["lhb_is_sustained"] = reason.str.contains("连续", regex=False)
        dt["lhb_is_drop"] = reason.str.contains("跌幅|跌停|下跌", regex=True)
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
            lhb_is_wave=("lhb_is_wave", "any"),
            lhb_is_sustained=("lhb_is_sustained", "any"),
            lhb_is_drop=("lhb_is_drop", "any"),
        ).reset_index()
        agg["lhb_present"] = (agg["lhb_present"] > 0).astype(np.float32)
        agg["lhb_buy_ratio"] = agg["lhb_buy_ratio"].fillna(0.5).astype(np.float32)
        agg["lhb_net_amount"] = agg["lhb_net_amount"].fillna(0.0).astype(np.float32)
        for c in ("lhb_is_wave", "lhb_is_sustained", "lhb_is_drop"):
            agg[c] = agg[c].fillna(False).astype(np.float32)
        df = df.merge(agg, on="date", how="left")
        flag_cols = ["lhb_is_wave", "lhb_is_sustained", "lhb_is_drop"]
        _batch_fill_shift(df, [c for c in DRAGON_TIGER_COLS if c in df.columns]
                          + [c for c in flag_cols if c in df.columns])
        # Past-5-trading-day LHB frequency (computed AFTER the PIT shift, so it
        # never looks ahead; must NOT be shifted again).
        if "lhb_present" in df.columns:
            df["lhb_count_5d"] = df["lhb_present"].rolling(5, min_periods=1).sum().astype("int16")
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
        # Fundamentals are a state snapshot — forward-fill any residual gap.
        _batch_fill_shift(df, available, policy="ffill", prefix="fundamental",
                          max_staleness=STATE_MAX_STALENESS["fundamental"])
        return df

    def _merge_earnings(self, df: pd.DataFrame,
                        earnings_df: pd.DataFrame | None) -> pd.DataFrame:
        """Merge the per-stock daily earnings-active frame (see EarningsStorage).

        The storage already forward-fills the net-profit band across trading
        days (a forecast stays active until superseded) and maps every
        announce_date to its next-trading-day ``effective_trade_date``; this
        method only merges on date and ZI-fills — no extra shift, so the signal
        first appears exactly on its effective date.
        """
        if not self.use_earnings:
            return df
        if earnings_df is None or earnings_df.empty:
            self._warn_if_missing("earnings")
            return df
        ed = earnings_df.copy()
        ed["date"] = pd.to_datetime(ed["date"])
        ed = ed.drop_duplicates(subset="date", keep="last")
        available = [c for c in EARNINGS_COLS if c in ed.columns]
        if not available:
            return df
        df = df.merge(ed[["date"] + available], on="date", how="left")
        # date is the storage-mapped effective_trade_date → no extra shift.
        # A forecast band is a state that persists until superseded → ffill.
        _batch_fill_shift(df, available, lag=False, policy="ffill",
                          prefix="earnings",
                          max_staleness=STATE_MAX_STALENESS["earnings"])
        return df

    def _merge_valuation(self, df: pd.DataFrame,
                         valuation_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_valuation:
            return df
        if valuation_df is None or valuation_df.empty:
            self._warn_if_missing("valuation")
            return df
        vd = valuation_df.copy()
        vd["date"] = pd.to_datetime(vd["date"])
        vd = vd.drop_duplicates(subset="date", keep="last")
        available = [c for c in VALUATION_COLS if c in vd.columns]
        if not available:
            return df
        df = df.merge(vd[["date"] + available], on="date", how="left")
        # Valuation ratios are a state snapshot — forward-fill, never zero.
        _batch_fill_shift(df, available, policy="ffill", prefix="valuation",
                          max_staleness=STATE_MAX_STALENESS["valuation"])
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
        _batch_fill_shift(df, available)
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
        g = g.drop_duplicates(subset="date", keep="last")
        available = [c for c in GUBA_COLS if c in g.columns]
        extra = [c for c in g.columns
                 if c not in GUBA_COLS and c not in ("date", "stock_code")
                 and not c.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(g[["date"] + available + extra], on="date", how="left")
        # GubaStorage maps post-close → next trading day; date is already the
        # effective_trade_date, so no extra shift.
        _batch_fill_shift(df, available + extra, lag=False)
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
        c = c.drop_duplicates(subset="date", keep="last")
        available = [col for col in COMMENT_COLS if col in c.columns]
        extra = [col for col in c.columns
                 if col not in COMMENT_COLS and col not in ("date", "stock_code")
                 and not col.startswith("has_")]
        if not available and not extra:
            return df
        df = df.merge(c[["date"] + available + extra], on="date", how="left")
        _batch_fill_shift(df, available + extra)
        # Guard: ensure has_comment exists (may be absent in sparse comment data)
        if "has_comment" not in df.columns:
            df["has_comment"] = df.get("comment_score", pd.Series(dtype=float)).notna()
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
        # Shareholder count is a state snapshot between disclosures.
        return _merge_daily_aux(df, sh_df, policy="ffill", prefix="shareholder",
                                max_staleness=STATE_MAX_STALENESS["shareholder"])

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

    def _merge_limit_up(self, df: pd.DataFrame,
                        limit_up_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_limit_up:
            return df
        if limit_up_df is None or limit_up_df.empty:
            self._warn_if_missing("limit_up")
            return df
        return _merge_daily_aux(df, limit_up_df)

    def _merge_pledge(self, df: pd.DataFrame,
                      pledge_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_pledge:
            return df
        if pledge_df is None or pledge_df.empty:
            self._warn_if_missing("pledge")
            return df
        # Pledge ratio is outstanding state — forward-fill between records.
        return _merge_daily_aux(df, pledge_df, policy="ffill", prefix="pledge",
                                max_staleness=STATE_MAX_STALENESS["pledge"])

    def _merge_index_membership(self, df: pd.DataFrame,
                                im_df: pd.DataFrame | None) -> pd.DataFrame:
        # §P1-9 (PIT caution): the index-membership source is a Baostock
        # MONTHLY-SNAPSHOT rebuild — each stock's in_date/out_date are the
        # boundaries of the monthly snapshot in which membership changed, NOT
        # the exact trading day the index committee acted.  A stock whose
        # membership flips mid-month therefore appears in/out only at the next
        # snapshot boundary, so a daily feature using these columns is a
        # MONTHLY approximation of true membership and can be late (or early)
        # by up to a month.  Treat index_membership features as slow-moving
        # state (like sector), not day-exact events; do not make decisions that
        # hinge on a single day's membership flag.
        if not self.use_index_membership:
            return df
        if im_df is None or im_df.empty:
            self._warn_if_missing("index_membership")
            return df
        return _merge_daily_aux(df, im_df)

    def _merge_macro(self, df: pd.DataFrame,
                     macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.use_macro:
            return df
        if macro_df is None:
            macro_df = getattr(self, '_macro_cache', None)
            if macro_df is None:
                from stoke_ml.config import load_config
                macro_df = _load_macro_features(load_config().project.data_dir)
                if macro_df is None:
                    self._warn_if_missing("macro")
                    return df
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
        # Macro rates/levels are state — forward-fill gaps, never zero.
        _batch_fill_shift(df, available, policy="ffill", prefix="macro",
                          max_staleness=STATE_MAX_STALENESS["macro"])
        return df

    def _merge_market_env(self, df: pd.DataFrame,
                          market_env_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.use_market_env:
            return df
        if market_env_df is None:
            market_env_df = self._market_env_cache
            if market_env_df is None:
                import os
                from stoke_ml.config import load_config
                cfg = load_config()
                path = os.path.join(cfg.project.data_dir, "a_shares", "market_breadth",
                                    "market_env_daily.parquet")
                if not os.path.exists(path):
                    self._warn_if_missing("market_env")
                    return df
                market_env_df = pd.read_parquet(path)
                self._market_env_cache = market_env_df
        if market_env_df is None or market_env_df.empty:
            return df
        me = market_env_df.copy()
        # Mirror _merge_macro's defensive date handling: named "date" index,
        # unnamed DatetimeIndex, or date-as-column all resolve to a date col.
        if "date" not in me.columns:
            if isinstance(me.index, pd.DatetimeIndex):
                me = me.reset_index()
                me = me.rename(columns={"index": "date"})
            else:
                return df
        me["date"] = pd.to_datetime(me["date"]).dt.normalize()
        me = me.drop_duplicates(subset="date", keep="last")
        available = [c for c in MARKET_ENV_COLS if c in me.columns]
        if not available:
            return df
        df = df.merge(me[["date"] + available], on="date", how="left")
        # Market-breadth stats are state — forward-fill gaps, never zero.
        _batch_fill_shift(df, available, policy="ffill", prefix="market_env",
                          max_staleness=STATE_MAX_STALENESS["market_env"])
        return df

    def _merge_industry(self, df: pd.DataFrame,
                        industry_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Merge industry-level cross-sectional stats (NOT per-stock membership).

        The industry-level columns (ind_pct_up / ind_return_* / ind_dispersion_20d)
        are daily cross-sectional statistics over all industry indexes and are
        PIT-safe.  The per-stock industry-relative columns (ind_matched_return /
        stock_vs_industry) were removed: they mapped each
        stock onto its industry through the current-snapshot sector_map.json,
        backfilling today's classification onto historical rows.
        """
        if not self.use_industry:
            return df
        if industry_df is None:
            industry_df = self._industry_cache
            if industry_df is None:
                import os
                from stoke_ml.config import load_config
                cfg = load_config()
                ind_dir = os.path.join(cfg.project.data_dir, "a_shares", "industry")
                path = os.path.join(ind_dir, "industry_returns.parquet")
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
                ind_float_cols = [c for c in INDUSTRY_COLS if c in industry_df.columns]
                if ind_float_cols:
                    industry_df[ind_float_cols] = industry_df[ind_float_cols].astype(np.float32)
                self._industry_cache = industry_df
        if industry_df.empty:
            return df
        ind = industry_df.copy()
        ind["date"] = pd.to_datetime(ind["date"])
        available = [c for c in INDUSTRY_COLS if c in ind.columns]
        if not available:
            return df

        df = df.merge(ind[["date"] + available], on="date", how="left")
        # Industry cross-sectional stats are state — forward-fill, never zero.
        _batch_fill_shift(df, available, policy="ffill", prefix="industry",
                          max_staleness=STATE_MAX_STALENESS["industry"])
        return df
